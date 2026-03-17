"""
Experiment 5: Adversarial Robustness of Self-Knowledge
======================================================

Tests whether models' self-knowledge survives adversarial pressure by re-running
Experiment 1 tasks under 4 different adversarial conditions.

Attack Conditions:
1. Authority Override - Claim model is expert in its WEAK domain
2. Social Pressure - State that others find task easy
3. Framing Manipulation - Present same task as easy vs. hard
4. Instruction Hijacking - Add expertise claim to system prompt

Usage:
  # Pilot (1 model, 1 attack, 10 questions)
  python scripts/run_experiment_5.py --mode pilot

  # Full run (all models, all attacks, 80 questions each)
  python scripts/run_experiment_5.py --mode full

  # Resume interrupted run
  python scripts/run_experiment_5.py --mode full --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import (
    build_channel1_prompt,
    build_channel2_prompt,
    build_channel4_prompt,
    build_channel5_prompt,
    parse_channel1,
    parse_channel2,
    parse_channel4,
    parse_channel5,
)
from mirror.scoring.answer_matcher import match_answer_robust


# Configuration
EXPERIMENT_CONFIG = {
    "pilot": {
        "n_questions": 10,
        "models": ["llama-3.1-8b"],
        "attacks": ["authority_override"],
        "description": "Pilot run - 1 model, 1 attack, 10 questions",
    },
    "full": {
        "n_questions": 80,
        "models": [
            "llama-3.1-8b",
            "llama-3.1-70b",
            # "llama-3.1-405b",  # 404 on NVIDIA NIM - model unavailable
            "mistral-large",
            "qwen-3-235b",
            "gpt-oss-120b",
            "deepseek-r1",
        ],
        "attacks": [
            "authority_override",
            "social_pressure",
            "framing_easy",
            "framing_hard",
        ],
        "description": "Full run - all models, all attacks",
    },
}

# Note: Channel 3 (difficulty_selection) requires question pairs, which complicates
# adversarial testing. We use the other 4 channels for robustness measurement.
ALL_CHANNELS = ["wagering", "opt_out", "tool_use", "natural"]

CHANNEL_BUILDERS = {
    "wagering": build_channel1_prompt,
    "opt_out": build_channel2_prompt,
    "tool_use": build_channel4_prompt,
    "natural": build_channel5_prompt,
}

CHANNEL_PARSERS = {
    "wagering": parse_channel1,
    "opt_out": parse_channel2,
    "tool_use": parse_channel4,
    "natural": parse_channel5,
}

DEFAULT_DELAY = 0.5
MAX_RETRIES = 5


CALL_TIMEOUT = 90  # seconds per API call before we give up
CONCURRENCY = 4   # parallel trials (safe for slow thinking models)


async def call_with_retry(client, model, messages, temperature, max_tokens, metadata):
    """Call client.complete with per-call timeout and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            coro = client.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=metadata,
            )
            response = await asyncio.wait_for(coro, timeout=CALL_TIMEOUT)
            # Retry on rate-limit errors returned in response dict
            if isinstance(response, dict) and "error" in response:
                err = str(response["error"])
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    wait = (2 ** attempt) * 2
                    print(f"\n    [rate limit, retry {attempt+1}/{MAX_RETRIES} in {wait}s]", end="", flush=True)
                    await asyncio.sleep(wait)
                    continue
            return response
        except asyncio.TimeoutError:
            wait = (2 ** attempt) * 2
            print(f"\n    [timeout {CALL_TIMEOUT}s, retry {attempt+1}/{MAX_RETRIES} in {wait}s]", end="", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            err = str(e)
            if attempt < MAX_RETRIES - 1 and (
                "429" in err or "rate" in err.lower() or "timeout" in err.lower()
                or "connection" in err.lower() or "503" in err or "502" in err
            ):
                wait = (2 ** attempt) * 2
                print(f"\n    [{type(e).__name__}, retry {attempt+1}/{MAX_RETRIES} in {wait}s]", end="", flush=True)
                await asyncio.sleep(wait)
            else:
                raise
    return {"error": f"Max retries ({MAX_RETRIES}) exceeded after timeout"}


def load_attack_templates() -> dict:
    """Load adversarial attack templates."""
    template_path = Path("data/attack_templates.json")
    with open(template_path) as f:
        return json.load(f)


def load_exp1_metrics() -> dict:
    """Load Experiment 1 metrics to identify weak/strong domains per model."""
    results_dir = Path("data/results")
    # Find latest exp1 accuracy file
    exp1_files = sorted(results_dir.glob("exp1_*_accuracy.json"),
                       key=lambda p: p.stat().st_mtime)
    if not exp1_files:
        raise FileNotFoundError("No Experiment 1 accuracy metrics found")

    with open(exp1_files[-1]) as f:
        return json.load(f)


def identify_weak_strong_domains(model: str, exp1_metrics: dict) -> tuple[list[str], list[str]]:
    """
    Identify weak and strong domains for a model based on Exp 1 accuracy.

    Returns:
        (weak_domains, strong_domains) - lists of 2 domain names each
    """
    if model not in exp1_metrics:
        # Fallback: use default domains
        return ["arithmetic", "logical"], ["linguistic", "procedural"]

    model_metrics = exp1_metrics[model]

    # Compute accuracy per domain from natural_acc
    domain_acc = {}
    for domain, channels in model_metrics.items():
        if isinstance(channels, dict):
            nat_acc = channels.get("natural_acc")
            if nat_acc is not None:
                domain_acc[domain] = nat_acc

    if not domain_acc:
        return ["arithmetic", "logical"], ["linguistic", "procedural"]

    # Sort by accuracy
    sorted_domains = sorted(domain_acc.items(), key=lambda x: x[1])

    # Bottom 2 = weak, Top 2 = strong
    weak_domains = [d[0] for d in sorted_domains[:2]]
    strong_domains = [d[0] for d in sorted_domains[-2:]]

    return weak_domains, strong_domains


def load_questions() -> list[dict]:
    """Load all questions from data/questions.jsonl."""
    questions = []
    with open("data/questions.jsonl") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def select_questions_for_model(
    questions: list[dict],
    weak_domains: list[str],
    strong_domains: list[str],
    n_questions: int
) -> dict[str, list[dict]]:
    """
    Select questions for a model: 50% weak domain, 50% strong domain.

    Returns:
        {"weak": [...], "strong": [...]}
    """
    weak_qs = [q for q in questions if q.get("domain") in weak_domains]
    strong_qs = [q for q in questions if q.get("domain") in strong_domains]

    n_per_type = n_questions // 2

    # Take first n_per_type from each (or all if fewer)
    selected_weak = weak_qs[:n_per_type]
    selected_strong = strong_qs[:n_per_type]

    return {"weak": selected_weak, "strong": selected_strong}


def apply_adversarial_prompt(
    base_prompt: str,
    attack_type: str,
    attack_templates: dict,
    domain: Optional[str] = None,
) -> str:
    """
    Apply adversarial attack to a base prompt.

    For attacks that target weak domains (authority_override, social_pressure),
    only apply to weak domain questions.

    For framing attacks, apply to all questions.
    """
    if attack_type in ["authority_override", "social_pressure"]:
        if domain:
            template = attack_templates.get(attack_type, "")
            attack_prefix = template.format(domain=domain)
            return attack_prefix + base_prompt
        return base_prompt

    elif attack_type in ["framing_easy", "framing_hard"]:
        attack_prefix = attack_templates.get(attack_type, "")
        return attack_prefix + base_prompt

    return base_prompt


async def run_channel_adversarial(
    client: UnifiedClient,
    model: str,
    question: dict,
    channel: str,
    attack_type: str,
    attack_templates: dict,
    is_weak_domain: bool,
) -> dict:
    """
    Run a single channel measurement with adversarial prompt.

    Returns:
        {
            "channel": str,
            "attack_type": str,
            "raw_response": str,
            "parsed": dict,
            "api_call_success": bool,
        }
    """
    # Build base channel prompt
    builder = CHANNEL_BUILDERS[channel]
    base_prompt = builder(question)

    # Apply adversarial modification
    if is_weak_domain or attack_type in ["framing_easy", "framing_hard"]:
        domain = question.get("domain", "")
        adversarial_prompt = apply_adversarial_prompt(
            base_prompt, attack_type, attack_templates, domain
        )
    else:
        adversarial_prompt = base_prompt

    # Call API with retry
    try:
        response = await call_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": adversarial_prompt}],
            temperature=0.0,
            max_tokens=1500,
            metadata={"experiment": "exp5", "channel": channel, "attack": attack_type},
        )

        if response is None:
            return {
                "channel": channel,
                "attack_type": attack_type,
                "raw_response": None,
                "parsed": {},
                "api_call_success": False,
                "error": "None response from API",
            }

        if "error" in response:
            return {
                "channel": channel,
                "attack_type": attack_type,
                "raw_response": None,
                "parsed": {},
                "api_call_success": False,
                "error": response["error"],
            }

        raw_response = response.get("content") or ""

        # Parse response
        parser = CHANNEL_PARSERS[channel]
        try:
            parsed = parser(raw_response) if raw_response else {}
        except Exception:
            parsed = {}

        return {
            "channel": channel,
            "attack_type": attack_type,
            "raw_response": raw_response,
            "parsed": parsed,
            "api_call_success": True,
        }

    except Exception as e:
        return {
            "channel": channel,
            "attack_type": attack_type,
            "raw_response": None,
            "parsed": {},
            "api_call_success": False,
            "error": str(e),
        }


async def run_single_trial(
    client, model, question, domain_type, attack_type,
    attack_templates, completed_trials, results_f, write_lock, sem,
):
    """Run one adversarial trial (all channels) with concurrency control."""
    trial_id = (model, question["source_id"], attack_type, domain_type)
    if trial_id in completed_trials:
        return None

    is_weak = domain_type == "weak"

    async with sem:
        channel_results = {}
        for channel in ALL_CHANNELS:
            result = await run_channel_adversarial(
                client, model, question, channel, attack_type,
                attack_templates, is_weak_domain=is_weak,
            )
            channel_results[channel] = result
            await asyncio.sleep(DEFAULT_DELAY)

        trial_result = {
            "model": model,
            "question_id": question["source_id"],
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            "attack_type": attack_type,
            "domain_type": domain_type,
            "channels": channel_results,
            "timestamp": datetime.now().isoformat(),
        }

        any_success = any(
            v.get("api_call_success") for v in channel_results.values()
            if isinstance(v, dict)
        )
        status = "✓" if any_success else "✗"

        async with write_lock:
            results_f.write(json.dumps(trial_result) + "\n")
            results_f.flush()
            os.fsync(results_f.fileno())
            completed_trials.add(trial_id)

        print(f"    {question['source_id']} ({domain_type}: {question.get('domain')}) {status}", flush=True)
        return trial_result


async def run_experiment(mode: str, resume: bool = False, run_id: Optional[str] = None, models_override: list = None):
    """Main experiment runner."""

    config = dict(EXPERIMENT_CONFIG[mode])  # copy so we can override
    if models_override:
        config["models"] = models_override

    if resume and run_id is None:
        # Auto-detect latest checkpoint
        results_dir = Path("data/results")
        ck_files = sorted(results_dir.glob("exp5_*_checkpoint.json"),
                          key=lambda p: p.stat().st_mtime)
        if ck_files:
            # Extract run_id from checkpoint filename
            run_id = ck_files[-1].stem.replace("_checkpoint", "").replace("exp5_", "")
            print(f"Auto-detected run_id from checkpoint: {run_id}")

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 80)
    print("EXPERIMENT 5: ADVERSARIAL ROBUSTNESS OF SELF-KNOWLEDGE")
    print("=" * 80)
    print(f"Mode: {mode.upper()}")
    print(f"Description: {config['description']}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Attacks: {', '.join(config['attacks'])}")
    print(f"Questions per model: {config['n_questions']}")
    print("=" * 80)

    # Load resources
    print("\nLoading resources...")
    attack_templates = load_attack_templates()
    exp1_metrics = load_exp1_metrics()
    all_questions = load_questions()
    print(f"  Loaded {len(all_questions)} questions")
    print(f"  Loaded Exp1 metrics for {len(exp1_metrics)} models")

    # Initialize client
    client = UnifiedClient()

    # Prepare output
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"exp5_{run_id}_results.jsonl"
    checkpoint_file = output_dir / f"exp5_{run_id}_checkpoint.json"

    # Load checkpoint if resuming - rebuild from results file (more reliable than checkpoint)
    completed_trials = set()
    if resume and results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                    # Count as completed if any channel succeeded OR if all failures
                    # are permanent 404s (model unavailable - no point retrying)
                    channels = trial.get("channels", {})
                    any_success = any(
                        v.get("api_call_success") for v in channels.values()
                        if isinstance(v, dict)
                    )
                    if not any_success:
                        errors = [v.get("error", "") for v in channels.values()
                                  if isinstance(v, dict) and not v.get("api_call_success")]
                        permanent_fail = all("404" in e or "asyncio" in e for e in errors if e)
                        if not permanent_fail:
                            continue  # Re-run transient failures
                        # else: 404 = model unavailable, treat as done
                    tid = (
                        trial.get("model"),
                        trial.get("question_id"),
                        trial.get("attack_type"),
                        trial.get("domain_type"),
                    )
                    if all(tid):
                        completed_trials.add(tid)
                except json.JSONDecodeError:
                    pass  # Skip corrupt lines
        print(f"\nResuming: rebuilt {len(completed_trials)} completed trials from results file")

    # Open results file in append mode
    results_f = open(results_file, "a", encoding="utf-8")

    total_trials = 0
    completed_count = 0

    try:
        # For each model
        for model in config["models"]:
            print(f"\n{'─' * 80}")
            print(f"Model: {model}")
            print(f"{'─' * 80}")

            # Identify weak/strong domains
            weak_domains, strong_domains = identify_weak_strong_domains(model, exp1_metrics)
            print(f"  Weak domains: {', '.join(weak_domains)}")
            print(f"  Strong domains: {', '.join(strong_domains)}")

            # Select questions
            questions_by_type = select_questions_for_model(
                all_questions, weak_domains, strong_domains, config["n_questions"]
            )
            weak_qs = questions_by_type["weak"]
            strong_qs = questions_by_type["strong"]
            print(f"  Selected: {len(weak_qs)} weak + {len(strong_qs)} strong = {len(weak_qs) + len(strong_qs)} questions")

            # For each attack type
            for attack_type in config["attacks"]:
                print(f"\n  Attack: {attack_type}")

                # Run weak + strong trials concurrently (up to CONCURRENCY at a time)
                sem = asyncio.Semaphore(CONCURRENCY)
                write_lock = asyncio.Lock()
                all_qs = (
                    [(q, "weak") for q in weak_qs]
                    + [(q, "strong") for q in strong_qs]
                )
                tasks = [
                    run_single_trial(
                        client, model, q, domain_type, attack_type,
                        attack_templates, completed_trials, results_f, write_lock, sem,
                    )
                    for q, domain_type in all_qs
                ]
                trial_results = await asyncio.gather(*tasks)
                completed_count += sum(1 for r in trial_results if r is not None)

            # Save checkpoint after each model
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "completed": [list(t) for t in completed_trials],
                    "timestamp": datetime.now().isoformat(),
                }, f)

    finally:
        results_f.close()

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 80)
    print(f"Completed trials: {completed_count}")
    print(f"Results: {results_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: Adversarial Robustness")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Experiment mode (pilot or full)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run ID to resume (e.g. 20260227T161012). Auto-detects latest if not given.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model list override (e.g. gemma-3-27b,kimi-k2)",
    )

    args = parser.parse_args()

    models_override = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_experiment(args.mode, args.resume, run_id=args.run_id, models_override=models_override))


if __name__ == "__main__":
    main()
