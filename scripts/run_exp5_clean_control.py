"""
Experiment 5: Clean Control Condition

Runs the same questions and models as the adversarial Exp 5 run
but WITHOUT any adversarial manipulation (attack_type="clean").

Used to establish baseline behaviour for computing the Adversarial
Robustness Score (ARS) — how much does adversarial framing change
model behaviour relative to a clean baseline?

Usage:
    python scripts/run_exp5_clean_control.py
    python scripts/run_exp5_clean_control.py --resume
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-use the full Exp 5 infrastructure
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

# Import helpers from run_experiment_5 (they only define functions/data)
from scripts.run_experiment_5 import (
    call_with_retry,
    identify_weak_strong_domains,
    load_exp1_metrics,
    load_questions,
    select_questions_for_model,
    DEFAULT_DELAY,
    ALL_CHANNELS,
    CHANNEL_BUILDERS,
    CHANNEL_PARSERS,
    CONCURRENCY,
)

# Models that have adversarial data AND functional APIs
CLEAN_CONTROL_MODELS = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "mistral-large",
    "gpt-oss-120b",
    "deepseek-r1",
]

N_QUESTIONS = 80   # same as exp5 full
ATTACK_TYPE = "clean"


async def run_channel_clean(
    client: UnifiedClient,
    model: str,
    question: dict,
    channel: str,
) -> dict:
    """Run one channel measurement with NO adversarial modification."""
    builder = CHANNEL_BUILDERS[channel]
    prompt = builder(question)

    try:
        response = await call_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
            metadata={"experiment": "exp5_clean", "channel": channel},
        )
        if response is None or "error" in response:
            return {"channel": channel, "attack_type": ATTACK_TYPE,
                    "raw_response": None, "parsed": {}, "api_call_success": False,
                    "error": (response or {}).get("error", "no response")}

        raw = response.get("content") or ""
        parser = CHANNEL_PARSERS[channel]
        try:
            parsed = parser(raw) if raw else {}
        except Exception:
            parsed = {}

        return {"channel": channel, "attack_type": ATTACK_TYPE,
                "raw_response": raw, "parsed": parsed, "api_call_success": True}

    except Exception as e:
        return {"channel": channel, "attack_type": ATTACK_TYPE,
                "raw_response": None, "parsed": {}, "api_call_success": False,
                "error": str(e)}


async def run_single_trial(
    client, model, question, domain_type,
    completed_trials, results_f, write_lock, sem,
) -> Optional[dict]:
    trial_id = (model, question["source_id"], ATTACK_TYPE, domain_type)
    if trial_id in completed_trials:
        return None

    async with sem:
        channel_results = {}
        for channel in ALL_CHANNELS:
            result = await run_channel_clean(client, model, question, channel)
            channel_results[channel] = result
            await asyncio.sleep(DEFAULT_DELAY)

        trial_result = {
            "model": model,
            "question_id": question["source_id"],
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            "attack_type": ATTACK_TYPE,
            "domain_type": domain_type,
            "channels": channel_results,
            "timestamp": datetime.now().isoformat(),
        }

        any_success = any(
            v.get("api_call_success") for v in channel_results.values()
        )
        status = "✓" if any_success else "✗"

        async with write_lock:
            results_f.write(json.dumps(trial_result) + "\n")
            results_f.flush()
            os.fsync(results_f.fileno())
            completed_trials.add(trial_id)

        print(f"    {question['source_id']} ({domain_type}: {question.get('domain')}) {status}",
              flush=True)
        return trial_result


async def run_clean_control(resume: bool = False, run_id: Optional[str] = None,
                            models_override: list = None, n_questions: int = None) -> None:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    models = models_override if models_override else CLEAN_CONTROL_MODELS
    n_q = n_questions if n_questions else N_QUESTIONS

    print("=" * 80)
    print("EXPERIMENT 5: CLEAN CONTROL CONDITION")
    print("=" * 80)
    print(f"Run ID:  {run_id}")
    print(f"Models:  {', '.join(models)}")
    print(f"Attack:  {ATTACK_TYPE} (no manipulation)")
    print(f"Questions per model: {n_q}")
    print("=" * 80)

    exp1_metrics = load_exp1_metrics()
    all_questions = load_questions()
    print(f"\nLoaded {len(all_questions)} questions, Exp1 metrics for {len(exp1_metrics)} models")

    client = UnifiedClient(experiment=f"exp5_clean_{run_id}")
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"exp5_clean_{run_id}_results.jsonl"

    completed_trials: set = set()
    if resume and results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    tid = (t.get("model"), t.get("question_id"),
                           t.get("attack_type"), t.get("domain_type"))
                    if all(tid):
                        completed_trials.add(tid)
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(completed_trials)} trials already done")

    with open(results_file, "a", encoding="utf-8") as results_f:
        for model in models:
            print(f"\n{'─' * 80}\nModel: {model}\n{'─' * 80}")

            weak_domains, strong_domains = identify_weak_strong_domains(model, exp1_metrics)
            print(f"  Weak:   {', '.join(weak_domains)}")
            print(f"  Strong: {', '.join(strong_domains)}")

            questions_by_type = select_questions_for_model(
                all_questions, weak_domains, strong_domains, n_q
            )
            weak_qs = questions_by_type["weak"]
            strong_qs = questions_by_type["strong"]
            print(f"  Selected: {len(weak_qs)} weak + {len(strong_qs)} strong")

            sem = asyncio.Semaphore(CONCURRENCY)
            write_lock = asyncio.Lock()
            all_qs = ([(q, "weak") for q in weak_qs]
                      + [(q, "strong") for q in strong_qs])

            tasks = [
                run_single_trial(client, model, q, dt,
                                 completed_trials, results_f, write_lock, sem)
                for q, dt in all_qs
            ]
            await asyncio.gather(*tasks)

    print(f"\n{'=' * 80}")
    print(f"Clean control complete — results: {results_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp5 clean control (no adversarial attack)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list override")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Questions per model (default 80, B1 target is 320)")
    args = parser.parse_args()
    models_override = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_clean_control(
        resume=args.resume, run_id=args.run_id,
        models_override=models_override, n_questions=args.n_questions
    ))
