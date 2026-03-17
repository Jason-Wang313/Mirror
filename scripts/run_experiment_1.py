"""
Experiment 1: Baseline Metacognitive Calibration
=================================================

Orchestrates the full Experiment 1 run across all models, questions,
and channels. Handles checkpointing, rate limiting, Channel 4 tool
execution, and triggers analysis after completion.

Usage:
  # Quick sanity check (1 model × 10 questions × 5 channels = 50 calls)
  python scripts/run_experiment_1.py --mode test

  # Pilot run (2 models × 50 questions × 5 channels = 500 calls)
  python scripts/run_experiment_1.py --mode pilot

  # Full run (7 models × 400 questions × 5 channels = 14,000 calls)
  python scripts/run_experiment_1.py --mode full

  # Resume interrupted run
  python scripts/run_experiment_1.py --mode full --resume

  # Layer 2 only (run after Layer 1 is complete)
  python scripts/run_experiment_1.py --mode full --layer2-only

  # Specific models only
  python scripts/run_experiment_1.py --mode full --models llama-3.1-8b,deepseek-r1
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import build_prompt, parse_response
from mirror.experiments.layers import (
    build_channel3_pairs_index,
    pair_questions_for_difficulty_selection,
)
from mirror.experiments.tool_executor import ToolExecutor
from mirror.scoring.answer_matcher import match_answer_robust

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    "test": {
        "n_questions": 10,
        "models": ["llama-3.1-8b"],
        "description": "Quick sanity check",
    },
    "pilot": {
        "n_questions": 50,
        "models": ["llama-3.1-8b", "deepseek-r1"],
        "description": "Pilot run — verify everything works before full scale",
    },
    "full": {
        "n_questions": 400,
        "models": [
            "llama-3.1-8b",
            "llama-3.1-70b",
            "llama-3.1-405b",
            "mistral-large",
            "qwen-3-235b",
            "gpt-oss-120b",
            "deepseek-r1",
            "gemini-2.5-pro",
        ],
        "description": "Full Experiment 1 — all models, all questions",
    },
}

# Ordered list of channels (Layer 1) plus layer2
ALL_CHANNELS = ["wagering", "opt_out", "difficulty_selection", "tool_use", "natural"]
LAYER2_CHANNEL = "layer2"

CHANNEL_MAP = {
    "wagering": 1,
    "opt_out": 2,
    "difficulty_selection": 3,
    "tool_use": 4,
    "natural": 5,
    "layer2": "layer2",
}

# Provider delays (seconds between calls)
PROVIDER_DELAYS = {
    "nvidia_nim": 0.5,
    "deepseek": 0.5,
    "google_ai": 1.0,
}
DEFAULT_DELAY = 0.5

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0
RETRY_MAX_DELAY = 120.0

# Format calibration: 5 trivially easy arithmetic questions
_CALIBRATION_QUESTIONS = [
    {"question_id": "_cal_0", "question_text": "What is 2 + 2?",
     "correct_answer": "4", "answer_type": "exact_numeric",
     "domain": "arithmetic", "difficulty": "easy", "metadata": {}},
    {"question_id": "_cal_1", "question_text": "What is 5 times 6?",
     "correct_answer": "30", "answer_type": "exact_numeric",
     "domain": "arithmetic", "difficulty": "easy", "metadata": {}},
    {"question_id": "_cal_2", "question_text": "What is 100 divided by 4?",
     "correct_answer": "25", "answer_type": "exact_numeric",
     "domain": "arithmetic", "difficulty": "easy", "metadata": {}},
    {"question_id": "_cal_3", "question_text": "What is 7 minus 3?",
     "correct_answer": "4", "answer_type": "exact_numeric",
     "domain": "arithmetic", "difficulty": "easy", "metadata": {}},
    {"question_id": "_cal_4", "question_text": "What is 2 to the power of 8?",
     "correct_answer": "256", "answer_type": "exact_numeric",
     "domain": "arithmetic", "difficulty": "easy", "metadata": {}},
]

DOMAINS = [
    "arithmetic", "spatial", "temporal", "linguistic",
    "logical", "social", "factual", "procedural",
]


# ---------------------------------------------------------------------------
# Question loading and sampling
# ---------------------------------------------------------------------------

def _qid(q: dict) -> str:
    """Return the question's unique identifier.

    Supports both 'question_id' (calibration questions) and
    'source_id' (pipeline-generated questions from data/questions.jsonl).
    """
    return q.get("question_id") or q.get("source_id") or "unknown"


def load_all_questions(path: Path) -> list[dict]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def sample_stratified(questions: list[dict], n: int) -> list[dict]:
    """Sample n questions with equal domain representation.

    Divides n evenly across domains. Within each domain, samples
    proportionally across difficulty levels. Uses no randomness —
    takes the first available questions in source order, ensuring
    reproducibility.
    """
    by_domain: dict[str, list[dict]] = {}
    for q in questions:
        d = q.get("domain", "unknown")
        by_domain.setdefault(d, []).append(q)

    active_domains = [d for d in DOMAINS if d in by_domain]
    if not active_domains:
        return questions[:n]

    per_domain = max(1, n // len(active_domains))
    selected: list[dict] = []

    for domain in active_domains:
        pool = by_domain[domain]
        # Sort by difficulty so we get balanced representation
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2, "adversarial": 3}
        pool_sorted = sorted(pool, key=lambda q: difficulty_order.get(q.get("difficulty", "medium"), 1))
        selected.extend(pool_sorted[:per_domain])

    # Fill up to n if rounding left gaps
    if len(selected) < n:
        used_ids = {_qid(q) for q in selected}
        for q in questions:
            if len(selected) >= n:
                break
            if _qid(q) not in used_ids:
                selected.append(q)
                used_ids.add(_qid(q))

    return selected[:n]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def checkpoint_path(run_id: str, results_dir: Path) -> Path:
    return results_dir / f"exp1_{run_id}_checkpoint.json"


def load_checkpoint(run_id: str, results_dir: Path) -> dict:
    cp = checkpoint_path(run_id, results_dir)
    if cp.exists():
        data = json.loads(cp.read_text(encoding="utf-8"))
        # Convert completed list to a set of tuples for fast lookup
        data["completed"] = set(
            (c["model"], c["question_id"] or c.get("source_id", "unknown"), c["channel"])
            for c in data.get("completed", [])
        )
        return data
    return {
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "completed": set(),
    }


def save_checkpoint(checkpoint: dict, run_id: str, results_dir: Path) -> None:
    serializable = dict(checkpoint)
    serializable["completed"] = [
        {"model": m, "question_id": qid, "channel": ch}
        for (m, qid, ch) in sorted(checkpoint["completed"])
    ]
    checkpoint_path(run_id, results_dir).write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# API call with retry / exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(client: UnifiedClient, model: str, messages: list[dict],
                    max_tokens: int = 1024) -> dict:
    """Synchronous API call with exponential backoff on 429 / transient errors."""
    delay = RETRY_BASE_DELAY
    last_error = None

    for attempt in range(MAX_RETRIES):
        result = client.complete_sync(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if "error" not in result:
            return result

        err = result["error"]
        last_error = err
        is_rate_limit = "429" in str(err) or "rate limit" in str(err).lower()

        if attempt < MAX_RETRIES - 1:
            sleep = min(delay, RETRY_MAX_DELAY)
            if is_rate_limit:
                print(f"    [rate limit] backing off {sleep:.0f}s …")
            else:
                print(f"    [error] {err[:80]} — retry in {sleep:.0f}s")
            time.sleep(sleep)
            delay *= 2
        else:
            print(f"    [error] giving up after {MAX_RETRIES} attempts: {err[:80]}")

    return {"error": last_error or "unknown error after retries"}


# ---------------------------------------------------------------------------
# Format calibration check
# ---------------------------------------------------------------------------

def run_format_calibration_check(model: str, client: UnifiedClient) -> bool:
    """Run 5 trivial questions through Channel 4 to verify format compliance.

    Returns True if ≥3/5 produce parseable tool commands (format reliable).
    Logs a warning if fewer than 3 succeed.
    """
    successes = 0
    for cal_q in _CALIBRATION_QUESTIONS:
        prompt_text = build_prompt(4, cal_q)
        messages = [{"role": "user", "content": prompt_text}]
        result = client.complete_sync(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        if "error" in result:
            continue
        parsed = parse_response(4, result.get("content", ""))
        if parsed.get("parse_success"):
            successes += 1

    reliable = successes >= 3
    if not reliable:
        print(
            f"  [WARNING] {model}: Channel 4 format calibration FAILED "
            f"({successes}/5 parseable). Marking channel 4 results as "
            f"format_unreliable=True."
        )
    else:
        print(f"  [calibration] {model}: Channel 4 OK ({successes}/5).")
    return reliable


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def get_provider_delay(model: str) -> float:
    """Return inter-call delay (seconds) based on model's provider."""
    ml = model.lower()
    if "deepseek" in ml:
        return PROVIDER_DELAYS["deepseek"]
    if "gemini" in ml:
        return PROVIDER_DELAYS["google_ai"]
    return PROVIDER_DELAYS["nvidia_nim"]


def get_max_tokens(model: str, channel_name: str) -> int:
    """Return appropriate max_tokens for this model + channel."""
    if channel_name == "layer2":
        return 1536
    if "qwen" in model.lower() and channel_name in {"wagering", "opt_out", "difficulty_selection", "tool_use"}:
        return 4096
    return 1024


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------

def build_result_record(
    model: str,
    question: dict,
    channel_name: str,
    parsed: dict,
    answer_correct: bool | None,
    api_result: dict,
    extra: dict | None = None,
) -> dict:
    record = {
        "question_id": _qid(question),
        "model": model,
        "channel": CHANNEL_MAP[channel_name],
        "channel_name": channel_name,
        "layer": 2 if channel_name == "layer2" else 1,
        "prompt_variant": channel_name,
        "timestamp": datetime.utcnow().isoformat(),
        "raw_response": api_result.get("content"),
        "parsed": parsed,
        "answer_correct": answer_correct,
        "parse_success": parsed.get("parse_success", False),
        "latency_ms": api_result.get("latency_ms"),
        "domain": question.get("domain"),
        "difficulty": question.get("difficulty"),
        "error": api_result.get("error"),
    }
    if extra:
        record.update(extra)
    return record


def append_result(result: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def log_progress(
    model: str,
    q_idx: int,
    n_questions: int,
    channel_results: dict[str, dict | None],
) -> None:
    parts = []
    for ch in ALL_CHANNELS + [LAYER2_CHANNEL]:
        r = channel_results.get(ch)
        if r is None:
            continue
        ch_abbr = {"wagering": "Ch1", "opt_out": "Ch2",
                   "difficulty_selection": "Ch3", "tool_use": "Ch4",
                   "natural": "Ch5", "layer2": "L2"}[ch]
        p = r.get("parsed", {})
        if r.get("error"):
            parts.append(f"{ch_abbr}: ERR")
        elif ch == "wagering":
            correct = "✓" if r.get("answer_correct") else "✗"
            bet = p.get("bet", "?")
            parts.append(f"{ch_abbr}: {correct} bet={bet}")
        elif ch == "opt_out":
            if p.get("skipped"):
                parts.append(f"{ch_abbr}: SKIP")
            else:
                correct = "✓" if r.get("answer_correct") else "✗"
                parts.append(f"{ch_abbr}: {correct}")
        elif ch == "difficulty_selection":
            choice = p.get("choice", "?")
            correct = "✓" if r.get("answer_correct") else "✗"
            parts.append(f"{ch_abbr}: {choice} {correct}")
        elif ch == "tool_use":
            tools = p.get("tools_used", [])
            n_calc = sum(1 for t in tools if t.get("tool_name") == "calculator")
            tag = f"calc×{n_calc}" if n_calc else "no-tool"
            correct = "✓" if r.get("answer_correct") else "✗"
            parts.append(f"{ch_abbr}: {tag} {correct}")
        elif ch == "natural":
            correct = "✓" if r.get("answer_correct") else "✗"
            parts.append(f"{ch_abbr}: {correct}")
        elif ch == "layer2":
            conf = p.get("confidence", "?")
            parts.append(f"L2: conf={conf}")

    print(f"  [{model}] Q {q_idx+1:>3}/{n_questions} | " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Baseline Metacognitive Calibration"
    )
    parser.add_argument(
        "--mode", choices=["test", "pilot", "full"], default="pilot",
        help="Experiment scale"
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model overrides (e.g. llama-3.1-8b,deepseek-r1)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint"
    )
    parser.add_argument(
        "--layer2-only", action="store_true",
        help="Run only the Layer 2 (structured self-report) channel"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip automatic analysis after data collection"
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Explicit run ID (used with --resume)"
    )
    parser.add_argument(
        "--questions-path", default="data/questions.jsonl",
        help="Path to questions JSONL"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EXPERIMENT_CONFIG[args.mode]

    # Model list
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = config["models"]

    # Channel list
    if args.layer2_only:
        channels_to_run = [LAYER2_CHANNEL]
    else:
        channels_to_run = ALL_CHANNELS + [LAYER2_CHANNEL]

    # Paths
    repo_root = Path(__file__).resolve().parent.parent
    questions_path = repo_root / args.questions_path
    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run ID
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    results_path = results_dir / f"exp1_{run_id}_results.jsonl"

    print(f"\n{'='*60}")
    print(f"  Experiment 1: Baseline Metacognitive Calibration")
    print(f"  Mode:     {args.mode} — {config['description']}")
    print(f"  Models:   {', '.join(models)}")
    print(f"  Channels: {', '.join(channels_to_run)}")
    print(f"  Run ID:   {run_id}")
    print(f"  Output:   {results_path}")
    print(f"{'='*60}\n")

    # Load questions
    print(f"Loading questions from {questions_path} …")
    all_questions = load_all_questions(questions_path)
    if args.mode == "full":
        questions = all_questions
    else:
        questions = sample_stratified(all_questions, config["n_questions"])
    print(f"  {len(questions)} questions selected.\n")

    # Pre-compute Channel 3 pairs
    pairs = pair_questions_for_difficulty_selection(questions)
    pairs_index = build_channel3_pairs_index(pairs)

    # Save pairs for reproducibility
    pairs_path = results_dir / f"exp1_{run_id}_difficulty_pairs.json"
    pairs_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    # Build question lookup
    questions_dict = {_qid(q): q for q in questions}

    # Checkpoint
    checkpoint = load_checkpoint(run_id, results_dir) if args.resume else {
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "completed": set(),
    }

    # Client + tool executor
    client = UnifiedClient(experiment=f"exp1_{run_id}")
    tool_executor = ToolExecutor()

    total_calls = len(models) * len(questions) * len(channels_to_run)
    print(f"Estimated API calls: {total_calls}\n")

    completed_this_run = 0
    skipped_checkpoint = 0

    for model in models:
        print(f"\n{'─'*60}")
        print(f"  Model: {model}")
        print(f"{'─'*60}")

        # Format calibration check for Channel 4
        if "tool_use" in channels_to_run:
            ch4_reliable = run_format_calibration_check(model, client)
        else:
            ch4_reliable = True

        delay = get_provider_delay(model)

        for q_idx, question in enumerate(questions):
            qid = _qid(question)
            channel_results: dict[str, dict | None] = {}

            for channel_name in channels_to_run:
                key = (model, qid, channel_name)
                if key in checkpoint["completed"]:
                    skipped_checkpoint += 1
                    continue

                channel_id = CHANNEL_MAP[channel_name]

                # Build prompt
                if channel_name == "difficulty_selection":
                    pairs_for_q = pairs_index.get(qid, [])
                    if not pairs_for_q:
                        # No pair available — skip this channel for this question
                        channel_results[channel_name] = None
                        checkpoint["completed"].add(key)
                        continue
                    pair = pairs_for_q[0]
                    easy_sid = (pair["easy_question"].get("source_id")
                                or pair["easy_question"].get("question_id"))
                    # Determine which question is the easy/hard partner
                    if easy_sid == qid:
                        easy_q_for_prompt = question
                        hard_q_for_prompt = pair["hard_question"]
                    else:
                        easy_q_for_prompt = pair["easy_question"]
                        hard_q_for_prompt = question
                    prompt_text = build_prompt(
                        3, question,
                        easy_question=easy_q_for_prompt,
                        hard_question=hard_q_for_prompt,
                    )
                    extra = {"ch3_pair_id": pair["pair_id"]}
                    # Store pair kwargs for parse_response
                    ch3_kwargs = {
                        "easy_question": easy_q_for_prompt,
                        "hard_question": hard_q_for_prompt,
                    }
                else:
                    prompt_text = build_prompt(channel_id, question)
                    extra = {}
                    ch3_kwargs = {}

                # Rate limiting
                time.sleep(delay)

                # API call
                max_tokens = get_max_tokens(model, channel_name)
                messages = [{"role": "user", "content": prompt_text}]
                api_result = call_with_retry(client, model, messages, max_tokens)

                raw_response = api_result.get("content") or ""

                # Guard: if the API returned no content at all, skip this cell
                if not raw_response and "error" not in api_result:
                    print(f"    [warning] {model} | {channel_name} | {qid}: API returned empty content — skipping")
                    checkpoint["completed"].add(key)
                    continue

                # Parse response
                if "error" in api_result:
                    parsed = {"parse_success": False, "answer": None}
                elif channel_name == "difficulty_selection":
                    parsed = parse_response(channel_id, raw_response, **ch3_kwargs)
                else:
                    parsed = parse_response(channel_id, raw_response)

                # Channel 4: tool execution
                if channel_name == "tool_use" and parsed.get("parse_success"):
                    parsed = tool_executor.process_channel4_response(
                        parsed_response=parsed,
                        model=model,
                        prompt_base=prompt_text,
                        client=client,
                        raw_response=raw_response,
                        max_tokens=max_tokens,
                    )
                    if not ch4_reliable:
                        parsed["format_unreliable"] = True
                    # If tool was executed, a second API call was made — add delay
                    if parsed.get("tool_executed"):
                        time.sleep(delay)

                # Score
                answer = parsed.get("final_answer") or parsed.get("answer")
                answer_correct: bool | None = None
                if answer is not None and not parsed.get("refused"):
                    try:
                        answer_correct = match_answer_robust(
                            predicted=answer,
                            correct=question["correct_answer"],
                            answer_type=question.get("answer_type", "short_text"),
                            metadata=question.get("metadata", {}),
                        )
                    except Exception as score_exc:
                        answer_correct = None
                        print(f"    [score error] {type(score_exc).__name__}: {score_exc}")

                # Build and save result
                result = build_result_record(
                    model=model,
                    question=question,
                    channel_name=channel_name,
                    parsed=parsed,
                    answer_correct=answer_correct,
                    api_result=api_result,
                    extra=extra if extra else None,
                )
                channel_results[channel_name] = result
                append_result(result, results_path)

                # Update checkpoint
                checkpoint["completed"].add(key)
                completed_this_run += 1

                # Periodic checkpoint save (every 20 calls)
                if completed_this_run % 20 == 0:
                    save_checkpoint(checkpoint, run_id, results_dir)

            # Log progress after all channels for this question
            if any(v is not None for v in channel_results.values()):
                log_progress(model, q_idx, len(questions), channel_results)

        # Save checkpoint after each model
        save_checkpoint(checkpoint, run_id, results_dir)

    print(f"\n{'='*60}")
    print(f"  Data collection complete.")
    print(f"  Completed this run: {completed_this_run} API calls")
    print(f"  Skipped (checkpoint): {skipped_checkpoint}")
    print(f"  Results: {results_path}")
    print(f"{'='*60}\n")

    # Trigger analysis
    if not args.skip_analysis:
        print("Running analysis …")
        import subprocess
        subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "analyze_experiment_1.py"),
             "--run-id", run_id],
            check=False,
        )


if __name__ == "__main__":
    main()
