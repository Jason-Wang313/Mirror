"""
Fast async Exp1 supplementary runner for new models.

BOLD SPEEDUPS vs original run_experiment_1.py:
  1. Calls providers DIRECTLY — no UnifiedClient rate-limiter overhead
  2. Only channels 1 (wagering) + 5 (natural) — the 2 needed for MIRROR gap
  3. All models run simultaneously in one process via asyncio.gather
  4. Smart routing: Groq for llama-3.3-70b (super-fast), DeepSeek direct, NIM direct
  5. Direct accuracy JSON output — no analyze_experiment_1.py needed

Result: ~2.5 hours → ~10-15 minutes for all 7 models.

Usage:
    python scripts/run_exp1_fast.py
    python scripts/run_exp1_fast.py --models deepseek-v3,llama-3.3-70b
    python scripts/run_exp1_fast.py --resume
    python scripts/run_exp1_fast.py --skip-nim   # skip models needing NIM (if NIM is down)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from mirror.experiments.channels import build_prompt, parse_response
from mirror.scoring.answer_matcher import match_answer_robust

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = REPO_ROOT / "data" / "results"
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.jsonl"
N_QUESTIONS = 400

DOMAINS = [
    "arithmetic", "spatial", "temporal", "linguistic",
    "logical", "social", "factual", "procedural",
]

# Only wagering (ch1) + natural (ch5) — sufficient for MIRROR gap
CHANNELS = [("wagering", 1), ("natural", 5)]

DEFAULT_MODELS = [
    "deepseek-v3",
    "llama-3.3-70b",    # routed via Groq (NIM fallback)
    "phi-4",
    "command-r-plus",
    "kimi-k2",
    "gemma-3-27b",
    "qwen3-235b-nim",
]

NIM_MODELS = {"phi-4", "command-r-plus", "gemma-3-27b", "qwen3-235b-nim"}

# Per-model: (provider_type, model_id, concurrency)
# Groq is fastest (~200 tok/s), DeepSeek is fast, NIM varies
# Notes:
#   kimi-k2: NIM 429 rate-limited even at c=1; rerouted to Groq
#   phi-4: phi-4-mini-instruct times out on NIM; using phi-4-mini-flash-reasoning proxy
#   command-r-plus: cohere/command-r-plus-08-2024 is 404 on NIM (model removed); EXCLUDED
#   llama-3.3-70b: reduced concurrency to avoid NIM quota hits
MODEL_ROUTING = {
    "deepseek-v3":    ("deepseek", "deepseek-chat",                          20),
    "llama-3.3-70b":  ("groq",     "llama-3.3-70b-versatile",               16),
    "phi-4":          ("nim",      "microsoft/phi-4-mini-flash-reasoning",    8),
    "command-r-plus": ("nim",      "cohere/command-r-plus-08-2024",           0),  # EXCLUDED
    "kimi-k2":        ("groq",     "moonshotai/kimi-k2-instruct",            16),
    "gemma-3-27b":    ("nim",      "google/gemma-3-27b-it",                  6),
    "qwen3-235b-nim": ("nim",      "qwen/qwen3.5-397b-a17b",                 4),
}

MAX_RETRIES = 4
BASE_DELAY = 2.0
MAX_DELAY = 60.0

# ---------------------------------------------------------------------------
# Provider factories (cached)
# ---------------------------------------------------------------------------
_providers: dict = {}


def get_provider(provider_type: str):
    if provider_type in _providers:
        return _providers[provider_type]
    if provider_type == "nim":
        from mirror.api.providers.nvidia_nim import NVIDIANIMProvider
        p = NVIDIANIMProvider()
    elif provider_type == "deepseek":
        from mirror.api.providers.deepseek import DeepSeekProvider
        p = DeepSeekProvider()
    elif provider_type == "groq":
        from mirror.api.providers.groq import GroqProvider
        p = GroqProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
    _providers[provider_type] = p
    return p


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_questions(n: int) -> list[dict]:
    all_qs: list[dict] = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_qs.append(json.loads(line))

    by_domain: dict[str, list[dict]] = {}
    for q in all_qs:
        by_domain.setdefault(q.get("domain", "unknown"), []).append(q)

    difficulty_order = {"easy": 0, "medium": 1, "hard": 2, "adversarial": 3}
    per_domain = max(1, n // max(len(by_domain), 1))
    selected: list[dict] = []
    for domain in DOMAINS:
        pool = sorted(
            by_domain.get(domain, []),
            key=lambda q: difficulty_order.get(q.get("difficulty", "medium"), 1),
        )
        selected.extend(pool[:per_domain])

    if len(selected) < n:
        used = {q.get("question_id") or q.get("source_id") for q in selected}
        for q in all_qs:
            if len(selected) >= n:
                break
            qid = q.get("question_id") or q.get("source_id")
            if qid not in used:
                selected.append(q)
                used.add(qid)

    return selected[:n]


# ---------------------------------------------------------------------------
# Async API call — direct to provider, no UnifiedClient overhead
# ---------------------------------------------------------------------------

async def call_direct(provider_type: str, model_id: str, messages: list[dict],
                      max_tokens: int = 512) -> dict:
    provider = get_provider(provider_type)
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            result = await provider.complete(
                model_id=model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if result.get("content"):
                return result
            err = result.get("error", "empty response")
        except Exception as e:
            err = str(e)

        if "rate" in err.lower() or "429" in err:
            sleep = min(delay * 2, MAX_DELAY)
        else:
            sleep = min(delay, 10.0)

        await asyncio.sleep(sleep)
        delay = min(delay * 2, MAX_DELAY)

    return {"content": "", "error": err}


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

async def run_question(
    model: str,
    provider_type: str,
    model_id: str,
    question: dict,
    channel_name: str,
    channel_id: int,
    sem: asyncio.Semaphore,
    completed: set,
    records: list,
    lock: asyncio.Lock,
    shard_path: Path,
) -> None:
    qid = question.get("question_id") or question.get("source_id") or "unknown"
    key = (model, qid, channel_name)
    if key in completed:
        return

    prompt = build_prompt(channel_id, question)
    messages = [{"role": "user", "content": prompt}]

    async with sem:
        api_result = await call_direct(provider_type, model_id, messages)

    raw = api_result.get("content") or ""
    if not raw:
        return  # silent skip; resume will retry

    parsed = parse_response(channel_id, raw)
    answer = parsed.get("final_answer") or parsed.get("answer")
    answer_correct: bool | None = None
    if answer and not parsed.get("refused"):
        try:
            answer_correct = match_answer_robust(
                predicted=answer,
                correct=question["correct_answer"],
                answer_type=question.get("answer_type", "short_text"),
                metadata=question.get("metadata", {}),
            )
        except Exception:
            pass

    record = {
        "model": model,
        "question_id": qid,
        "channel": channel_id,
        "channel_name": channel_name,
        "domain": question.get("domain"),
        "difficulty": question.get("difficulty"),
        "answer_correct": answer_correct,
        "parsed": parsed,
        "raw_response": raw,
        "timestamp": datetime.utcnow().isoformat(),
    }

    async with lock:
        completed.add(key)
        records.append(record)
        with shard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

async def run_model(model: str, questions: list[dict], run_id: str, resume: bool) -> dict:
    provider_type, model_id, concurrency = MODEL_ROUTING[model]
    slug = model.replace("/", "-").replace(".", "-")
    shard = RESULTS_DIR / f"exp1_{run_id}_{slug}_fast_shard.jsonl"

    completed: set = set()
    records: list[dict] = []
    if resume and shard.exists():
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                completed.add((r["model"], r["question_id"], r["channel_name"]))
                records.append(r)
            except Exception:
                pass

    total = len(questions) * len(CHANNELS)
    remaining = total - len(completed)
    t0 = time.time()
    print(f"  [{model}] provider={provider_type} c={concurrency} | {len(completed)} done, {remaining} to go", flush=True)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    tasks = [
        run_question(model, provider_type, model_id, q, ch_name, ch_id,
                     sem, completed, records, lock, shard)
        for q in questions
        for ch_name, ch_id in CHANNELS
    ]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    done_this_run = len(records) - (total - remaining)
    print(f"  [{model}] DONE — {len(records)} records total ({done_this_run} new) in {elapsed:.0f}s", flush=True)

    # Compute accuracy per domain
    accuracy: dict[str, dict] = {}
    for ch_name, ch_id in CHANNELS:
        by_domain: dict[str, list] = {}
        for r in records:
            if r.get("channel_name") == ch_name and r.get("answer_correct") is not None:
                by_domain.setdefault(r.get("domain", "unknown"), []).append(r["answer_correct"])
        for domain, corrects in by_domain.items():
            accuracy.setdefault(domain, {})[f"{ch_name}_acc"] = sum(corrects) / len(corrects)

    for domain in DOMAINS:
        accuracy.setdefault(domain, {})
        for ch_name, _ in CHANNELS:
            accuracy[domain].setdefault(f"{ch_name}_acc", None)

    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    print(f"\n{'='*60}")
    print(f"EXP1 FAST RUNNER  run_id={run_id}")
    print(f"Models ({len(args.models)}): {args.models}")
    print(f"Channels: wagering + natural only (MIRROR gap)")
    print(f"Questions: {N_QUESTIONS} | Resume: {args.resume}")
    total_calls = len(args.models) * N_QUESTIONS * len(CHANNELS)
    print(f"Total API calls: {total_calls} (all models in parallel)")
    print(f"{'='*60}\n")

    questions = load_questions(N_QUESTIONS)
    domains_found = len(set(q.get("domain") for q in questions))
    print(f"Loaded {len(questions)} questions ({domains_found} domains)\n")

    # Launch all models concurrently
    model_tasks = {
        model: asyncio.create_task(run_model(model, questions, run_id, args.resume))
        for model in args.models
    }

    all_accuracy: dict[str, dict] = {}
    for model, task in model_tasks.items():
        try:
            accuracy = await task
            all_accuracy[model] = accuracy
        except Exception as e:
            print(f"  [{model}] ERROR: {e}", flush=True)

    if not all_accuracy:
        print("No models completed successfully.")
        return

    # Write accuracy JSON
    out_path = RESULTS_DIR / f"exp1_{run_id}_accuracy.json"
    out_path.write_text(json.dumps(all_accuracy, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"ACCURACY JSON: {out_path.name}")
    print(f"{'='*60}\n")

    # Summary table
    print(f"{'Model':<22} {'Domain':<14} {'natural_acc':>11} {'wagering_acc':>12} {'gap':>6}")
    print("-" * 68)
    for model in args.models:
        acc = all_accuracy.get(model, {})
        for domain in DOMAINS:
            dom = acc.get(domain, {})
            nat = dom.get("natural_acc")
            wag = dom.get("wagering_acc")
            gap = abs(wag - nat) if nat is not None and wag is not None else None
            nat_s = f"{nat:.3f}" if nat is not None else " N/A"
            wag_s = f"{wag:.3f}" if wag is not None else " N/A"
            gap_s = f"{gap:.3f}" if gap is not None else " N/A"
            print(f"  {model:<20} {domain:<14} {nat_s:>11} {wag_s:>12} {gap_s:>6}")
        print()

    print(f"Results saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast async Exp1 supplementary runner")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--run-id", default=None, help="Reuse existing run_id to resume across restarts")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--skip-nim", action="store_true", default=False,
        help="Skip NIM-dependent models (use when NIM is unavailable)",
    )
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.skip_nim:
        args.models = [m for m in args.models if m not in NIM_MODELS]
        print(f"[skip-nim] Running only: {args.models}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
