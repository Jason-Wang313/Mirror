"""
Format-Matched Control Experiment (Appendix N)
===============================================

Runs 200 Exp1 questions through a structured prompt WITHOUT wagering
incentive to isolate the effect of format vs. incentive on accuracy.

Prompt template:
    [question text]
    Format your response EXACTLY as:
    ANSWER: [your answer]
    CONFIDENCE: [1-10]

Models: mistral-large, llama-3.1-70b, llama-3.1-8b
(deepseek-r1 excluded: DeepSeek API 402, NIM EOL; phi-4 excluded: NIM 504)

Usage:
    python scripts/run_format_matched_control.py
    python scripts/run_format_matched_control.py --resume
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from mirror.scoring.answer_matcher import match_answer_robust

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = REPO_ROOT / "data" / "results"
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.jsonl"
OUTPUT_PATH = RESULTS_DIR / "format_matched_control_results.jsonl"

RANDOM_SEED = 42
N_PER_DOMAIN = 25
DOMAINS = sorted([
    "arithmetic", "factual", "linguistic", "logical",
    "procedural", "social", "spatial", "temporal",
])

CONCURRENCY = 6
MAX_RETRIES = 4
BASE_DELAY = 2.0
MAX_DELAY = 60.0

# Model routing: (provider_type, model_id, concurrency_override)
# NOTE: deepseek-r1 excluded — DeepSeek API 402 (insufficient balance) and
#       NIM endpoint EOL'd 2026-01-26.  Replaced with mistral-large (NIM).
# llama-3.1-70b: NVIDIA NIM
# llama-3.1-8b: NVIDIA NIM
# mistral-large: NVIDIA NIM
MODEL_ROUTING = {
    "mistral-large": ("nim",      "mistralai/mistral-large-3-675b-instruct-2512", CONCURRENCY),
    "llama-3.1-70b": ("nim",      "meta/llama-3.1-70b-instruct",   CONCURRENCY),
    "llama-3.1-8b":  ("nim",      "meta/llama-3.1-8b-instruct",    CONCURRENCY),
}

MODELS = list(MODEL_ROUTING.keys())

# Existing Exp1 result files to pull Nat.Acc and Wag.Acc from
EXP1_RESULT_FILES = [
    "exp1_20260220T090109_results.jsonl",
    "exp1_20260217T210412_results.jsonl",
    "exp1_20260313T140826_results.jsonl",
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """{question_text}
Format your response EXACTLY as:
ANSWER: [your answer]
CONFIDENCE: [1-10]"""

# ---------------------------------------------------------------------------
# Provider factory (cached)
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
# Question loading: 25 per domain, seed=42, deduplicated by question_id
# ---------------------------------------------------------------------------

def load_sampled_questions() -> list[dict]:
    """Load all questions, deduplicate by question_id, sample 25 per domain."""
    all_qs = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_qs.append(json.loads(line))

    # Deduplicate by question_id, keeping first occurrence
    seen = set()
    unique_qs = []
    for q in all_qs:
        qid = q["question_id"]
        if qid not in seen:
            seen.add(qid)
            unique_qs.append(q)

    by_domain: dict[str, list[dict]] = {}
    for q in unique_qs:
        by_domain.setdefault(q.get("domain", "unknown"), []).append(q)

    rng = random.Random(RANDOM_SEED)
    sampled: list[dict] = []
    for domain in DOMAINS:
        pool = by_domain.get(domain, [])
        s = rng.sample(pool, min(N_PER_DOMAIN, len(pool)))
        sampled.extend(s)
        print(f"  {domain}: sampled {len(s)} from {len(pool)}")

    print(f"  Total: {len(sampled)} questions ({len(set(q['question_id'] for q in sampled))} unique IDs)")
    return sampled


# ---------------------------------------------------------------------------
# Async API call with retries
# ---------------------------------------------------------------------------

async def call_api(provider_type: str, model_id: str, messages: list[dict],
                   max_tokens: int = 300) -> dict:
    provider = get_provider(provider_type)
    delay = BASE_DELAY
    last_err = "unknown"
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
            last_err = result.get("error", "empty response")
        except Exception as e:
            last_err = str(e)

        # Backoff
        if "rate" in last_err.lower() or "429" in last_err:
            sleep = min(delay * 2, MAX_DELAY)
        elif "504" in last_err or "502" in last_err:
            sleep = min(delay * 3, MAX_DELAY)
        else:
            sleep = min(delay, 10.0)

        print(f"    Retry {attempt+1}/{MAX_RETRIES}: {last_err[:80]}... sleeping {sleep:.1f}s")
        await asyncio.sleep(sleep)
        delay = min(delay * 2, MAX_DELAY)

    return {"content": "", "error": last_err}


# ---------------------------------------------------------------------------
# Parse ANSWER and CONFIDENCE from response
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> dict:
    """Parse ANSWER: and CONFIDENCE: fields from model response."""
    result = {
        "answer": None,
        "confidence": None,
        "parse_success": False,
    }

    if not raw:
        return result

    # Extract ANSWER field
    answer_match = re.search(r"ANSWER\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()

    # Extract CONFIDENCE field
    conf_match = re.search(r"CONFIDENCE\s*:\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    result["parse_success"] = result["answer"] is not None
    return result


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

async def run_one(
    model: str,
    provider_type: str,
    model_id: str,
    question: dict,
    sem: asyncio.Semaphore,
    completed: set,
    records: list,
    lock: asyncio.Lock,
) -> None:
    qid = question.get("question_id", "unknown")
    key = (model, qid)
    if key in completed:
        return

    prompt = PROMPT_TEMPLATE.format(question_text=question["question_text"])
    messages = [{"role": "user", "content": prompt}]

    async with sem:
        api_result = await call_api(provider_type, model_id, messages)

    raw = api_result.get("content") or ""
    if not raw:
        # Record the error but don't skip
        record = {
            "model": model,
            "question_id": qid,
            "domain": question.get("domain"),
            "channel_name": "structured_control",
            "raw_response": "",
            "parsed": {"answer": None, "confidence": None, "parse_success": False},
            "answer_correct": None,
            "error": api_result.get("error", "empty response"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        async with lock:
            completed.add(key)
            records.append(record)
        return

    parsed = parse_structured_response(raw)
    answer = parsed.get("answer")
    answer_correct = None
    if answer:
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
        "domain": question.get("domain"),
        "channel_name": "structured_control",
        "raw_response": raw,
        "parsed": parsed,
        "answer_correct": answer_correct,
        "error": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    async with lock:
        completed.add(key)
        records.append(record)


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

async def run_model(model: str, questions: list[dict], resume: bool) -> list[dict]:
    provider_type, model_id, concurrency = MODEL_ROUTING[model]
    print(f"\n  [{model}] provider={provider_type}, model_id={model_id}, concurrency={concurrency}")

    completed: set = set()
    records: list = []

    # Resume from existing output file
    if resume and OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("model") == model:
                    completed.add((r["model"], r["question_id"]))
                    records.append(r)
            except Exception:
                pass

    remaining = len(questions) - len(completed)
    print(f"    {len(completed)} already done, {remaining} remaining")

    if remaining == 0:
        return records

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.time()

    tasks = [
        run_one(model, provider_type, model_id, q, sem, completed, records, lock)
        for q in questions
    ]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    done_count = sum(1 for r in records if r.get("answer_correct") is not None)
    correct_count = sum(1 for r in records if r.get("answer_correct") is True)
    acc = correct_count / done_count if done_count > 0 else 0.0
    print(f"    Finished in {elapsed:.1f}s — {done_count} scored, accuracy={acc:.3f}")

    return records


# ---------------------------------------------------------------------------
# Load existing Exp1 results for Nat.Acc and Wag.Acc comparison
# ---------------------------------------------------------------------------

def load_exp1_results(sample_qids: set[str]) -> dict:
    """Load existing Exp1 results for the sampled question IDs.

    Returns dict: {(model, channel_name, question_id): answer_correct}
    """
    lookup = {}
    for fname in EXP1_RESULT_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  Warning: {fname} not found, skipping")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                qid = r.get("question_id")
                model = r.get("model")
                ch = r.get("channel_name")
                if qid in sample_qids and model in MODELS and ch in ("natural", "wagering"):
                    key = (model, ch, qid)
                    # Keep the latest result if duplicates
                    lookup[key] = r.get("answer_correct")
    return lookup


def compute_exp1_accuracy(exp1_lookup: dict, model: str, channel: str, sample_qids: set[str]) -> tuple[float | None, int]:
    """Compute accuracy for a model+channel from existing Exp1 results.

    Returns (accuracy, n_questions_with_data)
    """
    correct = 0
    total = 0
    for qid in sample_qids:
        key = (model, channel, qid)
        if key in exp1_lookup and exp1_lookup[key] is not None:
            total += 1
            if exp1_lookup[key]:
                correct += 1
    if total == 0:
        return None, 0
    return correct / total, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Format-Matched Control Experiment")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    print("=" * 70)
    print("FORMAT-MATCHED CONTROL EXPERIMENT (Appendix N)")
    print("=" * 70)

    # 1. Load and sample questions
    print("\n[1] Loading questions...")
    questions = load_sampled_questions()
    sample_qids = set(q["question_id"] for q in questions)

    # 2. Run structured control on each model
    print("\n[2] Running structured control prompts...")
    all_records: list[dict] = []

    for model in MODELS:
        model_records = await run_model(model, questions, resume=args.resume)
        all_records.extend(model_records)

    # 3. Save results
    print(f"\n[3] Saving {len(all_records)} records to {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # 4. Load existing Exp1 results for comparison
    print("\n[4] Loading existing Exp1 results for Nat.Acc and Wag.Acc...")
    exp1_lookup = load_exp1_results(sample_qids)
    print(f"    Loaded {len(exp1_lookup)} existing results")

    # 5. Compute and display results table
    print("\n" + "=" * 70)
    print("RESULTS: Format-Matched Control Experiment")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Nat.Acc':>8} {'Struct.Acc':>10} {'Wag.Acc':>8} {'FmtEff':>8} {'WagEff':>8} {'n_struct':>8} {'n_nat':>6} {'n_wag':>6}")
    print("-" * 90)

    # Accumulators for mean
    sum_nat, sum_struct, sum_wag, sum_fmt, sum_wageff = 0.0, 0.0, 0.0, 0.0, 0.0
    n_models_with_data = 0

    for model in MODELS:
        # Struct.Acc from our new results
        model_records = [r for r in all_records if r["model"] == model]
        scored = [r for r in model_records if r.get("answer_correct") is not None]
        correct = sum(1 for r in scored if r["answer_correct"] is True)
        struct_acc = correct / len(scored) if scored else None

        # Nat.Acc and Wag.Acc from existing Exp1
        nat_acc, n_nat = compute_exp1_accuracy(exp1_lookup, model, "natural", sample_qids)
        wag_acc, n_wag = compute_exp1_accuracy(exp1_lookup, model, "wagering", sample_qids)

        # Format Effect = Struct.Acc - Nat.Acc
        fmt_effect = None
        if struct_acc is not None and nat_acc is not None:
            fmt_effect = struct_acc - nat_acc

        # Wager Effect = Wag.Acc - Struct.Acc
        wag_effect = None
        if wag_acc is not None and struct_acc is not None:
            wag_effect = wag_acc - struct_acc

        def fmt_pct(v):
            return f"{v*100:.1f}%" if v is not None else "N/A"

        def fmt_diff(v):
            return f"{v*100:+.1f}pp" if v is not None else "N/A"

        print(f"{model:<20} {fmt_pct(nat_acc):>8} {fmt_pct(struct_acc):>10} {fmt_pct(wag_acc):>8} "
              f"{fmt_diff(fmt_effect):>8} {fmt_diff(wag_effect):>8} "
              f"{len(scored):>8} {n_nat:>6} {n_wag:>6}")

        if struct_acc is not None and nat_acc is not None and wag_acc is not None:
            sum_nat += nat_acc
            sum_struct += struct_acc
            sum_wag += wag_acc
            sum_fmt += fmt_effect
            sum_wageff += wag_effect
            n_models_with_data += 1

    # Mean row
    if n_models_with_data > 0:
        print("-" * 90)
        mean_nat = sum_nat / n_models_with_data
        mean_struct = sum_struct / n_models_with_data
        mean_wag = sum_wag / n_models_with_data
        mean_fmt = sum_fmt / n_models_with_data
        mean_wageff = sum_wageff / n_models_with_data
        def fmt_pct(v):
            return f"{v*100:.1f}%" if v is not None else "N/A"
        def fmt_diff(v):
            return f"{v*100:+.1f}pp" if v is not None else "N/A"
        print(f"{'Mean':<20} {fmt_pct(mean_nat):>8} {fmt_pct(mean_struct):>10} {fmt_pct(mean_wag):>8} "
              f"{fmt_diff(mean_fmt):>8} {fmt_diff(mean_wageff):>8}")

    # 6. Per-domain breakdown for structured control
    print("\n\nPer-Domain Struct.Acc Breakdown:")
    print(f"{'Domain':<15}", end="")
    for model in MODELS:
        print(f" {model:>15}", end="")
    print()
    print("-" * (15 + 16 * len(MODELS)))
    for domain in DOMAINS:
        print(f"{domain:<15}", end="")
        for model in MODELS:
            dom_records = [r for r in all_records if r["model"] == model and r.get("domain") == domain and r.get("answer_correct") is not None]
            if dom_records:
                acc = sum(1 for r in dom_records if r["answer_correct"]) / len(dom_records)
                print(f" {acc*100:>14.1f}%", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
