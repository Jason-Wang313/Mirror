"""
Retry all api_success=False trials from an Exp9 run.

Reads the main results file, finds every record where api_success=False,
re-runs those trials, and merges successful retries back into the main file
(replacing the failed records in-place).

Usage:
    python scripts/retry_failed_exp9.py --run-id 20260312T140842
    python scripts/retry_failed_exp9.py --run-id 20260312T140842 --concurrency 8
    python scripts/retry_failed_exp9.py --run-id 20260312T140842 --models llama-3.1-8b,deepseek-r1
    python scripts/retry_failed_exp9.py --run-id 20260312T140842 --merge-only

Notes:
    - qwen-3-235b is skipped by default (100% failure — provider issue, not transient)
    - Successful retries are written to a shard file first, then merged into main
    - The merge overwrites only records that were api_success=False
    - Safe to re-run: resume logic skips already-retried trials
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.agentic_paradigms import (
    get_paradigm,
    build_condition_prefix,
    build_false_score_prefix,
    NoToolBehavioralParadigm,
    classify_error_type,
)
from mirror.scoring.answer_matcher import match_answer_robust, extract_answer_from_response

# Import helpers from run_experiment_9 (avoid code duplication)
from scripts.run_experiment_9 import (
    load_exp1_metrics,
    load_tasks,
    get_domain_accuracy,
    get_domain_mirror_gap,
    identify_strength,
    split_response_into_parts,
    apply_condition4_routing,
    call_with_retry,
    CALL_TIMEOUT,
    TOOL_RE,
    DEFER_RE,
    HEDGE_WORDS,
    DECOMP_RE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("data/results")
TASKS_FILE = Path("data/exp9_tasks.jsonl")

# Skip qwen-3-235b by default: 100% failure rate indicates provider configuration
# issue rather than transient errors; retrying would waste quota and time.
DEFAULT_SKIP_MODELS = {"qwen-3-235b"}

# Model-specific settings (must match run_experiment_9.py)
def _get_max_tokens(model: str) -> int:
    if "deepseek-r1" in model:
        return 8000
    if "mistral-large" in model:
        return 600
    return 4000

def _get_timeout(model: str) -> int:
    slow = ("mistral-large", "llama-3.1-405b", "qwen-3-235b", "gpt-oss-120b",
            "command-r-plus", "phi-4")
    return 300 if any(s in model for s in slow) else CALL_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# Parsing (identical to run_experiment_9.py)
# ─────────────────────────────────────────────────────────────────────────────

def classify_section_p1(section: str, gt: str, atype: str) -> dict:
    deferred = bool(DEFER_RE.search(section))
    tool_used = bool(TOOL_RE.search(section))
    if deferred:
        return {"decision": "defer", "extracted_answer": None, "correct": False,
                "tool_used": tool_used, "deferred": True}
    extracted = extract_answer_from_response(section, answer_type=atype) or ""
    correct = match_answer_robust(extracted, gt, atype)
    return {"decision": "use_tool" if tool_used else "proceed", "extracted_answer": extracted,
            "correct": correct, "tool_used": tool_used, "deferred": False}

def extract_checkpoint_decision(text: str) -> str:
    m = re.search(r'(?:YOUR\s+)?DECISION\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if not m:
        if re.search(r'FLAG_FOR_REVIEW|FLAG\s+FOR\s+REVIEW', text, re.IGNORECASE): return "defer"
        if re.search(r'\bUSE_TOOL\b|USE TOOL', text, re.IGNORECASE): return "use_tool"
        return "proceed"
    d = m.group(1).strip().upper()
    if "FLAG" in d or "REVIEW" in d: return "defer"
    if "TOOL" in d: return "use_tool"
    return "proceed"

def extract_checkpoint_answer(text: str) -> str:
    m = re.search(r'(?:YOUR\s+)?ANSWER\s*:\s*([^\n]+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_behavioral_signals(section: str) -> dict:
    lower = section.lower()
    return {
        "hedge_count": sum(lower.count(w) for w in HEDGE_WORDS),
        "decomp_count": len(DECOMP_RE.findall(section)),
        "token_count": len(section.split()),
    }

def parse_slot(section: str, gt: str, atype: str, forced: Optional[dict],
               paradigm_id: int, sec_a: str) -> dict:
    if forced is not None:
        return forced
    if paradigm_id == 1:
        return classify_section_p1(section, gt, atype)
    if paradigm_id == 2:
        decision = extract_checkpoint_decision(section)
        ans = extract_checkpoint_answer(section)
        correct = False if decision in ("defer", "use_tool") else match_answer_robust(ans, gt, atype)
        return {"decision": decision, "extracted_answer": ans, "correct": correct,
                "tool_used": decision == "use_tool", "deferred": decision == "defer"}
    if paradigm_id == 3:
        m = re.search(r'ANSWER\s*[12]?\s*[:\-]\s*([^\n]{1,300})', section, re.IGNORECASE)
        ans = m.group(1).strip() if m else ""
        correct = match_answer_robust(ans, gt, atype)
        return {"decision": "proceed", "extracted_answer": ans, "correct": correct,
                "tool_used": False, "deferred": False, **extract_behavioral_signals(section)}
    return {"decision": "proceed", "extracted_answer": "", "correct": False,
            "tool_used": False, "deferred": False}


# ─────────────────────────────────────────────────────────────────────────────
# Single trial (retry version)
# ─────────────────────────────────────────────────────────────────────────────

async def run_retry_trial(
    client: UnifiedClient,
    model: str,
    task: dict,
    condition: int,
    paradigm_id: int,
    exp1_metrics: dict,
    is_false_score_control: bool,
    already_retried: set,
    results_f,
    write_lock: asyncio.Lock,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    trial_key = (model, task["task_id"], condition, paradigm_id,
                 "c2_false" if is_false_score_control else "real")
    if trial_key in already_retried:
        return None  # Already successfully retried in a previous run

    domain_a = task.get("domain_a", "")
    domain_b = task.get("domain_b", "")
    acc_a = get_domain_accuracy(model, domain_a, exp1_metrics)
    acc_b = get_domain_accuracy(model, domain_b, exp1_metrics)
    gap_a = get_domain_mirror_gap(model, domain_a, exp1_metrics)
    gap_b = get_domain_mirror_gap(model, domain_b, exp1_metrics)
    strength_a = identify_strength(model, domain_a, exp1_metrics)
    strength_b = identify_strength(model, domain_b, exp1_metrics)
    if task.get("task_type") == "tailored":
        strength_a, strength_b = "strong", "weak"

    forced_a = forced_b = None
    if condition == 4:
        forced_a = apply_condition4_routing(task, "a", acc_a, task["correct_answer_a"], task["answer_type_a"])
        forced_b = apply_condition4_routing(task, "b", acc_b, task["correct_answer_b"], task["answer_type_b"])

    if is_false_score_control:
        false_acc_a = (1.0 - acc_a) if acc_a is not None else 0.31
        false_acc_b = (1.0 - acc_b) if acc_b is not None else 0.92
        prefix = build_false_score_prefix(domain_a, domain_b, false_acc_a, false_acc_b)
    else:
        prefix = build_condition_prefix(condition, domain_a, domain_b, acc_a, acc_b)

    paradigm = get_paradigm(paradigm_id)
    prompt = paradigm.format_prompt(task, condition_prefix=prefix)

    async with sem:
        response = await call_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=_get_max_tokens(model),
            metadata={"experiment": "exp9_retry", "task_id": task["task_id"],
                      "condition": condition, "paradigm": paradigm_id},
            timeout=_get_timeout(model),
        )

        raw_response = ""
        api_success = False
        if response and "error" not in response:
            raw_response = response.get("content") or ""
            api_success = True

        sec_a, sec_b = split_response_into_parts(raw_response)
        result_a = parse_slot(sec_a, task["correct_answer_a"], task["answer_type_a"],
                               forced_a, paradigm_id, sec_a)
        result_b = parse_slot(sec_b, task["correct_answer_b"], task["answer_type_b"],
                               forced_b, paradigm_id, sec_a)

        error_type_a = error_type_b = None
        if paradigm_id == 3:
            error_type_a = classify_error_type(result_a.get("extracted_answer", ""),
                                               result_a["correct"], result_a.get("hedge_count", 0))
            error_type_b = classify_error_type(result_b.get("extracted_answer", ""),
                                               result_b["correct"], result_b.get("hedge_count", 0))

        record = {
            "model": model, "task_id": task["task_id"], "condition": condition,
            "paradigm": paradigm_id, "is_false_score_control": is_false_score_control,
            "task_type": task.get("task_type"), "circularity_free": task.get("circularity_free", False),
            "domain_a": domain_a, "domain_b": domain_b,
            "subcategory_a": task.get("subcategory_a"), "subcategory_b": task.get("subcategory_b"),
            "difficulty_a": task.get("difficulty_a"), "difficulty_b": task.get("difficulty_b"),
            "strength_a": strength_a, "strength_b": strength_b,
            "component_a_decision": result_a["decision"], "component_a_correct": result_a["correct"],
            "component_a_answer": result_a.get("extracted_answer", ""),
            "component_a_tool_used": result_a.get("tool_used", False),
            "component_a_deferred": result_a.get("deferred", False),
            "component_a_externally_routed": result_a.get("externally_routed", False),
            "component_b_decision": result_b["decision"], "component_b_correct": result_b["correct"],
            "component_b_answer": result_b.get("extracted_answer", ""),
            "component_b_tool_used": result_b.get("tool_used", False),
            "component_b_deferred": result_b.get("deferred", False),
            "component_b_externally_routed": result_b.get("externally_routed", False),
            "exp1_accuracy_a": acc_a, "exp1_accuracy_b": acc_b,
            "mirror_gap_a": gap_a, "mirror_gap_b": gap_b,
            "hedge_count_a": result_a.get("hedge_count"), "hedge_count_b": result_b.get("hedge_count"),
            "decomp_count_a": result_a.get("decomp_count"), "decomp_count_b": result_b.get("decomp_count"),
            "token_count_a": result_a.get("token_count"), "token_count_b": result_b.get("token_count"),
            "error_type_a": error_type_a, "error_type_b": error_type_b,
            "raw_response": raw_response, "api_success": api_success,
            "timestamp": datetime.now().isoformat(),
            "_retry": True,
        }

        status = "OK" if api_success else "FAIL"
        dec_a = result_a["decision"][0].upper()
        dec_b = result_b["decision"][0].upper()
        print(f"    [{status}] C{condition}P{paradigm_id} {task['task_id'][:20]:<20} "
              f"A:{dec_a} B:{dec_b}", flush=True)

        if api_success:
            async with write_lock:
                results_f.write(json.dumps(record) + "\n")
                results_f.flush()
                os.fsync(results_f.fileno())
                already_retried.add(trial_key)

    return record if api_success else None


# ─────────────────────────────────────────────────────────────────────────────
# Merge: replace failed records with successful retries
# ─────────────────────────────────────────────────────────────────────────────

def merge_retries_into_main(run_id: str, retry_shard: Path) -> dict:
    """
    Replace api_success=False records in the main results file with successful
    retries from the shard. Returns stats dict.
    """
    main_file = RESULTS_DIR / f"exp9_{run_id}_results.jsonl"
    if not main_file.exists():
        print(f"ERROR: Main file not found: {main_file}")
        return {}

    # Load retry shard — keep only successful retries
    retry_map: dict = {}
    if retry_shard.exists():
        for line in retry_shard.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                r = json.loads(line)
                if r.get("api_success"):
                    k = (r["model"], r["task_id"], r["condition"], r["paradigm"],
                         "c2_false" if r.get("is_false_score_control") else "real")
                    retry_map[k] = line
            except Exception:
                pass
    print(f"Merge: {len(retry_map)} successful retries to merge")

    # Rebuild main file: replace failed records with retries
    replaced = 0
    kept_failed = 0
    kept_ok = 0
    new_lines: list[str] = []

    for line in main_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            k = (r.get("model"), r.get("task_id"), r.get("condition"), r.get("paradigm"),
                 "c2_false" if r.get("is_false_score_control") else "real")
            if not r.get("api_success") and k in retry_map:
                new_lines.append(retry_map[k])
                replaced += 1
            else:
                new_lines.append(line)
                if r.get("api_success"):
                    kept_ok += 1
                else:
                    kept_failed += 1
        except Exception:
            new_lines.append(line)

    # Write atomically (temp file + rename)
    tmp = main_file.with_suffix(".tmp")
    tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    tmp.replace(main_file)

    stats = {"replaced": replaced, "kept_failed": kept_failed, "kept_ok": kept_ok,
             "total": len(new_lines)}
    print(f"Merge complete: replaced={replaced}, still_failed={kept_failed}, "
          f"ok={kept_ok}, total={len(new_lines)}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def retry_run(
    run_id: str,
    models_filter: Optional[list],
    concurrency: int,
    skip_models: set,
    retry_shard: Path,
) -> None:
    print("=" * 80)
    print("EXPERIMENT 9: RETRY FAILED TRIALS")
    print("=" * 80)
    print(f"Run ID:       {run_id}")
    print(f"Retry shard:  {retry_shard}")
    print(f"Concurrency:  {concurrency}")

    main_file = RESULTS_DIR / f"exp9_{run_id}_results.jsonl"
    if not main_file.exists():
        print(f"ERROR: {main_file} not found")
        return

    # Build failed_keys: (model, task_id, condition, paradigm, false_flag) → task metadata
    failed_trials: list[dict] = []
    seen: set = set()
    for line in main_file.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            if r.get("api_success"):
                continue
            model = r.get("model", "")
            if model in skip_models:
                continue
            if models_filter and model not in models_filter:
                continue
            k = (model, r.get("task_id"), r.get("condition"), r.get("paradigm"),
                 "c2_false" if r.get("is_false_score_control") else "real")
            if k in seen: continue
            seen.add(k)
            failed_trials.append({
                "model": model,
                "task_id": r["task_id"],
                "condition": r["condition"],
                "paradigm": r["paradigm"],
                "is_false_score_control": r.get("is_false_score_control", False),
            })
        except Exception:
            pass

    # Count by model
    from collections import Counter
    by_model = Counter(t["model"] for t in failed_trials)
    print(f"\nFailed trials to retry: {len(failed_trials)}")
    for m, c in sorted(by_model.items()):
        print(f"  {m}: {c}")
    print()

    if not failed_trials:
        print("Nothing to retry.")
        return

    # Load already-retried keys from existing shard
    already_retried: set = set()
    if retry_shard.exists():
        for line in retry_shard.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                r = json.loads(line)
                if r.get("api_success"):
                    k = (r["model"], r["task_id"], r["condition"], r["paradigm"],
                         "c2_false" if r.get("is_false_score_control") else "real")
                    already_retried.add(k)
            except Exception:
                pass
    if already_retried:
        print(f"Already retried (skipping): {len(already_retried)}")

    # Load resources
    exp1_metrics = load_exp1_metrics()
    all_tasks = load_tasks()
    task_by_id = {t["task_id"]: t for t in all_tasks}
    print(f"Tasks loaded: {len(all_tasks)}, Exp1 metrics: {len(exp1_metrics)} models")

    client = UnifiedClient(experiment=f"exp9_retry_{run_id}")
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    # Group by model for cleaner progress output
    models_ordered = list(dict.fromkeys(t["model"] for t in failed_trials))

    results_f = open(retry_shard, "a", encoding="utf-8")
    total_ok = 0
    try:
        for model in models_ordered:
            model_trials = [t for t in failed_trials if t["model"] == model]
            print(f"\n{'─' * 60}")
            print(f"Model: {model}  ({len(model_trials)} to retry)")
            print(f"{'─' * 60}")

            coros = []
            for t in model_trials:
                task = task_by_id.get(t["task_id"])
                if task is None:
                    print(f"  WARNING: task {t['task_id']} not found in tasks file — skipping")
                    continue
                coros.append(run_retry_trial(
                    client=client, model=model, task=task,
                    condition=t["condition"], paradigm_id=t["paradigm"],
                    exp1_metrics=exp1_metrics,
                    is_false_score_control=t["is_false_score_control"],
                    already_retried=already_retried,
                    results_f=results_f, write_lock=write_lock, sem=sem,
                ))

            results = await asyncio.gather(*coros)
            ok = sum(1 for r in results if r is not None)
            total_ok += ok
            print(f"  Done: {ok}/{len(model_trials)} successful retries for {model}")
    finally:
        results_f.close()

    print("\n" + "=" * 80)
    print(f"RETRY COMPLETE: {total_ok}/{len(failed_trials)} successful")
    print(f"Shard: {retry_shard}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Retry api_success=False trials from Exp9")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g. 20260312T140842)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated models to retry (default: all except qwen-3-235b)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent requests per model (default: 8)")
    parser.add_argument("--skip-models", default="qwen-3-235b",
                        help="Comma-separated models to skip (default: qwen-3-235b)")
    parser.add_argument("--output-file", default=None,
                        help="Retry shard file (default: exp9_{run_id}_retry_shard.jsonl)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip retry run; only merge existing shard into main results")
    parser.add_argument("--no-merge", action="store_true",
                        help="Run retry but do not merge into main results after")
    args = parser.parse_args()

    skip_models = set(m.strip() for m in args.skip_models.split(",") if m.strip())
    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    retry_shard = (Path(args.output_file) if args.output_file
                   else RESULTS_DIR / f"exp9_{args.run_id}_retry_shard.jsonl")

    if not args.merge_only:
        asyncio.run(retry_run(
            run_id=args.run_id,
            models_filter=models_filter,
            concurrency=args.concurrency,
            skip_models=skip_models,
            retry_shard=retry_shard,
        ))

    if not args.no_merge:
        print("\n" + "=" * 80)
        print("MERGING RETRIES INTO MAIN RESULTS")
        print("=" * 80)
        stats = merge_retries_into_main(args.run_id, retry_shard)
        remaining_failed = stats.get("kept_failed", "?")
        print(f"\nRemaining api_success=False records: {remaining_failed}")
        if remaining_failed == 0:
            print("All failed trials resolved!")
        else:
            print("Re-run retry script to attempt remaining failures.")


if __name__ == "__main__":
    main()
