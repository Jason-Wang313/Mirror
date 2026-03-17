"""
Parallel retry launcher for Exp9 failed trials.

Runs one subprocess per model in parallel (each with its own retry shard),
then merges all shards into the main results file.

Concurrency tuned per model:
  - llama-3.1-8b/70b:    concurrency=16  (NIM, small models — handle high load)
  - llama-3.1-405b:      concurrency=12  (NIM, larger)
  - gpt-oss-120b:        concurrency=12  (NIM)
  - deepseek-r1:         concurrency=12  (DeepSeek API — separate provider)
  - mistral-large:       concurrency=4   (NIM 675B — hard ~23 req/min limit)

Usage:
    python scripts/launch_retry_exp9.py --run-id 20260312T140842
    python scripts/launch_retry_exp9.py --run-id 20260312T140842 --merge-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

RUN_ID_DEFAULT = "20260312T140842"
RESULTS_DIR = Path("data/results")
CWD = str(Path(__file__).parent.parent)

# Per-model concurrency — tuned to provider capacity
MODEL_CONCURRENCY: dict[str, int] = {
    "llama-3.1-8b":   16,
    "llama-3.1-70b":  16,
    "llama-3.1-405b": 12,
    "gpt-oss-120b":   12,
    "deepseek-r1":    12,
    "mistral-large":   4,  # NIM 675B hard limit
}

# Skip these entirely
SKIP_MODELS = {"qwen-3-235b"}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[RETRY {ts}] {msg}", flush=True)


def count_failed(run_id: str) -> dict[str, int]:
    main_file = RESULTS_DIR / f"exp9_{run_id}_results.jsonl"
    if not main_file.exists():
        return {}
    counts: dict[str, int] = {}
    for line in main_file.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            if not r.get("api_success"):
                m = r.get("model", "?")
                counts[m] = counts.get(m, 0) + 1
        except Exception:
            pass
    return counts


def merge_all_shards(run_id: str, shards: list[Path]) -> dict:
    """Merge successful retries from all shards into main results."""
    main_file = RESULTS_DIR / f"exp9_{run_id}_results.jsonl"

    # Collect all successful retries across shards (last write wins per key)
    retry_map: dict = {}
    for shard in shards:
        if not shard.exists(): continue
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                r = json.loads(line)
                if r.get("api_success"):
                    k = (r["model"], r["task_id"], r["condition"], r["paradigm"],
                         "c2_false" if r.get("is_false_score_control") else "real")
                    retry_map[k] = line
            except Exception:
                pass
    log(f"Total successful retries across all shards: {len(retry_map)}")

    if not retry_map:
        log("No successful retries to merge.")
        return {"replaced": 0, "kept_failed": 0}

    # Replace failed records in main file
    replaced = kept_failed = kept_ok = 0
    new_lines: list[str] = []
    for line in main_file.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            k = (r.get("model"), r.get("task_id"), r.get("condition"), r.get("paradigm"),
                 "c2_false" if r.get("is_false_score_control") else "real")
            if not r.get("api_success") and k in retry_map:
                new_lines.append(retry_map[k])
                replaced += 1
            else:
                new_lines.append(line)
                if r.get("api_success"): kept_ok += 1
                else: kept_failed += 1
        except Exception:
            new_lines.append(line)

    tmp = main_file.with_suffix(".tmp")
    tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    tmp.replace(main_file)
    log(f"Merge: replaced={replaced}, still_failed={kept_failed}, ok={kept_ok}")
    return {"replaced": replaced, "kept_failed": kept_failed}


def run_model_retry(
    run_id: str,
    model: str,
    concurrency: int,
    shard: Path,
    results: dict,
) -> None:
    cmd = [
        sys.executable, "scripts/retry_failed_exp9.py",
        "--run-id", run_id,
        "--models", model,
        "--concurrency", str(concurrency),
        "--output-file", str(shard),
        "--no-merge",  # merge is done centrally after all complete
    ]
    log(f"  {model}: starting (concurrency={concurrency}, shard={shard.name})")
    try:
        r = subprocess.run(cmd, cwd=CWD)
        results[model] = r.returncode
        status = "OK" if r.returncode == 0 else f"exit={r.returncode}"
        log(f"  {model}: {status}")
    except Exception as e:
        results[model] = -1
        log(f"  {model}: ERROR {e}")


def main():
    parser = argparse.ArgumentParser(description="Parallel retry launcher for Exp9")
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--models", default=None,
                        help="Comma-separated models (default: all with failures)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip retry; only merge existing shards")
    args = parser.parse_args()

    run_id = args.run_id

    log("=" * 60)
    log(f"EXP9 PARALLEL RETRY  run_id={run_id}")
    log("=" * 60)

    # Count current failures
    failed_counts = count_failed(run_id)
    log("Current failed counts:")
    for m, c in sorted(failed_counts.items()):
        skip = " [SKIP]" if m in SKIP_MODELS else ""
        log(f"  {m}: {c}{skip}")

    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
    else:
        models_to_run = [m for m, c in failed_counts.items()
                         if m not in SKIP_MODELS and c > 0]

    if not models_to_run:
        log("No models to retry.")
        return

    log(f"\nModels to retry: {models_to_run}")

    # Build shard paths
    shards = {
        m: RESULTS_DIR / f"exp9_{run_id}_{m.replace('.', '-').replace('/', '-')}_retry_shard.jsonl"
        for m in models_to_run
    }
    # Always include existing shards from skip list in merge (in case they were retried manually)
    all_shard_files = list(RESULTS_DIR.glob(f"exp9_{run_id}_*_retry_shard.jsonl"))

    if not args.merge_only:
        # Launch all models in parallel
        log(f"\nLaunching {len(models_to_run)} parallel retry processes...")
        threads = []
        proc_results: dict = {}
        for model in models_to_run:
            concurrency = MODEL_CONCURRENCY.get(model, 8)
            t = threading.Thread(
                target=run_model_retry,
                args=(run_id, model, concurrency, shards[model], proc_results),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        log("\nAll retry processes complete.")
        log(f"Results: {proc_results}")

    # Merge all shards
    log("\n" + "=" * 60)
    log("MERGING ALL RETRY SHARDS INTO MAIN RESULTS")
    log("=" * 60)
    # Collect all retry shards (including any from previous runs)
    all_shards = list(RESULTS_DIR.glob(f"exp9_{run_id}_*_retry_shard.jsonl"))
    log(f"Found {len(all_shards)} shard files to merge")
    stats = merge_all_shards(run_id, all_shards)

    log("\n" + "=" * 60)
    log("PARALLEL RETRY COMPLETE")
    log(f"  Replaced: {stats.get('replaced', 0)} failed records")
    log(f"  Remaining failures: {stats.get('kept_failed', 0)}")
    log(f"  Main file: data/results/exp9_{run_id}_results.jsonl")
    log("=" * 60)

    if stats.get("kept_failed", 0) > 0:
        log("\nThere are still failed records. Re-run to retry remaining failures.")
        log("For mistral-large specifically, consider separate NIM quota allocation.")


if __name__ == "__main__":
    main()
