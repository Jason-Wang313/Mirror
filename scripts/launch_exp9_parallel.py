"""
Parallel Experiment 9 launcher with automatic merge + control2 + analysis.

Strategy:
  - Launch one subprocess per model, each writing to its own shard file.
  - All shards run concurrently (no file-write conflicts).
  - Each shard uses CONCURRENCY=32 (set in run_experiment_9.py).
  - After all shards complete, merge into the main results file.
  - Then run control2 (same parallel approach) and full analysis.

Speedup vs sequential:
  - 7 parallel shards × 4× concurrency increase = ~10-15× faster overall.
  - Expected: ~1-2 hours total vs ~18 hours sequential.

Usage:
  python scripts/launch_exp9_parallel.py --run-id 20260312T140842
  python scripts/launch_exp9_parallel.py --run-id 20260312T140842 --start-phase control2
  python scripts/launch_exp9_parallel.py --run-id 20260312T140842 --models m1,m2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("data/results/pipeline_log.txt")
RETRY_DELAY = 30
MAX_MODEL_ATTEMPTS = 10

MODELS_7 = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "qwen-3-235b",
    "gpt-oss-120b",
    "deepseek-r1",
]

# Per-model concurrency overrides (lower for slow/large models to avoid rate limiting)
MODEL_CONCURRENCY: dict[str, int] = {
    "mistral-large": 8,  # 675B model — NIM peak throughput at 8 concurrent with max_tokens=1500
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[PARALLEL {ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def model_slug(model: str) -> str:
    return model.replace(".", "-").replace("/", "-")


def shard_path(run_id: str, model: str, mode: str = "full") -> Path:
    suffix = "c2" if mode == "control2" else "full"
    return Path("data/results") / f"exp9_{run_id}_{model_slug(model)}_{suffix}_shard.jsonl"


def run_model_shard(
    model: str,
    run_id: str,
    mode: str,
    resume: bool,
    max_attempts: int = MAX_MODEL_ATTEMPTS,
) -> bool:
    """Run a single model shard with auto-retry."""
    shard = shard_path(run_id, model, mode)
    for attempt in range(1, max_attempts + 1):
        cmd = [
            sys.executable, "scripts/run_experiment_9.py",
            "--mode", mode,
            "--run-id", run_id,
            "--models", model,
            "--output-file", str(shard),
        ]
        if attempt > 1 or resume:
            cmd.append("--resume")
        if model in MODEL_CONCURRENCY:
            cmd += ["--concurrency", str(MODEL_CONCURRENCY[model])]
        log(f"[{model}] {mode} attempt {attempt}/{max_attempts}")
        try:
            r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
            if r.returncode == 0:
                log(f"[{model}] {mode} DONE")
                return True
            log(f"[{model}] {mode} exit={r.returncode}. Retry in {RETRY_DELAY}s...")
        except Exception as e:
            log(f"[{model}] {mode} exception: {e}. Retry in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    log(f"[{model}] {mode} FAILED after {max_attempts} attempts")
    return False


def merge_shards(run_id: str, models: list[str], mode: str = "full") -> int:
    """
    Merge all model shard files into the main results file.
    Deduplicates by (model, task_id, condition, paradigm, is_false_score_control).
    Preserves existing records in the main file.
    Returns number of new records added.
    """
    main_file = Path("data/results") / f"exp9_{run_id}_results.jsonl"

    # Load existing main file
    existing_keys: set = set()
    existing_records: list = []
    if main_file.exists():
        for line in main_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                k = (r.get("model"), r.get("task_id"), r.get("condition"),
                     r.get("paradigm"), r.get("is_false_score_control"))
                existing_keys.add(k)
                existing_records.append(line)
            except json.JSONDecodeError:
                pass

    log(f"Merge: {len(existing_records)} existing records in main file")

    new_lines = []
    for model in models:
        shard = shard_path(run_id, model, mode)
        if not shard.exists():
            log(f"  [{model}] shard not found: {shard}")
            continue
        count = 0
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                k = (r.get("model"), r.get("task_id"), r.get("condition"),
                     r.get("paradigm"), r.get("is_false_score_control"))
                if k not in existing_keys:
                    existing_keys.add(k)
                    new_lines.append(line)
                    count += 1
            except json.JSONDecodeError:
                pass
        log(f"  [{model}] {count} new records from shard")

    if new_lines:
        with open(main_file, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")
        log(f"Merge complete: {len(new_lines)} new records appended to {main_file}")
    else:
        log("Merge: no new records to add")

    return len(new_lines)


def run_parallel_phase(
    models: list[str],
    run_id: str,
    mode: str,
    resume: bool,
    max_workers: int = 7,
) -> dict[str, bool]:
    """Run all models in parallel using a thread pool. Returns {model: success}."""
    results: dict[str, bool] = {}
    log(f"=== Starting {mode.upper()} parallel phase: {len(models)} models, {max_workers} workers ===")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_model_shard, m, run_id, mode, resume): m
            for m in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                ok = future.result()
                results[model] = ok
                log(f"[{model}] finished: {'OK' if ok else 'FAILED'}")
            except Exception as e:
                results[model] = False
                log(f"[{model}] exception in future: {e}")
    return results


def run_analysis(run_id: str) -> bool:
    log("=== Running analysis ===")
    cmd = [sys.executable, "scripts/analyze_experiment_9.py", "--run-id", run_id]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        if r.returncode == 0:
            log("Analysis DONE")
            return True
        log(f"Analysis exit={r.returncode}. Retry in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    log("Analysis FAILED")
    return False


def progress_report(run_id: str, models: list[str], mode: str) -> None:
    """Quick progress snapshot."""
    for m in models:
        shard = shard_path(run_id, m, mode)
        if shard.exists():
            lines = [l for l in shard.read_text(encoding="utf-8").splitlines() if l.strip()]
            log(f"  [{m}] {len(lines)} records in shard")
        else:
            log(f"  [{m}] no shard yet")


def main():
    parser = argparse.ArgumentParser(description="Parallel Exp9 launcher")
    parser.add_argument("--run-id", default="20260312T140842")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: 7 ready models)")
    parser.add_argument("--start-phase",
                        choices=["full", "control2", "analysis"],
                        default="full")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=7,
                        help="Max parallel model workers (default: 7 = one per model)")
    args = parser.parse_args()

    run_id = args.run_id
    models = [m.strip() for m in args.models.split(",")] if args.models else MODELS_7

    log("=" * 60)
    log(f"PARALLEL EXP9 PIPELINE  run_id={run_id}")
    log(f"  Models ({len(models)}): {', '.join(models)}")
    log(f"  Start phase: {args.start_phase}")
    log(f"  Workers: {args.workers}")
    log("=" * 60)

    phases = ["full", "control2", "analysis"]
    start_idx = phases.index(args.start_phase)

    for phase in phases[start_idx:]:
        if phase == "analysis":
            ok = run_analysis(run_id)
            if not ok:
                sys.exit(1)
            continue

        # Parallel model shards
        results = run_parallel_phase(models, run_id, phase, args.resume, args.workers)
        failed = [m for m, ok in results.items() if not ok]
        if failed:
            log(f"WARNING: {len(failed)} models failed in {phase}: {failed}")

        # Progress snapshot before merge
        progress_report(run_id, models, phase)

        # Merge shards into main file
        n_new = merge_shards(run_id, models, phase)
        log(f"Merged {n_new} new records for phase={phase}")

        # Clean up shard files on success
        for m in models:
            shard = shard_path(run_id, m, phase)
            if shard.exists() and results.get(m, False):
                shard.unlink()
                log(f"  Cleaned shard: {shard.name}")

    log("=" * 60)
    log("PARALLEL PIPELINE COMPLETE")
    log(f"  Results: data/results/exp9_{run_id}_results.jsonl")
    log(f"  Analysis: data/results/exp9_{run_id}_analysis/")
    log("=" * 60)


if __name__ == "__main__":
    main()
