"""
Monitor exp4 v2 expanded run and trigger analysis on completion.
Usage: python scripts/monitor_exp4_v2.py --run-id 20260314T135731
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"

MODELS = [
    "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
    "mistral-large", "gpt-oss-120b", "deepseek-r1", "deepseek-v3",
    # "gemini-2.5-pro",  # excluded: daily quota exhausted
    "phi-4",
    # "command-r-plus",  # excluded: NIM 404
    # "llama-3.3-70b",   # excluded: NIM rate limited
    "kimi-k2", "gemma-3-27b",
]
TARGET = 320
# Per-model overrides — llama-3.1-70b is NIM rate-limited; we have 242+ records
# already, so accept that as sufficient for analysis rather than waiting 4+ more hours
TARGET_OVERRIDE = {
    "llama-3.1-70b": 242,   # already met in both conditions
    "mistral-large":  320,  # full target — final analysis on completion
}
N_MODELS = len(MODELS)
COND_TARGET = TARGET * N_MODELS  # 4160 per condition (approximate, used for display only)


def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def count_by_model(path: Path) -> Counter:
    counts = Counter()
    if not path.exists():
        return counts
    for line in path.open():
        if line.strip():
            try:
                r = json.loads(line)
                counts[r.get("model", "?")] += 1
            except Exception:
                pass
    return counts


def check_done(counts: Counter) -> bool:
    return all(counts.get(m, 0) >= TARGET_OVERRIDE.get(m, TARGET) for m in MODELS)


def print_progress(run_id: str, counts_a: Counter, counts_b: Counter):
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    pct_a = total_a / COND_TARGET * 100
    pct_b = total_b / COND_TARGET * 100
    log(f"COND-A: {total_a}/{COND_TARGET} ({pct_a:.1f}%)  COND-B: {total_b}/{COND_TARGET} ({pct_b:.1f}%)")
    incomplete = [(m, counts_a.get(m, 0), counts_b.get(m, 0)) for m in MODELS
                  if counts_a.get(m, 0) < TARGET or counts_b.get(m, 0) < TARGET]
    for m, ca, cb in sorted(incomplete, key=lambda x: -(x[1] + x[2])):
        log(f"  {m:<24} A={ca:>3}  B={cb:>3}")


def run_analysis(run_id: str) -> bool:
    log("Running analysis...")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_exp4_expanded.py"),
         "--run-id", run_id],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        for line in (result.stdout or "").strip().split("\n")[-15:]:
            log(f"  {line}")
        log("Analysis complete.")
        return True
    else:
        log(f"Analysis FAILED: {(result.stderr or '')[-500:]}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    run_id = args.run_id
    log(f"=== Monitoring exp4_v2 run_id={run_id} ===")
    log(f"Target: {TARGET} trials/model × {N_MODELS} models × 2 conditions = {COND_TARGET*2} total")

    path_a = RESULTS_DIR / f"exp4_v2_{run_id}_condition_a_results.jsonl"
    path_b = RESULTS_DIR / f"exp4_v2_{run_id}_condition_b_results.jsonl"

    analysis_done = False
    last_counts_a = Counter()
    last_counts_b = Counter()
    stall_count = 0

    while True:
        time.sleep(args.interval)

        counts_a = count_by_model(path_a)
        counts_b = count_by_model(path_b)

        done_a = check_done(counts_a)
        done_b = check_done(counts_b)

        print_progress(run_id, counts_a, counts_b)

        # Stall detection
        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())
        prev_a = sum(last_counts_a.values())
        prev_b = sum(last_counts_b.values())
        if total_a == prev_a and total_b == prev_b:
            stall_count += 1
            if stall_count >= 5:
                log(f"WARNING: No progress for {stall_count * args.interval}s. Processes may have stalled.")
        else:
            stall_count = 0
        last_counts_a = counts_a
        last_counts_b = counts_b

        if done_a and done_b and not analysis_done:
            log("=== BOTH CONDITIONS COMPLETE — running analysis ===")
            run_analysis(run_id)
            analysis_done = True
            log("=== ALL DONE ===")
            break
        elif done_a and not analysis_done:
            log("Condition A complete — waiting for Condition B...")
        elif done_b and not analysis_done:
            log("Condition B complete — waiting for Condition A...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
