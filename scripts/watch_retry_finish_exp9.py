"""
Waits for all retry shards to stabilize, then merges and re-runs analysis.
Run this after launch_retry_exp9.py completes and mistral is still running.
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

RUN_ID = "20260312T140842"
RESULTS_DIR = Path("data/results")
CWD = str(Path(__file__).parent.parent)

SHARDS = {
    "mistral-large":  (1633, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_retry_shard.jsonl"),
    "deepseek-r1":    (779,  RESULTS_DIR / f"exp9_{RUN_ID}_deepseek-r1_retry_shard.jsonl"),
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[WATCH {ts}] {msg}", flush=True)


def count(path):
    if not path.exists(): return 0
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l).get("api_success"))


def main():
    log(f"Watching retry shards for run {RUN_ID}...")
    prev = {}
    stable_rounds = {}

    while True:
        time.sleep(60)
        curr = {slug: count(path) for slug, (_, path) in SHARDS.items()}
        done_all = all(curr[s] >= target for s, (target, _) in SHARDS.items())

        for slug, (target, _) in SHARDS.items():
            c = curr[slug]
            pct = 100 * c / target
            changed = c != prev.get(slug, -1)
            if changed:
                stable_rounds[slug] = 0
            else:
                stable_rounds[slug] = stable_rounds.get(slug, 0) + 1
            log(f"  {slug}: {c}/{target} ({pct:.1f}%) stable={stable_rounds[slug]}")
        prev = curr.copy()

        all_stable = all(stable_rounds.get(s, 0) >= 3 for s in SHARDS)

        if done_all or all_stable:
            log(f"All shards done or stable (done={done_all}, stable={all_stable})")
            break

    # Merge all retry shards into main results
    log("Merging all retry shards...")
    r = subprocess.run(
        [sys.executable, "scripts/launch_retry_exp9.py",
         "--run-id", RUN_ID, "--merge-only"],
        cwd=CWD
    )
    log(f"Merge exit={r.returncode}")

    # Re-run analysis
    log("Running final analysis...")
    r = subprocess.run(
        [sys.executable, "scripts/analyze_experiment_9.py",
         "--run-id", RUN_ID],
        cwd=CWD
    )
    log(f"Analysis exit={r.returncode}")
    log("DONE — check data/results/exp9_20260312T140842_analysis/analysis.json")


if __name__ == "__main__":
    main()
