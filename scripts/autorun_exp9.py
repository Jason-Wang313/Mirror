"""
Auto-restart wrapper for Experiment 9 phases.
Runs exp9 full → control2 → analysis sequentially with unlimited retries.

Usage:
  python scripts/autorun_exp9.py --run-id 20260312T140842
  python scripts/autorun_exp9.py --run-id 20260312T140842 --start-phase control2
  python scripts/autorun_exp9.py --run-id 20260312T140842 --models m1,m2,m3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RETRY_DELAY = 60
LOG_FILE = Path("data/results/pipeline_log.txt")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[AUTORUN {ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_phase(cmd: list[str], name: str, max_attempts: int = 999) -> bool:
    for attempt in range(1, max_attempts + 1):
        log(f"{name}: attempt {attempt}  cmd={' '.join(str(c) for c in cmd)}")
        try:
            r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
            if r.returncode == 0:
                log(f"{name}: SUCCESS")
                return True
            log(f"{name}: exit code {r.returncode}. Retry in {RETRY_DELAY}s...")
        except Exception as e:
            log(f"{name}: exception {e}. Retry in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default=None,
                        help="Comma-separated models override")
    parser.add_argument("--start-phase",
                        choices=["full", "control2", "analysis"],
                        default="full")
    args = parser.parse_args()

    run_id = args.run_id
    models_flag = ["--models", args.models] if args.models else []

    log(f"Starting autorun  run_id={run_id}  start={args.start_phase}  models={args.models or 'config default'}")

    phases = ["full", "control2", "analysis"]
    start_idx = phases.index(args.start_phase)

    for phase in phases[start_idx:]:
        if phase == "full":
            ok = run_phase(
                [sys.executable, "scripts/run_experiment_9.py",
                 "--mode", "full",
                 "--run-id", run_id,
                 "--resume"] + models_flag,
                "Exp9-Full"
            )
            if not ok:
                log("CRITICAL: Exp9 full failed permanently. Stopping.")
                sys.exit(1)

        elif phase == "control2":
            ok = run_phase(
                [sys.executable, "scripts/run_experiment_9.py",
                 "--mode", "control2",
                 "--run-id", run_id,
                 "--resume"] + models_flag,
                "Exp9-Control2"
            )
            if not ok:
                log("WARNING: Control2 failed. Proceeding to analysis without it.")

        elif phase == "analysis":
            ok = run_phase(
                [sys.executable, "scripts/analyze_experiment_9.py",
                 "--run-id", run_id],
                "Exp9-Analysis",
                max_attempts=5
            )
            if not ok:
                log("Analysis failed.")
                sys.exit(1)

    log(f"AUTORUN COMPLETE  run_id={run_id}")
    log(f"Results: data/results/exp9_{run_id}_results.jsonl")
    log(f"Analysis: data/results/exp9_{run_id}_analysis/")


if __name__ == "__main__":
    main()
