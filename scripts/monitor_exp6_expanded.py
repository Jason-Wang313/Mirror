"""
Monitor for Exp6 Expanded run — crash-resilient, auto-restarts stalled runner,
triggers analysis on completion.

Usage:
  python scripts/monitor_exp6_expanded.py --run-id 20260314T120000
  python scripts/monitor_exp6_expanded.py   # auto-detects latest
  python scripts/monitor_exp6_expanded.py --no-restart  # monitoring only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def count_records(path: Path):
    if not path.exists():
        return 0, {}
    sub_counts = {}
    total = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                sub = r.get("sub_experiment", "?")
                sub_counts[sub] = sub_counts.get(sub, 0) + 1
                total += 1
            except Exception:
                pass
    return total, sub_counts


def expected_total(models, tasks_6a=35, tasks_6b_flawed=30, tasks_6b_ctrl=30, tasks_6c=20):
    per_model = tasks_6a * 4 + tasks_6b_flawed + tasks_6b_ctrl + tasks_6c * 2
    return len(models) * per_model


def launch_runner(run_id, extra_args=None):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_exp6_expanded.py"),
        "--run-id", run_id,
        "--resume",
        "--workers", "20",
    ]
    if extra_args:
        cmd.extend(extra_args)
    log_path = LOG_DIR / f"exp6_expanded_{run_id}_runner.log"
    log_f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    print(f"  [MONITOR] Launched runner PID={proc.pid}  log={log_path}", flush=True)
    return proc


def launch_analysis(run_id):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_exp6_expanded.py"),
        "--run-id", run_id,
    ]
    log_path = LOG_DIR / f"exp6_expanded_{run_id}_analysis.log"
    log_f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    print(f"  [MONITOR] Analysis launched PID={proc.pid}  log={log_path}", flush=True)
    return proc


MODELS_ALL = [
    "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
    "mistral-large", "gpt-oss-120b",
    "deepseek-r1", "deepseek-v3", "gemini-2.5-pro",
    "phi-4", "command-r-plus",
    "llama-3.3-70b", "kimi-k2", "gemma-3-27b",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--stall-timeout", type=int, default=300,
                        help="Seconds with no new records before restart")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    models = args.models or MODELS_ALL

    # Resolve run_id
    if args.run_id:
        run_id = args.run_id
    else:
        files = sorted(RESULTS_DIR.glob("exp6_expanded_*_results.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            run_id = files[0].name.replace("exp6_expanded_", "").replace("_results.jsonl", "")
            print(f"[MONITOR] Auto-detected run_id={run_id}")
        else:
            run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            print(f"[MONITOR] No existing run found. Starting new run_id={run_id}")

    results_path = RESULTS_DIR / f"exp6_expanded_{run_id}_results.jsonl"
    total_expected = expected_total(models)

    print(f"[MONITOR] run_id={run_id}")
    print(f"[MONITOR] Results: {results_path}")
    print(f"[MONITOR] Expected: {total_expected} records ({len(models)} models)")
    print(f"[MONITOR] Poll: {args.poll_interval}s  Stall timeout: {args.stall_timeout}s")

    # Launch runner if not no-restart
    runner_proc = None
    if not args.no_restart:
        runner_proc = launch_runner(run_id)

    last_count = 0
    last_change_time = time.time()
    analysis_done = False
    check_count = 0

    while True:
        time.sleep(args.poll_interval)
        check_count += 1

        total, sub_counts = count_records(results_path)
        now_str = datetime.utcnow().strftime("%H:%M:%S")
        pct = 100 * total / total_expected if total_expected > 0 else 0

        print(f"[{now_str}] {total}/{total_expected} ({pct:.1f}%) — "
              f"6a={sub_counts.get('6a',0)} "
              f"6b={sub_counts.get('6b',0)} "
              f"6c_cap={sub_counts.get('6c_cap',0)} "
              f"6c_val={sub_counts.get('6c_val',0)}", flush=True)

        if total > last_count:
            last_count = total
            last_change_time = time.time()
        elif not args.no_restart and runner_proc is not None:
            stall_sec = time.time() - last_change_time
            if stall_sec > args.stall_timeout:
                # Check if runner is still alive
                if runner_proc.poll() is not None:
                    print(f"[MONITOR] Runner died (exit={runner_proc.returncode}). Restarting...",
                          flush=True)
                    runner_proc = launch_runner(run_id)
                    last_change_time = time.time()
                else:
                    print(f"[MONITOR] Stalled {stall_sec:.0f}s — runner alive PID={runner_proc.pid}",
                          flush=True)

        # Check completion
        if total >= total_expected and not analysis_done:
            print(f"[MONITOR] COMPLETE! {total} records. Triggering final analysis...", flush=True)
            launch_analysis(run_id)
            analysis_done = True
            # Kill runner if still running
            if runner_proc and runner_proc.poll() is None:
                runner_proc.terminate()
            print("[MONITOR] Done. Exiting.", flush=True)
            break

        # Periodic analysis every 30 checks (~30 minutes at 60s poll)
        if check_count % 30 == 0 and total > 0:
            print(f"[MONITOR] Periodic analysis at {total} records...", flush=True)
            launch_analysis(run_id)


if __name__ == "__main__":
    main()
