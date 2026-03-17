"""
Crash-resilient watcher for Exp6 expanded run.
Runs as foreground process — auto-restarts runner if stalled/dead,
triggers analysis on completion.

Usage:
  python scripts/watch_exp6_expanded.py 20260314T203446
"""
import json, os, subprocess, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260314T203446"
RESULTS = ROOT / "data" / "results" / f"exp6_expanded_{RUN_ID}_results.jsonl"
LOG = ROOT / "logs" / f"exp6_expanded_{RUN_ID}_watcher.log"
RUNNER_LOG = ROOT / "logs" / f"exp6_expanded_{RUN_ID}_runner_watch.log"
TOTAL_EXPECTED = 13 * 240  # 13 models × 240 records each

runner_proc = None
analysis_launched = False

def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def count_records():
    if not RESULTS.exists():
        return 0, {}
    recs = []
    with open(RESULTS) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    model_counts = Counter(r.get("model") for r in recs)
    return len(recs), model_counts

def runner_alive(proc):
    if proc is None:
        return False
    return proc.poll() is None

def launch_runner():
    log("Launching runner (resume mode)...")
    f = open(RUNNER_LOG, "a")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_exp6_expanded.py"),
         "--run-id", RUN_ID, "--resume", "--workers", "20"],
        stdout=f, stderr=f
    )
    log(f"Runner PID={proc.pid}")
    return proc

def launch_analysis():
    log("Triggering final analysis...")
    subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "analyze_exp6_expanded.py"),
         "--run-id", RUN_ID],
        stdout=open(ROOT / "logs" / f"exp6_expanded_{RUN_ID}_analysis.log", "w"),
        stderr=subprocess.STDOUT
    )

log(f"Watcher started. run_id={RUN_ID}  target={TOTAL_EXPECTED} records")
last_count = 0
last_change = time.time()
STALL_TIMEOUT = 360  # 6 minutes

# Don't launch runner here — assume it's already running
# Watcher will restart if it detects stall or death

while True:
    time.sleep(60)
    total, model_counts = count_records()
    done_models = sum(1 for c in model_counts.values() if c >= 240)
    rate_str = f"{(total - last_count)}/min"

    log(f"{total}/{TOTAL_EXPECTED} ({100*total/TOTAL_EXPECTED:.1f}%)  "
        f"models_done={done_models}/13  rate={rate_str}  "
        f"runner={'alive' if runner_alive(runner_proc) else 'not_detected'}")

    if total > last_count:
        last_count = total
        last_change = time.time()
    else:
        stall_sec = time.time() - last_change
        if stall_sec > STALL_TIMEOUT and not runner_alive(runner_proc):
            log(f"Stalled {stall_sec:.0f}s + runner not detected. Restarting runner...")
            runner_proc = launch_runner()
            last_change = time.time()

    if total >= TOTAL_EXPECTED and not analysis_launched:
        log("COMPLETE! All records collected.")
        if runner_alive(runner_proc):
            runner_proc.terminate()
        launch_analysis()
        analysis_launched = True
        log("Watcher exiting.")
        break
