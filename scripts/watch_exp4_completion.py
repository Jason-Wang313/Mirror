"""
Crash-resilient watcher for Exp4 remaining trials.
Auto-restarts throttled llama-70b and r1 retry runners if they stall.
Polls every 90s. Exits when both models reach 320/320.

Usage:
  python scripts/watch_exp4_completion.py
"""
import json, os, subprocess, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "20260314T135731"
RESULTS_DIR = ROOT / "data" / "results"
LOG = ROOT / "logs" / "exp4_completion_watcher.log"
THROTTLED_LOG = ROOT / "logs" / "exp4_throttled_runner.log"
R1_LOG = ROOT / "logs" / "exp4_r1_retry.log"
STALL_TIMEOUT = 300  # 5 minutes without file change = restart

throttled_proc = None
r1_proc = None


def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def count_valid(cond):
    p = RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_{cond}_results.jsonl"
    if not p.exists():
        return {}, 0
    done = {}
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
            m = r.get("model", "")
            if not r.get("error") and r.get("phase_a"):
                done.setdefault(m, set()).add(r["trial_id"])
        except Exception:
            pass
    return done, os.path.getmtime(p)


def alive(proc):
    return proc is not None and proc.poll() is None


def launch_throttled():
    log("Launching throttled llama-70b runner...")
    f = open(THROTTLED_LOG, "a")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "throttled_exp4_llama70b.py"),
         "--run-id", RUN_ID, "--rpm", "3", "--condition", "both"],
        stdout=f, stderr=f
    )
    log(f"  Throttled PID={proc.pid}")
    return proc


def launch_r1():
    log("Launching deepseek-r1 retry runner...")
    f = open(R1_LOG, "a")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "retry_deepseekr1.py")],
        stdout=f, stderr=f
    )
    log(f"  R1 retry PID={proc.pid}")
    return proc


log(f"Watcher started. run_id={RUN_ID}")
throttled_proc = launch_throttled()
r1_proc = launch_r1()

last_cond_a_mtime = 0
last_cond_b_mtime = 0
last_change = time.time()

while True:
    time.sleep(90)

    done_a, mtime_a = count_valid("a")
    done_b, mtime_b = count_valid("b")

    llama_a = len(done_a.get("llama-3.1-70b", set()))
    llama_b = len(done_b.get("llama-3.1-70b", set()))
    r1_a = len(done_a.get("deepseek-r1", set()))
    r1_b = len(done_b.get("deepseek-r1", set()))

    log(f"llama-3.1-70b: {llama_a}/320 (A)  {llama_b}/320 (B) | "
        f"deepseek-r1: {r1_a}/320 (A)  {r1_b}/320 (B) | "
        f"throttled={'alive' if alive(throttled_proc) else 'DEAD'} "
        f"r1={'alive' if alive(r1_proc) else 'DEAD'}")

    if mtime_a > last_cond_a_mtime or mtime_b > last_cond_b_mtime:
        last_cond_a_mtime = mtime_a
        last_cond_b_mtime = mtime_b
        last_change = time.time()

    stale = time.time() - last_change

    # Restart throttled runner if dead or stalled
    if not alive(throttled_proc):
        if llama_a < 320 or llama_b < 320:
            log(f"Throttled runner dead. Restarting...")
            throttled_proc = launch_throttled()
            last_change = time.time()
    elif stale > STALL_TIMEOUT and (llama_a < 320 or llama_b < 320):
        log(f"Stalled {stale:.0f}s, restarting throttled runner...")
        throttled_proc.terminate()
        time.sleep(2)
        throttled_proc = launch_throttled()
        last_change = time.time()

    # Restart r1 if dead and still has work
    if not alive(r1_proc) and (r1_a < 320 or r1_b < 320):
        log("R1 retry runner dead. Restarting...")
        r1_proc = launch_r1()
        last_change = time.time()

    # Exit condition: both models done
    if llama_a >= 320 and llama_b >= 320 and r1_a >= 320 and r1_b >= 320:
        log("Both models at 320/320! Triggering final analysis...")
        subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "analyze_exp4_expanded.py"),
             "--run-id", RUN_ID],
            stdout=open(ROOT / "logs" / "exp4_final_analysis.log", "w"),
            stderr=subprocess.STDOUT
        )
        log("Analysis triggered. Watcher exiting.")
        break

    # Also exit if only llama remains and it's clearly going to take hours
    # (just keep running - user can Ctrl+C)
