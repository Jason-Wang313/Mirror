"""
Watcher for gemma-3-12b backfill across Exp1/2/3/5.

Every 90s:
  - Checks completion of each experiment
  - Restarts dead Google AI runners (with -u flag)
  - Triggers analysis when an experiment hits its target
  - Bumps google_ai rate limit as experiments finish (fewer processes → higher RPM each)
  - Updates logs/gemma_backfill_watcher.log

Run IDs:
  exp1: 20260315T154652  target: 2640 records
  exp2: 20260315T154653  target: 1200 records
  exp3: 20260315T154654  target: 215 records
  exp5: 20260315T154655  target: 320 records

NIM runners (old config, slow fallback): PIDs 83312 / 82684 / 91656 / 86268
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLL_INTERVAL = 90  # seconds

EXPERIMENTS = {
    "exp1": {
        "run_id": "20260315T154652",
        "results": Path("data/results/exp1_20260315T154652_results.jsonl"),
        "target": 2640,
        "cmd_extra": ["--mode", "full", "--models", "gemma-3-12b", "--skip-analysis",
                      "--resume", "--run-id", "20260315T154652"],
        "script": "scripts/run_experiment_1.py",
        "analysis_cmd": ["python", "scripts/analyze_experiment_1.py",
                         "--run-id", "20260315T154652"],
    },
    "exp2": {
        "run_id": "20260315T154653",
        "results": Path("data/results/exp2_20260315T154653_results.jsonl"),
        "target": 1179,
        "cmd_extra": ["--mode", "full", "--models", "gemma-3-12b",
                      "--resume", "--run-id", "20260315T154653"],
        "script": "scripts/run_experiment_2.py",
        "analysis_cmd": ["python", "scripts/analyze_experiment_2.py",
                         "--run-id", "20260315T154653"],
    },
    "exp3": {
        "run_id": "20260315T154654",
        "results": Path("data/results/exp3_20260315T154654_results.jsonl"),
        "target": 215,
        "cmd_extra": ["--mode", "full", "--models", "gemma-3-12b",
                      "--resume", "--run-id", "20260315T154654"],
        "script": "scripts/run_experiment_3.py",
        "analysis_cmd": ["python", "scripts/analyze_experiment_3.py",
                         "--run-id", "20260315T154654"],
    },
    "exp5": {
        "run_id": "20260315T154655",
        "results": Path("data/results/exp5_20260315T154655_results.jsonl"),
        "target": 320,
        "cmd_extra": ["--mode", "full", "--models", "gemma-3-12b",
                      "--resume", "--run-id", "20260315T154655"],
        "script": "scripts/run_experiment_5.py",
        "analysis_cmd": ["python", "scripts/analyze_experiment_5.py",
                         "data/results/exp5_20260315T154655_results.jsonl"],
    },
}

# RPM per Google AI runner as a function of remaining active experiments
RPM_SCHEDULE = {4: 3, 3: 4, 2: 6, 1: 12}

# Rate schedule for when we need more RPM: how many RPM to set per process
RATE_LIMITER_PATH = Path("mirror/api/rate_limiter.py")

LOG_PATH = Path("logs/gemma_backfill_watcher.log")


def log(msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_success(results_path: Path, model: str = "gemma-3-12b") -> int:
    if not results_path.exists():
        return 0
    count = 0
    for line in open(results_path, encoding="utf-8"):
        line = line.strip()
        if not line or model not in line:
            continue
        try:
            r = json.loads(line)
            if r.get("model") == model and not r.get("error"):
                count += 1
        except Exception:
            pass
    return count


def get_running_pids() -> dict:
    """Return dict of script_basename -> list of PIDs (with -u flag = Google AI)."""
    result = {"nim": {}, "googleai": {}}
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             "Get-WmiObject Win32_Process -Filter \"Name='python.exe'\" "
             "| Select-Object ProcessId,CommandLine | Format-List"],
            capture_output=True, text=True, errors="replace", timeout=15
        )
        block = {}
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("ProcessId"):
                block["pid"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("CommandLine"):
                block["cmd"] = line.split(":", 1)[1].strip()
                cmd = block.get("cmd", "")
                pid = block.get("pid")
                for exp in EXPERIMENTS:
                    script = EXPERIMENTS[exp]["script"].replace("/", "\\").split("\\")[-1]
                    if script in cmd and "run_experiment" in cmd:
                        bucket = "googleai" if " -u " in cmd else "nim"
                        result[bucket].setdefault(exp, []).append(pid)
                block = {}
    except Exception as e:
        log(f"WARNING: could not enumerate processes: {e}")
    return result


def kill_pid(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def start_googleai_runner(exp: str) -> int:
    cfg = EXPERIMENTS[exp]
    log_name = f"logs/{exp}_gemma_googleai_{time.strftime('%Y%m%dT%H%M%S')}.log"
    cmd = [sys.executable, "-u", cfg["script"]] + cfg["cmd_extra"]
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_name, "a"),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    log(f"Started {exp} Google AI runner PID={proc.pid} → {log_name}")
    return proc.pid


def update_rate_limit(rpm: int):
    """Patch rate_limiter.py google_ai RPM in-place."""
    text = RATE_LIMITER_PATH.read_text(encoding="utf-8")
    import re
    new_text = re.sub(
        r'("google_ai":\s*RateLimiter\(requests_per_minute=)\d+(\))',
        rf'\g<1>{rpm}\g<2>',
        text,
    )
    if new_text != text:
        RATE_LIMITER_PATH.write_text(new_text, encoding="utf-8")
        log(f"Updated rate_limiter.py: google_ai → {rpm} RPM")
        return True
    return False


def trigger_analysis(exp: str):
    cfg = EXPERIMENTS[exp]
    log_path = f"logs/{exp}_analysis_{time.strftime('%Y%m%dT%H%M%S')}.log"
    log(f"Triggering analysis for {exp}: {' '.join(cfg['analysis_cmd'])}")
    subprocess.Popen(
        cfg["analysis_cmd"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


def main():
    LOG_PATH.parent.mkdir(exist_ok=True)
    log("=== gemma backfill watcher started ===")
    targets_str = ", ".join(f"{e}:{c['target']}" for e, c in EXPERIMENTS.items())
    log(f"Targets: {targets_str}")

    completed = set()      # experiments whose target has been reached
    analysis_triggered = set()
    active_rpm = 3         # current google_ai RPM setting
    last_restart = {}      # exp -> timestamp of last restart attempt

    while True:
        pids = get_running_pids()
        active_exps = [e for e in EXPERIMENTS if e not in completed]
        counts = {e: count_success(EXPERIMENTS[e]["results"]) for e in active_exps}

        # --- Check completion ---
        newly_done = []
        for exp in active_exps:
            n = counts[exp]
            tgt = EXPERIMENTS[exp]["target"]
            log(f"{exp}: {n}/{tgt}")
            if n >= tgt:
                log(f"✅ {exp} COMPLETE ({n}/{tgt})")
                completed.add(exp)
                newly_done.append(exp)

        # --- Trigger analysis for newly completed experiments ---
        for exp in newly_done:
            if exp not in analysis_triggered:
                trigger_analysis(exp)
                analysis_triggered.add(exp)
            # Kill its Google AI runner (experiment is done)
            for pid in pids["googleai"].get(exp, []):
                log(f"Killing finished {exp} Google AI runner PID={pid}")
                kill_pid(pid)

        # --- Bump rate limit as load decreases ---
        still_active = [e for e in EXPERIMENTS if e not in completed]
        n_active = len(still_active)
        if n_active > 0:
            desired_rpm = RPM_SCHEDULE.get(n_active, 3)
            if desired_rpm != active_rpm:
                if update_rate_limit(desired_rpm):
                    active_rpm = desired_rpm
                    # Restart Google AI runners to pick up new rate limit
                    log(f"Restarting Google AI runners at {desired_rpm} RPM "
                        f"({n_active} active experiments)")
                    for exp in still_active:
                        for pid in pids["googleai"].get(exp, []):
                            kill_pid(pid)
                        time.sleep(1)
                        start_googleai_runner(exp)
                    time.sleep(5)

        # --- Restart dead Google AI runners ---
        now = time.time()
        for exp in still_active:
            running = pids["googleai"].get(exp, [])
            if not running:
                last = last_restart.get(exp, 0)
                if now - last > 120:  # don't restart more than once per 2 min
                    log(f"⚠️  {exp} Google AI runner is dead — restarting")
                    start_googleai_runner(exp)
                    last_restart[exp] = now

        # --- All done ---
        if not still_active:
            log("🎉 All experiments complete! Watcher exiting.")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
