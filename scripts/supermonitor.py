"""
Supermonitor: watches all active tasks, detects stalls/stopped processes,
auto-restarts, triggers analyses on completion. Runs until everything is done.
"""
import glob
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/supermonitor.log")
LOG_PATH.parent.mkdir(exist_ok=True)


def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def count_model(path, model):
    n = 0
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("model") == model:
                        n += 1
    except Exception:
        pass
    return n


def is_win_pid_alive(pid):
    if not pid:
        return False
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in r.stdout.strip()
    except Exception:
        return False


def find_win_pid(keyword):
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if keyword in line:
                parts = line.split()
                if parts:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        pass
    except Exception:
        pass
    return None


def launch_bg(cmd, log_path):
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    return proc.pid


def run_analysis(cmd, name):
    log(f"  [analysis] {name}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            lines = (r.stdout or "").strip().split("\n")
            for line in lines[-5:]:
                log(f"    {line}")
            log(f"  [analysis] DONE: {name}")
        else:
            log(f"  [analysis] FAILED: {name} — {r.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        log(f"  [analysis] TIMEOUT: {name}")


def merge_exp6():
    orig = Path("data/results/exp6_20260313T201832_results.jsonl")
    new  = Path("data/results/exp6_20260313T224756_results.jsonl")
    merged = Path("data/results/exp6_combined_results.jsonl")
    records = {}
    for src in [orig, new]:
        if not src.exists():
            continue
        with open(src) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        key = (r.get("model"), r.get("task_id") or r.get("id"),
                               r.get("sub_experiment") or r.get("task_type"))
                        records[key] = line
                    except Exception:
                        pass
    with open(merged, "w", encoding="utf-8") as out:
        for line in records.values():
            out.write(line if line.endswith("\n") else line + "\n")
    log(f"  Merged {len(records)} records -> {merged}")
    return merged


# ── task definitions ────────────────────────────────────────────────────────

class Task:
    def __init__(self, name, file, model, target, runner, log_path,
                 pid_keyword, analysis_cmd=None):
        self.name = name
        self.file = Path(file)
        self.model = model
        self.target = target
        self.runner = runner          # list of args or None
        self.log_path = Path(log_path) if log_path else None
        self.pid_keyword = pid_keyword  # substring to find in wmic output
        self.analysis_cmd = analysis_cmd
        self.done = False
        self.win_pid = None
        self.prev_count = -1
        self.stall_ticks = 0

    def count(self):
        return count_model(self.file, self.model)

    def detect_pid(self):
        if self.pid_keyword:
            self.win_pid = find_win_pid(self.pid_keyword)

    def alive(self):
        return is_win_pid_alive(self.win_pid)

    def launch(self):
        if self.runner is None:
            return
        log_p = self.log_path or Path(f"logs/{self.name}.log")
        log(f"  Launching {self.name} ...")
        bash_pid = launch_bg(self.runner, log_p)
        time.sleep(3)
        self.detect_pid()
        log(f"  {self.name} started (bash PID {bash_pid}, win PID {self.win_pid})")
        self.stall_ticks = 0

    def kill(self):
        if self.win_pid:
            try:
                subprocess.run(
                    ["powershell", "-Command",
                     f"Stop-Process -Id {self.win_pid} -Force"],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass


STALL_LIMIT = 5  # consecutive no-progress checks before restart

tasks = [
    Task(
        name="exp3_dsv3",
        file="data/results/exp3_20260313T205339_results.jsonl",
        model="deepseek-v3",
        target=215,
        runner=["python", "-u", "scripts/turbo_exp3_deepseekv3.py"],
        log_path="logs/turbo_exp3.log",
        pid_keyword="turbo_exp3_deepseekv3",
        analysis_cmd=["python", "scripts/analyze_experiment_3.py",
                      "--run-id", "20260313T205339"],
    ),
    Task(
        name="exp5_8b",
        file="data/results/exp5_clean_20260314T092857_results.jsonl",
        model="llama-3.1-8b",
        target=320,
        runner=["python", "-u", "scripts/run_exp5_clean_control.py",
                "--n-questions", "320", "--models", "llama-3.1-8b"],
        log_path="logs/exp5_clean_llama8b_v2.log",
        pid_keyword="llama-3.1-8b",
        analysis_cmd=None,
    ),
    Task(
        name="exp6_gemma",
        file="data/results/exp6_20260313T224756_results.jsonl",
        model="gemma-3-27b",
        target=520,
        runner=["python", "-u", "scripts/run_experiment_6.py",
                "--mode", "full", "--resume", "--models", "gemma-3-27b"],
        log_path="logs/exp6_gemma_only.log",
        pid_keyword="gemma-3-27b",
        analysis_cmd=["python", "scripts/analyze_experiment_6.py",
                      "--file", "data/results/exp6_20260313T224756_results.jsonl"],
    ),
]

# separate tracking for exp5_70b (managed by PID 59004, no restart needed if alive)
exp5_70b_file = Path("data/results/exp5_clean_20260314T092857_results.jsonl")
exp5_70b_done = False

log("=== Supermonitor starting ===")

# Detect existing processes
for t in tasks:
    t.detect_pid()
    n = t.count()
    t.prev_count = n
    status = "ALIVE" if t.alive() else "DEAD"
    log(f"  {t.name}: {n}/{t.target}, PID={t.win_pid} [{status}]")

# Launch any that aren't running yet and aren't done
for t in tasks:
    n = t.count()
    if n >= t.target:
        t.done = True
        log(f"  {t.name} already DONE ({n}/{t.target})")
    elif not t.alive() and t.runner:
        t.launch()

log("Entering monitoring loop (60s interval)...")

while True:
    time.sleep(60)

    all_done = True

    for t in tasks:
        if t.done:
            continue

        n = t.count()
        progress = n - t.prev_count if t.prev_count >= 0 else 0
        t.prev_count = n

        if n >= t.target:
            log(f"✓ {t.name}: {n}/{t.target} COMPLETE")
            t.done = True
            if t.analysis_cmd:
                run_analysis(t.analysis_cmd, t.name)
            if t.name == "exp6_gemma":
                merged = merge_exp6()
                run_analysis(
                    ["python", "scripts/analyze_experiment_6.py",
                     "--file", str(merged)],
                    "Exp6 combined (12 models)",
                )
            continue

        all_done = False

        # stall detection
        if progress == 0:
            t.stall_ticks += 1
        else:
            t.stall_ticks = 0

        alive = t.alive()
        stalled = t.stall_ticks >= STALL_LIMIT

        rate = f"{progress}/min" if progress > 0 else "—"
        eta = f"~{(t.target - n) // max(progress, 1)}min" if progress > 0 else "?"
        status = "STALLED" if stalled else ("OK" if alive else "STOPPED")
        log(f"  {t.name}: {n}/{t.target} +{progress} [{status}] {rate} ETA={eta}")

        if (stalled or not alive) and t.runner:
            log(f"  → Restarting {t.name} (stall={t.stall_ticks}, alive={alive})")
            t.kill()
            time.sleep(2)
            t.launch()

    # exp5_70b — passive monitoring only
    if not exp5_70b_done:
        n70 = count_model(exp5_70b_file, "llama-3.1-70b")
        if n70 >= 320:
            log(f"✓ exp5_70b: {n70}/320 COMPLETE")
            exp5_70b_done = True
        else:
            alive_59004 = is_win_pid_alive(59004)
            if not alive_59004 and n70 < 320:
                log(f"  exp5_70b: {n70}/320 — runner stopped, restarting...")
                p = launch_bg(
                    ["python", "-u", "scripts/run_exp5_clean_control.py",
                     "--n-questions", "320", "--models", "llama-3.1-70b"],
                    Path("logs/exp5_70b_restart.log"),
                )
                log(f"  exp5_70b relaunched bash PID {p}")
            else:
                log(f"  exp5_70b: {n70}/320 [{'RUNNING' if alive_59004 else 'OK(maybe done)'}]")
        if not exp5_70b_done:
            all_done = False

    if all_done and exp5_70b_done:
        log("=== ALL TASKS COMPLETE ===")
        # Final exp5 clean analysis for llama models
        n8b_max = 0
        for fp in glob.glob("data/results/exp5_clean_*_results.jsonl"):
            n8b_max = max(n8b_max, count_model(Path(fp), "llama-3.1-8b"))
        log(f"  llama-3.1-8b clean records: {n8b_max}")
        if n8b_max >= 300:
            run_analysis(
                ["python", "scripts/analyze_experiment_5.py",
                 "data/results/exp5_20260313T205347_results.jsonl",
                 "--clean-baseline", "data/results/exp5_clean_20260313T205910_results.jsonl"],
                "Exp5 final re-analysis",
            )
        break

log("Supermonitor finished.")
