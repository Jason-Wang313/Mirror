"""
Auto-trigger analyses once data files stabilize.
Monitors Exp3 deepseek-v3 completion and Exp6 gemma-3-27b completion.
"""
import json
import time
import subprocess
from pathlib import Path
from collections import Counter

EXP3_FILE = Path("data/results/exp3_20260313T205339_results.jsonl")
EXP6_ORIG_FILE = Path("data/results/exp6_20260313T201832_results.jsonl")
EXP6_NEW_FILE = Path("data/results/exp6_20260313T224756_results.jsonl")
EXP3_RUN_ID = "20260313T205339"


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


def run(cmd, name):
    print(f"\n>>> Running: {name}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        lines = (result.stdout or "").strip().split("\n")
        for line in lines[-8:]:
            print(f"  {line}")
        print(f"  DONE: {name}")
    else:
        print(f"  FAILED: {name}: {result.stderr[-300:]}")


def merge_exp6_files():
    merged_path = Path("data/results/exp6_combined_results.jsonl")
    records = {}
    for src in [EXP6_ORIG_FILE, EXP6_NEW_FILE]:
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
    with open(merged_path, "w", encoding="utf-8") as out:
        for line in records.values():
            out.write(line if line.endswith("\n") else line + "\n")
    print(f"  Merged {len(records)} unique records -> {merged_path}")
    return merged_path


exp3_done = False
exp6_done = False

print("Watching for Exp3 and Exp6 completion...", flush=True)
while not (exp3_done and exp6_done):
    time.sleep(30)

    if not exp3_done:
        n = count_model(EXP3_FILE, "deepseek-v3")
        print(f"  [Exp3] deepseek-v3: {n}/215", flush=True)
        if n >= 215:
            exp3_done = True
            run(["python", "scripts/analyze_experiment_3.py", "--run-id", EXP3_RUN_ID], "Exp3 analysis")

    if not exp6_done:
        n = count_model(EXP6_NEW_FILE, "gemma-3-27b")
        print(f"  [Exp6] gemma-3-27b: {n}/520", flush=True)
        if n >= 520:
            exp6_done = True
            run(["python", "scripts/analyze_experiment_6.py", "--file",
                 str(EXP6_NEW_FILE)], "Exp6 analysis (4 new models)")
            merged = merge_exp6_files()
            run(["python", "scripts/analyze_experiment_6.py", "--file",
                 str(merged)], "Exp6 combined analysis (all 12 models)")

print("All done!", flush=True)
