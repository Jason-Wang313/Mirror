"""
Watch Phase 3 Track A backfill files for stability, then run analysis.

Monitors: Exp2, Exp3, Exp4 full, Exp5 adversarial, Exp5 clean backfills.
When each stabilizes (no size change for 5+ minutes), runs its analysis script.
"""
import time
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("data/results")
LOG_FILE = Path("logs/phase3_completion_watcher.log")

# Track which experiments to watch and their expected output
WATCHED = {
    "exp2": {
        "glob": "exp2_20260313T205532_results.jsonl",
        "min_records": 1000,  # 5 models × ~200+ records
        "analysis": ["python", "scripts/analyze_experiment_2.py", "--latest"],
        "done": False,
    },
    "exp3": {
        "glob": "exp3_20260313T205339_results.jsonl",
        "min_records": 800,   # 5 models × ~175 records
        "analysis": ["python", "scripts/analyze_experiment_3.py", "--latest"],
        "done": False,
    },
    "exp4_full": {
        "glob": "exp4_20260313T205349_results.jsonl",
        "min_records": 200,   # 5 models × ~46 records
        "analysis": ["python", "scripts/analyze_experiment_4.py", "--latest"],
        "done": False,
    },
    "exp5_adv": {
        "glob": "exp5_20260313T205347_results.jsonl",
        "min_records": 1500,  # 5 models × 320 records
        "analysis": None,  # Analysis done after clean too
        "done": False,
    },
    "exp5_clean": {
        "glob": "exp5_clean_20260313T205910_results.jsonl",
        "min_records": 2500,  # 9 models × 320 records, but many might not complete
        "analysis": ["python", "scripts/analyze_experiment_5.py", "--latest"],
        "done": False,
    },
}

# Stability tracking: {exp_name: [size1, size2, size3]}
stability = {k: [] for k in WATCHED}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_analysis(cmd, name):
    log(f"Running analysis for {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        log(f"  {name} analysis COMPLETE")
        # Print last few lines
        lines = (result.stdout or "").strip().split("\n")
        for line in lines[-5:]:
            log(f"  > {line}")
    else:
        log(f"  {name} analysis FAILED: {result.stderr[-500:]}")


def count_records(filepath):
    if not filepath.exists():
        return 0
    try:
        return sum(1 for _ in open(filepath, encoding="utf-8") if _.strip())
    except Exception:
        return 0


def check_model_coverage(filepath, required_models):
    """Return set of models with ≥20 records."""
    models = {}
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                m = r.get("model", "?")
                models[m] = models.get(m, 0) + 1
    except Exception:
        pass
    return {m for m, cnt in models.items() if cnt >= 20 and m != "?"}


TARGET_MODELS = {"gemma-3-27b", "kimi-k2", "phi-4", "llama-3.3-70b", "deepseek-v3"}


def main():
    log("Phase 3 Track A completion watcher started")
    log(f"Watching: {list(WATCHED.keys())}")

    all_done = False
    check_count = 0

    while not all_done:
        time.sleep(60)
        check_count += 1
        all_done = True

        for name, cfg in WATCHED.items():
            if cfg["done"]:
                continue
            all_done = False

            filepath = RESULTS_DIR / cfg["glob"]
            size = filepath.stat().st_size if filepath.exists() else 0
            n = count_records(filepath)

            # Track stability (last 5 checks)
            stability[name].append(size)
            if len(stability[name]) > 5:
                stability[name].pop(0)

            # Check if stable (no size change for 5+ checks)
            is_stable = (
                len(stability[name]) >= 5
                and len(set(stability[name])) == 1
                and size > 0
            )

            # Check model coverage
            covered = check_model_coverage(filepath, TARGET_MODELS)
            coverage_ok = TARGET_MODELS.issubset(covered)

            if check_count % 5 == 0:
                log(f"  {name}: {n} records, {len(covered)}/5 models, stable={is_stable}")

            # Mark done when stable AND has all 5 models (or stable for long time)
            if is_stable and (coverage_ok or len(stability[name]) >= 5):
                log(f"  {name}: COMPLETE! {n} records, models={sorted(covered)}")
                cfg["done"] = True

                if cfg["analysis"]:
                    run_analysis(cfg["analysis"], name)

                # Special: run exp5 analysis after BOTH adv + clean are done
                if name == "exp5_clean" and WATCHED["exp5_adv"]["done"]:
                    run_analysis(
                        ["python", "scripts/analyze_experiment_5.py", "--latest"],
                        "exp5 (combined adv+clean)",
                    )
                elif name == "exp5_adv" and WATCHED["exp5_clean"]["done"]:
                    run_analysis(
                        ["python", "scripts/analyze_experiment_5.py", "--latest"],
                        "exp5 (combined adv+clean)",
                    )

        if all_done:
            log("All Phase 3 Track A experiments complete! Final analysis done.")
            # Run exp4 analysis one more time with all data
            run_analysis(
                ["python", "scripts/analyze_experiment_4.py", "--latest"],
                "exp4 (final with syco)",
            )
            break

    log("Phase 3 watcher finished.")


if __name__ == "__main__":
    main()
