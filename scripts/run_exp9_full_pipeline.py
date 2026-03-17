"""
Experiment 9 Full Pipeline Master Runner
=========================================

Runs all phases in sequence with automatic crash recovery:

  Phase 0 — Exp 1 supplemental (deepseek-v3, gemini-2.5-pro, claude-3.5-sonnet,
             phi-4, command-r-plus): fills MIRROR score gaps before Exp 9.
  Phase 1 — Exp 9 full (12 models, 597 tasks, 4 conditions, 3 paradigms).
  Phase 2 — Exp 9 Control 2 (false score injection, 150 tasks per model).
  Phase 3 — Full analysis pipeline.

Each phase auto-resumes on failure up to MAX_ATTEMPTS times.
All intermediate results are crash-resistant (fsync per trial).

Usage:
  python scripts/run_exp9_full_pipeline.py
  python scripts/run_exp9_full_pipeline.py --run-id 20260312T140842
  python scripts/run_exp9_full_pipeline.py --skip-exp1  # skip Phase 0
  python scripts/run_exp9_full_pipeline.py --start-phase 2  # resume from Phase 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Fixed run_id for this pipeline so phases share the same results file
DEFAULT_RUN_ID = "20260312T140842"

# Models missing from existing exp1 accuracy data (requires API key config)
# Note: claude-3.5-sonnet excluded — requires ANTHROPIC_API_KEY (not configured)
MISSING_EXP1_MODELS = [
    "deepseek-v3",
    "gemini-2.5-pro",
    "phi-4",
    "command-r-plus",
]

ALL_EXP9_MODELS = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "qwen-3-235b",
    "gpt-oss-120b",
    "deepseek-r1",
    "deepseek-v3",
    "gemini-2.5-pro",
    "phi-4",
    "command-r-plus",
    # "claude-3.5-sonnet",  # requires ANTHROPIC_API_KEY — add when key is available
]

MAX_ATTEMPTS = 20
RETRY_DELAY = 60  # seconds between retries


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[PIPELINE {ts}] {msg}"
    print(line, flush=True)
    # Append to pipeline log file
    with open("data/results/pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_with_retry(cmd: list[str], phase_name: str, max_attempts: int = MAX_ATTEMPTS) -> bool:
    """Run a command, retrying on non-zero exit up to max_attempts times."""
    for attempt in range(1, max_attempts + 1):
        log(f"{phase_name}: attempt {attempt}/{max_attempts}  cmd={' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
            if result.returncode == 0:
                log(f"{phase_name}: SUCCESS")
                return True
            log(f"{phase_name}: exit code {result.returncode}. Retrying in {RETRY_DELAY}s...")
        except Exception as e:
            log(f"{phase_name}: exception {e}. Retrying in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
    log(f"{phase_name}: FAILED after {max_attempts} attempts.")
    return False


def phase0_exp1_supplement(exp1_run_id: str) -> bool:
    """Run Experiment 1 for models missing from existing accuracy data."""
    log("=" * 60)
    log("PHASE 0: Exp 1 supplemental for 5 missing models")
    log(f"  Models: {', '.join(MISSING_EXP1_MODELS)}")
    log("=" * 60)

    models_arg = ",".join(MISSING_EXP1_MODELS)
    # Phase 0a: Layer 1
    cmd_layer1 = [
        sys.executable, "scripts/run_experiment_1.py",
        "--mode", "full",
        "--models", models_arg,
        "--run-id", exp1_run_id,
        "--resume",
    ]
    if not run_with_retry(cmd_layer1, "Exp1-Layer1"):
        log("PHASE 0 Layer 1 failed permanently. Continuing to exp9 with available data.")
        return False

    # Phase 0b: Layer 2 only
    cmd_layer2 = [
        sys.executable, "scripts/run_experiment_1.py",
        "--mode", "full",
        "--models", models_arg,
        "--run-id", exp1_run_id,
        "--resume",
        "--layer2-only",
    ]
    if not run_with_retry(cmd_layer2, "Exp1-Layer2"):
        log("PHASE 0 Layer 2 failed. Continuing with Layer 1 data only.")

    # Phase 0c: Analyze to produce accuracy JSON
    cmd_analyze = [
        sys.executable, "scripts/analyze_experiment_1.py",
        "--run-id", exp1_run_id,
    ]
    if not run_with_retry(cmd_analyze, "Exp1-Analyze", max_attempts=3):
        log("PHASE 0 analysis failed. Exp9 will use only existing 7-model data.")
        return False

    log("PHASE 0 complete. Exp1 accuracy data now available for all 12 models.")
    return True


def phase1_exp9_full(run_id: str, models: list[str]) -> bool:
    """Run Experiment 9 full."""
    log("=" * 60)
    log(f"PHASE 1: Experiment 9 FULL  (run_id={run_id})")
    log(f"  Models: {', '.join(models)}")
    log("=" * 60)

    cmd = [
        sys.executable, "scripts/run_experiment_9.py",
        "--mode", "full",
        "--run-id", run_id,
        "--resume",
        "--models", ",".join(models),
    ]
    return run_with_retry(cmd, "Exp9-Full")


def phase2_control2(run_id: str, models: list[str]) -> bool:
    """Run Experiment 9 Control 2 (false score injection)."""
    log("=" * 60)
    log(f"PHASE 2: Experiment 9 Control 2  (run_id={run_id})")
    log("=" * 60)

    cmd = [
        sys.executable, "scripts/run_experiment_9.py",
        "--mode", "control2",
        "--run-id", run_id,
        "--resume",
        "--models", ",".join(models),
    ]
    return run_with_retry(cmd, "Exp9-Control2")


def phase3_analysis(run_id: str) -> bool:
    """Run full analysis pipeline."""
    log("=" * 60)
    log(f"PHASE 3: Analysis  (run_id={run_id})")
    log("=" * 60)

    cmd = [
        sys.executable, "scripts/analyze_experiment_9.py",
        "--run-id", run_id,
    ]
    return run_with_retry(cmd, "Exp9-Analysis", max_attempts=3)


def main():
    parser = argparse.ArgumentParser(description="Experiment 9 Full Pipeline")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--exp1-run-id", default=None,
                        help="Run ID for supplemental exp1 (default: exp9_run_id + '_exp1')")
    parser.add_argument("--skip-exp1", action="store_true",
                        help="Skip Phase 0 (exp1 supplement for missing models)")
    parser.add_argument("--start-phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Start from this phase (0=exp1, 1=full, 2=control2, 3=analysis)")
    args = parser.parse_args()

    run_id = args.run_id
    exp1_run_id = args.exp1_run_id or f"{run_id}_exp1supp"

    # Ensure output dir and log file exist
    Path("data/results").mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("EXPERIMENT 9 FULL PIPELINE START")
    log(f"  Run ID:       {run_id}")
    log(f"  Exp1 Run ID:  {exp1_run_id}")
    log(f"  Start phase:  {args.start_phase}")
    log(f"  Skip exp1:    {args.skip_exp1}")
    log("=" * 60)

    start = args.start_phase
    ok = True

    # Phase 0: Supplemental Exp 1
    if start <= 0 and not args.skip_exp1:
        ok = phase0_exp1_supplement(exp1_run_id)
        if not ok:
            log("Phase 0 failed or skipped — will run Exp9 with available 7-model data.")

    # Determine which models have exp1 data now
    from pathlib import Path as P
    import json as _json
    exp1_files = sorted(P("data/results").glob("exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
    models_with_data: set = set()
    for fp in exp1_files:
        try:
            models_with_data.update(_json.loads(fp.read_text()).keys())
        except Exception:
            pass

    models_to_run = [m for m in ALL_EXP9_MODELS if m in models_with_data]
    log(f"Models with exp1 data ({len(models_to_run)}): {', '.join(models_to_run)}")
    missing = [m for m in ALL_EXP9_MODELS if m not in models_with_data]
    if missing:
        log(f"Models WITHOUT exp1 data (skipped): {', '.join(missing)}")

    if not models_to_run:
        log("ERROR: No models with exp1 data. Cannot run Exp9. Exiting.")
        sys.exit(1)

    # Phase 1: Exp9 Full
    if start <= 1:
        ok = phase1_exp9_full(run_id, models_to_run)
        if not ok:
            log("CRITICAL: Phase 1 (full run) failed. Aborting pipeline.")
            sys.exit(1)

    # Phase 2: Control 2
    if start <= 2:
        ok = phase2_control2(run_id, models_to_run)
        if not ok:
            log("WARNING: Phase 2 (control2) failed. Analysis will proceed without control 2 data.")

    # Phase 3: Analysis
    if start <= 3:
        ok = phase3_analysis(run_id)
        if not ok:
            log("ERROR: Phase 3 (analysis) failed.")
            sys.exit(1)

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log(f"  Results: data/results/exp9_{run_id}_results.jsonl")
    log(f"  Analysis: data/results/exp9_{run_id}_analysis/")
    log("=" * 60)


if __name__ == "__main__":
    main()
