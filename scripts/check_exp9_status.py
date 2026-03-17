"""
Quick status check for Experiment 9 full pipeline.

Usage:
  python scripts/check_exp9_status.py
  python scripts/check_exp9_status.py --run-id 20260312T140842
"""

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def check_exp1_progress(run_id_supp: str, results_dir: Path):
    f = results_dir / f"exp1_{run_id_supp}_results.jsonl"
    if not f.exists():
        print("  [exp1 supp] No results file yet.")
        return

    records = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_model = Counter(r.get("model", "?") for r in records)

    # Estimate: 4 models × 1087 questions × 6 channels = 26,088 total
    target_total = 4 * 1087 * 6
    print(f"  [exp1 supp] Records: {len(records)} / ~{target_total}  ({100*len(records)/target_total:.1f}%)")
    print(f"  By model: {dict(sorted(by_model.items()))}")

    # Rate
    if records:
        ts_first = records[0].get("timestamp", "")
        ts_last = records[-1].get("timestamp", "")
        if ts_first and ts_last:
            try:
                t0 = datetime.fromisoformat(ts_first)
                t1 = datetime.fromisoformat(ts_last)
                elapsed = (t1 - t0).total_seconds()
                if elapsed > 0:
                    rate = len(records) / elapsed
                    remaining = (target_total - len(records)) / rate
                    print(f"  Rate: {rate:.2f} rec/s — ETA: {remaining/3600:.1f} hours")
            except Exception:
                pass

    accuracy_file = results_dir / f"exp1_{run_id_supp}_accuracy.json"
    if accuracy_file.exists():
        accuracy = json.loads(accuracy_file.read_text())
        print(f"  [exp1 accuracy] Written for {len(accuracy)} models: {sorted(accuracy.keys())}")


def check_exp9_progress(run_id: str, results_dir: Path):
    f = results_dir / f"exp9_{run_id}_results.jsonl"
    if not f.exists():
        print("  [exp9 full] No results file yet.")
        return

    records = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = [r for r in records if not r.get("is_false_score_control")]
    c2 = [r for r in records if r.get("is_false_score_control")]

    by_model = Counter(r.get("model", "?") for r in real)
    # Estimate: 11 models × (297 fixed + ~25 tailored) × 11 combos ≈ 35,420
    target_real = 11 * 322 * 11
    print(f"  [exp9 full]  Real records: {len(real)} / ~{target_real}  ({100*len(real)/target_real:.1f}%)")
    print(f"  By model: {dict(sorted(by_model.items()))}")
    print(f"  [control2]   C2 records:   {len(c2)}")

    # CFR preview (condition 1, paradigm 1)
    c1p1 = [r for r in real if r.get("condition") == 1 and r.get("paradigm") == 1]
    if c1p1:
        weak_fail = sum(
            1 for r in c1p1
            if r.get("strength_a") == "weak"
            and r.get("component_a_decision") == "proceed"
            and not r.get("component_a_correct")
        )
        weak_total = sum(1 for r in c1p1 if r.get("strength_a") == "weak")
        cfr = weak_fail / weak_total if weak_total else float("nan")
        print(f"  Preview CFR (C1P1 weak): {cfr:.3f} (n_weak={weak_total})")

    # Rate
    if real:
        ts_last = real[-1].get("timestamp", "")
        print(f"  Last record: {ts_last[:19]}")


def check_pipeline_log(results_dir: Path):
    log = results_dir / "pipeline_log.txt"
    if not log.exists():
        return
    lines = log.read_text(encoding="utf-8").splitlines()
    print(f"\n  Pipeline log (last 6 lines):")
    for l in lines[-6:]:
        print(f"  {l}")


def check_processes():
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
            capture_output=True, text=True
        )
        lines = [l.strip() for l in result.stdout.splitlines() if "run_experiment" in l or "pipeline" in l]
        if lines:
            print(f"\n  Running processes ({len(lines)}):")
            for l in lines:
                pid = l.rsplit(None, 1)[-1]
                script = [p for p in l.split("python.exe") if p.strip()]
                print(f"    PID {pid}: {script[0][:80].strip() if script else '?'}")
        else:
            print("\n  WARNING: No experiment processes running!")
    except Exception as e:
        print(f"\n  Could not check processes: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260312T140842")
    args = parser.parse_args()

    results_dir = Path("data/results")
    run_id = args.run_id
    exp1_supp_run_id = f"{run_id}_exp1supp"

    print(f"{'='*60}")
    print(f"EXPERIMENT 9 PIPELINE STATUS  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}")

    print("\n[PHASE 0] Exp1 supplemental:")
    check_exp1_progress(exp1_supp_run_id, results_dir)

    print("\n[PHASE 1+2] Exp9 full + Control2:")
    check_exp9_progress(run_id, results_dir)

    check_pipeline_log(results_dir)
    check_processes()

    print(f"\n{'='*60}")
    print("To resume if pipeline dies:")
    print(f"  python scripts/run_exp9_full_pipeline.py --run-id {run_id} --start-phase 0")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
