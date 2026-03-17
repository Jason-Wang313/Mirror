"""Quick status check for Experiment 5."""

import sys
from pathlib import Path

def check_status():
    results_file = Path("data/results/exp5_20260227T161012_results.jsonl")
    output_file = Path(r"C:\Users\wangz\AppData\Local\Temp\claude\C--Users-wangz-MIRROR\tasks\b88c9d2.output")

    # Count completed trials
    if results_file.exists():
        with open(results_file) as f:
            count = sum(1 for _ in f)
    else:
        count = 0

    pct = (count / 2240) * 100 if count > 0 else 0

    # Check for errors in output
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            recent = lines[-100:] if len(lines) >= 100 else lines

        # Check for completion
        if any("EXPERIMENT 5 COMPLETE" in line for line in recent):
            print(f"✓ COMPLETE: {count} trials")
            return "complete"

        # Check for errors
        error_lines = [line for line in recent if "error" in line.lower() or "traceback" in line.lower()]
        if error_lines:
            print(f"⚠ ERROR at {count} trials:")
            for line in error_lines[:3]:
                print(f"  {line.rstrip()}")
            return "error"

        # Find current progress
        progress = [line for line in recent if line.strip().startswith('[') and '] Q' in line]
        if progress:
            current = progress[-1].strip()
            print(f"Progress: {count}/2240 ({pct:.1f}%) - {current}")
        else:
            print(f"Progress: {count}/2240 ({pct:.1f}%)")

    return "running"

if __name__ == "__main__":
    check_status()
