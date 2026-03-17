"""Final monitoring script - reports milestones and completion."""

import time
import subprocess
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path("data/results/exp5_20260227T161012_results.jsonl")
POLL_OUTPUT = Path(r"C:\Users\wangz\AppData\Local\Temp\claude\C--Users-wangz-MIRROR\tasks\bf5970e.output")

def get_count():
    """Get current trial count."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return sum(1 for _ in f)
    return 0

def main():
    print("Final monitoring - will report key milestones until completion")
    print("="*80)

    milestones = {500: False, 1000: False, 1500: False, 2000: False, 2240: False}
    last_count = get_count()

    while True:
        count = get_count()

        # Report milestones
        for m, reported in milestones.items():
            if count >= m and not reported:
                pct = (m / 2240) * 100
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✓✓ MILESTONE: {m} trials ({pct:.1f}%)")
                milestones[m] = True

                # Estimate remaining time
                if last_count > 0:
                    # Get rate from poll output
                    if POLL_OUTPUT.exists():
                        lines = POLL_OUTPUT.read_text(errors='ignore').splitlines()
                        if len(lines) >= 2:
                            print(f"    Latest status from poll: {lines[-2]}")

        # Check for completion
        if count >= 2240:
            print(f"\n{'='*80}")
            print(f"✓✓✓ EXPERIMENT 5 COMPLETE ✓✓✓")
            print(f"{'='*80}")
            print(f"Total trials: {count}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}\n")
            break

        time.sleep(120)  # Check every 2 minutes

if __name__ == "__main__":
    main()
