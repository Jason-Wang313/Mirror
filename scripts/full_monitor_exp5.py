"""Full monitoring for Experiment 5 - runs until completion or error."""

import time
import sys
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path("data/results/exp5_20260227T161012_results.jsonl")
OUTPUT_FILE = Path(r"C:\Users\wangz\AppData\Local\Temp\claude\C--Users-wangz-MIRROR\tasks\b88c9d2.output")
TARGET_TRIALS = 2240
CHECK_INTERVAL = 900  # 15 minutes

milestones = [50, 100, 200, 500, 1000, 1500, 2000, 2200]
reported_milestones = set()

print("=" * 80)
print("EXPERIMENT 5 - FULL MONITORING")
print("=" * 80)
print(f"Target: {TARGET_TRIALS} trials")
print(f"Check interval: {CHECK_INTERVAL//60} minutes")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 80)
print()

last_count = 0
stall_checks = 0

while True:
    try:
        # Count trials
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                count = sum(1 for _ in f)
        else:
            count = 0

        pct = (count / TARGET_TRIALS) * 100

        # Check for stalling
        if count == last_count:
            stall_checks += 1
            if stall_checks >= 4:  # 1 hour of no progress
                print(f"\n⚠ WARNING: No progress for {stall_checks * CHECK_INTERVAL // 60} minutes")
                print(f"Stuck at {count} trials")
        else:
            stall_checks = 0

        last_count = count

        # Report milestones
        for milestone in milestones:
            if count >= milestone and milestone not in reported_milestones:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ MILESTONE: {milestone}/{TARGET_TRIALS} trials ({milestone/TARGET_TRIALS*100:.1f}%)")
                reported_milestones.add(milestone)

        # Check for completion
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                recent = lines[-20:] if len(lines) >= 20 else lines

            if any("EXPERIMENT 5 COMPLETE" in line for line in recent):
                print(f"\n{'='*80}")
                print(f"✓✓✓ EXPERIMENT COMPLETE at {datetime.now().strftime('%H:%M:%S')} ✓✓✓")
                print(f"{'='*80}")
                print(f"Total trials: {count}")
                print(f"Duration: {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*80}")
                sys.exit(0)

            # Check for errors
            error_lines = [line for line in recent if "error" in line.lower() or "traceback" in line.lower()]
            if error_lines:
                print(f"\n{'='*80}")
                print(f"⚠⚠⚠ ERROR DETECTED at {datetime.now().strftime('%H:%M:%S')} ⚠⚠⚠")
                print(f"{'='*80}")
                print(f"Trials completed before error: {count}")
                print("\nError details:")
                for line in error_lines[:5]:
                    print(f"  {line.rstrip()}")
                print(f"\n{'='*80}")
                sys.exit(1)

        # Periodic status update
        if count > 0:
            rate = count / ((time.time() - start_time) / 60) if 'start_time' in locals() else 0
            if rate > 0:
                remaining = (TARGET_TRIALS - count) / rate
                eta = datetime.now().timestamp() + (remaining * 60)
                eta_str = datetime.fromtimestamp(eta).strftime('%H:%M:%S')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {count}/{TARGET_TRIALS} ({pct:.1f}%) - ETA: {eta_str}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {count}/{TARGET_TRIALS} ({pct:.1f}%)")

        if 'start_time' not in locals():
            start_time = time.time()

        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped at {count} trials")
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠ Monitor error: {e}")
        time.sleep(CHECK_INTERVAL)
