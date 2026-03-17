"""Wait for Experiment 5 to complete, reporting progress and catching errors."""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

RESULTS_FILE = Path("data/results/exp5_20260227T161012_results.jsonl")
OUTPUT_FILE = Path(r"C:\Users\wangz\AppData\Local\Temp\claude\C--Users-wangz-MIRROR\tasks\b88c9d2.output")

def wait_for_completion():
    """Wait for experiment to complete, reporting milestones."""

    milestones = {50: False, 100: False, 200: False, 500: False, 1000: False, 1500: False, 2000: False}
    last_report_time = datetime.now()
    start_time = datetime.now()
    last_count = 0

    print(f"Waiting for Experiment 5 to complete...")
    print(f"Started monitoring at: {start_time.strftime('%H:%M:%S')}")
    print(f"Target: 2240 trials\n")

    while True:
        # Count trials
        if RESULTS_FILE.exists():
            count = sum(1 for _ in open(RESULTS_FILE))
        else:
            count = 0

        # Check for milestones
        for milestone, reported in milestones.items():
            if count >= milestone and not reported:
                pct = (milestone / 2240) * 100
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = count / elapsed if elapsed > 0 else 0
                remaining_mins = (2240 - count) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(minutes=remaining_mins)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ MILESTONE: {milestone} trials ({pct:.1f}%)")
                print(f"    Rate: {rate:.1f} trials/min | ETA: {eta.strftime('%H:%M:%S')}")
                milestones[milestone] = True
                last_report_time = datetime.now()

        # Periodic status (every 30 minutes if no milestones)
        if (datetime.now() - last_report_time).total_seconds() >= 1800:
            pct = (count / 2240) * 100
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {count}/2240 ({pct:.1f}%)")
            last_report_time = datetime.now()

        # Check for completion
        if count >= 2240 or (OUTPUT_FILE.exists() and "EXPERIMENT 5 COMPLETE" in OUTPUT_FILE.read_text(errors='ignore')):
            print(f"\n{'='*80}")
            print(f"✓✓✓ EXPERIMENT 5 COMPLETE ✓✓✓")
            print(f"{'='*80}")
            print(f"Total trials: {count}")
            print(f"Start time: {start_time.strftime('%H:%M:%S')}")
            print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
            duration = datetime.now() - start_time
            print(f"Duration: {duration.total_seconds()/3600:.1f} hours")
            print(f"{'='*80}\n")
            return True

        # Check for errors
        if OUTPUT_FILE.exists():
            recent_lines = OUTPUT_FILE.read_text(errors='ignore').splitlines()[-50:]
            error_lines = [l for l in recent_lines if 'error' in l.lower() or 'traceback' in l.lower()]
            if error_lines:
                print(f"\n{'='*80}")
                print(f"⚠⚠⚠ ERROR DETECTED ⚠⚠⚠")
                print(f"{'='*80}")
                print(f"Trials completed: {count}")
                print(f"\nRecent errors:")
                for line in error_lines[:5]:
                    print(f"  {line}")
                print(f"{'='*80}\n")
                return False

        # Check for stalling
        if count == last_count:
            # No progress - wait longer
            time.sleep(300)  # 5 minutes
        else:
            time.sleep(60)  # 1 minute

        last_count = count

if __name__ == "__main__":
    try:
        success = wait_for_completion()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        sys.exit(0)
