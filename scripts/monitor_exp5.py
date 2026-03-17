"""Monitor Experiment 5 progress and check for errors."""

import time
import sys
from pathlib import Path

def monitor_progress(output_file: str, check_interval: int = 60):
    """Monitor experiment progress from output file."""
    output_path = Path(output_file)

    if not output_path.exists():
        print(f"Output file not found: {output_file}")
        return

    last_size = 0
    stall_count = 0
    last_line = ""

    print("Monitoring Experiment 5...")
    print("Press Ctrl+C to stop monitoring (experiment continues in background)\n")

    try:
        while True:
            current_size = output_path.stat().st_size

            # Read last 50 lines
            with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) >= 50 else lines

            # Check for completion
            if any("EXPERIMENT 5 COMPLETE" in line for line in recent_lines):
                print("\n✓ EXPERIMENT COMPLETED!")
                print("Last lines:")
                for line in recent_lines[-10:]:
                    print(line.rstrip())
                return

            # Check for errors
            error_lines = [line for line in recent_lines if "error" in line.lower() or "traceback" in line.lower()]
            if error_lines:
                print("\n⚠ ERROR DETECTED:")
                for line in error_lines[:5]:
                    print(line.rstrip())
                return

            # Find progress indicator
            progress_lines = [line for line in recent_lines if line.strip().startswith('[') and ']' in line]
            if progress_lines:
                current_line = progress_lines[-1].strip()
                if current_line != last_line:
                    print(f"\r{current_line}", end='', flush=True)
                    last_line = current_line

            # Check for stalls
            if current_size == last_size:
                stall_count += 1
                if stall_count >= 5:  # 5 minutes of no activity
                    print(f"\n⚠ WARNING: No output for {stall_count * check_interval} seconds")
                    print("Last 10 lines:")
                    for line in recent_lines[-10:]:
                        print(line.rstrip())
            else:
                stall_count = 0

            last_size = current_size
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped (experiment continues in background)")
        print(f"Check progress: tail -50 {output_file}")

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not output_file:
        print("Usage: python scripts/monitor_exp5.py <output_file>")
        sys.exit(1)

    monitor_progress(output_file)
