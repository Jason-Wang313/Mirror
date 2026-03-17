#!/bin/bash
# Watch Experiment 5 progress and report milestones

OUTPUT_FILE="$1"
RESULTS_FILE="/c/Users/wangz/MIRROR/data/results/exp5_20260227T161012_results.jsonl"

echo "Watching Experiment 5..."
echo "Target: ~2,240 trials (7 models × 4 attacks × 80 questions)"
echo ""

last_count=0
while true; do
    # Count completed trials
    if [ -f "$RESULTS_FILE" ]; then
        count=$(wc -l < "$RESULTS_FILE")

        # Report progress every 50 trials
        if [ $((count / 50)) -gt $((last_count / 50)) ]; then
            pct=$((count * 100 / 2240))
            echo "[$(date +%H:%M:%S)] Progress: $count/2240 ($pct%)"
        fi

        last_count=$count
    fi

    # Check for errors
    if tail -100 "$OUTPUT_FILE" 2>/dev/null | grep -i "error\|traceback" > /dev/null; then
        echo ""
        echo "⚠ ERROR DETECTED at $(date +%H:%M:%S)"
        tail -30 "$OUTPUT_FILE" | grep -A 10 -i "error\|traceback"
        exit 1
    fi

    # Check for completion
    if tail -10 "$OUTPUT_FILE" 2>/dev/null | grep "EXPERIMENT 5 COMPLETE" > /dev/null; then
        echo ""
        echo "✓ EXPERIMENT COMPLETE at $(date +%H:%M:%S)"
        echo "Total trials: $count"
        exit 0
    fi

    sleep 300  # Check every 5 minutes
done
