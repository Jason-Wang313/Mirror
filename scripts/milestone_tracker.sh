#!/bin/bash
# Track major milestones for Experiment 5

RESULTS="/c/Users/wangz/MIRROR/data/results/exp5_20260227T161012_results.jsonl"

echo "Tracking milestones: 500, 1000, 1500, 2000, completion"
echo ""

milestones=(500 1000 1500 2000)
reported=()

while true; do
  count=$(wc -l < "$RESULTS" 2>/dev/null || echo "0")

  # Check each milestone
  for m in "${milestones[@]}"; do
    if [ "$count" -ge "$m" ]; then
      # Check if already reported
      already_reported=false
      for r in "${reported[@]}"; do
        if [ "$r" == "$m" ]; then
          already_reported=true
          break
        fi
      done

      if [ "$already_reported" = false ]; then
        pct=$((m * 100 / 2240))
        echo "[$(date +%H:%M:%S)] ✓✓ MILESTONE: $m trials ($pct%)"
        reported+=("$m")
      fi
    fi
  done

  # Check for completion
  if [ "$count" -ge 2240 ]; then
    echo ""
    echo "[$(date +%H:%M:%S)] ✓✓✓ ALL 2240 TRIALS COMPLETE ✓✓✓"
    exit 0
  fi

  sleep 300  # Check every 5 minutes
done
