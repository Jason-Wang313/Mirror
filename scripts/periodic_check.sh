#!/bin/bash
# Periodic checks for Experiment 5

RESULTS="/c/Users/wangz/MIRROR/data/results/exp5_20260227T161012_results.jsonl"

while true; do
  count=$(wc -l < "$RESULTS" 2>/dev/null || echo "0")
  pct=$((count * 100 / 2240))
  timestamp=$(date +"%H:%M:%S")
  
  echo "[$timestamp] $count/2240 trials ($pct%)"
  
  # Check for completion or error
  if [ "$count" -ge 2240 ]; then
    echo "✓ All trials complete!"
    break
  fi
  
  sleep 1800  # Check every 30 minutes
done
