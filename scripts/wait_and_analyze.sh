#!/bin/bash
# Wait for Experiment 5 to complete, then run analysis

RESULTS="data/results/exp5_20260227T161012_results.jsonl"

echo "Waiting for Experiment 5 to complete..."
echo "Checking every 15 minutes for completion..."
echo ""

while true; do
  count=$(wc -l < "$RESULTS" 2>/dev/null || echo "0")
  pct=$((count * 100 / 2240))
  timestamp=$(date +"%H:%M:%S")
  
  echo "[$timestamp] $count/2240 ($pct%)"
  
  # Check for completion
  if [ "$count" -ge 2240 ]; then
    echo ""
    echo "================================================================================"
    echo "✓✓✓ EXPERIMENT 5 COMPLETE ✓✓✓"
    echo "================================================================================"
    echo "Total trials: $count"
    echo "Completion time: $timestamp"
    echo "================================================================================"
    echo ""
    echo "Running analysis..."
    python scripts/analyze_experiment_5.py
    exit 0
  fi
  
  # Also check for major milestones
  if [ "$count" -ge 500 ] && [ "$count" -lt 550 ]; then
    echo "  ✓ Passed 500-trial milestone"
  fi
  if [ "$count" -ge 1000 ] && [ "$count" -lt 1050 ]; then
    echo "  ✓ Passed 1000-trial milestone (halfway!)"
  fi
  if [ "$count" -ge 1500 ] && [ "$count" -lt 1550 ]; then
    echo "  ✓ Passed 1500-trial milestone"
  fi
  if [ "$count" -ge 2000 ] && [ "$count" -lt 2050 ]; then
    echo "  ✓ Passed 2000-trial milestone (almost done!)"
  fi
  
  sleep 900  # Check every 15 minutes
done
