#!/bin/bash
# Run Exp1 (channels 1+5) for gemini-2.5-pro, then Exp9.
# Expected total time: ~9-12 hours (Exp1 ~2h + Exp9 ~8h)

set -e
cd "$(dirname "$0")/.."

RUN_ID="20260323_gemini"

echo "=== Step 1: Exp1 for gemini-2.5-pro (wagering + natural, 400 questions) ==="
python scripts/run_exp1_fast.py --models gemini-2.5-pro --run-id "$RUN_ID" --resume

echo ""
echo "=== Step 2: Exp9 for gemini-2.5-pro (full: 597 tasks x 4 conditions x 3 paradigms) ==="
python scripts/run_experiment_9.py --mode full --models gemini-2.5-pro --run-id "$RUN_ID" --resume
