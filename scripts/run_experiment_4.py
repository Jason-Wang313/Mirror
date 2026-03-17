"""
Experiment 4 Runner: The Adaptation Crucible (Level 3)

Modes:
  --mode pilot        2 trials × 2 models (quick test)
  --mode full         All 46 trials × 7 models
  --mode sycophancy   8 sycophancy controls × 7 models

Usage:
  python scripts/run_experiment_4.py --mode pilot
  python scripts/run_experiment_4.py --mode full
  python scripts/run_experiment_4.py --mode full --resume --run-id <ID>
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api import UnifiedClient
from mirror.experiments.burn_test_runner import BurnTestRunner


def load_trials(trials_path: str, trial_type_filter=None) -> list[dict]:
    """Load trials from JSONL, optionally filtering by type."""
    trials = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trial = json.loads(line)

                # Filter by type if specified
                if trial_type_filter:
                    if trial.get("trial_type") not in trial_type_filter:
                        continue

                trials.append(trial)

    return trials


def load_existing_results(results_path: Path) -> set:
    """Load existing results to enable resume."""
    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = (
                        record.get("model"),
                        record.get("trial_id")
                    )
                    completed.add(key)
                except:
                    continue
    return completed


async def run_experiment_4(
    mode: str,
    run_id: str = None,
    resume: bool = False,
    models_override: list = None,
):
    """Run Experiment 4 data collection."""
    # Validate mode
    if mode not in ["pilot", "full", "sycophancy"]:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

    # Setup run ID
    if resume and run_id is None:
        print("❌ Error: --resume requires --run-id")
        sys.exit(1)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    else:
        resume = True

    # Determine configuration
    if mode == "pilot":
        models = ["llama-3.1-8b", "deepseek-r1"]
        trial_filter = ["standard"]  # Only standard trials
        max_trials = 2
    elif mode == "sycophancy":
        models = [
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
            "deepseek-r1", "mistral-large", "qwen-3-235b", "gpt-oss-120b"
        ]
        trial_filter = ["sycophancy_control"]
        max_trials = None
    else:  # full
        models = [
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
            "deepseek-r1", "mistral-large", "qwen-3-235b", "gpt-oss-120b"
        ]
        trial_filter = None  # All trial types
        max_trials = None

    # Override models if specified
    if models_override:
        models = models_override

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4: THE ADAPTATION CRUCIBLE (LEVEL 3)")
    print(f"{'='*80}")
    print(f"Mode: {mode.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(models)}")
    print(f"Resume: {resume}")
    print(f"{'='*80}\n")

    # Load trials
    trials_path = "data/exp4/trials.jsonl"
    if not Path(trials_path).exists():
        print(f"❌ Trials not found: {trials_path}")
        print(f"   Run: python scripts/generate_exp4_tasks.py")
        sys.exit(1)

    print("Loading trials...")
    trials = load_trials(trials_path, trial_filter)
    if max_trials:
        trials = trials[:max_trials]
    print(f"  Loaded {len(trials)} trials\n")

    # Setup results file
    results_path = Path(f"data/results/exp4_{run_id}_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup conversation logs directory
    conv_dir = Path(f"data/results/exp4_{run_id}_conversations")
    conv_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    completed = load_existing_results(results_path) if resume else set()
    if completed:
        print(f"📋 Loaded checkpoint: {len(completed)} trials completed\n")

    # Run collection
    client = UnifiedClient(experiment=f"exp4_{run_id}")
    total = len(models) * len(trials)
    completed_count = len(completed)
    skipped_count = 0
    start_time = time.time()

    with open(results_path, "a", encoding="utf-8") as f:
        for model in models:
            print(f"\n{'─'*80}")
            print(f"Model: {model}")
            print(f"{'─'*80}")

            runner = BurnTestRunner(client, model)

            for trial_idx, trial in enumerate(trials, 1):
                print(f"\nTrial {trial_idx}/{len(trials)}: {trial['trial_id']}")

                key = (model, trial["trial_id"])

                if key in completed:
                    print(f"  ✓ Skipped (checkpoint)")
                    skipped_count += 1
                    continue

                print(f"  Running multi-turn conversation...")
                print(f"  Type: {trial.get('trial_type', 'standard')}")

                # Run the complete burn-and-test trial
                result = await runner.run_trial(trial)

                if "error" in result:
                    print(f"  ❌ Error in phase {result.get('phase', '?')}: {result['error']}")
                    # Still write the partial result
                else:
                    print(f"  ✅ Complete ({result.get('conversation_length', 0)} messages)")

                # Write result immediately with fsync
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())

                # Save full conversation for qualitative analysis
                conv_path = conv_dir / f"{model}_{trial['trial_id']}.json"
                with open(conv_path, "w", encoding="utf-8") as cf:
                    json.dump({
                        "model": model,
                        "trial_id": trial["trial_id"],
                        "messages": runner.messages,
                        "result": result,
                    }, cf, indent=2)

                completed_count += 1
                progress = completed_count / total
                elapsed = time.time() - start_time
                if completed_count > len(completed):
                    eta_min = (elapsed / (completed_count - len(completed)) *
                               (total - completed_count)) / 60
                    print(f"  Progress: {progress:.1%} | ETA: {eta_min:.0f}min")

                await asyncio.sleep(0.5)  # Longer delay for multi-turn

    duration_min = (time.time() - start_time) / 60
    completed_this_run = completed_count - len(completed)

    print(f"\n{'='*80}")
    print(f"Experiment 4 complete")
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"Completed this run: {completed_this_run} trials")
    if skipped_count > 0:
        print(f"Skipped (checkpoint): {skipped_count} trials")
    print(f"Results: {results_path}")
    print(f"Conversations: {conv_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Adaptation Crucible")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full", "sycophancy"],
        required=True,
        help="Experiment mode"
    )
    parser.add_argument("--run-id", help="Resume from existing run ID")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list override")
    args = parser.parse_args()

    models_override = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_experiment_4(
        mode=args.mode,
        run_id=args.run_id,
        resume=args.resume,
        models_override=models_override,
    ))


if __name__ == "__main__":
    main()
