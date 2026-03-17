"""
Experiment 2: Cross-Domain Transfer Tests

Tests whether self-knowledge in domain A influences behavior on tasks that
implicitly require domain A skills but are framed as domain B.

Modes:
  test   — 1 model × 5 tasks × 5 channels (smoke test)
  pilot  — 2 models × 25 tasks × 5 channels
  full   — 7 models × 230 tasks × 6 channels (5 behavioral + L2)

Usage:
  python scripts/run_experiment_2.py --mode test
  python scripts/run_experiment_2.py --mode pilot
  python scripts/run_experiment_2.py --mode full
  python scripts/run_experiment_2.py --mode full --resume --run-id <RUN_ID>
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api import UnifiedClient
from mirror.experiments.channels import (
    build_prompt,
    build_layer2_prompt,
    parse_response,
    parse_layer2,
)
from mirror.experiments.transfer_tasks import load_transfer_tasks
from mirror.scoring.answer_matcher import match_answer


MODELS_TEST = ["llama-3.1-8b"]
MODELS_PILOT = ["llama-3.1-8b", "deepseek-r1"]
MODELS_FULL = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "deepseek-r1",
    "mistral-large",
    "qwen-3-235b",
    "gpt-oss-120b",
]

CHANNELS_BEHAVIORAL = [
    ("wagering", 1),
    ("opt_out", 2),
    ("difficulty_selection", 3),
    ("tool_use", 4),
    ("natural", 5),
]


def load_checkpoint(run_id: str) -> dict:
    """Load checkpoint data from previous run."""
    checkpoint_file = Path(f"data/results/exp2_{run_id}_checkpoint.json")
    if not checkpoint_file.exists():
        return {}

    with open(checkpoint_file, "r") as f:
        return json.load(f)


def save_checkpoint(run_id: str, checkpoint_data: dict):
    """Save checkpoint data."""
    checkpoint_file = Path(f"data/results/exp2_{run_id}_checkpoint.json")
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_existing_results(results_path: Path) -> dict:
    """Load existing results from JSONL file."""
    completed = set()
    if results_path.exists():
        with open(results_path, "r") as f:
            for line in f:
                record = json.loads(line)
                key = (record["model"], record["task_id"], record["channel"])
                completed.add(key)
    return completed


async def run_transfer_task_on_channel(
    client: UnifiedClient,
    model: str,
    task: dict,
    channel_name: str,
    channel_id: int,
) -> dict:
    """
    Run a single transfer task through one channel.

    Returns:
        Result record dict
    """
    # Convert task to question format expected by channels
    question = {
        "question_id": task["task_id"],
        "question_text": task["task_text"],
        "correct_answer": task["correct_answer"],
        "answer_type": "short_text",
        "domain": task["source_domain"],
    }

    # Build prompt (channel 3 needs special handling for variants)
    if channel_id == 3:
        # Create variant A (reduced dependency) and variant B (full dependency)
        easy_question = {
            **question,
            "question_text": f"{task['task_text']} [Simplified: key values provided]",
        }
        hard_question = question
        prompt = build_prompt(3, question, easy_question=easy_question, hard_question=hard_question)
        extra_context = {"easy_question": easy_question, "hard_question": hard_question}
    else:
        prompt = build_prompt(channel_id, question)
        extra_context = {}

    # Call model
    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 if channel_id in [1, 2, 3, 4] else 0.7,
        max_tokens=600,
        metadata={"task": "transfer_experiment", "channel": channel_id},
    )

    if "error" in response:
        return {
            "task_id": task["task_id"],
            "model": model,
            "channel": channel_id,
            "channel_name": channel_name,
            "error": response["error"],
            "source_domain": task["source_domain"],
            "surface_domain": task["surface_domain"],
            "hidden_dependency": task["hidden_dependency"],
        }

    response_text = response.get("content", "")

    # Parse response
    if channel_id == 3:
        parsed = parse_response(3, response_text, **extra_context)
    else:
        parsed = parse_response(channel_id, response_text)

    # Check answer correctness
    answer_correct = match_answer(
        parsed.get("answer", ""),
        task["correct_answer"],
        task.get("answer_type", "short_text"),
    )

    # Build result record
    result = {
        "task_id": task["task_id"],
        "model": model,
        "channel": channel_id,
        "channel_name": channel_name,
        "source_domain": task["source_domain"],
        "surface_domain": task["surface_domain"],
        "hidden_dependency": task["hidden_dependency"],
        "prompt": prompt,
        "response": response_text,
        "parsed": parsed,
        "answer_correct": answer_correct,
        "parse_success": parsed.get("parse_success", True),
    }

    return result


async def run_layer2_on_transfer_task(
    client: UnifiedClient,
    model: str,
    task: dict,
) -> dict:
    """
    Run Layer 2 verbal self-report on transfer task.

    Prompt asks model to:
    1. List sub-skills required
    2. Flag which they're least confident in
    3. Recommend external verification
    """
    # Convert task to question format
    question = {
        "question_id": task["task_id"],
        "question_text": task["task_text"],
        "correct_answer": task["correct_answer"],
        "answer_type": "short_text",
        "domain": task["source_domain"],
    }

    # Build Layer 2 prompt
    prompt = build_layer2_prompt(question)

    # Call model
    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
        metadata={"task": "layer2_transfer", "task_id": task["task_id"]},
    )

    if "error" in response:
        return {
            "task_id": task["task_id"],
            "model": model,
            "channel": "layer2",
            "channel_name": "layer2",
            "error": response["error"],
            "source_domain": task["source_domain"],
            "surface_domain": task["surface_domain"],
            "hidden_dependency": task["hidden_dependency"],
        }

    response_text = response.get("content", "")

    # Parse Layer 2 response
    parsed = parse_layer2(response_text)

    return {
        "task_id": task["task_id"],
        "model": model,
        "channel": "layer2",
        "channel_name": "layer2",
        "source_domain": task["source_domain"],
        "surface_domain": task["surface_domain"],
        "hidden_dependency": task["hidden_dependency"],
        "prompt": prompt,
        "response": response_text,
        "parsed": parsed,
        "parse_success": parsed.get("parse_success", True),
    }


async def run_experiment_2(
    mode: str = "test",
    run_id: str = None,
    resume: bool = False,
    models_override: list = None,
):
    """
    Run Experiment 2 data collection.

    Args:
        mode: 'test', 'pilot', or 'full'
        run_id: Resume from existing run ID
        resume: If True, skip already-completed tasks
    """
    # Determine run parameters
    if mode == "test":
        models = MODELS_TEST
        tasks_limit = 5
        channels = CHANNELS_BEHAVIORAL[:2]  # Just wagering + opt-out
        include_layer2 = False
    elif mode == "pilot":
        models = MODELS_PILOT
        tasks_limit = 25
        channels = CHANNELS_BEHAVIORAL
        include_layer2 = False
    elif mode == "full":
        models = MODELS_FULL
        tasks_limit = None  # All tasks
        channels = CHANNELS_BEHAVIORAL
        include_layer2 = True
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

    # Override models if specified
    if models_override:
        models = models_override

    # Generate or use existing run ID
    if resume and run_id is None:
        print("❌ Error: --resume requires --run-id to specify which run to resume")
        sys.exit(1)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    else:
        resume = True  # If run_id provided, assume resume

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Cross-Domain Transfer")
    print(f"{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(models)}")
    print(f"Channels: {', '.join(c[0] for c in channels)}{' + Layer 2' if include_layer2 else ''}")
    print(f"Resume: {resume}")
    print(f"{'='*60}\n")

    # Load transfer tasks
    tasks = load_transfer_tasks("data/transfer_tasks.jsonl")
    if not tasks:
        print("❌ No transfer tasks found. Run:")
        print("   python -m mirror.experiments.transfer_tasks --generate --pilot")
        sys.exit(1)

    if tasks_limit:
        tasks = tasks[:tasks_limit]

    print(f"Loaded {len(tasks)} transfer tasks\n")

    # Setup results file
    results_path = Path(f"data/results/exp2_{run_id}_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    completed = load_existing_results(results_path) if resume else set()
    if completed:
        print(f"📋 Loaded checkpoint: {len(completed)} tasks already completed\n")
    elif resume:
        print(f"⚠️  No existing results found for run {run_id}\n")

    # Run data collection
    client = UnifiedClient(experiment=f"exp2_{run_id}")
    total_tasks = len(models) * len(tasks) * (len(channels) + (1 if include_layer2 else 0))
    completed_count = len(completed)
    skipped_count = 0
    start_time = time.time()

    with open(results_path, "a", encoding="utf-8") as f:
        for model in models:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")

            for task_idx, task in enumerate(tasks, 1):
                print(f"\nTask {task_idx}/{len(tasks)}: {task['task_id']}")
                print(f"  Source: {task['source_domain']} → Surface: {task['surface_domain']}")

                # Run behavioral channels
                for channel_name, channel_id in channels:
                    key = (model, task["task_id"], channel_id)
                    if key in completed:
                        print(f"  [Ch{channel_id}: {channel_name}] ✓ Skipped (checkpoint)")
                        skipped_count += 1
                        continue

                    print(f"  [Ch{channel_id}: {channel_name}] Running...")
                    result = await run_transfer_task_on_channel(
                        client, model, task, channel_name, channel_id
                    )

                    # Write result immediately to prevent data loss
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                    completed_count += 1
                    progress = completed_count / total_tasks
                    elapsed = time.time() - start_time
                    eta_min = (elapsed / completed_count * (total_tasks - completed_count)) / 60
                    print(f"    Progress: {progress:.1%} | ETA: {eta_min:.0f}min")

                    await asyncio.sleep(0.3)

                # Run Layer 2 if enabled
                if include_layer2:
                    key = (model, task["task_id"], "layer2")
                    if key in completed:
                        print(f"  [Layer 2] ✓ Skipped (checkpoint)")
                        skipped_count += 1
                    else:
                        print(f"  [Layer 2] Running...")
                        result = await run_layer2_on_transfer_task(client, model, task)
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        completed_count += 1

                    await asyncio.sleep(0.3)

    duration_min = (time.time() - start_time) / 60
    completed_this_run = completed_count - len(completed)
    print(f"\n{'='*60}")
    print(f"Data collection complete")
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"Completed this run: {completed_this_run} tasks")
    if skipped_count > 0:
        print(f"Skipped (checkpoint): {skipped_count} tasks")
    print(f"Results: {results_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Cross-Domain Transfer")
    parser.add_argument("--mode", choices=["test", "pilot", "full"], required=True,
                        help="Experiment mode")
    parser.add_argument("--run-id", help="Resume from existing run ID")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip completed tasks)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list override (e.g. gemma-3-27b,kimi-k2)")
    args = parser.parse_args()

    models_override = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_experiment_2(
        mode=args.mode,
        run_id=args.run_id,
        resume=args.resume,
        models_override=models_override,
    ))


if __name__ == "__main__":
    main()
