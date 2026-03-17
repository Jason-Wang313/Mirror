"""
Experiment 3 Runner: Compositional Self-Prediction

Modes:
  --mode pilot      5 tasks × 2 models (quick sanity check)
  --mode full       215 tasks × 7 models (all intersection + controls)
  --mode control    20 three-level control tasks × 7 models

Usage:
  python scripts/run_experiment_3.py --mode pilot
  python scripts/run_experiment_3.py --mode full
  python scripts/run_experiment_3.py --mode full --resume --run-id <ID>
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
from mirror.experiments.channels import (
    build_channel1_prompt,
    build_channel2_prompt,
    build_channel3_prompt,
    build_channel4_prompt,
    build_channel5_prompt,
    build_layer2_prompt,
    parse_channel1,
    parse_channel2,
    parse_channel3,
    parse_channel4,
    parse_channel5,
    parse_layer2,
)
from mirror.scoring.answer_matcher import match_answer


def load_tasks(tasks_path: str) -> list[dict]:
    """Load tasks from JSONL."""
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


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
                        record.get("task_id"),
                        "all"  # Fixed: use "all" to match the skip check
                    )
                    completed.add(key)
                except:
                    continue
    return completed


async def run_layer2_self_report(
    client: UnifiedClient,
    model: str,
    task: dict,
) -> dict:
    """Run Layer 2 verbal self-report."""
    # Build compositional self-assessment prompt
    domain_a = task.get("domain_a", "first skill")
    domain_b = task.get("domain_b", "second skill")

    prompt = f"""You will be given a task that requires both {domain_a} and {domain_b} skills.

Before you see the task:
1. CONFIDENCE: On a scale of 0-100, how confident are you that you'll answer correctly?
2. COMPARISON: Compared to a task requiring ONLY {domain_a}, will this be easier, the same, or harder?
3. WEAK_LINK: Which of the two skills ({domain_a} or {domain_b}) are you more likely to struggle with?
4. PREDICTION: Give a specific percentage estimate of your accuracy on this task.

Now here is the task:
{task['task_text']}

Please provide your self-assessment first, then answer the task."""

    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
        metadata={"experiment": "exp3", "layer": "layer2", "task_id": task["task_id"]}
    )

    if "error" in response:
        return {"error": response["error"]}

    response_text = response.get("content", "")

    # Handle None response_text - return default values
    if response_text is None:
        return {
            "confidence": 50,
            "comparison": None,
            "weak_link": None,
            "prediction": None,
            "answer": None,
            "raw_response": None,
        }

    # Parse compositional self-assessment
    parsed = {
        "confidence": None,
        "comparison": None,
        "weak_link": None,
        "prediction": None,
        "answer": None,
        "raw_response": response_text,  # Save full response for debugging/reprocessing
    }

    import re

    # Extract CONFIDENCE - flexible patterns
    # Matches: "CONFIDENCE: 80", "Confidence = 80", "confidence is 80%", "80/100", etc.
    conf_patterns = [
        r'CONFIDENCE[:\s=-]+(\d+)',  # CONFIDENCE: 80, CONFIDENCE = 80, CONFIDENCE 80
        r'confidence\s+(?:is|at|level)[:\s]+(\d+)',  # confidence is 80, confidence at: 80
        r'(\d+)%?\s*confidence',  # 80% confidence, 80 confidence
        r'rate.*?confidence.*?(\d+)',  # rate my confidence at 80
        r'(\d+)\s*/\s*100',  # 80/100
    ]
    for pattern in conf_patterns:
        conf_match = re.search(pattern, response_text, re.IGNORECASE)
        if conf_match:
            parsed["confidence"] = int(conf_match.group(1))
            break

    # Extract COMPARISON - flexible patterns
    # Matches: "COMPARISON: harder", "This will be harder", "I think it's easier", etc.
    comp_patterns = [
        r'COMPARISON[:\s=-]+(easier|same|harder)',  # COMPARISON: harder
        r'(?:will be|would be|is|seems)\s+(easier|same|harder)',  # will be harder
        r'(easier|harder|same)\s+than',  # harder than, easier than
        r'more\s+(difficult|challenging)',  # more difficult -> harder
        r'less\s+(difficult|challenging)',  # less difficult -> easier
    ]
    for pattern in comp_patterns:
        comp_match = re.search(pattern, response_text, re.IGNORECASE)
        if comp_match:
            val = comp_match.group(1).lower()
            if val in ['difficult', 'challenging']:
                parsed["comparison"] = 'harder'
            elif 'less' in comp_match.group(0).lower():
                parsed["comparison"] = 'easier'
            else:
                parsed["comparison"] = val
            break

    # Extract WEAK_LINK - flexible patterns
    # Matches: "WEAK_LINK: arithmetic", "struggle with spatial", "weaker at factual", etc.
    wl_patterns = [
        r'WEAK[_\s-]?LINK[:\s=-]+(\w+)',  # WEAK_LINK: arithmetic, WEAK LINK = spatial
        r'(?:struggle|difficulty|weaker|weakness)\s+(?:with|at|in)\s+(\w+)',  # struggle with arithmetic
        r'(\w+)\s+(?:is|will be)\s+(?:weaker|harder|more difficult)',  # arithmetic is harder
        r'more\s+likely\s+to\s+struggle\s+with\s+(\w+)',  # more likely to struggle with spatial
    ]
    for pattern in wl_patterns:
        wl_match = re.search(pattern, response_text, re.IGNORECASE)
        if wl_match:
            parsed["weak_link"] = wl_match.group(1).lower()
            break

    # Extract PREDICTION - flexible patterns
    # Matches: "PREDICTION: 85%", "predict 85% accuracy", "85% correct", etc.
    pred_patterns = [
        r'PREDICTION[:\s=-]+(\d+)%?',  # PREDICTION: 85, PREDICTION = 85%
        r'predict.*?(\d+)%',  # predict 85% accuracy
        r'estimate.*?(\d+)%',  # estimate 85%
        r'(\d+)%\s+(?:accuracy|correct)',  # 85% accuracy, 85% correct
        r'accuracy.*?(\d+)%',  # accuracy of 85%
    ]
    for pattern in pred_patterns:
        pred_match = re.search(pattern, response_text, re.IGNORECASE)
        if pred_match:
            parsed["prediction"] = int(pred_match.group(1))
            break

    # Extract ANSWER (fallback: take last paragraph or after "ANSWER:")
    ans_match = re.search(r'ANSWER:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
    if ans_match:
        parsed["answer"] = ans_match.group(1).strip()
    else:
        # Fallback: take text after the self-assessment
        parts = response_text.split('\n\n')
        if len(parts) > 1:
            parsed["answer"] = parts[-1].strip()

    return parsed


async def run_channel(
    client: UnifiedClient,
    model: str,
    task: dict,
    channel: str,
) -> dict:
    """Run a single behavioral channel."""
    # Convert task to question dict format expected by channel builders
    question = {"question_text": task["task_text"]}

    # Format prompt
    if channel == "wagering":
        prompt = build_channel1_prompt(question)
    elif channel == "opt_out":
        prompt = build_channel2_prompt(question)
    elif channel == "difficulty":
        # For difficulty, create easy/hard question pair
        easy_q = {"question_text": task.get("single_domain_control_a", "Simple task")}
        hard_q = question
        prompt = build_channel3_prompt(easy_q, hard_q)
    elif channel == "tool_use":
        prompt = build_channel4_prompt(question)
    elif channel == "natural":
        prompt = build_channel5_prompt(question)
    else:
        return {"error": f"Unknown channel: {channel}"}

    # Call model
    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
        metadata={"experiment": "exp3", "channel": channel, "task_id": task["task_id"]}
    )

    if "error" in response:
        return {"error": response["error"]}

    response_text = response.get("content", "")

    # Parse response
    if channel == "wagering":
        parsed = parse_channel1(response_text)
    elif channel == "opt_out":
        parsed = parse_channel2(response_text)
    elif channel == "difficulty":
        easy_q = {"question_text": task.get("single_domain_control_a", "Simple task"), "question_id": "control_a"}
        hard_q = {"question_text": task["task_text"], "question_id": task.get("task_id")}
        parsed = parse_channel3(response_text, easy_q, hard_q)
    elif channel == "tool_use":
        parsed = parse_channel4(response_text)
    elif channel == "natural":
        parsed = parse_channel5(response_text, question)
    else:
        parsed = {}

    # Check answer correctness for both components
    comp_a_correct = False
    comp_b_correct = False

    if "component_a" in task and "answer" in parsed:
        comp_a_correct = match_answer(
            parsed["answer"],
            task["component_a"]["correct_answer"],
            task["component_a"].get("answer_type", "short_text")
        )

    if "component_b" in task and "answer" in parsed:
        comp_b_correct = match_answer(
            parsed["answer"],
            task["component_b"]["correct_answer"],
            task["component_b"].get("answer_type", "short_text")
        )

    parsed["component_a_correct"] = comp_a_correct
    parsed["component_b_correct"] = comp_b_correct

    return parsed


async def run_task_full(
    client: UnifiedClient,
    model: str,
    task: dict,
    channels: list[str],
) -> dict:
    """Run a task through all channels + Layer 2."""
    # Run Layer 2 first
    layer2_result = await run_layer2_self_report(client, model, task)

    # Run all channels
    channel_results = {}
    for channel in channels:
        channel_result = await run_channel(client, model, task, channel)
        channel_results[channel] = channel_result
        await asyncio.sleep(0.2)

    # Build result record
    result = {
        "task_id": task["task_id"],
        "model": model,
        "domain_a": task.get("domain_a"),
        "domain_b": task.get("domain_b"),
        "tier": task.get("tier"),
        "intersection_types": task.get("intersection_types", {}),
        "layer2": layer2_result,
        "channels": channel_results,
    }

    return result


async def run_experiment_3(
    mode: str,
    run_id: str = None,
    resume: bool = False,
    models_override: list = None,
):
    """Run Experiment 3 data collection."""
    # Validate mode
    if mode not in ["pilot", "full", "control"]:
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
        max_tasks = 5
        channels = ["wagering", "natural"]  # Just 2 channels for speed
        use_controls = False
    elif mode == "control":
        models = [
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
            "deepseek-r1", "mistral-large", "qwen-3-235b", "gpt-oss-120b"
        ]
        max_tasks = None
        channels = []  # Controls don't use behavioral channels
        use_controls = True
    else:  # full
        models = [
            "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
            "deepseek-r1", "mistral-large", "qwen-3-235b", "gpt-oss-120b"
        ]
        max_tasks = None
        channels = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]
        use_controls = True

    # Override models if specified
    if models_override:
        models = models_override

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3: COMPOSITIONAL SELF-PREDICTION")
    print(f"{'='*80}")
    print(f"Mode: {mode.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(models)}")
    print(f"Channels: {', '.join(channels) if channels else 'None (controls only)'}")
    print(f"Resume: {resume}")
    print(f"{'='*80}\n")

    # Load tasks
    intersection_path = "data/exp3/intersection_tasks.jsonl"
    control_path = "data/exp3/control_tasks.jsonl"

    if not Path(intersection_path).exists():
        print(f"❌ Intersection tasks not found: {intersection_path}")
        print(f"   Run: python scripts/generate_exp3_tasks.py")
        sys.exit(1)

    if use_controls and not Path(control_path).exists():
        print(f"❌ Control tasks not found: {control_path}")
        print(f"   Run: python scripts/generate_exp3_tasks.py")
        sys.exit(1)

    print("Loading tasks...")
    intersection_tasks = load_tasks(intersection_path)
    control_tasks = load_tasks(control_path) if use_controls else []

    if max_tasks:
        intersection_tasks = intersection_tasks[:max_tasks]
        control_tasks = control_tasks[:max_tasks] if control_tasks else []

    all_tasks = intersection_tasks + control_tasks
    print(f"  Loaded {len(intersection_tasks)} intersection tasks")
    if control_tasks:
        print(f"  Loaded {len(control_tasks)} control tasks")
    print(f"  Total: {len(all_tasks)} tasks\n")

    # Setup results file
    results_path = Path(f"data/results/exp3_{run_id}_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    completed = load_existing_results(results_path) if resume else set()
    if completed:
        print(f"📋 Loaded checkpoint: {len(completed)} tasks completed\n")

    # Run collection
    client = UnifiedClient(experiment=f"exp3_{run_id}")
    total = len(models) * len(all_tasks)
    completed_count = len(completed)
    skipped_count = 0
    start_time = time.time()

    with open(results_path, "a", encoding="utf-8") as f:
        for model in models:
            print(f"\n{'─'*80}")
            print(f"Model: {model}")
            print(f"{'─'*80}")

            for task_idx, task in enumerate(all_tasks, 1):
                print(f"\nTask {task_idx}/{len(all_tasks)}: {task['task_id']}")

                key = (model, task["task_id"], "all")

                if key in completed:
                    print(f"  ✓ Skipped (checkpoint)")
                    skipped_count += 1
                    continue

                print(f"  Running...")

                result = await run_task_full(client, model, task, channels)

                # Write immediately with fsync
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())

                completed_count += 1
                progress = completed_count / total
                elapsed = time.time() - start_time
                if completed_count > len(completed):
                    eta_min = (elapsed / (completed_count - len(completed)) *
                               (total - completed_count)) / 60
                    print(f"  Progress: {progress:.1%} | ETA: {eta_min:.0f}min")

                await asyncio.sleep(0.3)

    duration_min = (time.time() - start_time) / 60
    completed_this_run = completed_count - len(completed)

    print(f"\n{'='*80}")
    print(f"Experiment 3 complete")
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"Completed this run: {completed_this_run} tasks")
    if skipped_count > 0:
        print(f"Skipped (checkpoint): {skipped_count} tasks")
    print(f"Results: {results_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Compositional Self-Prediction")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full", "control"],
        required=True,
        help="Experiment mode"
    )
    parser.add_argument("--run-id", help="Resume from existing run ID")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list override")
    args = parser.parse_args()

    models_override = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_experiment_3(
        mode=args.mode,
        run_id=args.run_id,
        resume=args.resume,
        models_override=models_override,
    ))


if __name__ == "__main__":
    main()
