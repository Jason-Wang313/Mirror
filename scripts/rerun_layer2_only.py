"""
Re-run ONLY Layer 2 self-reports for Experiment 3.
Keeps existing behavioral channel data intact.
"""
import asyncio
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.api.client import UnifiedClient


async def run_layer2_self_report(
    client: UnifiedClient,
    model: str,
    task: dict,
) -> dict:
    """Run Layer 2 verbal self-report - SAME AS IN run_experiment_3.py."""
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
        metadata={"experiment": "exp3_layer2_rerun", "layer": "layer2", "task_id": task["task_id"]}
    )

    if "error" in response:
        return {"error": response["error"], "raw_response": None}

    response_text = response.get("content", "")

    # Handle None response_text
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
        "raw_response": response_text,  # Save full response
    }

    import re

    # Extract CONFIDENCE - flexible patterns (handle markdown ** and other formatting)
    conf_patterns = [
        r'\*{0,2}CONFIDENCE\*{0,2}[:\s=-]+\*{0,2}(\d+)',  # **CONFIDENCE:** 85 or CONFIDENCE: 85
        r'confidence\s+(?:is|at|level)[:\s]+(\d+)',
        r'rate.*?confidence.*?(\d+)',
        r'I.*?estimate.*?(\d+)%?',  # "I'd estimate 80"
        r'about\s+(\d+)%(?!\d)',  # "about 80%" but not "2023"
        r'(\d+)/100',  # 80/100
    ]
    for pattern in conf_patterns:
        conf_match = re.search(pattern, response_text, re.IGNORECASE)
        if conf_match:
            parsed["confidence"] = int(conf_match.group(1))
            break

    # Extract COMPARISON - flexible patterns (handle markdown)
    comp_patterns = [
        r'\*{0,2}COMPARISON\*{0,2}[:\s=-]+\*{0,2}(easier|same|harder)',  # **COMPARISON:** harder
        r'(?:will be|would be|is|seems)\s+(easier|same|harder)',
        r'(easier|harder|same)\s+than',
        r'(?:Slightly\s+)?(easier|harder)',  # "Slightly harder"
        r'more\s+(difficult|challenging)',
        r'less\s+(difficult|challenging)',
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

    # Extract WEAK_LINK - flexible patterns (handle markdown and various phrasings)
    wl_patterns = [
        r'\*{0,2}WEAK[_\s-]?LINK\*{0,2}[:\s=-]+\*{0,2}(\w+)',  # **WEAK_LINK:** factual
        r'(?:struggle|difficulty|weaker|weakness)\s+(?:with|at|in)\s+(\w+)',
        r'(\w+)\s+(?:is|will be)\s+(?:weaker|harder|more difficult)',
        r'more\s+likely\s+to\s+struggle\s+with\s+(\w+)',
        r'weak.*?point.*?(?:is|be)\s+(\w+)',
        r'challenging.*?(?:part|aspect|component)[:\s]+(\w+)',
        r'(\w+)\s+recall',  # "Factual recall" or "Arithmetic recall"
    ]
    for pattern in wl_patterns:
        wl_match = re.search(pattern, response_text, re.IGNORECASE)
        if wl_match:
            parsed["weak_link"] = wl_match.group(1).lower()
            break

    # Extract PREDICTION - flexible patterns (handle markdown and various phrasings)
    pred_patterns = [
        r'\*{0,2}PREDICTION\*{0,2}[:\s=-]+\*{0,2}(\d+)%?',  # **PREDICTION:** 90%
        r'predict.*?(\d+)%',
        r'estimate.*?(\d+)%',
        r'(\d+)%\s+(?:accuracy|correct|chance)',
        r'accuracy.*?(\d+)%',
        r'expect.*?(\d+)%',
        r'(\d+)%\s+(?:probability|likely)',
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
        else:
            parsed["answer"] = response_text.strip()

    return parsed


async def rerun_layer2_for_model(
    client: UnifiedClient,
    model: str,
    tasks: list[dict],
    results: dict,
    results_file: Path,
    show_examples: bool = False,
) -> int:
    """Re-run Layer 2 for all tasks for a given model."""
    print(f"\n{'='*80}")
    print(f"Re-running Layer 2 for {model}")
    print(f"{'='*80}")

    updated_count = 0
    examples_shown = 0

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]

        # Find existing result
        result_key = f"{model}_{task_id}"
        if result_key not in results:
            print(f"  Warning: No existing result for {task_id}, skipping")
            continue

        # Run Layer 2
        print(f"  [{i}/{len(tasks)}] {task_id}...", end=" ", flush=True)
        layer2_result = await run_layer2_self_report(client, model, task)

        # Update the result's layer2 field (keep channels intact)
        results[result_key]["layer2"] = layer2_result
        updated_count += 1

        # Save immediately after each task (append mode)
        with open(results_file, 'w') as f:
            for result in results.values():
                f.write(json.dumps(result) + '\n')

        # Show parsing status
        conf = "✓" if layer2_result.get("confidence") else "✗"
        comp = "✓" if layer2_result.get("comparison") else "✗"
        wl = "✓" if layer2_result.get("weak_link") else "✗"
        pred = "✓" if layer2_result.get("prediction") else "✗"
        print(f"conf:{conf} comp:{comp} wl:{wl} pred:{pred}")

        # Print example raw responses for first model (up to 5)
        if show_examples and examples_shown < 5 and layer2_result.get("raw_response"):
            examples_shown += 1
            print(f"\n{'─'*80}")
            print(f"RAW RESPONSE EXAMPLE {examples_shown}/{5}")
            print(f"Task: {task_id}")
            print(f"{'─'*80}")
            print(layer2_result["raw_response"][:600])
            print("...")
            print(f"{'─'*80}\n")

    print(f"\n✅ Updated {updated_count} results for {model}")
    print(f"✅ Saved to {results_file} after each task")
    return updated_count


async def main(run_id: str):
    """Main re-run logic."""
    results_file = Path(f"data/results/{run_id}_results.jsonl")

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    print(f"Loading existing results from: {results_file}")

    # Load existing results into dict keyed by model_taskid
    results = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            key = f"{r['model']}_{r['task_id']}"
            results[key] = r

    print(f"Loaded {len(results)} existing results")

    # Extract unique tasks (assume all models have same tasks)
    tasks_by_id = {}
    for r in results.values():
        task_id = r["task_id"]
        if task_id not in tasks_by_id:
            tasks_by_id[task_id] = {
                "task_id": task_id,
                "task_text": r.get("question", ""),  # May need to load from task file
                "domain_a": r.get("domain_a"),
                "domain_b": r.get("domain_b"),
            }

    # Load task texts from original task file
    task_file = Path("data/exp3/intersection_tasks.jsonl")
    if task_file.exists():
        print(f"Loading task texts from: {task_file}")
        with open(task_file) as f:
            for line in f:
                task = json.loads(line)
                if task["task_id"] in tasks_by_id:
                    tasks_by_id[task["task_id"]]["task_text"] = task["task_text"]
    else:
        print(f"Warning: Task file not found at {task_file}")

    tasks = list(tasks_by_id.values())
    print(f"Found {len(tasks)} unique tasks")

    # Get models
    models = sorted(set(r["model"] for r in results.values()))
    print(f"Found {len(models)} models: {', '.join(models)}")

    # Initialize client
    client = UnifiedClient()

    # Re-run Layer 2 for each model
    total_updated = 0
    for idx, model in enumerate(models):
        # Show examples for first model only
        show_examples = (idx == 0)
        updated = await rerun_layer2_for_model(
            client, model, tasks, results, results_file, show_examples
        )
        total_updated += updated

    print(f"\n{'='*80}")
    print(f"Total updated: {total_updated} results")
    print(f"{'='*80}")
    print(f"✅ Complete! All results saved incrementally to {results_file}")

    # Print final statistics
    print("\nLayer2 parsing statistics by model:")
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "conf": 0, "comp": 0, "wl": 0, "pred": 0})

    for result in results.values():
        model = result["model"]
        layer2 = result.get("layer2", {})
        stats[model]["total"] += 1
        if layer2.get("confidence") is not None:
            stats[model]["conf"] += 1
        if layer2.get("comparison") is not None:
            stats[model]["comp"] += 1
        if layer2.get("weak_link") is not None:
            stats[model]["wl"] += 1
        if layer2.get("prediction") is not None:
            stats[model]["pred"] += 1

    for model in sorted(stats.keys()):
        s = stats[model]
        print(f"\n{model}:")
        print(f"  Confidence: {s['conf']}/{s['total']} ({100*s['conf']/s['total']:.1f}%)")
        print(f"  Comparison: {s['comp']}/{s['total']} ({100*s['comp']/s['total']:.1f}%)")
        print(f"  Weak-link:  {s['wl']}/{s['total']} ({100*s['wl']/s['total']:.1f}%)")
        print(f"  Prediction: {s['pred']}/{s['total']} ({100*s['pred']/s['total']:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rerun_layer2_only.py <run_id>")
        print("Example: python scripts/rerun_layer2_only.py exp3_20260224T120251")
        sys.exit(1)

    run_id = sys.argv[1]
    asyncio.run(main(run_id))
