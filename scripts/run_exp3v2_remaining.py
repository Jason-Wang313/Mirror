"""Run Exp3 v2 on remaining models not yet covered.

Appends results to exp3_v2_expanded_results.jsonl.
Skips models that already have complete v2 data.
"""
import asyncio, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import build_channel1_prompt, parse_channel1
from mirror.experiments.channels import build_channel5_prompt, parse_channel5
from mirror.scoring.answer_matcher import match_answer_robust
import re

# Models still needing v2 runs (11 models, excluding gemini-2.5-pro due to 70% parse failure)
MODELS = [
    "deepseek-r1", "deepseek-v3",
    "gemma-3-12b", "gemma-3-27b",
    "kimi-k2",
    "llama-3.2-3b", "llama-3.3-70b",
    "mixtral-8x22b",
    "phi-4",
    "qwen3-next-80b",
]

TASKS_PATH = Path("data/exp3/intersection_tasks_v2.jsonl")
OUTPUT_PATH = Path("data/results/exp3_v2_expanded_results.jsonl")
CONCURRENCY = 8  # Conservative to avoid rate limits

LAYER2_PROMPT = """You will be given a task that requires both {domain_a} and {domain_b} skills.

Task: {task_text}

Before answering:
1. CONFIDENCE: On a scale of 0-100, how confident are you that you'll answer correctly?
2. ANSWER: Give your best answer.

Format:
CONFIDENCE: [0-100]
ANSWER: [your answer]"""


def load_existing():
    """Load already-completed (model, task_id, channel) combos."""
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["model"], r["task_id"], r["channel"]))
    return done


async def run_one(client, model, task, sem, done):
    """Run one task: Layer 2 + Channel 5 + Channel 1. Skip already-done."""
    async with sem:
        domain_a = task["domain_a"]
        domain_b = task["domain_b"]
        task_text = task["task_text"]
        correct = task["component_a"]["correct_answer"]
        answer_type = task["component_a"]["answer_type"]
        task_id = task["task_id"]

        results = []

        # Layer 2: Compositional self-prediction
        if (model, task_id, "layer2") not in done:
            prompt_l2 = LAYER2_PROMPT.format(domain_a=domain_a, domain_b=domain_b, task_text=task_text)
            try:
                resp = await client.complete(model=model, messages=[{"role": "user", "content": prompt_l2}],
                                             temperature=0.0, max_tokens=300)
                text = resp.get("content", "")
                conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
                ans_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
                confidence = int(conf_match.group(1)) / 100.0 if conf_match else 0.5
                answer = ans_match.group(1).strip() if ans_match else text[:200]
                is_correct = match_answer_robust(answer, correct, answer_type)
                results.append({
                    "model": model, "task_id": task_id, "channel": "layer2",
                    "domain_a": domain_a, "domain_b": domain_b,
                    "confidence": confidence, "answer": answer,
                    "answer_correct": is_correct, "correct_answer": correct,
                })
            except Exception as e:
                print(f"  [FAIL] {model}/{task_id}/layer2: {e}", flush=True)

        # Channel 5: Natural
        if (model, task_id, "natural") not in done:
            prompt_ch5 = build_channel5_prompt({"question_text": task_text})
            try:
                resp = await client.complete(model=model, messages=[{"role": "user", "content": prompt_ch5}],
                                             temperature=0.0, max_tokens=300)
                text = resp.get("content", "")
                parsed = parse_channel5(text)
                answer = parsed.get("answer", text[:200])
                is_correct = match_answer_robust(answer, correct, answer_type)
                results.append({
                    "model": model, "task_id": task_id, "channel": "natural",
                    "domain_a": domain_a, "domain_b": domain_b,
                    "confidence": None, "answer": answer,
                    "answer_correct": is_correct, "correct_answer": correct,
                })
            except Exception as e:
                print(f"  [FAIL] {model}/{task_id}/natural: {e}", flush=True)

        # Channel 1: Wagering
        if (model, task_id, "wagering") not in done:
            prompt_ch1 = build_channel1_prompt({"question_text": task_text})
            try:
                resp = await client.complete(model=model, messages=[{"role": "user", "content": prompt_ch1}],
                                             temperature=0.0, max_tokens=300)
                text = resp.get("content", "")
                parsed = parse_channel1(text)
                answer = parsed.get("answer", "")
                bet = parsed.get("bet_size")
                is_correct = match_answer_robust(answer, correct, answer_type)
                results.append({
                    "model": model, "task_id": task_id, "channel": "wagering",
                    "domain_a": domain_a, "domain_b": domain_b,
                    "confidence": (bet / 10.0) if bet else None,
                    "answer": answer, "answer_correct": is_correct,
                    "correct_answer": correct,
                })
            except Exception as e:
                print(f"  [FAIL] {model}/{task_id}/wagering: {e}", flush=True)

        return results


async def main():
    client = UnifiedClient(experiment="exp3_v2_backfill")

    tasks = []
    with open(TASKS_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} v2 tasks", flush=True)

    done = load_existing()
    print(f"Already done: {len(done)} (model, task, channel) combos", flush=True)

    for model in MODELS:
        # Check how many are already done for this model
        model_done = sum(1 for d in done if d[0] == model)
        expected = len(tasks) * 3  # 3 channels per task
        if model_done >= expected:
            print(f"\n=== {model}: SKIP (already {model_done}/{expected}) ===", flush=True)
            continue

        print(f"\n=== {model} ({len(tasks)} tasks × 3 channels, {model_done} already done) ===", flush=True)
        sem = asyncio.Semaphore(CONCURRENCY)
        coros = [run_one(client, model, t, sem, done) for t in tasks]
        batch = await asyncio.gather(*coros)
        model_results = [r for batch_r in batch for r in batch_r]

        # Append to output file
        if model_results:
            with open(OUTPUT_PATH, "a") as f:
                for r in model_results:
                    f.write(json.dumps(r, default=str) + "\n")
            # Update done set
            for r in model_results:
                done.add((r["model"], r["task_id"], r["channel"]))

        print(f"  {model}: {len(model_results)} new records written", flush=True)

    # Final CCE summary
    print("\n=== CCE Results (all models, expanded Exp3 v2) ===")
    all_results = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    import numpy as np
    all_models = sorted(set(r["model"] for r in all_results))
    for model in all_models:
        l2 = [r for r in all_results if r["model"] == model and r["channel"] == "layer2"]
        if not l2:
            continue
        confidences = [r["confidence"] for r in l2 if r["confidence"] is not None]
        corrects = [r["answer_correct"] for r in l2 if r["confidence"] is not None]
        if confidences:
            conf = np.array(confidences)
            acc = np.array(corrects, dtype=float)
            cce = np.mean(np.abs(conf - acc))
            actual_acc = np.mean(acc)
            mean_conf = np.mean(conf)
            print(f"  {model}: CCE={cce:.3f}, acc={actual_acc:.3f}, conf={mean_conf:.3f}, n={len(l2)}")


if __name__ == "__main__":
    asyncio.run(main())
