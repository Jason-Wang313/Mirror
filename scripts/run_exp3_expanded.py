"""Run Exp3 on the expanded v2 task bank (112 tasks, all 28 domain pairs).

Runs Layer 2 self-report + Channel 5 (natural) for CCE computation.
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

MODELS = [
    "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
    "mistral-large", "gpt-oss-120b",
    "gemma-3-27b", "kimi-k2", "llama-3.3-70b",
]
TASKS_PATH = Path("data/exp3/intersection_tasks_v2.jsonl")
OUTPUT_PATH = Path("data/results/exp3_v2_expanded_results.jsonl")
CONCURRENCY = 12

# Layer 2 prompt: ask model to predict its composite accuracy
LAYER2_PROMPT = """You will be given a task that requires both {domain_a} and {domain_b} skills.

Task: {task_text}

Before answering:
1. CONFIDENCE: On a scale of 0-100, how confident are you that you'll answer correctly?
2. ANSWER: Give your best answer.

Format:
CONFIDENCE: [0-100]
ANSWER: [your answer]"""


async def run_one(client, model, task, sem):
    """Run one task: Layer 2 + Channel 5."""
    async with sem:
        domain_a = task["domain_a"]
        domain_b = task["domain_b"]
        task_text = task["task_text"]
        correct = task["component_a"]["correct_answer"]
        answer_type = task["component_a"]["answer_type"]
        task_id = task["task_id"]

        results = []

        # Layer 2: Compositional self-prediction
        prompt_l2 = LAYER2_PROMPT.format(domain_a=domain_a, domain_b=domain_b, task_text=task_text)
        try:
            resp = await client.complete(model=model, messages=[{"role": "user", "content": prompt_l2}],
                                         temperature=0.0, max_tokens=300)
            text = resp.get("content", "")
            import re
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
            pass

        # Channel 5: Natural (unconstrained)
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
            pass

        # Channel 1: Wagering
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
            pass

        return results


async def main():
    client = UnifiedClient()

    # Load tasks
    tasks = []
    with open(TASKS_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks", flush=True)

    # Run per model
    all_results = []
    for model in MODELS:
        print(f"\n=== {model} ({len(tasks)} tasks × 3 channels) ===", flush=True)
        sem = asyncio.Semaphore(CONCURRENCY)
        coros = [run_one(client, model, t, sem) for t in tasks]
        batch = await asyncio.gather(*coros)
        model_results = [r for batch_r in batch for r in batch_r]
        all_results.extend(model_results)
        print(f"  {model}: {len(model_results)} records", flush=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(all_results)} records to {OUTPUT_PATH}", flush=True)

    # Compute CCE per model
    print("\n=== CCE Results (expanded Exp3 v2) ===")
    for model in MODELS:
        l2 = [r for r in all_results if r["model"] == model and r["channel"] == "layer2"]
        if not l2:
            continue
        # CCE = |predicted_confidence - actual_accuracy|
        confidences = [r["confidence"] for r in l2 if r["confidence"] is not None]
        corrects = [r["answer_correct"] for r in l2 if r["confidence"] is not None]
        if confidences:
            import numpy as np
            conf = np.array(confidences)
            acc = np.array(corrects, dtype=float)
            cce = np.mean(np.abs(conf - acc))
            actual_acc = np.mean(acc)
            mean_conf = np.mean(conf)
            print(f"  {model}: CCE={cce:.3f}, acc={actual_acc:.3f}, conf={mean_conf:.3f}, n={len(l2)}")

    # Check MCI (do channels agree on which tasks are hard?)
    print("\n=== Channel Agreement ===")
    for model in MODELS:
        wag = {r["task_id"]: r["answer_correct"] for r in all_results
               if r["model"] == model and r["channel"] == "wagering"}
        nat = {r["task_id"]: r["answer_correct"] for r in all_results
               if r["model"] == model and r["channel"] == "natural"}
        common = set(wag.keys()) & set(nat.keys())
        if len(common) >= 10:
            agree = sum(1 for t in common if wag[t] == nat[t])
            print(f"  {model}: wagering-natural agreement = {agree}/{len(common)} = {agree/len(common):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
