"""Temperature ablation: Run Exp1 (Ch1+Ch5) and Exp9 (C1+C4, P1) at temp=0.7.

Usage: python scripts/temperature_ablation.py
"""
import asyncio, json, os, sys, time, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import build_channel1_prompt, parse_channel1
from mirror.experiments.channels import build_channel5_prompt, parse_channel5
from mirror.scoring.answer_matcher import match_answer_robust

MODELS = ["llama-3.1-8b", "llama-3.1-70b", "mistral-large"]
MODEL_SLUGS = ["llama-3.1-8b", "llama-3.1-70b", "mistral-large"]
TEMPERATURE = 0.7
N_QUESTIONS = 200  # stratified across 8 domains (reduced for speed)
CONCURRENCY = 16  # parallel API calls
OUTPUT_DIR = Path("paper/supplementary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load question bank
QUESTIONS_PATH = Path("data/questions.jsonl")

def load_questions(n=500):
    """Load stratified sample of questions."""
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            q = json.loads(line)
            questions.append(q)
    # Stratify by domain
    by_domain = {}
    for q in questions:
        d = q.get("domain", "unknown")
        by_domain.setdefault(d, []).append(q)
    per_domain = n // len(by_domain)
    sampled = []
    for domain, qs in by_domain.items():
        random.seed(42)
        random.shuffle(qs)
        sampled.extend(qs[:per_domain])
    return sampled[:n]


async def run_one_exp1(client, model_id, model_slug, q, i, channel):
    """Run a single Exp1 question on one channel."""
    qtext = q.get("question_text", q.get("question", ""))
    correct_answer = q.get("correct_answer", q.get("answer", ""))
    answer_type = q.get("answer_type", "short_text")
    domain = q.get("domain", "unknown")
    if channel == 1:
        prompt = build_channel1_prompt({"question_text": qtext})
    else:
        prompt = build_channel5_prompt({"question_text": qtext})
    try:
        resp = await client.complete(model=model_id, messages=[{"role": "user", "content": prompt}], temperature=TEMPERATURE, max_tokens=512)
        text = resp.get("content", resp.get("text", ""))
        if channel == 1:
            parsed = parse_channel1(text)
            answer = parsed.get("answer", "")
            bet = parsed.get("bet_size")
        else:
            parsed = parse_channel5(text)
            answer = parsed.get("answer", "")
            bet = None
        correct = match_answer_robust(answer, correct_answer, answer_type)
        return {"model": model_slug, "domain": domain, "channel": channel, "answer_correct": correct, "bet_size": bet, "temperature": TEMPERATURE}
    except Exception as e:
        return None

async def run_exp1_temp(client, model_id, model_slug, questions):
    """Run Exp1 Ch1 + Ch5 at temperature=0.7 with concurrency."""
    sem = asyncio.Semaphore(CONCURRENCY)
    async def bounded(coro):
        async with sem:
            return await coro
    tasks = []
    for i, q in enumerate(questions):
        for ch in [1, 5]:
            tasks.append(bounded(run_one_exp1(client, model_id, model_slug, q, i, ch)))
    results_raw = await asyncio.gather(*tasks)
    results = [r for r in results_raw if r is not None]
    print(f"  {model_slug} Exp1: {len(results)}/{len(tasks)} succeeded", flush=True)
    return results


async def run_exp9_temp(client, model_id, model_slug):
    """Run simplified Exp9 C1 + C4 at temperature=0.7.

    C1: Model decides autonomously (just answer the question).
    C4: If model's Exp1 accuracy < 0.50, force tool use (oracle correct).
    """
    # Load exp1 accuracy for this model
    exp1_files = sorted(Path("data/results").glob("exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
    exp1_acc = {}
    for f in exp1_files:
        data = json.load(open(f))
        if model_slug in data:
            for domain, metrics in data[model_slug].items():
                if isinstance(metrics, dict):
                    exp1_acc[domain] = metrics.get("natural_acc", 0.5)

    # Load fixed tasks
    tasks_path = Path("data/exp9_tasks.jsonl")
    if not tasks_path.exists():
        print(f"  {model_slug}: No exp9_tasks.jsonl found, skipping Exp9")
        return []

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            t = json.loads(line)
            if t.get("circularity_free", False):  # fixed tasks only
                tasks.append(t)

    random.seed(42)
    random.shuffle(tasks)
    tasks = tasks[:100]  # subset for speed

    async def run_one_exp9(task, i, condition):
        domain_a = task.get("domain_a", "")
        part1 = task.get("part1_text", task.get("task_text", ""))
        correct_a = task.get("correct_answer_a", "")
        answer_type_a = task.get("answer_type_a", "short_text")
        acc_a = exp1_acc.get(domain_a, 0.5)
        if condition == 4 and acc_a < 0.50:
            return {"model": model_slug, "task_id": task.get("task_id", i), "condition": 4, "domain": domain_a, "decision": "use_tool", "correct": True, "externally_routed": True, "temperature": TEMPERATURE}
        prompt = f"Answer this question concisely.\n\n{part1}\n\nANSWER:"
        try:
            resp = await client.complete(model=model_id, messages=[{"role": "user", "content": prompt}], temperature=TEMPERATURE, max_tokens=256)
            text = resp.get("content", resp.get("text", ""))
            correct = match_answer_robust(text, correct_a, answer_type_a)
            return {"model": model_slug, "task_id": task.get("task_id", i), "condition": condition, "domain": domain_a, "decision": "proceed", "correct": correct, "externally_routed": False, "temperature": TEMPERATURE}
        except:
            return None

    sem = asyncio.Semaphore(CONCURRENCY)
    async def bounded(coro):
        async with sem:
            return await coro
    coros = []
    for i, task in enumerate(tasks):
        for cond in [1, 4]:
            coros.append(bounded(run_one_exp9(task, i, cond)))
    raw = await asyncio.gather(*coros)
    results = [r for r in raw if r is not None]
    print(f"  {model_slug} Exp9: {len(results)}/{len(coros)} succeeded", flush=True)
    return results


async def main():
    client = UnifiedClient()
    questions = load_questions(N_QUESTIONS)
    print(f"Loaded {len(questions)} questions")

    all_exp1 = []
    all_exp9 = []

    for model_id, model_slug in zip(MODELS, MODEL_SLUGS):
        print(f"\n=== {model_slug} (temp={TEMPERATURE}) ===")

        # Exp1
        print(f"Running Exp1 ({len(questions)} questions, Ch1+Ch5)...")
        exp1_results = await run_exp1_temp(client, model_id, model_slug, questions)
        all_exp1.extend(exp1_results)
        print(f"  Got {len(exp1_results)} Exp1 records")

        # Exp9
        print(f"Running Exp9 (C1+C4, fixed tasks)...")
        exp9_results = await run_exp9_temp(client, model_id, model_slug)
        all_exp9.extend(exp9_results)
        print(f"  Got {len(exp9_results)} Exp9 records")

    # Compute metrics
    print("\n=== RESULTS ===")

    # Exp1: MIRROR gap at temp=0.7
    print("\nExp1 MIRROR Gap (temp=0.7):")
    import numpy as np
    for slug in MODEL_SLUGS:
        ch1 = [r for r in all_exp1 if r["model"] == slug and r["channel"] == 1]
        ch5 = [r for r in all_exp1 if r["model"] == slug and r["channel"] == 5]
        if ch1 and ch5:
            wag_acc = sum(1 for r in ch1 if r["answer_correct"]) / len(ch1) if ch1 else 0
            nat_acc = sum(1 for r in ch5 if r["answer_correct"]) / len(ch5) if ch5 else 0
            gap = abs(wag_acc - nat_acc)
            print(f"  {slug}: nat_acc={nat_acc:.3f}, wag_acc={wag_acc:.3f}, gap={gap:.3f}")

    # Exp9: CFR at temp=0.7
    print("\nExp9 CFR (temp=0.7):")
    for slug in MODEL_SLUGS:
        for cond in [1, 4]:
            recs = [r for r in all_exp9 if r["model"] == slug and r["condition"] == cond]
            proceed = [r for r in recs if r["decision"] == "proceed"]
            if proceed:
                cfr = sum(1 for r in proceed if not r["correct"]) / len(proceed)
                print(f"  {slug} C{cond}: CFR={cfr:.3f} (n={len(proceed)} proceed, {len(recs)} total)")
            else:
                print(f"  {slug} C{cond}: no proceed records (all routed)")

    # Save
    output = {
        "temperature": TEMPERATURE,
        "models": MODEL_SLUGS,
        "exp1_records": len(all_exp1),
        "exp9_records": len(all_exp9),
        "exp1_results": all_exp1,
        "exp9_results": all_exp9,
    }
    outpath = OUTPUT_DIR / "temperature_ablation.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
