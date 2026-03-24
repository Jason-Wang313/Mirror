"""Extended temperature ablation: Run 6 additional models at temp=0.7.

Adds to existing 3-model data (llama-3.1-8b, llama-3.1-70b, mistral-large)
to create a 9-model temperature ablation across 6 labs.

New models: deepseek-r1, gemma-3-27b, kimi-k2, phi-4, gpt-oss-120b, llama-3.3-70b

Usage: python scripts/temperature_ablation_extended.py
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

# Only the NEW 6 models (existing 3 already done)
# deepseek-r1 removed — API balance exhausted
MODELS = [
    "gemma-3-27b", "kimi-k2",
    "phi-4", "gpt-oss-120b", "llama-3.3-70b",
]
TEMPERATURE = 0.7
N_QUESTIONS = 200
CONCURRENCY = 24  # Higher concurrency for speed
OUTPUT_DIR = Path("paper/supplementary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_PATH = Path("data/questions.jsonl")


def load_questions(n=200):
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line))
    by_domain = {}
    for q in questions:
        by_domain.setdefault(q.get("domain", "unknown"), []).append(q)
    per_domain = n // len(by_domain)
    sampled = []
    for domain, qs in by_domain.items():
        random.seed(42)
        random.shuffle(qs)
        sampled.extend(qs[:per_domain])
    return sampled[:n]


async def run_one_exp1(client, model_id, q, channel):
    qtext = q.get("question_text", q.get("question", ""))
    correct_answer = q.get("correct_answer", q.get("answer", ""))
    answer_type = q.get("answer_type", "short_text")
    domain = q.get("domain", "unknown")
    if channel == 1:
        prompt = build_channel1_prompt({"question_text": qtext})
    else:
        prompt = build_channel5_prompt({"question_text": qtext})
    try:
        resp = await client.complete(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE, max_tokens=512
        )
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
        return {"model": model_id, "domain": domain, "channel": channel,
                "answer_correct": correct, "bet_size": bet, "temperature": TEMPERATURE}
    except Exception as e:
        return None


async def run_exp9_temp(client, model_id):
    exp1_files = sorted(Path("data/results").glob("exp1_*_accuracy.json"),
                        key=lambda p: p.stat().st_mtime)
    exp1_acc = {}
    for f in exp1_files:
        data = json.load(open(f))
        if model_id in data:
            for domain, metrics in data[model_id].items():
                if isinstance(metrics, dict):
                    exp1_acc[domain] = metrics.get("natural_acc", 0.5)

    tasks = []
    with open("data/exp9_tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t.get("circularity_free", False):
                tasks.append(t)
    random.seed(42)
    random.shuffle(tasks)
    tasks = tasks[:100]

    async def run_one(task, condition):
        domain_a = task.get("domain_a", "")
        part1 = task.get("part1_text", task.get("task_text", ""))
        correct_a = task.get("correct_answer_a", "")
        answer_type_a = task.get("answer_type_a", "short_text")
        acc_a = exp1_acc.get(domain_a, 0.5)
        if condition == 4 and acc_a < 0.50:
            return {"model": model_id, "task_id": task.get("task_id", ""),
                    "condition": 4, "domain": domain_a, "decision": "use_tool",
                    "correct": True, "externally_routed": True, "temperature": TEMPERATURE}
        prompt = f"Answer this question concisely.\n\n{part1}\n\nANSWER:"
        try:
            resp = await client.complete(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE, max_tokens=256
            )
            text = resp.get("content", resp.get("text", ""))
            correct = match_answer_robust(text, correct_a, answer_type_a)
            return {"model": model_id, "task_id": task.get("task_id", ""),
                    "condition": condition, "domain": domain_a, "decision": "proceed",
                    "correct": correct, "externally_routed": False, "temperature": TEMPERATURE}
        except:
            return None

    sem = asyncio.Semaphore(CONCURRENCY)
    async def bounded(coro):
        async with sem:
            return await coro
    coros = []
    for task in tasks:
        for cond in [1, 4]:
            coros.append(bounded(run_one(task, cond)))
    raw = await asyncio.gather(*coros)
    results = [r for r in raw if r is not None]
    print(f"  {model_id} Exp9: {len(results)}/{len(coros)} succeeded", flush=True)
    return results


async def main():
    client = UnifiedClient()
    questions = load_questions(N_QUESTIONS)
    print(f"Loaded {len(questions)} questions")
    print(f"Models: {MODELS}")
    print(f"Temperature: {TEMPERATURE}")

    all_exp1 = []
    all_exp9 = []

    for model_id in MODELS:
        print(f"\n=== {model_id} (temp={TEMPERATURE}) ===")

        # Exp1
        sem = asyncio.Semaphore(CONCURRENCY)
        async def bounded(coro):
            async with sem:
                return await coro
        tasks_exp1 = []
        for q in questions:
            for ch in [1, 5]:
                tasks_exp1.append(bounded(run_one_exp1(client, model_id, q, ch)))
        raw = await asyncio.gather(*tasks_exp1)
        exp1_results = [r for r in raw if r is not None]
        all_exp1.extend(exp1_results)
        print(f"  Exp1: {len(exp1_results)}/{len(tasks_exp1)} succeeded")

        # Exp9
        exp9_results = await run_exp9_temp(client, model_id)
        all_exp9.extend(exp9_results)
        print(f"  Exp9: {len(exp9_results)} records")

    # Merge with existing data
    existing_path = OUTPUT_DIR / "temperature_ablation.json"
    if existing_path.exists():
        existing = json.load(open(existing_path))
        existing_exp1 = existing.get("exp1_records", [])
        existing_exp9 = existing.get("exp9_records", [])
        existing_models = existing.get("models", [])
    else:
        existing_exp1 = []
        existing_exp9 = []
        existing_models = []

    merged = {
        "temperature": TEMPERATURE,
        "models": existing_models + MODELS,
        "exp1_records": existing_exp1 + all_exp1,
        "exp9_records": existing_exp9 + all_exp9,
        "exp1_results": {},
        "exp9_results": {},
    }

    # Compute per-model summary
    for model_id in merged["models"]:
        model_exp1 = [r for r in merged["exp1_records"] if r["model"] == model_id]
        if model_exp1:
            ch1 = [r for r in model_exp1 if r["channel"] == 1]
            ch5 = [r for r in model_exp1 if r["channel"] == 5]
            wag_acc = sum(1 for r in ch1 if r["answer_correct"]) / len(ch1) if ch1 else None
            nat_acc = sum(1 for r in ch5 if r["answer_correct"]) / len(ch5) if ch5 else None
            gap = abs(wag_acc - nat_acc) if wag_acc is not None and nat_acc is not None else None
            merged["exp1_results"][model_id] = {
                "wagering_acc": round(wag_acc, 3) if wag_acc else None,
                "natural_acc": round(nat_acc, 3) if nat_acc else None,
                "mirror_gap": round(gap, 3) if gap else None,
                "n_ch1": len(ch1), "n_ch5": len(ch5),
            }

        model_exp9 = [r for r in merged["exp9_records"] if r["model"] == model_id]
        if model_exp9:
            c1 = [r for r in model_exp9 if r["condition"] == 1]
            c4 = [r for r in model_exp9 if r["condition"] == 4]
            c1_fail = sum(1 for r in c1 if r["decision"] == "proceed" and not r["correct"])
            c1_total = len(c1)
            c4_fail = sum(1 for r in c4 if r["decision"] == "proceed" and not r["correct"])
            c4_total = len(c4)
            cfr_c1 = c1_fail / c1_total if c1_total else None
            cfr_c4 = c4_fail / c4_total if c4_total else None
            reduction = (cfr_c1 - cfr_c4) / cfr_c1 * 100 if cfr_c1 and cfr_c1 > 0 else None
            merged["exp9_results"][model_id] = {
                "cfr_c1": round(cfr_c1, 3) if cfr_c1 is not None else None,
                "cfr_c4": round(cfr_c4, 3) if cfr_c4 is not None else None,
                "reduction_pct": round(reduction, 1) if reduction is not None else None,
                "n_c1": c1_total, "n_c4": c4_total,
            }

    # Save merged
    out_path = OUTPUT_DIR / "temperature_ablation_9models.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved merged results to {out_path}")
    print(f"Total: {len(merged['models'])} models, {len(merged['exp1_records'])} exp1 records, {len(merged['exp9_records'])} exp9 records")

    # Print summary table
    print("\n=== TEMPERATURE ABLATION SUMMARY (t=0.7) ===")
    print(f"{'Model':20s} {'Gap(t=0.7)':>10s} {'CFR_C1':>8s} {'CFR_C4':>8s} {'Reduction':>10s}")
    for m in merged["models"]:
        e1 = merged["exp1_results"].get(m, {})
        e9 = merged["exp9_results"].get(m, {})
        gap = e1.get("mirror_gap", "?")
        cfr1 = e9.get("cfr_c1", "?")
        cfr4 = e9.get("cfr_c4", "?")
        red = e9.get("reduction_pct", "?")
        print(f"  {m:20s} {str(gap):>10s} {str(cfr1):>8s} {str(cfr4):>8s} {str(red):>10s}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
