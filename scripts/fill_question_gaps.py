"""
Fills question bank gaps per domain using timestamp-based unique IDs.
Generates exactly N new questions per domain needed to reach 625 unique.
"""
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirror.api.client import UnifiedClient

DOMAINS = ["arithmetic", "factual", "linguistic", "logical",
           "procedural", "social", "spatial", "temporal"]
TARGET = 625
BATCH_SIZE = 5
MAX_CONCURRENT = 16
GEN_MODEL = "llama-3.1-70b"


def count_unique(domain):
    seen = set()
    for path_str in [f"data/seeds_v2/{domain}.jsonl",
                     f"data/generated_v2/{domain}.jsonl"]:
        p = Path(path_str)
        if not p.exists():
            continue
        for line in open(p):
            if line.strip():
                q = json.loads(line)
                qid = q.get("source_id") or q.get("question_id") or "none"
                seen.add(qid)
    return seen


def load_seeds(domain):
    p = Path(f"data/seeds_v2/{domain}.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


async def generate_batch(client, seed, n, domain, ts_suffix):
    prompt = f"""Generate {n} controlled variations of this benchmark question.
Keep the SAME reasoning structure, change only surface details (names, numbers, context).
Domain: {domain}

Original: {seed.get('question_text', '')}
Answer: {seed.get('correct_answer', '')}

Respond ONLY with a JSON array:
[
  {{"question_text": "...", "correct_answer": "...", "transformation_description": "..."}}
]"""
    response = await client.complete(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
        metadata={"task": "gap_fill", "domain": domain},
    )
    if "error" in response:
        return []
    content = response.get("content") or ""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    try:
        items = json.loads(content.strip())
        if not isinstance(items, list):
            items = [items]
    except json.JSONDecodeError:
        return []

    seed_id = seed.get("source_id") or seed.get("question_id") or "seed"
    variations = []
    for i, item in enumerate(items[:n]):
        var = seed.copy()
        var["question_text"] = item.get("question_text", "")
        var["correct_answer"] = item.get("correct_answer", "")
        var["transformation"] = "surface_variation"
        var["parent_id"] = seed_id
        # Use timestamp suffix for uniqueness
        var["source_id"] = f"{seed_id}_gap_{ts_suffix}_{i}"
        var["metadata"] = {
            **seed.get("metadata", {}),
            "transformation_description": item.get("transformation_description", ""),
        }
        if var["question_text"]:
            variations.append(var)
    return variations


async def fill_domain(client, domain, semaphore):
    existing_ids = count_unique(domain)
    needed = TARGET - len(existing_ids)
    if needed <= 0:
        print(f"[{domain}] Already at {len(existing_ids)}/{TARGET} — skip.")
        return 0

    print(f"[{domain}] Need {needed} more (have {len(existing_ids)} unique).")
    seeds = load_seeds(domain)
    if not seeds:
        print(f"[{domain}] No seeds found!")
        return 0

    ts_suffix = str(int(time.time()))
    gen_path = Path(f"data/generated_v2/{domain}.jsonl")
    new_count = 0
    seed_idx = 0

    with open(gen_path, "a", encoding="utf-8") as f:
        while new_count < needed:
            batch_tasks = []
            for _ in range(min(MAX_CONCURRENT, needed - new_count)):
                seed = seeds[seed_idx % len(seeds)]
                seed_idx += 1
                n = min(BATCH_SIZE, needed - new_count - len(batch_tasks))
                if n <= 0:
                    break
                ts = f"{int(time.time())}_{seed_idx}"
                batch_tasks.append((seed, n, ts))

            if not batch_tasks:
                break

            async def bounded(seed, n, ts):
                async with semaphore:
                    return await generate_batch(client, seed, n, domain, ts)

            results = await asyncio.gather(*[bounded(s, n, ts) for s, n, ts in batch_tasks])
            for variations in results:
                for var in variations:
                    if new_count >= needed:
                        break
                    f.write(json.dumps(var) + "\n")
                    new_count += 1
                if new_count >= needed:
                    break
            f.flush()
            print(f"  [{domain}] +{new_count}/{needed}", flush=True)

    print(f"[{domain}] Done: added {new_count} new questions.")
    return new_count


async def main():
    print("Gap-fill question generation")
    print("=" * 50)

    client = UnifiedClient(experiment="fill_question_gaps")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [fill_domain(client, d, semaphore) for d in DOMAINS]
    totals = await asyncio.gather(*tasks)

    print(f"\nTotal new questions generated: {sum(totals)}")

    # Compile
    print("\nCompiling questions.jsonl...")
    import subprocess
    r = subprocess.run(
        ["python", "scripts/generate_questions_fast.py", "--compile-only", "--target", str(TARGET)],
        capture_output=True, text=True
    )
    print(r.stdout[-1000:] if r.stdout else "")
    if r.returncode == 0:
        from collections import Counter
        domains_count = Counter()
        for line in open("data/questions.jsonl"):
            if line.strip():
                q = json.loads(line)
                domains_count[q.get("domain", "?")] += 1
        total = sum(domains_count.values())
        print(f"questions.jsonl: {total} questions")
        for d, c in sorted(domains_count.items()):
            print(f"  {d}: {c}")
    else:
        print("Compile failed:", r.stderr[-500:])


if __name__ == "__main__":
    asyncio.run(main())
