"""
Fast parallel question generation — reaches 625 per domain.

Reads seeds from data/seeds_v2/, generates new variations in batches of 5,
appends to data/generated_v2/, then compiles everything to data/questions.jsonl.

Usage:
    python scripts/generate_questions_fast.py
    python scripts/generate_questions_fast.py --target 625
    python scripts/generate_questions_fast.py --compile-only
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient

DOMAINS = ["arithmetic", "factual", "linguistic", "logical",
           "procedural", "social", "spatial", "temporal"]
BATCH_SIZE = 5          # variations per API call (safe for max_tokens=2048)
MAX_CONCURRENT = 16     # parallel API calls across all domains


async def generate_batch(client: UnifiedClient, seed: dict, n: int,
                         domain: str, round_idx: int) -> list[dict]:
    """Generate n variations of a seed in one API call."""
    prompt = f"""Generate {n} controlled variations of this benchmark question.
Keep the SAME reasoning structure, change only surface details (names, numbers, context).
Domain: {domain}

Original: {seed['question_text']}
Answer: {seed['correct_answer']}

Respond ONLY with a JSON array:
[
  {{"question_text": "...", "correct_answer": "...", "transformation_description": "..."}}
]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
        metadata={"task": "fast_variation", "domain": domain},
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

    seed_id = seed.get("source_id") or seed.get("question_id") or "unknown"
    variations = []
    for i, item in enumerate(items[:n]):
        var = seed.copy()
        var["question_text"] = item.get("question_text", "")
        var["correct_answer"] = item.get("correct_answer", "")
        var["transformation"] = "surface_variation"
        var["parent_id"] = seed_id
        var["source_id"] = f"{seed_id}_fast_r{round_idx}_v{i}"
        var["metadata"] = {
            **seed.get("metadata", {}),
            "transformation_description": item.get("transformation_description", ""),
        }
        if var["question_text"]:
            variations.append(var)

    return variations


async def fill_domain(client: UnifiedClient, domain: str,
                      target: int, semaphore: asyncio.Semaphore) -> int:
    """Generate variations for one domain until `target` is reached."""
    seeds_path = Path(f"data/seeds_v2/{domain}.jsonl")
    gen_path = Path(f"data/generated_v2/{domain}.jsonl")
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [json.loads(l) for l in seeds_path.open() if l.strip()]
    existing_count = sum(1 for l in gen_path.open() if l.strip()) if gen_path.exists() else 0
    current = len(seeds) + existing_count
    needed = max(0, target - current)

    if needed <= 0:
        print(f"[{domain}] Already at {current}/{target} — skipping.")
        return existing_count

    print(f"[{domain}] Need {needed} more (current={current}, seeds={len(seeds)}).")

    # Rounds: keep generating batches of BATCH_SIZE per seed until done
    per_seed = max(1, -(-needed // len(seeds)))   # ceil division
    rounds = -(-per_seed // BATCH_SIZE)            # ceil

    new_count = 0
    with gen_path.open("a", encoding="utf-8") as f:
        for round_idx in range(rounds):
            tasks = []
            for seed in seeds:
                remaining = needed - new_count
                if remaining <= 0:
                    break
                batch = min(BATCH_SIZE, remaining)
                tasks.append((seed, batch, round_idx))

            if not tasks:
                break

            async def bounded(seed, batch, round_idx):
                async with semaphore:
                    return await generate_batch(client, seed, batch, domain, round_idx)

            results = await asyncio.gather(*[bounded(s, b, r) for s, b, r in tasks])
            for variations in results:
                for var in variations:
                    f.write(json.dumps(var) + "\n")
                    new_count += 1
                    if new_count >= needed:
                        break
                if new_count >= needed:
                    break
            f.flush()
            print(f"  [{domain}] Round {round_idx+1}: +{sum(len(r) for r in results)} "
                  f"(total new={new_count}/{needed})")

    print(f"[{domain}] Done: {existing_count + new_count} variations written.")
    return existing_count + new_count


def compile_questions(target: int) -> int:
    """Merge seeds_v2 + generated_v2 into data/questions.jsonl (up to target/domain)."""
    print("\nCompiling to data/questions.jsonl …")
    all_questions: list[dict] = []
    seen_ids: set = set()

    for domain in DOMAINS:
        domain_qs: list[dict] = []

        for path in [Path(f"data/seeds_v2/{domain}.jsonl"),
                     Path(f"data/generated_v2/{domain}.jsonl")]:
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                q = json.loads(line)
                qid = q.get("source_id") or q.get("question_id") or id(q)
                if qid not in seen_ids:
                    seen_ids.add(qid)
                    q.setdefault("domain", domain)
                    domain_qs.append(q)

        selected = domain_qs[:target]
        all_questions.extend(selected)
        print(f"  {domain}: {len(selected)}/{len(domain_qs)} selected")

    with open("data/questions.jsonl", "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    totals = Counter(q.get("domain", "?") for q in all_questions)
    print(f"\nTotal: {len(all_questions)} questions")
    for d, n in sorted(totals.items()):
        print(f"  {d}: {n}")
    return len(all_questions)


async def main(target: int, compile_only: bool) -> None:
    if not compile_only:
        print(f"Generating questions — target {target}/domain")
        print("="*60)

        client = UnifiedClient(experiment="generate_questions_fast")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        domain_tasks = [fill_domain(client, d, target, semaphore) for d in DOMAINS]
        await asyncio.gather(*domain_tasks)

    compile_questions(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast question generation to target per domain")
    parser.add_argument("--target", type=int, default=625)
    parser.add_argument("--compile-only", action="store_true",
                        help="Skip generation, just recompile questions.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args.target, args.compile_only))
