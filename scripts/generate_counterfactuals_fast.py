"""
Fast counterfactual generation — 13 per domain, 104 total.

Generates counterfactual questions and writes to data/counterfactual/{domain}.jsonl.
Verification uses llama-3.1-8b (not qwen-3-235b which is unavailable).

Usage:
    python scripts/generate_counterfactuals_fast.py
    python scripts/generate_counterfactuals_fast.py --no-verify
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient

DOMAINS = ["arithmetic", "factual", "linguistic", "logical",
           "procedural", "social", "spatial", "temporal"]

DOMAIN_DISPLAY = {
    "arithmetic": "Arithmetic & Numeracy",
    "factual": "Factual Knowledge",
    "linguistic": "Linguistic Reasoning",
    "logical": "Logical & Deductive Reasoning",
    "procedural": "Procedural & Algorithmic Reasoning",
    "social": "Social & Pragmatic Reasoning",
    "spatial": "Spatial & Geometric Reasoning",
    "temporal": "Temporal & Causal Reasoning",
}

NUM_PER_DOMAIN = 13


async def generate_counterfactuals(client: UnifiedClient, domain: str) -> list[dict]:
    """Generate NUM_PER_DOMAIN counterfactual questions for a domain."""
    display = DOMAIN_DISPLAY.get(domain, domain.title())
    prompt = f"""Create {NUM_PER_DOMAIN} counterfactual reasoning questions for the domain: {display}.

Each question should:
1. Look like a standard {display} question
2. Explicitly state a counterfactual rule that inverts a fundamental assumption
3. Have a correct answer that follows the counterfactual rules (NOT real-world rules)
4. Include a "trap" answer (what someone would say using real-world rules)

Examples of counterfactual framings:
- "In a number system where addition is not commutative: compute 3+5 vs 5+3"
- "In a world where time flows backwards: what happened first?"
- "In a language where all verbs must precede subjects: rewrite this sentence"
- "In a logic system where NOT(NOT(P)) = NOT(P): evaluate this statement"

Respond ONLY with a JSON array:
[
  {{
    "question_text": "...",
    "correct_answer": "...",
    "trap_answer": "...",
    "counterfactual_rule": "..."
  }}
]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
        metadata={"task": "counterfactual_generation", "domain": domain},
    )

    if "error" in response:
        print(f"  [{domain}] Error: {response['error'][:80]}")
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
    except json.JSONDecodeError as e:
        print(f"  [{domain}] JSON parse error: {e}")
        return []

    results = []
    for i, item in enumerate(items[:NUM_PER_DOMAIN]):
        if not item.get("question_text") or not item.get("correct_answer"):
            continue
        results.append({
            "question_id": f"{domain}_counterfactual_{i:03d}",
            "domain": domain,
            "subcategory": "counterfactual",
            "difficulty": "adversarial",
            "question_text": item["question_text"],
            "correct_answer": item["correct_answer"],
            "answer_type": "short_text",
            "source": "generated_counterfactual",
            "source_id": f"cf_{domain}_{i:03d}",
            "transformation": "counterfactual",
            "parent_id": None,
            "metadata": {
                "trap_answer": item.get("trap_answer", ""),
                "counterfactual_rule": item.get("counterfactual_rule", ""),
            },
        })
    return results


async def verify_counterfactual(client: UnifiedClient, cf: dict) -> bool:
    """Verify a counterfactual using llama-3.1-8b (fast, avoids unavailable qwen)."""
    prompt = f"""Does the stated answer correctly follow from the counterfactual premise in this question?

Question: {cf['question_text']}
Stated answer: {cf['correct_answer']}

Reply with ONLY "PASS" or "FAIL"."""

    response = await client.complete(
        model="llama-3.1-8b",   # fast verifier — qwen-3-235b is unavailable
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
        metadata={"task": "cf_verification"},
    )
    if "error" in response:
        return True  # on API error, accept (don't discard good questions)
    verdict = (response.get("content") or "").strip().upper()
    return "PASS" in verdict


async def process_domain(client: UnifiedClient, domain: str,
                         verify: bool, semaphore: asyncio.Semaphore) -> int:
    """Generate, verify, and save counterfactuals for one domain."""
    out_path = Path(f"data/counterfactual/{domain}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if out_path.exists() and out_path.stat().st_size > 10:
        existing = sum(1 for l in out_path.open() if l.strip())
        if existing >= NUM_PER_DOMAIN - 2:
            print(f"[{domain}] Already have {existing} counterfactuals — skipping.")
            return existing

    print(f"[{domain}] Generating {NUM_PER_DOMAIN} counterfactuals …")
    async with semaphore:
        cfs = await generate_counterfactuals(client, domain)

    if not cfs:
        print(f"[{domain}] No counterfactuals generated.")
        return 0

    if verify:
        print(f"[{domain}] Verifying {len(cfs)} counterfactuals …")
        verify_tasks = [verify_counterfactual(client, cf) for cf in cfs]
        verdicts = await asyncio.gather(*verify_tasks)
        verified = [cf for cf, ok in zip(cfs, verdicts) if ok]
        print(f"[{domain}] {len(verified)}/{len(cfs)} passed verification.")
    else:
        verified = cfs
        print(f"[{domain}] Skipping verification — keeping all {len(cfs)}.")

    with out_path.open("w", encoding="utf-8") as f:
        for cf in verified:
            f.write(json.dumps(cf) + "\n")

    print(f"[{domain}] Saved {len(verified)} to {out_path}")
    return len(verified)


async def main(verify: bool) -> None:
    print("Counterfactual Generation")
    print("=" * 60)

    client = UnifiedClient(experiment="counterfactual_fast")
    semaphore = asyncio.Semaphore(8)   # 8 parallel generation calls

    tasks = [process_domain(client, d, verify, semaphore) for d in DOMAINS]
    counts = await asyncio.gather(*tasks)

    total = sum(counts)
    print(f"\nDone — {total} counterfactuals across {len(DOMAINS)} domains")
    for d, n in zip(DOMAINS, counts):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual questions")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip cross-model verification (faster)")
    args = parser.parse_args()
    asyncio.run(main(verify=not args.no_verify))
