"""Generate expanded Exp3 composite tasks for all 28 domain pairs.

Generates 4 unique composite tasks per pair = 112 total intersection tasks.
Uses LLM to generate then cross-verifies answers.
"""
import asyncio, json, os, sys, itertools, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from mirror.api.client import UnifiedClient

DOMAINS = ["arithmetic", "spatial", "temporal", "linguistic", "logical", "social", "factual", "procedural"]
TASKS_PER_PAIR = 4
GENERATOR_MODEL = "llama-3.1-70b"
VERIFIER_MODEL = "mistral-large"
OUTPUT_PATH = Path("data/exp3/intersection_tasks_v2.jsonl")

GENERATION_PROMPT = """Generate a question that REQUIRES skills from BOTH {domain_a} AND {domain_b} domains to answer correctly.

Requirements:
- The question must be impossible to answer with only one domain's knowledge
- It should have a single, unambiguous correct answer
- Answer should be short (1-3 words or a number)
- Provide the answer type: one of "exact_numeric", "multiple_choice", "boolean", "short_text"

Format your response EXACTLY as:
SCENARIO: [brief scenario name, 3-5 words]
QUESTION: [the composite question, 1-3 sentences]
CORRECT_ANSWER: [the correct answer]
ANSWER_TYPE: [exact_numeric/multiple_choice/boolean/short_text]
DOMAIN_A_SKILL: [what {domain_a} skill is needed, 1 sentence]
DOMAIN_B_SKILL: [what {domain_b} skill is needed, 1 sentence]

Generate task #{task_num} of {total} for this pair. Make it DIFFERENT from previous tasks.
"""

VERIFY_PROMPT = """A question was generated that should require both {domain_a} and {domain_b} skills.

Question: {question}
Claimed answer: {answer}

1. Is the claimed answer correct? Reply YES or NO.
2. Does this question genuinely require BOTH {domain_a} AND {domain_b} skills? Reply YES or NO.
3. What is your answer to the question?

Format:
ANSWER_CORRECT: [YES/NO]
REQUIRES_BOTH: [YES/NO]
YOUR_ANSWER: [your answer]
"""


async def generate_task(client, domain_a, domain_b, task_num, total):
    """Generate one composite task."""
    prompt = GENERATION_PROMPT.format(
        domain_a=domain_a, domain_b=domain_b,
        task_num=task_num, total=total
    )
    try:
        resp = await client.complete(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        text = resp.get("content", "")
        # Parse response
        task = {"domain_a": domain_a, "domain_b": domain_b, "raw": text}
        for field in ["SCENARIO", "QUESTION", "CORRECT_ANSWER", "ANSWER_TYPE", "DOMAIN_A_SKILL", "DOMAIN_B_SKILL"]:
            import re
            match = re.search(rf"{field}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            task[field.lower()] = match.group(1).strip() if match else ""
        return task
    except Exception as e:
        print(f"  Gen error {domain_a}×{domain_b} #{task_num}: {e}")
        return None


async def verify_task(client, task):
    """Verify a generated task."""
    prompt = VERIFY_PROMPT.format(
        domain_a=task["domain_a"], domain_b=task["domain_b"],
        question=task.get("question", ""), answer=task.get("correct_answer", "")
    )
    try:
        resp = await client.complete(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.get("content", "")
        task["verify_answer_correct"] = "YES" in text.split("ANSWER_CORRECT:")[-1][:20].upper() if "ANSWER_CORRECT:" in text.upper() else None
        task["verify_requires_both"] = "YES" in text.split("REQUIRES_BOTH:")[-1][:20].upper() if "REQUIRES_BOTH:" in text.upper() else None
        return task
    except Exception as e:
        print(f"  Verify error: {e}")
        task["verify_answer_correct"] = None
        task["verify_requires_both"] = None
        return task


async def main():
    client = UnifiedClient()
    all_pairs = list(itertools.combinations(DOMAINS, 2))
    print(f"Generating {len(all_pairs)} pairs × {TASKS_PER_PAIR} tasks = {len(all_pairs) * TASKS_PER_PAIR} tasks")

    all_tasks = []
    sem = asyncio.Semaphore(8)

    async def gen_one(pair, num):
        async with sem:
            return await generate_task(client, pair[0], pair[1], num, TASKS_PER_PAIR)

    # Generate all tasks
    coros = []
    for pair in all_pairs:
        for i in range(1, TASKS_PER_PAIR + 1):
            coros.append(gen_one(pair, i))

    print(f"Generating {len(coros)} tasks...")
    results = await asyncio.gather(*coros)
    tasks = [t for t in results if t is not None and t.get("question")]
    print(f"Generated {len(tasks)} valid tasks")

    # Verify all tasks
    print("Verifying tasks...")
    sem2 = asyncio.Semaphore(8)

    async def ver_one(task):
        async with sem2:
            return await verify_task(client, task)

    verified = await asyncio.gather(*[ver_one(t) for t in tasks])

    # Filter to verified tasks
    good = [t for t in verified if t.get("verify_answer_correct") and t.get("verify_requires_both")]
    print(f"Verified: {len(good)}/{len(verified)} passed both checks")

    # Format for Exp3 pipeline
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    task_id = 0
    with open(OUTPUT_PATH, "w") as f:
        for t in verified:  # keep all, mark verification status
            task_id += 1
            record = {
                "task_id": f"exp3v2_{task_id:04d}",
                "domain_a": t["domain_a"],
                "domain_b": t["domain_b"],
                "tier": 1,
                "scenario": t.get("scenario", ""),
                "task_text": t.get("question", ""),
                "component_a": {
                    "domain": t["domain_a"],
                    "text": t.get("question", ""),
                    "correct_answer": t.get("correct_answer", ""),
                    "difficulty": "medium",
                    "answer_type": t.get("answer_type", "short_text"),
                },
                "component_b": {
                    "domain": t["domain_b"],
                    "text": t.get("question", ""),
                    "correct_answer": t.get("correct_answer", ""),
                    "difficulty": "medium",
                    "answer_type": t.get("answer_type", "short_text"),
                },
                "requires_both": True,
                "verified_answer_correct": t.get("verify_answer_correct"),
                "verified_requires_both": t.get("verify_requires_both"),
                "domain_a_skill": t.get("domain_a_skill", ""),
                "domain_b_skill": t.get("domain_b_skill", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {task_id} tasks to {OUTPUT_PATH}")

    # Summary
    from collections import Counter
    pair_counts = Counter()
    for t in verified:
        pair_counts[(t["domain_a"], t["domain_b"])] += 1
    print(f"\nPairs covered: {len(pair_counts)}/28")
    print(f"Tasks per pair: min={min(pair_counts.values())}, max={max(pair_counts.values())}, mean={sum(pair_counts.values())/len(pair_counts):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
