"""
Difficulty validator - calibrates difficulty using Llama 3.1 8B as baseline.

Assigns difficulty based on whether Llama 8B answers correctly.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from ..api import UnifiedClient
from .answer_matcher import match_answer


async def validate_question_difficulty(
    client: UnifiedClient,
    question: dict,
) -> tuple[str, bool]:
    """
    Validate difficulty of a question using Llama 8B.

    Args:
        client: UnifiedClient instance
        question: Question dict

    Returns:
        Tuple of (assigned_difficulty, is_correct)
    """
    prompt = f"""Answer this question concisely.

{question['question_text']}

Provide only the answer, nothing else."""

    response = await client.complete(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
        metadata={"task": "difficulty_validation", "question_id": question.get("question_id", "")}
    )

    if "error" in response:
        # Can't determine, use existing or default to medium
        return question.get("difficulty") or "medium", False

    # Check if answer is correct
    predicted = response["content"].strip()
    is_correct = match_answer(
        predicted,
        question["correct_answer"],
        question["answer_type"],
        question.get("metadata", {})
    )

    # Assign difficulty based on correctness + source info
    source_difficulty = question.get("difficulty")

    if is_correct:
        # Llama 8B got it right
        if source_difficulty in ["easy", "medium", None]:
            difficulty = "easy"
        else:
            difficulty = "medium"  # Hard question but 8B solved it
    else:
        # Llama 8B got it wrong
        if source_difficulty == "easy":
            difficulty = "medium"  # Should be easy but 8B failed
        else:
            difficulty = "hard"

    return difficulty, is_correct


async def validate_domain_difficulty(domain_name: str) -> dict:
    """
    Validate difficulty for all questions in a domain.

    Args:
        domain_name: Domain name

    Returns:
        Stats dict
    """
    print(f"\n{'='*60}")
    print(f"Validating difficulty for: {domain_name}")
    print(f"{'='*60}")

    # Load questions
    input_file = Path(f"data/verified/{domain_name}.jsonl")
    if not input_file.exists():
        print(f"No verified file for {domain_name}, skipping...")
        return {}

    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Validating {len(questions)} questions...")

    client = UnifiedClient(experiment="difficulty_validation")

    difficulty_counts = defaultdict(int)
    llama8b_correct_count = 0

    for question in tqdm(questions):
        difficulty, is_correct = await validate_question_difficulty(client, question)

        question["difficulty"] = difficulty
        difficulty_counts[difficulty] += 1

        if is_correct:
            llama8b_correct_count += 1

        if "verification" not in question:
            question["verification"] = {}
        question["verification"]["difficulty_validated"] = True

        await asyncio.sleep(0.2)  # Rate limiting

    # Save updated questions
    with open(input_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    stats = {
        "total": len(questions),
        "llama8b_accuracy": llama8b_correct_count / len(questions) if questions else 0,
        "difficulty_distribution": dict(difficulty_counts),
    }

    print(f"✅ Difficulty validation complete:")
    print(f"  Llama 8B accuracy: {stats['llama8b_accuracy']:.1%}")
    print(f"  Difficulty distribution:")
    for diff, count in difficulty_counts.items():
        print(f"    {diff}: {count}")

    return stats


async def validate_all_difficulties():
    """Validate difficulty for all domains."""
    import yaml

    with open("configs/domains.yaml", "r") as f:
        config = yaml.safe_load(f)

    all_stats = {}

    for domain_name in config["domains"].keys():
        stats = await validate_domain_difficulty(domain_name)
        all_stats[domain_name] = stats

    return all_stats


def run_difficulty_validation():
    """Run difficulty validation (sync wrapper)."""
    return asyncio.run(validate_all_difficulties())
