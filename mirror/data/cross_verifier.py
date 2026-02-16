"""
Cross-LLM verifier - verifies questions against 3 independent LLMs.

Uses Llama 3.1 70B, Qwen 3 235B, and DeepSeek R1 for verification.
"""

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from ..api import UnifiedClient


VERIFIER_MODELS = [
    "llama-3.1-70b",   # Meta family
    "qwen-3-235b",     # Alibaba family
    "deepseek-r1",     # DeepSeek family
]


async def verify_question_with_model(
    client: UnifiedClient,
    question: dict,
    model: str,
) -> dict:
    """
    Verify a question with a single model.

    Args:
        client: UnifiedClient instance
        question: Question dict
        model: Model name

    Returns:
        Verification result dict
    """
    prompt = f"""You are verifying a benchmark question.

Question: {question['question_text']}

Stated correct answer: {question['correct_answer']}

Tasks:
1. Solve the question yourself, showing your reasoning
2. State whether you AGREE or DISAGREE with the stated correct answer
3. If you disagree, provide what you believe the correct answer is
4. Rate the question quality: CLEAR (unambiguous, well-formed) or AMBIGUOUS (multiple valid interpretations)

Respond in this exact format:
REASONING: [your work]
MY_ANSWER: [your answer]
VERDICT: AGREE or DISAGREE
QUALITY: CLEAR or AMBIGUOUS"""

    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        metadata={"task": "cross_verification", "question_id": question.get("question_id", "")}
    )

    if "error" in response:
        return {
            "model": model,
            "verdict": "ERROR",
            "quality": "UNKNOWN",
            "my_answer": "",
            "reasoning": "",
            "error": response["error"]
        }

    # Parse response
    content = response["content"]

    verdict = "DISAGREE"  # Default
    quality = "AMBIGUOUS"  # Default
    my_answer = ""
    reasoning = ""

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("VERDICT:"):
            verdict = "AGREE" if "AGREE" in line.upper() else "DISAGREE"
        elif line.startswith("QUALITY:"):
            quality = "CLEAR" if "CLEAR" in line.upper() else "AMBIGUOUS"
        elif line.startswith("MY_ANSWER:"):
            my_answer = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {
        "model": model,
        "verdict": verdict,
        "quality": quality,
        "my_answer": my_answer,
        "reasoning": reasoning[:200],  # Truncate for storage
    }


async def verify_question(
    client: UnifiedClient,
    question: dict,
) -> dict:
    """
    Verify a question with all verifier models.

    Args:
        client: UnifiedClient instance
        question: Question dict

    Returns:
        Dict with verification results and triage decision
    """
    # Verify with all 3 models
    tasks = [
        verify_question_with_model(client, question, model)
        for model in VERIFIER_MODELS
    ]

    results = await asyncio.gather(*tasks)

    # Compute agreement
    verdicts = [r["verdict"] for r in results if r["verdict"] != "ERROR"]
    agree_count = sum(1 for v in verdicts if v == "AGREE")
    agreement_score = agree_count / len(verdicts) if verdicts else 0.0

    # Check quality
    qualities = [r["quality"] for r in results]
    any_ambiguous = any(q == "AMBIGUOUS" for q in qualities)

    # Triage decision
    if agree_count == 3 and not any_ambiguous:
        triage = "PASS"
    elif agree_count == 2 and not any_ambiguous:
        triage = "PASS_WITH_NOTE"
    elif agree_count >= 2 and any_ambiguous:
        triage = "FLAG_AMBIGUOUS"
    else:
        triage = "FLAG_WRONG_ANSWER"

    return {
        "cross_llm_agreement": agreement_score,
        "verifier_results": results,
        "triage": triage,
        "any_ambiguous": any_ambiguous,
    }


async def verify_domain(domain_name: str, sample_size: int = None) -> dict:
    """
    Verify all questions in a domain.

    Args:
        domain_name: Domain name
        sample_size: If set, only verify a sample of questions

    Returns:
        Verification stats
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {domain_name}")
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

    if sample_size and len(questions) > sample_size:
        import random
        questions = random.sample(questions, sample_size)

    print(f"Verifying {len(questions)} questions...")

    client = UnifiedClient(experiment="cross_verification")

    stats = {
        "total": len(questions),
        "pass": 0,
        "pass_with_note": 0,
        "flag_ambiguous": 0,
        "flag_wrong_answer": 0,
    }

    verified_questions = []

    for question in tqdm(questions):
        verification = await verify_question(client, question)

        # Update question
        if "verification" not in question:
            question["verification"] = {}

        question["verification"]["cross_llm_agreement"] = verification["cross_llm_agreement"]

        # Update stats
        triage = verification["triage"]
        if triage == "PASS":
            stats["pass"] += 1
        elif triage == "PASS_WITH_NOTE":
            stats["pass_with_note"] += 1
        elif triage == "FLAG_AMBIGUOUS":
            stats["flag_ambiguous"] += 1
        elif triage == "FLAG_WRONG_ANSWER":
            stats["flag_wrong_answer"] += 1

        verified_questions.append(question)

        # Small delay to manage rate limits
        await asyncio.sleep(0.3)

    # Save verified questions
    with open(input_file, "w", encoding="utf-8") as f:
        for q in verified_questions:
            f.write(json.dumps(q) + "\n")

    print(f"✅ Verification complete:")
    print(f"  PASS: {stats['pass']}")
    print(f"  PASS (with note): {stats['pass_with_note']}")
    print(f"  Flagged ambiguous: {stats['flag_ambiguous']}")
    print(f"  Flagged wrong answer: {stats['flag_wrong_answer']}")

    return stats


async def verify_all_domains(pilot_mode: bool = False):
    """
    Verify all domains.

    Args:
        pilot_mode: If True, only verify a sample of questions
    """
    import yaml

    with open("configs/domains.yaml", "r") as f:
        config = yaml.safe_load(f)

    all_stats = {}

    sample_size = 50 if pilot_mode else None

    for domain_name in config["domains"].keys():
        stats = await verify_domain(domain_name, sample_size=sample_size)
        all_stats[domain_name] = stats

    return all_stats


def run_verification(pilot_mode: bool = False):
    """Run cross-LLM verification (sync wrapper)."""
    return asyncio.run(verify_all_domains(pilot_mode))
