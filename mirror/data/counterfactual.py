"""
Counterfactual generator - creates questions that break standard reasoning templates.

Generates ~100 counterfactual questions (12-13 per domain).
"""

import asyncio
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from ..api import UnifiedClient


def load_domains_config() -> dict:
    """Load domain configuration from YAML."""
    with open("configs/domains.yaml", "r") as f:
        return yaml.safe_load(f)


async def generate_counterfactuals_for_domain(
    client: UnifiedClient,
    domain_name: str,
    domain_display: str,
    num_counterfactuals: int = 13,
) -> list[dict]:
    """
    Generate counterfactual questions for a domain.

    Args:
        client: UnifiedClient instance
        domain_name: Domain name
        domain_display: Display name for domain
        num_counterfactuals: Number to generate

    Returns:
        List of counterfactual question dicts
    """
    prompt = f"""You are creating counterfactual reasoning questions that break standard reasoning templates. These questions use familiar structures but invert fundamental rules or assumptions.

Domain: {domain_display}

Create {num_counterfactuals} counterfactual questions for this domain. Each question should:
1. Use a format that LOOKS like a standard {domain_display} question
2. BUT changes a fundamental rule or assumption
3. The correct answer should follow from the counterfactual rules, NOT from real-world rules

Examples of counterfactual framing:
- "In a world where gravity pushes objects upward..."
- "Assume multiplication is not commutative: compute 3×5 and 5×3"
- "In a language where adjectives come after verbs instead of before nouns..."
- "In a logic system where modus ponens is invalid..."

For each question provide:
1. The question text (clearly stating the counterfactual premise)
2. The correct answer under the counterfactual rules
3. The answer that would be correct under normal rules (the "trap" answer)

Respond in JSON format:
[
  {{
    "question_text": "...",
    "correct_answer": "...",
    "trap_answer": "...",
    "counterfactual_rule": "Brief description of what rule is changed"
  }}
]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,  # More creativity for counterfactuals
        max_tokens=3000,
        metadata={"task": "counterfactual_generation", "domain": domain_name}
    )

    if "error" in response:
        print(f"  ⚠️  Error generating counterfactuals: {response['error']}")
        return []

    # Parse JSON
    try:
        content = response["content"]

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        cf_data = json.loads(content.strip())

        if not isinstance(cf_data, list):
            cf_data = [cf_data]

        # Convert to full format
        counterfactuals = []
        for i, cf in enumerate(cf_data[:num_counterfactuals]):
            counterfactuals.append({
                "question_id": f"{domain_name}_counterfactual_{i:03d}",
                "domain": domain_name,
                "subcategory": "counterfactual",
                "difficulty": "adversarial",
                "question_text": cf["question_text"],
                "correct_answer": cf["correct_answer"],
                "answer_type": "short_text",
                "source": "generated_counterfactual",
                "source_id": f"cf_{domain_name}_{i}",
                "transformation": "counterfactual",
                "parent_id": None,
                "verification": {},
                "metadata": {
                    "trap_answer": cf.get("trap_answer", ""),
                    "counterfactual_rule": cf.get("counterfactual_rule", "")
                }
            })

        return counterfactuals

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Failed to parse JSON: {e}")
        return []


async def verify_counterfactual(
    client: UnifiedClient,
    cf: dict,
) -> bool:
    """
    Verify a counterfactual question with a different model.

    Args:
        client: UnifiedClient instance
        cf: Counterfactual question dict

    Returns:
        True if verification passed
    """
    prompt = f"""You are verifying a counterfactual reasoning question.

Question: {cf['question_text']}

Stated correct answer: {cf['correct_answer']}

Task: Does this answer correctly follow from the counterfactual premises stated in the question? Answer with ONLY "PASS" or "FAIL"."""

    response = await client.complete(
        model="qwen-3-235b",  # Different model for verification
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
        metadata={"task": "counterfactual_verification"}
    )

    if "error" in response:
        return False

    verdict = response["content"].strip().upper()
    return "PASS" in verdict


async def generate_all_counterfactuals(pilot_mode: bool = False):
    """
    Generate counterfactual questions for all domains.

    Args:
        pilot_mode: If True, generate fewer counterfactuals (5 per domain)
    """
    config = load_domains_config()
    client = UnifiedClient(experiment="counterfactual_generation")

    output_dir = Path("data/counterfactual")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_counterfactuals = []

    num_per_domain = 5 if pilot_mode else 13

    for domain_name, domain_config in config["domains"].items():
        print(f"\n{'='*60}")
        print(f"Generating counterfactuals for: {domain_name}")
        print(f"{'='*60}")

        counterfactuals = await generate_counterfactuals_for_domain(
            client,
            domain_name,
            domain_config["display_name"],
            num_counterfactuals=num_per_domain
        )

        # Verify each counterfactual
        print(f"Verifying {len(counterfactuals)} counterfactuals...")
        verified = []
        for cf in tqdm(counterfactuals):
            passed = await verify_counterfactual(client, cf)
            if passed:
                verified.append(cf)
            else:
                print(f"  ⚠️  Verification failed for: {cf['question_text'][:60]}...")

        print(f"✅ {len(verified)}/{len(counterfactuals)} counterfactuals verified")

        # Save
        with open(output_dir / f"{domain_name}.jsonl", "w", encoding="utf-8") as f:
            for cf in verified:
                f.write(json.dumps(cf) + "\n")

        all_counterfactuals.extend(verified)

    print(f"\n{'='*60}")
    print(f"Total counterfactuals generated: {len(all_counterfactuals)}")
    print(f"{'='*60}")


def run_counterfactual_generation(pilot_mode: bool = False):
    """Run counterfactual generation (sync wrapper)."""
    asyncio.run(generate_all_counterfactuals(pilot_mode))
