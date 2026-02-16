"""
Variation generator - creates controlled variations of seed questions.

Generates 4-5 variations per seed to reach target count per domain.
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


async def generate_variations_for_seed(
    client: UnifiedClient,
    seed: dict,
    num_variations: int,
) -> list[dict]:
    """
    Generate variations for a single seed question.

    Args:
        client: UnifiedClient instance
        seed: Seed question dict
        num_variations: Number of variations to generate

    Returns:
        List of variation question dicts
    """
    prompt = f"""You are generating controlled variations of benchmark questions for a metacognition research project. The variations must preserve the logical structure and difficulty of the original while changing surface content to prevent memorization.

Original question:
{seed['question_text']}

Original answer: {seed['correct_answer']}

Generate {num_variations} variations of this question. For each variation:
1. Keep the SAME logical structure and reasoning steps required
2. Change surface details: names, numbers, objects, context, scenario
3. Maintain the same difficulty level
4. Provide the correct answer for each variation

CRITICAL: The answer must be definitively correct. Do not create ambiguous questions.

Respond in this exact JSON format:
[
  {{
    "question_text": "...",
    "correct_answer": "...",
    "transformation_description": "Brief description of what changed"
  }}
]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # Some creativity for variation
        max_tokens=2048,
        metadata={"task": "variation_generation", "seed_id": seed.get("question_id", seed["source_id"])}
    )

    if "error" in response:
        print(f"  ⚠️  Error generating variations: {response['error']}")
        return []

    # Parse JSON response
    try:
        content = response["content"]

        # Extract JSON from markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        variations_data = json.loads(content.strip())

        if not isinstance(variations_data, list):
            variations_data = [variations_data]

        # Convert to full question format
        variations = []
        for i, var_data in enumerate(variations_data[:num_variations]):
            variation = seed.copy()
            variation["question_text"] = var_data["question_text"]
            variation["correct_answer"] = var_data["correct_answer"]
            variation["transformation"] = "surface_variation"
            variation["parent_id"] = seed.get("question_id", seed["source_id"])
            variation["source_id"] = f"{seed['source_id']}_var_{i+1}"
            variation["metadata"] = {
                **seed.get("metadata", {}),
                "transformation_description": var_data.get("transformation_description", "")
            }
            variations.append(variation)

        return variations

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Failed to parse JSON: {e}")
        print(f"  Response: {response['content'][:200]}")
        return []


async def generate_variations_for_domain(
    client: UnifiedClient,
    domain_name: str,
    target_total: int,
) -> list[dict]:
    """
    Generate variations for all seeds in a domain.

    Args:
        client: UnifiedClient instance
        domain_name: Domain name
        target_total: Target total questions (seeds + variations)

    Returns:
        List of all questions (seeds + variations)
    """
    print(f"\n{'='*60}")
    print(f"Generating variations for: {domain_name}")
    print(f"{'='*60}")

    # Load seeds
    seeds_file = Path(f"data/seeds/{domain_name}.jsonl")
    if not seeds_file.exists():
        print(f"⚠️  No seeds file for {domain_name}")
        return []

    seeds = []
    with open(seeds_file, "r", encoding="utf-8") as f:
        for line in f:
            seeds.append(json.loads(line))

    print(f"Loaded {len(seeds)} seeds")

    # Check existing generations
    generated_file = Path(f"data/generated/{domain_name}.jsonl")
    generated_file.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if generated_file.exists():
        with open(generated_file, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                if q.get("parent_id"):
                    existing.add(q["parent_id"])

    print(f"Already have variations for {len(existing)} seeds")

    # Calculate variations needed per seed
    total_variations_needed = target_total - len(seeds)
    variations_per_seed = (total_variations_needed + len(seeds) - 1) // len(seeds)  # Ceil division

    print(f"Need {total_variations_needed} variations ({variations_per_seed} per seed)")

    all_questions = seeds.copy()

    # Generate variations
    for seed in tqdm(seeds, desc="Generating variations"):
        seed_id = seed.get("question_id", seed["source_id"])

        # Skip if already generated
        if seed_id in existing:
            continue

        # Generate variations
        variations = await generate_variations_for_seed(
            client,
            seed,
            num_variations=variations_per_seed
        )

        all_questions.extend(variations)

        # Save incrementally (for resumability)
        with open(generated_file, "a", encoding="utf-8") as f:
            for var in variations:
                f.write(json.dumps(var) + "\n")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Add question IDs
    for i, q in enumerate(all_questions):
        if "question_id" not in q:
            q["question_id"] = f"{domain_name}_{q['subcategory']}_{i:04d}"

    print(f"✅ Total questions for {domain_name}: {len(all_questions)}")
    return all_questions


async def generate_all_variations(pilot_mode: bool = False):
    """
    Generate variations for all domains.

    Args:
        pilot_mode: If True, generate fewer variations (target ~200 per domain)
    """
    config = load_domains_config()
    client = UnifiedClient(experiment="variation_generation")

    for domain_name, domain_config in config["domains"].items():
        target = 200 if pilot_mode else domain_config["target_total"]

        await generate_variations_for_domain(
            client,
            domain_name,
            target_total=target
        )


def run_variation_generation(pilot_mode: bool = False):
    """
    Run variation generation (sync wrapper).

    Args:
        pilot_mode: Whether to run in pilot mode
    """
    asyncio.run(generate_all_variations(pilot_mode))
