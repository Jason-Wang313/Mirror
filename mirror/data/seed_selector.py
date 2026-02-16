"""
Seed selector - selects high-quality seed questions with LLM classification.

Selects 130 seeds per domain ensuring subcategory coverage.
"""

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from ..api import UnifiedClient


def load_domains_config() -> dict:
    """Load domain configuration from YAML."""
    with open("configs/domains.yaml", "r") as f:
        return yaml.safe_load(f)


async def classify_question(
    client: UnifiedClient,
    question: dict,
    domain_name: str,
    subcategories: list[str],
) -> str:
    """
    Use LLM to classify a question into the correct subcategory.

    Args:
        client: UnifiedClient instance
        question: Question dict
        domain_name: Domain name
        subcategories: List of possible subcategories

    Returns:
        Classified subcategory name
    """
    prompt = f"""You are classifying questions for a metacognition research benchmark.

Domain: {domain_name}
Subcategories: {", ".join(subcategories)}

Question: {question['question_text'][:500]}

Which subcategory does this question best belong to? Respond with ONLY the subcategory name, nothing else."""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
        metadata={"task": "seed_classification", "domain": domain_name}
    )

    if "error" in response:
        # Fall back to existing subcategory
        return question.get("subcategory", subcategories[0])

    # Parse response
    classification = response["content"].strip().lower().replace(" ", "_")

    # Validate
    if classification in [s.lower() for s in subcategories]:
        return classification
    else:
        # Fall back to existing
        return question.get("subcategory", subcategories[0])


async def select_seeds_for_domain(
    client: UnifiedClient,
    domain_name: str,
    domain_config: dict,
    all_questions: list[dict],
    target_seeds: int = 130,
) -> list[dict]:
    """
    Select seeds for a single domain.

    Args:
        client: UnifiedClient instance
        domain_name: Domain name
        domain_config: Domain configuration
        all_questions: All loaded questions
        target_seeds: Target number of seeds

    Returns:
        List of selected seed questions
    """
    print(f"\n{'='*60}")
    print(f"Selecting seeds for: {domain_name}")
    print(f"{'='*60}")

    # Filter questions for this domain
    domain_questions = [q for q in all_questions if q["domain"] == domain_name]
    print(f"Total questions in domain: {len(domain_questions)}")

    if len(domain_questions) == 0:
        print(f"⚠️  No questions found for {domain_name}")
        return []

    # Get subcategories
    subcategories = domain_config["subcategories"]
    seeds_per_subcat = target_seeds // len(subcategories)

    # Classify questions into subcategories (batch)
    print(f"Classifying questions into subcategories...")

    # For questions that already have subcategories, use them
    # For others, classify with LLM
    questions_to_classify = [
        q for q in domain_questions
        if not q.get("subcategory") or q["subcategory"] not in subcategories
    ]

    if questions_to_classify:
        # Classify in batches to avoid rate limits
        for i in tqdm(range(0, len(questions_to_classify), 20)):
            batch = questions_to_classify[i:i+20]

            # Classify batch
            tasks = [
                classify_question(client, q, domain_name, subcategories)
                for q in batch
            ]

            results = await asyncio.gather(*tasks)

            for q, subcat in zip(batch, results):
                q["subcategory"] = subcat

    # Group by subcategory
    by_subcategory = defaultdict(list)
    for q in domain_questions:
        by_subcategory[q["subcategory"]].append(q)

    # Select seeds from each subcategory
    selected_seeds = []

    for subcat in subcategories:
        available = by_subcategory.get(subcat, [])
        print(f"  {subcat}: {len(available)} available")

        if len(available) == 0:
            print(f"    ⚠️  No questions for subcategory {subcat}")
            continue

        # Sample with diversity
        if len(available) <= seeds_per_subcat:
            selected = available
        else:
            # Prefer variety in sources and answer types
            selected = []
            remaining = available.copy()

            # First, get one from each unique source if possible
            sources_seen = set()
            for q in remaining[:]:
                if q["source"] not in sources_seen:
                    selected.append(q)
                    sources_seen.add(q["source"])
                    remaining.remove(q)
                if len(selected) >= seeds_per_subcat:
                    break

            # Fill remaining with random sampling
            if len(selected) < seeds_per_subcat:
                random.shuffle(remaining)
                needed = seeds_per_subcat - len(selected)
                selected.extend(remaining[:needed])

        print(f"    Selected: {len(selected)}")
        selected_seeds.extend(selected)

    # If we're short, fill from largest subcategories
    if len(selected_seeds) < target_seeds:
        shortfall = target_seeds - len(selected_seeds)
        print(f"  Filling shortfall of {shortfall} from largest subcategories...")

        # Get extras from largest subcategories
        all_ids = {q["source_id"] for q in selected_seeds}
        candidates = [q for q in domain_questions if q["source_id"] not in all_ids]

        random.shuffle(candidates)
        selected_seeds.extend(candidates[:shortfall])

    # Mark as seeds
    for seed in selected_seeds:
        seed["transformation"] = "original"
        seed["parent_id"] = None

    print(f"Total seeds selected: {len(selected_seeds)}")
    return selected_seeds[:target_seeds]


async def select_all_seeds(all_questions: list[dict], pilot_mode: bool = False) -> dict[str, list[dict]]:
    """
    Select seeds for all domains.

    Args:
        all_questions: All loaded questions from all sources
        pilot_mode: If True, select 26 seeds per domain instead of 130

    Returns:
        Dict mapping domain name to list of seed questions
    """
    config = load_domains_config()
    client = UnifiedClient(experiment="seed_selection")

    all_seeds = {}

    for domain_name, domain_config in config["domains"].items():
        target = 26 if pilot_mode else domain_config["target_seeds"]

        seeds = await select_seeds_for_domain(
            client,
            domain_name,
            domain_config,
            all_questions,
            target_seeds=target
        )

        all_seeds[domain_name] = seeds

        # Save seeds for this domain
        output_file = Path(f"data/seeds/{domain_name}.jsonl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for seed in seeds:
                f.write(json.dumps(seed) + "\n")

        print(f"✅ Saved {len(seeds)} seeds to {output_file}")

    return all_seeds


def run_seed_selection(all_questions: list[dict], pilot_mode: bool = False):
    """
    Run seed selection (sync wrapper).

    Args:
        all_questions: All loaded questions
        pilot_mode: Whether to run in pilot mode (fewer seeds)
    """
    return asyncio.run(select_all_seeds(all_questions, pilot_mode))
