"""
LLM-based seed generator for domains with no or poor HuggingFace source coverage.

Used for:
- linguistic: replaces GLUE CoLA (too easy); covers all 5 subcategories
- procedural: no HuggingFace source exists; generates all 5 subcategories from scratch
"""

import asyncio
import json
import re
import time
from pathlib import Path

from ...api import UnifiedClient


# Per-domain generation configs: subcategory -> (count_per_subcat_pilot, examples, answer_type)
LINGUISTIC_SUBCATS = {
    "syntax_parsing": {
        "answer_type": "short_text",
        "guidance": (
            "Questions ask which word a modifier or clause attaches to, or require identifying "
            "the grammatical role of a word in a sentence. "
            "Example: 'In the sentence \"The chef quickly prepared the meal\", what word does "
            "\"quickly\" modify?' → 'prepared'"
        ),
    },
    "morphology": {
        "answer_type": "short_text",
        "guidance": (
            "Questions ask for root morphemes, affixes, derivational history, or morphological "
            "decomposition of complex words. Avoid trivially obvious cases. "
            "Example: 'What is the root morpheme of \"unbelievable\"?' → 'believe' | "
            "'What suffix in \"happiness\" turns the adjective into a noun?' → '-ness'"
        ),
    },
    "ambiguity_resolution": {
        "answer_type": "short_text",
        "guidance": (
            "Questions present an ambiguous sentence and ask for two distinct interpretations, "
            "or ask which reading is more natural given context. "
            "Example: 'Give two interpretations of: \"I saw the man with the telescope\"' → "
            "'(1) I used a telescope to see the man; (2) I saw a man who had a telescope'"
        ),
    },
    "register_formality": {
        "answer_type": "short_text",
        "guidance": (
            "Questions ask to identify the register of a phrase, rewrite it in a different register, "
            "or judge appropriateness in a given context. "
            "Example: 'Rewrite in formal register: \"Can you shoot me that report ASAP?\"' → "
            "'Could you please send me that report at your earliest convenience?'"
        ),
    },
    "grammar_correction": {
        "answer_type": "short_text",
        "guidance": (
            "Questions identify specific grammatical errors (not just 'is this acceptable?') "
            "and explain the correction. Must require genuine analysis, not surface pattern matching. "
            "Example: 'Identify the grammatical error in: \"Each of the students have submitted their work\"' → "
            "'\"have\" should be \"has\" — subject \"each\" is singular'"
        ),
    },
}

PROCEDURAL_SUBCATS = {
    "algorithm_execution": {
        "answer_type": "short_text",
        "guidance": (
            "Questions ask to trace through a well-known algorithm on a small input and state "
            "the result at a specific step or the final output. "
            "Example: 'Apply one pass of bubble sort to [4, 2, 1, 3]. What is the array after the pass?' "
            "→ '[2, 1, 3, 4]'"
        ),
    },
    "recipe_following": {
        "answer_type": "short_text",
        "guidance": (
            "Questions present a cooking/recipe scenario and ask what happens if a step is "
            "modified, doubled, omitted, or substituted. Answer must be factually grounded. "
            "Example: 'A recipe calls for 2 cups of flour and bakes at 350°F for 30 minutes. "
            "If you double the recipe, what baking time and temperature adjustment is needed?' "
            "→ 'Same temperature (350°F); increase time slightly to ~35-40 minutes'"
        ),
    },
    "protocol_compliance": {
        "answer_type": "short_text",
        "guidance": (
            "Questions describe a standard process or protocol (web form validation, file locking, "
            "medical triage, safety checklist) and ask what the correct next action is. "
            "Example: 'A form submission arrives with a missing required email field. "
            "Per standard web form protocol, what should happen?' "
            "→ 'Reject submission, display inline validation error on the email field'"
        ),
    },
    "multistep_planning": {
        "answer_type": "short_text",
        "guidance": (
            "Questions present a goal with preconditions and ask for the correct ordered sequence "
            "of steps to achieve it. "
            "Example: 'You need to install a Python package that requires Python 3.11+, but your "
            "system has Python 3.9. List the correct ordered steps.' "
            "→ '1. Install Python 3.11+, 2. Create a new virtual environment with 3.11, "
            "3. Activate venv, 4. pip install the package'"
        ),
    },
    "error_recovery": {
        "answer_type": "short_text",
        "guidance": (
            "Questions describe a system or process mid-failure and ask for the correct recovery "
            "action according to standard practice. "
            "Example: 'A database transaction has executed 3 of 5 steps when step 4 fails. "
            "What is the correct recovery action?' → 'Roll back the entire transaction to its start state'"
        ),
    },
}


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences around JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def generate_seeds_for_subcategory(
    client: UnifiedClient,
    domain_name: str,
    subcategory: str,
    subcat_config: dict,
    count: int,
) -> list[dict]:
    """
    Generate `count` seed questions for a single subcategory via LLM.

    Returns:
        List of normalized question dicts
    """
    answer_type = subcat_config["answer_type"]
    guidance = subcat_config["guidance"]

    prompt = f"""You are creating high-quality benchmark questions for an AI evaluation dataset.

Domain: {domain_name}
Subcategory: {subcategory}
Task guidance: {guidance}

Generate exactly {count} benchmark questions for this subcategory.

Each question must:
1. Have a single, definitively correct answer (verifiable by an expert)
2. Require genuine knowledge/reasoning — not trivially guessable
3. Be self-contained (no external context required)
4. Have a concise correct answer (1-3 sentences max)
5. Vary in surface content (different words, scenarios, contexts)

Respond with ONLY a JSON array, no other text:
[
  {{
    "question_text": "...",
    "correct_answer": "...",
    "subcategory": "{subcategory}"
  }}
]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=3000,
        metadata={"task": "llm_seed_generation", "domain": domain_name, "subcategory": subcategory},
    )

    if "error" in response:
        print(f"  ⚠️  LLM error for {domain_name}/{subcategory}: {response['error']}")
        return []

    content = response.get("content")
    if not content:
        print(f"  ⚠️  Empty response for {domain_name}/{subcategory}")
        return []

    # Parse JSON
    try:
        raw = json.loads(_strip_json_fences(content))
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            print(f"  ⚠️  Could not parse JSON for {domain_name}/{subcategory}")
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            print(f"  ⚠️  JSON parse failed for {domain_name}/{subcategory}")
            return []

    if not isinstance(raw, list):
        print(f"  ⚠️  Expected list for {domain_name}/{subcategory}, got {type(raw)}")
        return []

    questions = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        qt = item.get("question_text", "").strip()
        ca = item.get("correct_answer", "").strip()
        if not qt or not ca:
            continue

        questions.append({
            "question_text": qt,
            "correct_answer": ca,
            "answer_type": answer_type,
            "source": "llm_generated",
            "source_id": f"llm_{domain_name}_{subcategory}_{i:03d}_{int(time.time())}",
            "domain": domain_name,
            "subcategory": subcategory,
            "difficulty": None,
            "metadata": {"generation_model": "llama-3.1-70b"},
        })

    print(f"  ✅ Generated {len(questions)} seeds for {domain_name}/{subcategory}")
    return questions


async def generate_all_llm_seeds(pilot_mode: bool = False) -> dict[str, list[dict]]:
    """
    Generate LLM seeds for linguistic and procedural domains.

    In pilot mode: 8 seeds per subcategory (40 per domain).
    In full mode: 26 seeds per subcategory (130 per domain).

    Returns:
        Dict mapping domain name to list of seed questions
    """
    count_per_subcat = 8 if pilot_mode else 26

    client = UnifiedClient(experiment="llm_seed_generation")

    domain_configs = {
        "linguistic": LINGUISTIC_SUBCATS,
        "procedural": PROCEDURAL_SUBCATS,
    }

    all_seeds: dict[str, list[dict]] = {}

    for domain_name, subcats in domain_configs.items():
        print(f"\n{'='*60}")
        print(f"Generating LLM seeds for: {domain_name}")
        print(f"{'='*60}")

        domain_seeds = []

        for subcategory, subcat_config in subcats.items():
            seeds = await generate_seeds_for_subcategory(
                client,
                domain_name,
                subcategory,
                subcat_config,
                count=count_per_subcat,
            )
            domain_seeds.extend(seeds)
            await asyncio.sleep(0.5)

        # Assign question IDs and mark as original seeds
        for i, seed in enumerate(domain_seeds):
            seed["transformation"] = "original"
            seed["parent_id"] = None
            if "question_id" not in seed:
                seed["question_id"] = f"{domain_name}_{seed['subcategory']}_{i:04d}"

        # Write to data/seeds/{domain}.jsonl (overwrites any prior content)
        output_file = Path(f"data/seeds/{domain_name}.jsonl")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for seed in domain_seeds:
                f.write(json.dumps(seed) + "\n")

        print(f"✅ Saved {len(domain_seeds)} LLM seeds to {output_file}")
        all_seeds[domain_name] = domain_seeds

    return all_seeds


async def filter_trivial_social_seeds(
    client: UnifiedClient,
    questions: list[dict],
) -> list[dict]:
    """
    Pre-filter social seeds: keep only questions where Llama 8B answers incorrectly.

    If >80% of seeds would be filtered out (8B gets them all right), return all
    seeds to avoid over-filtering.

    Args:
        client: UnifiedClient instance
        questions: Social domain seed questions (multiple_choice)

    Returns:
        Filtered list
    """
    if not questions:
        return questions

    print(f"  Pre-filtering {len(questions)} social seeds (removing 8B-trivial questions)...")

    results = []
    for q in questions:
        prompt = (
            f"Answer this multiple choice question. Respond with ONLY the letter (A, B, or C), "
            f"nothing else.\n\n{q['question_text']}"
        )
        response = await client.complete(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            metadata={"task": "social_trivial_filter"},
        )

        if "error" in response or not response.get("content"):
            # Can't score, keep the question
            results.append((q, False))  # treat as 8B got it wrong
            continue

        predicted = response["content"].strip().upper()
        # Extract letter
        match = re.search(r"\b([A-Z])\b", predicted)
        letter = match.group(1) if match else predicted[:1]

        correct = q.get("correct_answer", "").strip().upper()
        is_correct = letter == correct
        results.append((q, is_correct))

        await asyncio.sleep(0.1)

    # Count how many 8B gets right (trivial)
    trivial_count = sum(1 for _, correct in results if correct)
    trivial_rate = trivial_count / len(results) if results else 0

    print(f"  8B trivial rate: {trivial_rate:.1%} ({trivial_count}/{len(results)})")

    if trivial_rate > 0.8:
        # Over-filtering protection: if 8B gets >80% right, the filter is too aggressive
        print("  ⚠️  >80% trivially correct — returning all seeds without filtering")
        return questions

    # Keep only non-trivial questions
    non_trivial = [q for q, correct in results if not correct]
    print(f"  Kept {len(non_trivial)} non-trivial seeds (removed {trivial_count} trivial)")
    return non_trivial


def run_llm_seed_generation(pilot_mode: bool = False) -> dict:
    """Run LLM seed generation for linguistic and procedural domains (sync wrapper)."""
    return asyncio.run(generate_all_llm_seeds(pilot_mode))


# ---------------------------------------------------------------------------
# Easy/medium subcategory configs for all 8 domains (used by expand pipeline)
# Designed so Llama-8B struggles but Llama-70B succeeds.
# ---------------------------------------------------------------------------

ARITHMETIC_EASY_SUBCATS = {
    "mental_calculation": {
        "answer_type": "exact_numeric",
        "guidance": (
            "2-3 digit numbers only. Single operation: add, subtract, multiply, or divide. "
            "Example: 'A store has 248 apples. 63 are sold. How many remain?' → '185'. "
            "AVOID: multi-step problems, unit conversions, algebra."
        ),
    },
    "word_problems": {
        "answer_type": "exact_numeric",
        "guidance": (
            "Simple real-world scenarios, 2 steps max, integers only. "
            "Example: 'Sam earns $12/hour and works 8 hours. He spends $40. How much remains?' → '$56'. "
            "AVOID: fractions, percentages, geometry, complex rates."
        ),
    },
    "estimation": {
        "answer_type": "short_text",
        "guidance": (
            "Round-number estimation with a single operation. "
            "Example: 'About how many minutes are in 3 hours?' → 'About 180 minutes'. "
            "AVOID: multi-step chains, unit conversions, square roots."
        ),
    },
    "unit_conversion": {
        "answer_type": "exact_numeric",
        "guidance": (
            "Single common unit conversion using whole numbers (cm↔m, minutes↔hours, kg↔g). "
            "Example: 'Convert 3.5 kg to grams.' → '3500'. "
            "AVOID: compound conversions, temperature, currency."
        ),
    },
    "numerical_reasoning": {
        "answer_type": "short_text",
        "guidance": (
            "Simple ordering or comparison of 3-5 numbers. "
            "Example: 'Which is largest: 1/2, 0.4, or 45%?' → '1/2 (= 0.5)'. "
            "AVOID: multi-step derivations, statistics, algebra."
        ),
    },
}

LOGICAL_EASY_SUBCATS = {
    "syllogisms": {
        "answer_type": "short_text",
        "guidance": (
            "Classic 2-premise syllogisms with concrete nouns (animals, objects). "
            "Example: 'All dogs are mammals. Rex is a dog. What is Rex?' → 'a mammal'. "
            "AVOID: negation, double negation, more than 2 premises."
        ),
    },
    "propositional_logic": {
        "answer_type": "short_text",
        "guidance": (
            "Simple if-then rules with 2 variables. "
            "Example: 'If it rains, the ground gets wet. It is raining. Is the ground wet?' → 'Yes'. "
            "AVOID: nested conditionals, biconditionals, more than 2 rules."
        ),
    },
    "set_relations": {
        "answer_type": "short_text",
        "guidance": (
            "Set membership or subset questions with 2-3 concrete sets. "
            "Example: 'All cats are animals. Whiskers is a cat. Is Whiskers an animal?' → 'Yes'. "
            "AVOID: set complements, Venn diagrams with 3+ sets."
        ),
    },
    "conditional_reasoning": {
        "answer_type": "short_text",
        "guidance": (
            "Modus ponens or modus tollens with everyday scenarios. "
            "Example: 'If you study hard, you pass. You did not pass. Did you study hard?' → 'No'. "
            "AVOID: disjunctions, more than 2 conditions."
        ),
    },
    "proof_evaluation": {
        "answer_type": "short_text",
        "guidance": (
            "Identify a single obvious logical error in a 2-step argument. "
            "Example: 'Premise: All birds can fly. Penguin is a bird. Conclusion: Penguin can fly. "
            "What is wrong?' → 'Not all birds can fly — penguins are a counterexample'. "
            "AVOID: subtle fallacies, arguments with 4+ steps."
        ),
    },
}

SPATIAL_EASY_SUBCATS = {
    "2d_geometry": {
        "answer_type": "exact_numeric",
        "guidance": (
            "2D shapes: area or perimeter of rectangle/square/triangle using whole numbers ≤ 20. "
            "Example: 'What is the area of a rectangle 8 m wide and 5 m tall?' → '40'. "
            "AVOID: 3D shapes, circles (π), composite figures."
        ),
    },
    "distance_direction": {
        "answer_type": "short_text",
        "guidance": (
            "Cardinal direction questions with 3 objects and simple left/right/north/south. "
            "Example: 'Ann is north of Bob. Bob is north of Carl. Who is furthest south?' → 'Carl'. "
            "AVOID: diagonals, more than 3 objects, ambiguous angles."
        ),
    },
    "relative_positioning": {
        "answer_type": "short_text",
        "guidance": (
            "Object position using 'above', 'below', 'left of', 'right of' with 3 objects. "
            "Example: 'The cup is left of the plate. The fork is right of the plate. "
            "What is between the cup and fork?' → 'the plate'. "
            "AVOID: 3D, rotation, perspective changes."
        ),
    },
    "spatial_transformation": {
        "answer_type": "short_text",
        "guidance": (
            "Simple 90-degree rotation or mirror reflection of a basic shape (arrow, L-shape). "
            "Example: 'An arrow pointing right is rotated 90 degrees clockwise. "
            "Which direction does it now point?' → 'down'. "
            "AVOID: non-90-degree angles, 3D transforms, compound transformations."
        ),
    },
    "3d_visualization": {
        "answer_type": "exact_numeric",
        "guidance": (
            "Count visible faces or blocks in a simple 3D arrangement (max 5 blocks). "
            "Example: 'A 2×2×1 box of unit cubes has how many unit cubes?' → '4'. "
            "AVOID: complex structures, hidden blocks, fractional dimensions."
        ),
    },
}

FACTUAL_EASY_SUBCATS = {
    "geography": {
        "answer_type": "short_text",
        "guidance": (
            "Major world capitals, well-known countries, largest continents/oceans. "
            "Example: 'What is the capital of France?' → 'Paris'. "
            "AVOID: obscure cities, disputed territories, historical capitals."
        ),
    },
    "science": {
        "answer_type": "short_text",
        "guidance": (
            "Fundamental science facts: basic chemistry, physics constants, biology. "
            "Example: 'What is the chemical formula for water?' → 'H2O'. "
            "AVOID: advanced derivations, obscure elements, cutting-edge research."
        ),
    },
    "history": {
        "answer_type": "short_text",
        "guidance": (
            "Major historical turning points and widely known figures. "
            "Example: 'In which year did World War II end?' → '1945'. "
            "AVOID: obscure events, disputed dates, local history."
        ),
    },
    "culture": {
        "answer_type": "short_text",
        "guidance": (
            "Widely recognized cultural facts: famous works, authors, inventors. "
            "Example: 'Who wrote Romeo and Juliet?' → 'William Shakespeare'. "
            "AVOID: niche subcultures, disputed attributions, recent events."
        ),
    },
    "cross_domain_trivia": {
        "answer_type": "short_text",
        "guidance": (
            "Common knowledge trivia combining two well-known domains. "
            "Example: 'What element has the atomic symbol Au and is used in jewelry?' → 'Gold'. "
            "AVOID: obscure connections, specialized jargon, trick questions."
        ),
    },
}

TEMPORAL_EASY_SUBCATS = {
    "ordering_events": {
        "answer_type": "short_text",
        "guidance": (
            "Order 3 events that are unambiguously before/after each other, same decade. "
            "Example: 'Which came first: moon landing (1969), Berlin Wall fall (1989), "
            "or WWII end (1945)?' → 'WWII end'. "
            "AVOID: events within same year, disputed timelines."
        ),
    },
    "duration_estimation": {
        "answer_type": "short_text",
        "guidance": (
            "Simple duration arithmetic using hours or days, single time zone. "
            "Example: 'A meeting starts at 2:00 PM and lasts 90 minutes. When does it end?' "
            "→ '3:30 PM'. "
            "AVOID: DST changes, multi-timezone, week-spanning calculations."
        ),
    },
    "timezone_arithmetic": {
        "answer_type": "short_text",
        "guidance": (
            "Single whole-hour timezone offset, no DST. "
            "Example: 'New York is UTC-5. It is 3 PM in New York. "
            "What time is it in UTC?' → '8 PM'. "
            "AVOID: fractional offsets, DST, multi-hop conversions."
        ),
    },
    "scheduling": {
        "answer_type": "short_text",
        "guidance": (
            "Schedule 2-3 events within a single day given simple constraints. "
            "Example: 'Lunch is at noon. A meeting is 2 hours after lunch. "
            "What time is the meeting?' → '2:00 PM'. "
            "AVOID: overnight spans, recurring events, conflict resolution with 4+ items."
        ),
    },
    "temporal_paradoxes": {
        "answer_type": "short_text",
        "guidance": (
            "Simple apparent paradox resolvable with one logical step. "
            "Example: 'A woman was born in 1985 and celebrated her 10th birthday in 1994. "
            "Is this possible?' → 'No, 1985 + 10 = 1995'. "
            "AVOID: leap-year edge cases, ambiguous birth dates, nested paradoxes."
        ),
    },
}

SOCIAL_EASY_SUBCATS = {
    "implicature": {
        "answer_type": "short_text",
        "guidance": (
            "Direct conversational implicature: what does a speaker clearly mean beyond literal words? "
            "Example: 'When asked if they liked the food, someone said \"It was interesting.\" "
            "What do they likely mean?' → 'They probably did not enjoy it'. "
            "AVOID: sarcasm-within-sarcasm, cultural idioms requiring specialist knowledge."
        ),
    },
    "theory_of_mind": {
        "answer_type": "short_text",
        "guidance": (
            "First-order theory of mind only: what does person A believe/want? "
            "Example: 'Sarah hid a ball in a box. Tom then moved it to a basket "
            "while Sarah was away. Where will Sarah look for the ball?' → 'In the box'. "
            "AVOID: second-order beliefs (what does A think B thinks), nested mental states."
        ),
    },
    "pragmatic_inference": {
        "answer_type": "short_text",
        "guidance": (
            "Simple pragmatic inference from a brief exchange. "
            "Example: 'A: Can you pass the salt? B: It's right next to you. "
            "What should A do?' → 'Reach for the salt themselves'. "
            "AVOID: indirect refusals, multi-turn inference chains."
        ),
    },
    "sarcasm_detection": {
        "answer_type": "short_text",
        "guidance": (
            "Unambiguous sarcasm: the tone contradicts the situation clearly. "
            "Example: 'After spilling coffee all over herself, Jane said \"Great start to the day!\" "
            "Was she being sincere or sarcastic?' → 'Sarcastic'. "
            "AVOID: dry humor, culturally dependent sarcasm, ambiguous cases."
        ),
    },
    "social_norm_violation": {
        "answer_type": "short_text",
        "guidance": (
            "Clear, widely agreed-upon social norm violation in a familiar context. "
            "Example: 'At a formal dinner, someone reaches across the table and takes food "
            "from another person's plate. Is this a social norm violation?' → 'Yes'. "
            "AVOID: culturally specific norms, contested etiquette, minor faux pas."
        ),
    },
}

# Maps domain name → easy/medium subcategory config dict
ALL_DOMAIN_EASY_SUBCATS: dict[str, dict] = {
    "arithmetic": ARITHMETIC_EASY_SUBCATS,
    "logical": LOGICAL_EASY_SUBCATS,
    "spatial": SPATIAL_EASY_SUBCATS,
    "factual": FACTUAL_EASY_SUBCATS,
    "temporal": TEMPORAL_EASY_SUBCATS,
    "social": SOCIAL_EASY_SUBCATS,
    "linguistic": LINGUISTIC_SUBCATS,     # reuse existing (already easy-medium calibrated)
    "procedural": PROCEDURAL_SUBCATS,     # reuse existing
}


async def generate_easy_medium_seeds_for_domain(
    domain_name: str,
    subcats: dict,
    n_per_subcat: int,
    pilot_mode: bool = False,
) -> list[dict]:
    """
    Generate easy/medium seeds for a single domain using the given subcategory configs.

    Saves to data/seeds_v2/{domain}.jsonl.

    Returns:
        List of seed dicts tagged with seed_tier="easy_medium"
    """
    client = UnifiedClient(experiment="easy_medium_seed_generation")

    domain_seeds = []

    for subcategory, subcat_config in subcats.items():
        seeds = await generate_seeds_for_subcategory(
            client,
            domain_name,
            subcategory,
            subcat_config,
            count=n_per_subcat,
        )
        domain_seeds.extend(seeds)
        await asyncio.sleep(0.5)

    # Assign question IDs, mark tier
    for i, seed in enumerate(domain_seeds):
        seed["transformation"] = "original"
        seed["parent_id"] = None
        seed["seed_tier"] = "easy_medium"
        if "question_id" not in seed:
            seed["question_id"] = f"{domain_name}_v2_{seed.get('subcategory', 'x')}_{i:04d}"

    output_file = Path(f"data/seeds_v2/{domain_name}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for seed in domain_seeds:
            f.write(json.dumps(seed) + "\n")

    print(f"✅ Saved {len(domain_seeds)} easy/medium seeds to {output_file}")
    return domain_seeds


async def _run_easy_medium_seed_generation_async(pilot_mode: bool = False) -> dict[str, int]:
    """Generate easy/medium seeds for all 8 domains."""
    n_per_subcat = 4 if pilot_mode else 10  # 10 × 5 subcats = 50 seeds/domain

    stats: dict[str, int] = {}

    for domain_name, subcats in ALL_DOMAIN_EASY_SUBCATS.items():
        print(f"\n{'='*60}")
        print(f"Generating easy/medium seeds for: {domain_name}")
        print(f"{'='*60}")

        seeds = await generate_easy_medium_seeds_for_domain(
            domain_name, subcats, n_per_subcat, pilot_mode
        )
        stats[domain_name] = len(seeds)

    return stats


def run_easy_medium_seed_generation(pilot_mode: bool = False) -> dict[str, int]:
    """
    Generate easy/medium seeds for all 8 domains and save to data/seeds_v2/.

    Args:
        pilot_mode: If True, generate 4 seeds per subcategory (20 per domain);
                    otherwise 10 per subcategory (50 per domain).

    Returns:
        Dict mapping domain name to seed count.
    """
    return asyncio.run(_run_easy_medium_seed_generation_async(pilot_mode))
