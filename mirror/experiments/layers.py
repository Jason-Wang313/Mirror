"""
Layer assembly and question pairing for MIRROR experiments.

Three measurement layers:
  Layer 1 — Multi-Paradigm Behavioral Choice (all 5 channels)
  Layer 2 — Structured Self-Report (confidence, approach, sub-skills)
  Layer 3 — Free-Form Ecological (reuses Channel 5 data)

Also provides Channel 3 pairing: matches easy questions with hard/adversarial
questions from the same domain.
"""

import random
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Channel 3 question pairing
# ---------------------------------------------------------------------------

def pair_questions_for_difficulty_selection(
    questions: list[dict],
    seed: int = 42,
) -> list[dict]:
    """
    Pre-compute easy/hard pairs for Channel 3 (Difficulty Selection).

    For each domain, pair easy questions with hard/adversarial questions.
    If a domain has insufficient easy questions, also pair with medium.

    Args:
        questions: Full question list from questions.jsonl
        seed: Random seed for reproducible pairing

    Returns:
        List of pair dicts:
        {
            "domain": str,
            "easy_question": dict,
            "hard_question": dict,
            "pair_id": str,        # "easy_source_id__hard_source_id"
        }
    """
    rng = random.Random(seed)

    # Bucket by domain × difficulty
    by_domain_diff: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for q in questions:
        domain = q.get("domain", "unknown")
        diff = q.get("difficulty", "unknown")
        by_domain_diff[domain][diff].append(q)

    pairs = []

    for domain, diff_map in by_domain_diff.items():
        easy_pool = list(diff_map.get("easy", []))
        hard_pool = list(diff_map.get("hard", []))
        adv_pool = list(diff_map.get("adversarial", []))
        medium_pool = list(diff_map.get("medium", []))

        # Combined hard side: hard + adversarial
        hard_combined = hard_pool + adv_pool

        # If easy pool is empty, skip pairing for this domain
        if not easy_pool:
            continue

        # If hard combined is empty, fall back to medium
        if not hard_combined:
            hard_combined = medium_pool

        if not hard_combined:
            continue

        # Shuffle both pools
        rng.shuffle(easy_pool)
        rng.shuffle(hard_combined)

        # Pair up: cycle through the smaller pool
        n_pairs = max(len(easy_pool), len(hard_combined))
        for i in range(n_pairs):
            easy_q = easy_pool[i % len(easy_pool)]
            hard_q = hard_combined[i % len(hard_combined)]

            # Avoid pairing a question with itself
            if easy_q.get("source_id") == hard_q.get("source_id"):
                # Try next hard question
                hard_q = hard_combined[(i + 1) % len(hard_combined)]
                if easy_q.get("source_id") == hard_q.get("source_id"):
                    continue  # Skip if still same

            easy_id = easy_q.get("source_id", f"easy_{i}")
            hard_id = hard_q.get("source_id", f"hard_{i}")

            pairs.append(
                {
                    "domain": domain,
                    "easy_question": easy_q,
                    "hard_question": hard_q,
                    "pair_id": f"{easy_id}__{hard_id}",
                }
            )

    return pairs


def get_channel3_pair_for_question(
    question: dict,
    all_pairs: list[dict],
) -> Optional[dict]:
    """
    Find a Channel 3 pair that involves the given question.

    Returns the pair if found, else None (question won't run in Channel 3).
    """
    qid = question.get("source_id")
    domain = question.get("domain")

    for pair in all_pairs:
        if pair["domain"] != domain:
            continue
        if (
            pair["easy_question"].get("source_id") == qid
            or pair["hard_question"].get("source_id") == qid
        ):
            return pair

    return None


def build_channel3_pairs_index(pairs: list[dict]) -> dict:
    """
    Build a lookup index: source_id → list of pairs containing that question.

    Args:
        pairs: Output of pair_questions_for_difficulty_selection()

    Returns:
        Dict mapping source_id → list[pair_dict]
    """
    index: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        easy_id = pair["easy_question"].get("source_id")
        hard_id = pair["hard_question"].get("source_id")
        if easy_id:
            index[easy_id].append(pair)
        if hard_id:
            index[hard_id].append(pair)
    return dict(index)


# ---------------------------------------------------------------------------
# Layer configuration
# ---------------------------------------------------------------------------

LAYER_CONFIGS = {
    1: {
        "name": "Multi-Paradigm Behavioral Choice",
        "channels": [1, 2, 3, 4, 5],
        "description": "All five behavioral channels — primary data",
    },
    2: {
        "name": "Structured Self-Report",
        "channels": ["layer2"],
        "description": "Explicit metacognitive self-assessment",
    },
    3: {
        "name": "Free-Form Ecological",
        "channels": [5],
        "description": "Natural response — reuses Channel 5 data, no extra API calls",
    },
}


def get_channels_for_layers(layers: list[int]) -> list:
    """
    Return the unique set of channels needed for the given layers.

    Layer 3 reuses Channel 5, so no additional calls.

    Args:
        layers: List of layer IDs (1, 2, 3)

    Returns:
        List of channel IDs to run (ints + "layer2" str)
    """
    channels = []
    for layer in layers:
        for ch in LAYER_CONFIGS.get(layer, {}).get("channels", []):
            if ch not in channels:
                channels.append(ch)
    return channels


def deduplicate_channels(layers: list[int]) -> list:
    """
    Get deduplicated channel list.
    If both Layer 1 (ch5) and Layer 3 (ch5) are requested, ch5 runs once.
    """
    return get_channels_for_layers(layers)


# ---------------------------------------------------------------------------
# Max tokens per model per channel
# ---------------------------------------------------------------------------

# Qwen 3 235B thinking model needs extra tokens for channels requiring
# structured output (thinking tokens consume budget before visible output).
QWEN_EXTENDED_CHANNELS = {1, 2, 3, 4}
QWEN_MAX_TOKENS = 4096
DEFAULT_MAX_TOKENS = 1024
LAYER2_MAX_TOKENS = 1536


def get_max_tokens(model: str, channel_id) -> int:
    """
    Return appropriate max_tokens for model × channel combination.

    Qwen 3 235B thinking model needs 4096 on structured channels.
    """
    if "qwen" in model.lower():
        if channel_id in QWEN_EXTENDED_CHANNELS:
            return QWEN_MAX_TOKENS
    if channel_id == "layer2":
        return LAYER2_MAX_TOKENS
    return DEFAULT_MAX_TOKENS
