"""
Deduplicator - removes near-duplicate questions using embeddings.

Uses sentence-transformers to detect semantic similarity.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_questions_for_domain(domain_name: str) -> list[dict]:
    """Load all questions for a domain (seeds + variations)."""
    questions = []

    # Load from generated file
    generated_file = Path(f"data/generated/{domain_name}.jsonl")
    if generated_file.exists():
        with open(generated_file, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line))

    return questions


def deduplicate_domain(
    domain_name: str,
    similarity_threshold: float = 0.85,
) -> tuple[list[dict], dict]:
    """
    Deduplicate questions within a domain.

    Args:
        domain_name: Domain name
        similarity_threshold: Cosine similarity threshold for duplicates

    Returns:
        Tuple of (deduplicated questions, stats dict)
    """
    print(f"\n{'='*60}")
    print(f"Deduplicating: {domain_name}")
    print(f"{'='*60}")

    questions = load_questions_for_domain(domain_name)

    if len(questions) == 0:
        print("No questions to deduplicate")
        return [], {}

    print(f"Loaded {len(questions)} questions")

    # Compute embeddings
    print("Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [q["question_text"] for q in questions]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Compute pairwise similarities
    print("Computing pairwise similarities...")
    similarities = cosine_similarity(embeddings)

    # Find duplicate clusters
    clusters = []
    visited = set()

    for i in range(len(questions)):
        if i in visited:
            continue

        cluster = [i]
        for j in range(i+1, len(questions)):
            if j in visited:
                continue

            if similarities[i, j] >= similarity_threshold:
                cluster.append(j)
                visited.add(j)

        if len(cluster) > 1:
            clusters.append(cluster)

        visited.add(i)

    print(f"Found {len(clusters)} duplicate clusters")

    # From each cluster, keep the highest quality question
    to_remove = set()

    for cluster in clusters:
        # Quality heuristic: prefer seeds > variations
        cluster_questions = [questions[i] for i in cluster]

        # Rank by: 1) transformation (original > surface_variation), 2) source quality
        ranked = sorted(cluster_questions, key=lambda q: (
            0 if q.get("transformation") == "original" else 1,
            q.get("source_id", "")
        ))

        # Keep first (highest quality), remove others
        keep_idx = cluster[cluster_questions.index(ranked[0])]
        remove_indices = [idx for idx in cluster if idx != keep_idx]

        to_remove.update(remove_indices)

        print(f"  Cluster of {len(cluster)}: keeping {keep_idx}, removing {remove_indices}")

    # Filter questions
    deduplicated = [q for i, q in enumerate(questions) if i not in to_remove]

    print(f"✅ Removed {len(to_remove)} duplicates, kept {len(deduplicated)} questions")

    # Stats
    stats = {
        "original_count": len(questions),
        "removed_count": len(to_remove),
        "final_count": len(deduplicated),
        "clusters_found": len(clusters),
        "similarity_threshold": similarity_threshold,
    }

    # Save deduplicated
    output_file = Path(f"data/verified/{domain_name}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for q in deduplicated:
            # Add dedup cluster ID if in a cluster
            for cluster_id, cluster in enumerate(clusters):
                if questions.index(q) in cluster:
                    if "verification" not in q:
                        q["verification"] = {}
                    q["verification"]["dedup_cluster"] = f"cluster_{cluster_id}"
                    break

            f.write(json.dumps(q) + "\n")

    return deduplicated, stats


def run_deduplication() -> dict:
    """
    Run deduplication for all domains.

    Returns:
        Dict of stats per domain
    """
    import yaml

    with open("configs/domains.yaml", "r") as f:
        config = yaml.safe_load(f)

    all_stats = {}

    for domain_name in config["domains"].keys():
        _, stats = deduplicate_domain(domain_name)
        all_stats[domain_name] = stats

    return all_stats
