"""
Experiment 9 Step 1: Extract Capability Profiles

Reads Experiment 1 results and extracts per-model capability profiles:
- Strong domains (accuracy >= threshold)
- Weak domains (accuracy <= inverse threshold)
- Medium domains (everything else)

Also generates cross-model dissociation pairs where one model is strong
and another is weak in the same domain.

Usage:
  python scripts/extract_capability_profiles.py
  python scripts/extract_capability_profiles.py --exp1-run-id 20260220T090109
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DOMAINS = [
    "arithmetic",
    "spatial",
    "temporal",
    "linguistic",
    "logical",
    "social",
    "factual",
    "procedural",
]

# Default thresholds
STRONG_THRESHOLD = 0.60
WEAK_THRESHOLD = 0.40
MIN_DOMAINS_REQUIRED = 2
THRESHOLD_RELAXATION_STEP = 0.05


def load_exp1_results(run_id: str) -> list[dict]:
    """Load Experiment 1 results from JSONL."""
    results_path = Path(f"data/results/exp1_{run_id}_results.jsonl")

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print(f"   Run Experiment 1 first or specify correct --exp1-run-id")
        sys.exit(1)

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    return results


def compute_domain_accuracy(results: list[dict]) -> dict:
    """
    Compute per-model, per-domain accuracy from Experiment 1 results.

    Returns:
        Dict[model][domain] -> accuracy (0.0-1.0)
    """
    # Group by model and domain
    model_domain_results = defaultdict(lambda: defaultdict(list))

    for record in results:
        model = record.get("model")
        domain = record.get("domain")
        answer_correct = record.get("answer_correct")

        if model and domain and answer_correct is not None:
            model_domain_results[model][domain].append(answer_correct)

    # Compute accuracy for each model-domain pair
    accuracy = {}
    for model, domains in model_domain_results.items():
        accuracy[model] = {}
        for domain, corrects in domains.items():
            if corrects:
                accuracy[model][domain] = sum(corrects) / len(corrects)

    return accuracy


def find_adaptive_thresholds(
    domain_accuracies: dict,
    initial_strong: float,
    initial_weak: float,
    min_required: int,
) -> tuple[float, float]:
    """
    Find thresholds that ensure at least min_required strong and weak domains.

    Relaxes thresholds by 5% increments until requirements are met.
    """
    strong_threshold = initial_strong
    weak_threshold = initial_weak

    while True:
        strong_count = sum(1 for acc in domain_accuracies.values() if acc >= strong_threshold)
        weak_count = sum(1 for acc in domain_accuracies.values() if acc <= weak_threshold)

        if strong_count >= min_required and weak_count >= min_required:
            return strong_threshold, weak_threshold

        # Relax thresholds
        if strong_count < min_required:
            strong_threshold -= THRESHOLD_RELAXATION_STEP
        if weak_count < min_required:
            weak_threshold += THRESHOLD_RELAXATION_STEP

        # Safety: don't let thresholds cross
        if strong_threshold <= weak_threshold:
            # Return best effort
            return strong_threshold + THRESHOLD_RELAXATION_STEP, weak_threshold - THRESHOLD_RELAXATION_STEP


def extract_capability_profiles(
    domain_accuracy: dict,
    strong_threshold: float,
    weak_threshold: float,
) -> dict:
    """
    Extract capability profiles for each model.

    Returns:
        Dict[model] -> {
            "strong": [domains],
            "weak": [domains],
            "medium": [domains],
            "thresholds": {"strong": float, "weak": float},
            "accuracy": Dict[domain -> float]
        }
    """
    profiles = {}

    for model, accuracies in domain_accuracy.items():
        # Adapt thresholds per model if needed
        model_strong, model_weak = find_adaptive_thresholds(
            accuracies,
            strong_threshold,
            weak_threshold,
            MIN_DOMAINS_REQUIRED,
        )

        strong = []
        weak = []
        medium = []

        for domain in DOMAINS:
            acc = accuracies.get(domain)
            if acc is None:
                continue

            if acc >= model_strong:
                strong.append(domain)
            elif acc <= model_weak:
                weak.append(domain)
            else:
                medium.append(domain)

        # Calculate overall accuracy
        overall_acc = sum(accuracies.get(d, 0.0) for d in DOMAINS) / len(DOMAINS)

        profiles[model] = {
            "strong_domains": sorted(strong),
            "weak_domains": sorted(weak),
            "medium_domains": sorted(medium),
            "domain_accuracy": {domain: round(accuracies.get(domain, 0.0), 3) for domain in DOMAINS},
            "overall_accuracy": round(overall_acc, 3),
            "thresholds_used": {
                "strong": model_strong,
                "weak": model_weak,
            },
        }

    return profiles


def generate_dissociation_pairs(profiles: dict) -> list[dict]:
    """
    Generate cross-model dissociation pairs.

    A dissociation pair consists of:
    - model_strong: model that is strong in domain D
    - model_weak: model that is weak in domain D
    - domain: the domain D
    - accuracy_gap: difference in accuracy

    Returns:
        List of dissociation pair dicts
    """
    pairs = []
    models = sorted(profiles.keys())

    for domain in DOMAINS:
        # Find models strong and weak in this domain
        strong_models = [
            (m, profiles[m]["domain_accuracy"][domain])
            for m in models
            if domain in profiles[m]["strong_domains"]
        ]
        weak_models = [
            (m, profiles[m]["domain_accuracy"][domain])
            for m in models
            if domain in profiles[m]["weak_domains"]
        ]

        # Generate all cross-model pairs
        for strong_model, strong_acc in strong_models:
            for weak_model, weak_acc in weak_models:
                if strong_model != weak_model:
                    pairs.append({
                        "domain": domain,
                        "model_strong": strong_model,
                        "model_weak": weak_model,
                        "accuracy_strong": round(strong_acc, 3),
                        "accuracy_weak": round(weak_acc, 3),
                        "gap": round(strong_acc - weak_acc, 3),
                    })

    # Sort by gap descending (largest dissociations first)
    pairs.sort(key=lambda p: p["gap"], reverse=True)

    return pairs


def print_summary_table(profiles: dict):
    """Print a summary table of capability profiles."""
    print("\n" + "="*90)
    print("CAPABILITY PROFILES SUMMARY")
    print("="*90)
    print()

    # Header
    header = (
        f"{'Model':<20} | {'Thresholds':<15} | {'Strong':<8} | {'Weak':<8} | "
        f"{'Strong Domains':<30}"
    )
    print(header)
    print("-"*90)

    for model in sorted(profiles.keys()):
        profile = profiles[model]
        strong_thresh = profile["thresholds_used"]["strong"]
        weak_thresh = profile["thresholds_used"]["weak"]
        thresh_str = f"{strong_thresh:.2f}/{weak_thresh:.2f}"

        strong_count = len(profile["strong_domains"])
        weak_count = len(profile["weak_domains"])
        strong_domains = ", ".join(profile["strong_domains"][:3])
        if len(profile["strong_domains"]) > 3:
            strong_domains += "..."

        row = (
            f"{model:<20} | {thresh_str:<15} | {strong_count:<8} | {weak_count:<8} | "
            f"{strong_domains:<30}"
        )
        print(row)

    print("="*90)
    print()


def print_dissociation_summary(pairs: list[dict]):
    """Print summary of dissociation pairs."""
    print("="*90)
    print("DISSOCIATION PAIRS SUMMARY")
    print("="*90)
    print()
    print(f"Total pairs: {len(pairs)}")
    print(f"Domains covered: {len(set(p['domain'] for p in pairs))}")
    print()

    if pairs:
        print("Top 5 dissociations by accuracy gap:")
        print(f"{'Domain':<12} | {'Strong Model':<20} | {'Weak Model':<20} | {'Gap':<6}")
        print("-"*75)
        for pair in pairs[:5]:
            print(
                f"{pair['domain']:<12} | {pair['model_strong']:<20} | "
                f"{pair['model_weak']:<20} | {pair['gap']:<6.3f}"
            )

    print("="*90)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract capability profiles from Experiment 1 results"
    )
    parser.add_argument(
        "--exp1-run-id",
        default="20260220T090109",
        help="Experiment 1 run ID to use",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exp9",
        help="Output directory for profiles and pairs",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"EXTRACTING CAPABILITY PROFILES")
    print(f"{'='*60}")
    print(f"Source: exp1_{args.exp1_run_id}_results.jsonl")
    print(f"Output: {args.output_dir}/")
    print(f"{'='*60}\n")

    # Load results
    print("Loading Experiment 1 results...")
    results = load_exp1_results(args.exp1_run_id)
    print(f"  Loaded {len(results)} records\n")

    # Compute domain accuracy
    print("Computing per-model domain accuracy...")
    domain_accuracy = compute_domain_accuracy(results)
    models = sorted(domain_accuracy.keys())
    print(f"  Found {len(models)} models: {', '.join(models)}\n")

    # Extract capability profiles
    print("Extracting capability profiles...")
    profiles = extract_capability_profiles(
        domain_accuracy,
        STRONG_THRESHOLD,
        WEAK_THRESHOLD,
    )
    print(f"  Extracted profiles for {len(profiles)} models\n")

    # Generate dissociation pairs
    print("Generating dissociation pairs...")
    pairs = generate_dissociation_pairs(profiles)
    print(f"  Generated {len(pairs)} dissociation pairs\n")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = output_dir / "capability_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"✅ Saved capability profiles: {profiles_path}")

    pairs_path = output_dir / "dissociation_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"✅ Saved dissociation pairs: {pairs_path}\n")

    # Print summaries
    print_summary_table(profiles)
    print_dissociation_summary(pairs)

    print("Extraction complete.\n")


if __name__ == "__main__":
    main()
