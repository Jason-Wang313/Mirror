"""
Meta-accuracy test: Ask models to self-rank their domain strengths and compare to actual performance.
"""

import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api import UnifiedClient

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

DOMAIN_DISPLAY_NAMES = {
    "arithmetic": "arithmetic",
    "spatial": "spatial reasoning",
    "temporal": "temporal reasoning",
    "linguistic": "linguistic reasoning",
    "logical": "logical deduction",
    "social": "social/pragmatic reasoning",
    "factual": "factual recall",
    "procedural": "procedural reasoning",
}


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation."""
    import math

    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def load_actual_domain_accuracy(run_id: str) -> dict[str, dict[str, float]]:
    """
    Load per-domain accuracy for each model from accuracy.json.

    Returns:
        Dict mapping model -> domain -> accuracy (from natural channel)
    """
    results_dir = Path("data/results")
    accuracy_file = results_dir / f"exp1_{run_id}_accuracy.json"

    with open(accuracy_file) as f:
        accuracy_data = json.load(f)

    # Extract natural channel accuracy per domain
    model_domain_acc = {}
    for model, domains in accuracy_data.items():
        model_domain_acc[model] = {}
        for domain, channels in domains.items():
            # Use natural channel (Ch5) accuracy
            acc = channels.get("natural_acc")
            if acc is not None:
                model_domain_acc[model][domain] = acc

    return model_domain_acc


def parse_domain_ranking(response_text: str) -> list[str] | None:
    """
    Parse model's self-ranking response.

    Expected format:
    1. domain_name
    2. domain_name
    ...
    8. domain_name

    Returns list of domain names in order (strongest to weakest), or None if parse fails.
    """
    lines = response_text.strip().split("\n")
    ranking = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to match numbered list: "1. domain_name" or "1) domain_name"
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            domain_text = match.group(1).strip().lower()

            # Match to canonical domain names
            matched = None
            for domain_key, display_name in DOMAIN_DISPLAY_NAMES.items():
                if domain_key in domain_text or display_name in domain_text:
                    matched = domain_key
                    break

            if matched and matched not in ranking:
                ranking.append(matched)

    # Valid ranking must have all 8 domains
    if len(ranking) == 8 and set(ranking) == set(DOMAINS):
        return ranking

    return None


async def query_model_self_ranking(client: UnifiedClient, model: str) -> dict:
    """
    Query a single model to self-rank its domain strengths.

    Returns:
        Dict with raw_response, parsed_ranking, and parse_success.
    """
    prompt = """You will be tested on 8 domains: arithmetic, spatial reasoning, temporal reasoning, linguistic reasoning, logical deduction, social/pragmatic reasoning, factual recall, and procedural reasoning.

Rank these 8 domains from your STRONGEST to your WEAKEST.

Return ONLY a numbered list 1-8, strongest first. Example format:
1. arithmetic
2. logical deduction
3. ...
8. social/pragmatic reasoning

Do not include any explanation or commentary."""

    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        metadata={"task": "meta_accuracy_test"},
    )

    if "error" in response:
        return {
            "raw_response": None,
            "parsed_ranking": None,
            "parse_success": False,
            "error": response["error"],
        }

    raw_response = response.get("content") or ""
    parsed_ranking = parse_domain_ranking(raw_response) if raw_response else None

    return {
        "raw_response": raw_response,
        "parsed_ranking": parsed_ranking,
        "parse_success": parsed_ranking is not None,
    }


async def run_meta_accuracy_test(run_id: str) -> dict:
    """
    Run meta-accuracy test for all models.

    For each model:
    1. Query model to self-rank domains
    2. Compare to actual domain accuracy
    3. Compute Spearman correlation
    """
    print("Loading actual domain accuracies...")
    actual_acc = load_actual_domain_accuracy(run_id)

    models = sorted(actual_acc.keys())
    print(f"Testing {len(models)} models...\n")

    client = UnifiedClient(experiment="meta_accuracy_test")
    results = {}

    for model in models:
        print(f"Querying {model}...")

        # Get model's self-ranking
        self_rank_data = await query_model_self_ranking(client, model)

        if not self_rank_data["parse_success"]:
            print(f"  ⚠️  Failed to parse response")
            results[model] = {
                **self_rank_data,
                "spearman_correlation": None,
                "actual_ranking": None,
            }
            continue

        self_ranking = self_rank_data["parsed_ranking"]
        print(f"  Self-ranking: {' > '.join(self_ranking[:3])} ... {' > '.join(self_ranking[-2:])}")

        # Get actual accuracy ranking (strongest to weakest)
        domain_accs = actual_acc[model]
        actual_ranking = sorted(
            domain_accs.keys(),
            key=lambda d: domain_accs[d],
            reverse=True  # Highest accuracy first
        )
        print(f"  Actual ranking: {' > '.join(actual_ranking[:3])} ... {' > '.join(actual_ranking[-2:])}")

        # Compute Spearman correlation
        # Convert rankings to scores (rank 1 = score 8, rank 8 = score 1)
        self_scores = {domain: 8 - i for i, domain in enumerate(self_ranking)}
        actual_scores = {domain: 8 - i for i, domain in enumerate(actual_ranking)}

        # Align by domain order
        domains_ordered = DOMAINS
        self_values = [self_scores[d] for d in domains_ordered]
        actual_values = [actual_scores[d] for d in domains_ordered]

        rho = _spearman(self_values, actual_values)
        print(f"  Spearman ρ = {rho:.3f}")

        results[model] = {
            **self_rank_data,
            "self_ranking": self_ranking,
            "actual_ranking": actual_ranking,
            "spearman_correlation": rho if not (rho is not None and abs(rho - rho) > 0) else None,  # NaN check
        }

        await asyncio.sleep(0.5)  # Rate limiting

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Meta-accuracy test")
    parser.add_argument("--run-id", required=True, help="Experiment run ID")
    args = parser.parse_args()

    results = asyncio.run(run_meta_accuracy_test(args.run_id))

    # Save results
    output_file = Path(f"data/results/exp1_{args.run_id}_meta_accuracy.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # Print summary
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║              META-ACCURACY TEST SUMMARY                   ║")
    print("╠═══════════════╦═══════════╦════════════════════════════════╣")
    print("║ Model         ║  ρ (self  ║  Interpretation                ║")
    print("║               ║   vs act) ║                                ║")
    print("╠═══════════════╬═══════════╬════════════════════════════════╣")

    for model in sorted(results.keys()):
        rho = results[model].get("spearman_correlation")
        parse_ok = results[model].get("parse_success", False)

        if not parse_ok:
            rho_str = "  PARSE   "
            interp = "Failed to parse response"
        elif rho is None:
            rho_str = "    —     "
            interp = "Insufficient data"
        else:
            rho_str = f"  {rho:>6.3f}  "
            if rho > 0.5:
                interp = "Good self-awareness"
            elif rho > 0.0:
                interp = "Weak self-awareness"
            elif rho > -0.3:
                interp = "Poor self-awareness"
            else:
                interp = "Inverted self-perception"

        short_model = model[:13]
        print(f"║ {short_model:<13} ║{rho_str}║ {interp:<26} ║")

    print("╚═══════════════╩═══════════╩════════════════════════════════╝")


if __name__ == "__main__":
    main()
