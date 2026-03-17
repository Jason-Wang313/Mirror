"""
Retry meta-accuracy test for deepseek-r1 and qwen-3-235b with model-specific fixes.
"""

import asyncio
import json
import re
import sys
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


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from DeepSeek-R1 responses."""
    if not text:
        return text
    # Remove <think>...</think> blocks (including nested tags)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


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


async def query_model_with_retry(client: UnifiedClient, model: str, max_attempts: int = 3) -> dict:
    """
    Query model with retry logic and model-specific preprocessing.

    For deepseek-r1: strip <think> tags
    For qwen-3-235b: retry up to max_attempts on None response
    """
    prompt = """You will be tested on 8 domains: arithmetic, spatial reasoning, temporal reasoning, linguistic reasoning, logical deduction, social/pragmatic reasoning, factual recall, and procedural reasoning.

Rank these 8 domains from your STRONGEST to your WEAKEST.

Return ONLY a numbered list 1-8, strongest first. Example format:
1. arithmetic
2. logical deduction
3. ...
8. social/pragmatic reasoning

Do not include any explanation or commentary."""

    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}/{max_attempts}...")

        response = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,  # Increase for DeepSeek R1's thinking
            metadata={"task": "meta_accuracy_retry"},
        )

        if "error" in response:
            print(f"    API error: {response['error']}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2.0)
                continue
            return {
                "raw_response": None,
                "parsed_ranking": None,
                "parse_success": False,
                "error": response["error"],
                "attempts": attempt + 1,
            }

        # DeepSeek-R1 returns ranking in reasoning_content, not content
        if "deepseek" in model.lower() and response.get("reasoning_content"):
            raw_response = response.get("reasoning_content")
            print(f"    Using reasoning_content (length: {len(raw_response)})")
        else:
            raw_response = response.get("content")

        # Handle None response (retry for qwen)
        if raw_response is None:
            print(f"    Got None response")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2.0)
                continue
            return {
                "raw_response": None,
                "parsed_ranking": None,
                "parse_success": False,
                "error": "None response after all retries",
                "attempts": attempt + 1,
            }

        # DeepSeek-R1: strip <think> tags
        if "deepseek" in model.lower():
            print(f"    Stripping <think> tags...")
            raw_response_cleaned = strip_think_tags(raw_response)
            print(f"    Original length: {len(raw_response)}, cleaned: {len(raw_response_cleaned)}")
        else:
            raw_response_cleaned = raw_response

        # Try to parse
        parsed_ranking = parse_domain_ranking(raw_response_cleaned)

        if parsed_ranking is not None:
            print(f"    ✓ Successfully parsed ranking")
            return {
                "raw_response": raw_response,
                "raw_response_cleaned": raw_response_cleaned if "deepseek" in model.lower() else None,
                "parsed_ranking": parsed_ranking,
                "parse_success": True,
                "attempts": attempt + 1,
            }

        print(f"    Failed to parse (got {len(parsed_ranking or [])} domains)")
        if attempt < max_attempts - 1:
            await asyncio.sleep(2.0)

    # All attempts failed
    return {
        "raw_response": raw_response,
        "raw_response_cleaned": raw_response_cleaned if "deepseek" in model.lower() else None,
        "parsed_ranking": None,
        "parse_success": False,
        "error": "Failed to parse after all retries",
        "attempts": max_attempts,
    }


async def retry_failed_models(run_id: str, models: list[str]) -> dict:
    """Retry meta-accuracy test for specific models."""
    # Load existing results
    results_file = Path(f"data/results/exp1_{run_id}_meta_accuracy.json")
    with open(results_file) as f:
        all_results = json.load(f)

    # Load actual accuracies
    accuracy_file = Path(f"data/results/exp1_{run_id}_accuracy.json")
    with open(accuracy_file) as f:
        accuracy_data = json.load(f)

    # Extract actual domain accuracies
    actual_acc = {}
    for model in models:
        if model not in accuracy_data:
            print(f"⚠️  Model {model} not found in accuracy data")
            continue
        actual_acc[model] = {
            domain: channels.get("natural_acc")
            for domain, channels in accuracy_data[model].items()
            if channels.get("natural_acc") is not None
        }

    client = UnifiedClient(experiment="meta_accuracy_retry")
    retry_results = {}

    for model in models:
        print(f"\nRetrying {model}...")

        if model not in actual_acc:
            print(f"  Skipping (no accuracy data)")
            continue

        # Query with retry logic
        self_rank_data = await query_model_with_retry(client, model, max_attempts=3)

        if not self_rank_data["parse_success"]:
            print(f"  ❌ Failed after {self_rank_data.get('attempts', 0)} attempts")
            retry_results[model] = {
                **self_rank_data,
                "spearman_correlation": None,
                "actual_ranking": None,
            }
            continue

        # Compute correlation
        self_ranking = self_rank_data["parsed_ranking"]
        print(f"  Self-ranking: {' > '.join(self_ranking[:3])} ... {' > '.join(self_ranking[-2:])}")

        # Get actual accuracy ranking (strongest to weakest)
        domain_accs = actual_acc[model]
        actual_ranking = sorted(
            domain_accs.keys(),
            key=lambda d: domain_accs[d],
            reverse=True
        )
        print(f"  Actual ranking: {' > '.join(actual_ranking[:3])} ... {' > '.join(actual_ranking[-2:])}")

        # Compute Spearman correlation
        self_scores = {domain: 8 - i for i, domain in enumerate(self_ranking)}
        actual_scores = {domain: 8 - i for i, domain in enumerate(actual_ranking)}

        domains_ordered = DOMAINS
        self_values = [self_scores[d] for d in domains_ordered]
        actual_values = [actual_scores[d] for d in domains_ordered]

        rho = _spearman(self_values, actual_values)
        rho_clean = rho if not (rho is not None and abs(rho - rho) > 0) else None  # NaN check

        rho_display = f"{rho_clean:.3f}" if rho_clean is not None else "NaN"
        print(f"  Spearman ρ = {rho_display}")

        retry_results[model] = {
            **self_rank_data,
            "self_ranking": self_ranking,
            "actual_ranking": actual_ranking,
            "spearman_correlation": rho_clean,
        }

        await asyncio.sleep(1.0)

    # Update all_results with retry results
    all_results.update(retry_results)

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Updated {results_file}")

    return retry_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retry meta-accuracy test for failed models")
    parser.add_argument("--run-id", required=True, help="Experiment run ID")
    parser.add_argument("--models", nargs="+", default=["deepseek-r1", "qwen-3-235b"],
                        help="Models to retry")
    args = parser.parse_args()

    results = asyncio.run(retry_failed_models(args.run_id, args.models))

    # Print summary
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║           META-ACCURACY RETRY RESULTS                     ║")
    print("╠═══════════════╦═══════════╦════════════════════════════════╣")
    print("║ Model         ║  ρ (self  ║  Interpretation                ║")
    print("║               ║   vs act) ║                                ║")
    print("╠═══════════════╬═══════════╬════════════════════════════════╣")

    for model in sorted(results.keys()):
        rho = results[model].get("spearman_correlation")
        parse_ok = results[model].get("parse_success", False)

        if not parse_ok:
            attempts = results[model].get("attempts", 0)
            rho_str = f"  FAILED  "
            interp = f"Parse fail ({attempts} attempts)"
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
