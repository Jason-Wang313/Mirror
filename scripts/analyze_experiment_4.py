"""
Experiment 4 Analysis: Adaptation Crucible

Computes:
1. Adaptation Index (AI) per model × trial
2. Sycophancy Adaptation Ratio (SAR)
3. Recency decay analysis
4. Specificity effect
5. Strategy fingerprints
6. Summary statistics

Usage:
  python scripts/analyze_experiment_4.py --run-id <RUN_ID>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.scoring.adaptation_metrics import (
    compute_all_ai,
    compute_sar,
    compute_recency_decay,
    compute_specificity_effect,
    compute_strategy_fingerprint,
    compute_behavioral_deltas,
)


def load_results(run_id: str) -> list[dict]:
    """Load Experiment 4 results."""
    results_path = Path(f"data/results/exp4_{run_id}_results.jsonl")

    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        sys.exit(1)

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except:
                    continue

    return results


def print_summary_table(metrics: dict):
    """Print summary table of results."""
    print(f"\n{'='*100}")
    print("EXPERIMENT 4 ANALYSIS SUMMARY")
    print(f"{'='*100}\n")

    # AI by model
    print("Mean Adaptation Index (AI) by Model:")
    print(f"{'Model':<25} {'Mean AI':<12} {'N Trials':<10} {'Positive AI?':<15}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        mean_ai = data.get("mean_ai", 0.0)
        n_trials = data.get("n_trials", 0)
        positive = "✓ Yes" if mean_ai > 0.05 else "✗ No"

        print(f"{model:<25} {mean_ai:>11.3f} {n_trials:<10} {positive:<15}")

    print()

    # SAR by model
    print("Sycophancy Adaptation Ratio (SAR) by Model:")
    print(f"{'Model':<25} {'SAR':<10} {'True AI':<10} {'False AI':<10} {'Interpretation':<30}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        sar_data = data.get("sar", {})
        sar = sar_data.get("sar", 0.0)
        true_ai = sar_data.get("mean_ai_true_failure", 0.0)
        false_ai = sar_data.get("mean_ai_false_failure", 0.0)
        interp = sar_data.get("interpretation", "N/A")

        print(f"{model:<25} {sar:>9.2f} {true_ai:>9.3f} {false_ai:>9.3f} {interp:<30}")

    print()

    # Recency decay
    print("Recency Decay Analysis:")
    print(f"{'Model':<25} {'Immediate AI':<15} {'Delayed AI':<15} {'Decay %':<12} {'Robust?':<10}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        recency_data = data.get("recency", {})
        ai_by_delay = recency_data.get("ai_by_delay", {})
        decay_pct = recency_data.get("decay_percentage", 0.0)
        robust = "✓ Yes" if recency_data.get("robust", False) else "✗ No"

        immediate = ai_by_delay.get(0, 0.0)
        delayed = ai_by_delay.get(max(ai_by_delay.keys()) if ai_by_delay else 0, 0.0)

        print(f"{model:<25} {immediate:>14.3f} {delayed:>14.3f} {decay_pct:>11.1f}% {robust:<10}")

    print()

    # Specificity effect
    print("Specificity Effect:")
    print(f"{'Model':<25} {'Vague AI':<12} {'Specific AI':<12} {'Effect':<12} {'Better?':<10}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        spec_data = data.get("specificity", {})
        vague = spec_data.get("vague_ai", 0.0)
        specific = spec_data.get("specific_ai", 0.0)
        effect = spec_data.get("specificity_effect", 0.0)
        better = "✓ Yes" if spec_data.get("specific_is_better", False) else "✗ No"

        print(f"{model:<25} {vague:>11.3f} {specific:>11.3f} {effect:>11.3f} {better:<10}")

    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 4 results")
    parser.add_argument("--run-id", required=True, help="Experiment 4 run ID")
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Output directory"
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4 ANALYSIS")
    print(f"{'='*80}")
    print(f"Run ID: {args.run_id}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading results...")
    results = load_results(args.run_id)
    print(f"  Loaded {len(results)} result records\n")

    # Get unique models
    models = sorted(set(r.get("model") for r in results if r.get("model")))
    print(f"Found {len(models)} models: {', '.join(models)}\n")

    # Separate results by trial type
    standard_results = [r for r in results if r.get("trial_type") == "standard"]
    sycophancy_results = [r for r in results if r.get("trial_type") == "sycophancy_control"]
    recency_results = [r for r in results if r.get("trial_type") in ["standard", "recency_control"]]
    specificity_results = [r for r in results if r.get("trial_type") in ["specificity_vague", "specificity_specific"]]

    print(f"Trial breakdown:")
    print(f"  Standard: {len(standard_results)}")
    print(f"  Sycophancy: {len(sycophancy_results)}")
    print(f"  Recency: {len(recency_results)}")
    print(f"  Specificity: {len(specificity_results)}\n")

    # Compute metrics per model
    all_metrics = {}

    for model in models:
        print(f"Computing metrics for {model}...")

        model_metrics = {}

        # AI for standard trials
        model_standard = [r for r in standard_results if r.get("model") == model]
        ais = []
        fingerprints = []

        for result in model_standard:
            ai_data = compute_all_ai(result)
            ais.append(ai_data["mean_ai"])

            # Strategy fingerprint
            if "phase_c_related" in result and "phase_b" in result:
                fingerprint = compute_strategy_fingerprint(
                    result["phase_c_related"].get("response", ""),
                    result.get("trial", {}).get("phase_b_feedback", "")
                )
                fingerprints.append(fingerprint)

        model_metrics["mean_ai"] = sum(ais) / len(ais) if ais else 0.0
        model_metrics["n_trials"] = len(ais)

        # Aggregate fingerprints
        if fingerprints:
            agg_fingerprint = {}
            for key in fingerprints[0].keys():
                agg_fingerprint[key] = sum(f[key] for f in fingerprints) / len(fingerprints)
            model_metrics["strategy_fingerprint"] = agg_fingerprint

        # SAR
        model_metrics["sar"] = compute_sar(standard_results, sycophancy_results, model)

        # Recency decay
        model_metrics["recency"] = compute_recency_decay(recency_results, model)

        # Specificity effect
        model_metrics["specificity"] = compute_specificity_effect(specificity_results, model)

        all_metrics[model] = model_metrics

    print(f"\n✅ Computed all metrics for {len(models)} models\n")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"exp4_{args.run_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ Saved metrics: {metrics_path}\n")

    # Print summary
    print_summary_table(all_metrics)

    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
