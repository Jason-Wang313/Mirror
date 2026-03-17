"""
Experiment 3 Analysis: Compositional Self-Prediction

Computes:
1. CCE (Compositional Calibration Error) per model × intersection type
2. BCI (Behavioral Composition Index) per model × channel
3. Weak-link identification accuracy
4. Compositional MCI
5. Three-level comparison (coins → agent → self)
6. Summary statistics

Usage:
  python scripts/analyze_experiment_3.py --run-id <RUN_ID>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.scoring.compositional_metrics import (
    compute_all_cce,
    compute_all_bci,
    compute_weak_link_accuracy,
    compute_compositional_mci,
    compute_three_level_comparison,
)


def load_results(run_id: str) -> list[dict]:
    """Load Experiment 3 results."""
    results_path = Path(f"data/results/exp3_{run_id}_results.jsonl")

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


def load_capability_profiles():
    """Load capability profiles for baseline comparisons."""
    profiles_path = Path("data/exp9/capability_profiles.json")

    if not profiles_path.exists():
        print(f"⚠️  Capability profiles not found: {profiles_path}")
        return {}

    with open(profiles_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary_table(metrics: dict):
    """Print summary table of results."""
    print(f"\n{'='*100}")
    print("EXPERIMENT 3 ANALYSIS SUMMARY")
    print(f"{'='*100}\n")

    # CCE by model and type
    print("Compositional Calibration Error (CCE) by Model:")
    print(f"{'Model':<25} {'Strong×Strong':<15} {'Strong×Weak':<15} {'Weak×Weak':<15} {'Mixed':<15}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        cce_data = data.get("cce", {})
        ss = cce_data.get("strong_strong", {}).get("mean_cce", 0.0)
        sw = cce_data.get("strong_weak", {}).get("mean_cce", 0.0)
        ww = cce_data.get("weak_weak", {}).get("mean_cce", 0.0)
        mixed = cce_data.get("mixed", {}).get("mean_cce", 0.0)

        print(f"{model:<25} {ss:>14.3f} {sw:>14.3f} {ww:>14.3f} {mixed:>14.3f}")

    print()

    # BCI by model
    print("Behavioral Composition Index (BCI) by Model:")
    print(f"{'Model':<25} {'Wagering':<12} {'Opt-out':<12} {'Difficulty':<12} {'Tool Use':<12} {'Natural':<12}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        bci_data = data.get("bci", {})
        wager = bci_data.get("wagering", {}).get("bci", 0.0)
        opt = bci_data.get("opt_out", {}).get("bci", 0.0)
        diff = bci_data.get("difficulty", {}).get("bci", 0.0)
        tool = bci_data.get("tool_use", {}).get("bci", 0.0)
        nat = bci_data.get("natural", {}).get("bci", 0.0)

        print(f"{model:<25} {wager:>11.3f} {opt:>11.3f} {diff:>11.3f} {tool:>11.3f} {nat:>11.3f}")

    print()

    # Weak-link accuracy
    print("Weak-Link Identification Accuracy:")
    print(f"{'Model':<25} {'Accuracy':<12} {'Above Chance?':<15} {'N Tasks'}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        wl_data = data.get("weak_link", {})
        acc = wl_data.get("accuracy", 0.0)
        above = "✓ Yes" if wl_data.get("above_chance", False) else "✗ No"
        n = wl_data.get("n_total", 0)

        print(f"{model:<25} {acc:>11.1%} {above:<15} {n}")

    print()

    # Three-level comparison
    print("Three-Level Decoupled Prediction (A: coins, B: agent, C: self):")
    print(f"{'Model':<25} {'Level A':<10} {'Level B':<10} {'Level C':<10} {'Monotonic?':<12}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        tl_data = data.get("three_level", {})
        level_a = tl_data.get("level_a", {}).get("accuracy", 0.0)
        level_b = tl_data.get("level_b", {}).get("accuracy", 0.0)
        level_c = tl_data.get("level_c", {}).get("accuracy", 0.0)
        mono = "✓ Yes" if tl_data.get("gradient", {}).get("monotonic_decrease", False) else "✗ No"

        print(f"{model:<25} {level_a:>9.1%} {level_b:>9.1%} {level_c:>9.1%} {mono:<12}")

    print()

    # MCI
    print("Compositional Meta-Cognitive Index (MCI):")
    print(f"{'Model':<25} {'MCI':<10} {'N Tasks'}")
    print("-"*100)

    for model, data in sorted(metrics.items()):
        mci_data = data.get("mci", {})
        mci = mci_data.get("mci", 0.0)
        n = mci_data.get("n_tasks", 0)

        print(f"{model:<25} {mci:>9.3f} {n}")

    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 3 results")
    parser.add_argument("--run-id", required=True, help="Experiment 3 run ID")
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Output directory"
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3 ANALYSIS")
    print(f"{'='*80}")
    print(f"Run ID: {args.run_id}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading results...")
    results = load_results(args.run_id)
    print(f"  Loaded {len(results)} result records\n")

    print("Loading capability profiles...")
    profiles = load_capability_profiles()
    print(f"  Loaded profiles for {len(profiles)} models\n")

    # Get unique models
    models = sorted(set(r.get("model") for r in results if r.get("model")))
    print(f"Found {len(models)} models: {', '.join(models)}\n")

    # Compute metrics per model
    all_metrics = {}

    for model in models:
        print(f"Computing metrics for {model}...")

        model_metrics = {}

        # CCE
        model_metrics["cce"] = compute_all_cce(results, model, profiles)

        # BCI
        model_metrics["bci"] = compute_all_bci(results, model)

        # Weak-link accuracy
        model_metrics["weak_link"] = compute_weak_link_accuracy(results, model)

        # MCI
        model_metrics["mci"] = compute_compositional_mci(results, model)

        # Three-level comparison
        model_metrics["three_level"] = compute_three_level_comparison(results, model)

        all_metrics[model] = model_metrics

    print(f"\n✅ Computed all metrics for {len(models)} models\n")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"exp3_{args.run_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ Saved metrics: {metrics_path}\n")

    # Print summary
    print_summary_table(all_metrics)

    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
