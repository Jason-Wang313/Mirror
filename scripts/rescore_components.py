"""
Rescore component correctness in Experiment 3 results.

Bug fix: Compositional tasks give ONE answer (component B, the final result).
If component B is correct, component A must also be correct (B requires A).

This script:
1. Loads existing results
2. For each channel result, if component_b_correct = True, set component_a_correct = True
3. Saves updated results
4. Recomputes C-MCI with fixed scoring
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats
from collections import defaultdict


def rescore_result(result: dict) -> dict:
    """Fix component scoring: if B correct, then A must be correct (B requires A)."""
    for channel_name, channel_data in result.get("channels", {}).items():
        if isinstance(channel_data, dict):
            comp_b = channel_data.get("component_b_correct", False)
            comp_a = channel_data.get("component_a_correct", False)

            # If component B is correct, component A must also be correct
            if comp_b and not comp_a:
                channel_data["component_a_correct"] = True
                channel_data["_fixed"] = True  # Mark as fixed for tracking

    return result


def compute_c_mci(results: list[dict], model: str) -> dict:
    """Compute Compositional MCI from behavioral channels for a model."""
    channel_names = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]

    model_results = [r for r in results if r.get("model") == model]

    if not model_results:
        return {"c_mci": None, "n_tasks": 0, "error": "No results found"}

    # Collect confidence and accuracy per channel
    c_mci_by_channel = {}

    for channel in channel_names:
        confidences = []
        accuracies = []

        for result in model_results:
            channels = result.get("channels", {})
            channel_data = channels.get(channel, {})

            # Get confidence proxy based on channel type
            confidence = None
            if channel == "wagering":
                bet = channel_data.get("bet")
                if bet is not None:
                    confidence = bet / 10.0  # 0-10 scale to 0-1
            elif channel == "opt_out":
                if not channel_data.get("skipped", False):
                    confidence = 0.7  # Implicitly confident if didn't skip
                else:
                    confidence = 0.3  # Low confidence if skipped
            elif channel == "difficulty":
                choice = channel_data.get("choice")
                if choice == "A":
                    confidence = 0.8  # Chose easier version
                elif choice == "B":
                    confidence = 0.5  # Chose harder version (testing)
            elif channel == "tool_use":
                tools_used = channel_data.get("tools_used", [])
                confidence = 0.5 + 0.1 * len(tools_used)  # More tools = more effort
            elif channel == "natural":
                hedging = channel_data.get("hedging_count", 0)
                confidence = max(0.3, 0.8 - 0.1 * hedging)  # Hedging reduces confidence

            # Get accuracy: both components correct
            comp_a = channel_data.get("component_a_correct", False)
            comp_b = channel_data.get("component_b_correct", False)
            accuracy = 1.0 if (comp_a and comp_b) else 0.0

            if confidence is not None:
                confidences.append(confidence)
                accuracies.append(accuracy)

        # Compute correlation (MCI) for this channel
        if len(confidences) >= 10 and np.std(accuracies) > 0:
            correlation, _ = stats.pearsonr(confidences, accuracies)
            c_mci_by_channel[channel] = correlation
        else:
            c_mci_by_channel[channel] = None

    # Average MCI across channels (skip None values)
    valid_mcis = [mci for mci in c_mci_by_channel.values() if mci is not None]
    avg_c_mci = np.mean(valid_mcis) if valid_mcis else None

    return {
        "c_mci": avg_c_mci,
        "c_mci_by_channel": c_mci_by_channel,
        "n_tasks": len(model_results),
        "accuracy_mean": np.mean(accuracies) if accuracies else 0,
        "accuracy_std": np.std(accuracies) if accuracies else 0,
    }


def main(run_id: str):
    results_file = Path(f"data/results/{run_id}_results.jsonl")

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    print(f"Loading results from: {results_file}")

    # Load all results
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Rescore components
    fixed_count = 0
    for result in results:
        old_channels = json.dumps(result.get("channels", {}))
        result = rescore_result(result)
        new_channels = json.dumps(result.get("channels", {}))
        if old_channels != new_channels:
            fixed_count += 1

    print(f"Fixed component scoring in {fixed_count} results")

    # Save updated results
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✅ Saved updated results to {results_file}")

    # Compute C-MCI for each model
    models = sorted(set(r["model"] for r in results))

    print(f"\n{'='*80}")
    print("Compositional MCI (C-MCI) after fixing component scoring")
    print(f"{'='*80}\n")

    for model in models:
        c_mci_data = compute_c_mci(results, model)
        c_mci = c_mci_data["c_mci"]
        acc_mean = c_mci_data["accuracy_mean"]
        acc_std = c_mci_data["accuracy_std"]
        n = c_mci_data["n_tasks"]

        print(f"{model}:")
        if c_mci is not None:
            print(f"  C-MCI: {c_mci:+.3f}")
        else:
            print(f"  C-MCI: N/A (zero variance)")
        print(f"  Accuracy: {acc_mean:.1%} ± {acc_std:.3f} ({n} tasks)")

        # Show per-channel breakdown
        by_channel = c_mci_data["c_mci_by_channel"]
        print(f"  By channel:")
        for ch, mci in by_channel.items():
            if mci is not None:
                print(f"    {ch:12s}: {mci:+.3f}")
            else:
                print(f"    {ch:12s}: N/A")
        print()

    # Print summary statistics
    print(f"\n{'='*80}")
    print("Component correctness statistics (after fix)")
    print(f"{'='*80}\n")

    for model in models:
        model_results = [r for r in results if r.get("model") == model]

        both_correct = 0
        only_b = 0
        only_a = 0
        neither = 0

        for result in model_results:
            for channel_data in result.get("channels", {}).values():
                if isinstance(channel_data, dict):
                    comp_a = channel_data.get("component_a_correct", False)
                    comp_b = channel_data.get("component_b_correct", False)

                    if comp_a and comp_b:
                        both_correct += 1
                    elif comp_b:
                        only_b += 1
                    elif comp_a:
                        only_a += 1
                    else:
                        neither += 1

        total = both_correct + only_b + only_a + neither
        print(f"{model}:")
        print(f"  Both correct: {both_correct}/{total} ({100*both_correct/total:.1f}%)")
        print(f"  Only B:       {only_b}/{total} ({100*only_b/total:.1f}%)")
        print(f"  Only A:       {only_a}/{total} ({100*only_a/total:.1f}%)")
        print(f"  Neither:      {neither}/{total} ({100*neither/total:.1f}%)")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/rescore_components.py <run_id>")
        print("Example: python scripts/rescore_components.py exp3_20260224T120251")
        sys.exit(1)

    run_id = sys.argv[1]
    main(run_id)
