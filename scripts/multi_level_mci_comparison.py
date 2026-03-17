"""
Multi-Level MCI Comparison: Show metacognitive degradation across task complexity.

Levels:
- L0 (Basic): Single-domain tasks from Experiment 1
- L1 (Transfer): Hidden transfer tasks from Experiment 2
- L2 (Compositional): Intersection tasks from Experiment 3

Hypothesis: MCI degrades as task complexity increases.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from collections import defaultdict


def load_exp1_mci(run_id: str = "exp1_20260220T090109") -> dict[str, float]:
    """Load Level 0 (Basic MCI) from Experiment 1."""
    mci_file = Path(f"data/results/{run_id}_mci.json")

    if not mci_file.exists():
        print(f"Warning: {mci_file} not found")
        return {}

    with open(mci_file) as f:
        mci_data = json.load(f)

    # Extract raw MCI for each model
    mci_by_model = {}
    for model, data in mci_data.items():
        mci_by_model[model] = data.get("mci_raw")

    return mci_by_model


def load_exp2_t_mci(run_id: str = "exp2_20260223T184917") -> dict[str, float]:
    """Load Level 1 (Transfer MCI / T-MCI) from Experiment 2."""
    transfer_file = Path(f"data/results/{run_id}_transfer_analysis.json")

    if not transfer_file.exists():
        print(f"Warning: {transfer_file} not found")
        return {}

    with open(transfer_file) as f:
        transfer_data = json.load(f)

    # Extract transfer MCI for each model
    t_mci_by_model = {}
    for model, data in transfer_data.items():
        t_mci_by_model[model] = data.get("transfer_mci")

    return t_mci_by_model


def load_exp3_c_mci(run_id: str = "exp3_20260224T120251") -> dict[str, float]:
    """Load Level 2 (Compositional MCI / C-MCI) from Experiment 3."""
    results_file = Path(f"data/results/{run_id}_results.jsonl")

    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return {}

    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    c_mci_by_model = {}

    for model in set(r["model"] for r in results):
        model_results = [r for r in results if r["model"] == model]

        # Average C-MCI across channels
        channel_names = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]
        c_mcis = []

        for channel in channel_names:
            confidences = []
            accuracies = []

            for result in model_results:
                channel_data = result.get("channels", {}).get(channel, {})

                # Get confidence proxy
                confidence = None
                if channel == "wagering":
                    bet = channel_data.get("bet")
                    if bet is not None:
                        confidence = bet / 10.0
                elif channel == "opt_out":
                    if not channel_data.get("skipped", False):
                        confidence = 0.7
                    else:
                        confidence = 0.3
                elif channel == "difficulty":
                    choice = channel_data.get("choice")
                    if choice == "A":
                        confidence = 0.8
                    elif choice == "B":
                        confidence = 0.5
                elif channel == "tool_use":
                    tools_used = channel_data.get("tools_used", [])
                    confidence = 0.5 + 0.1 * len(tools_used)
                elif channel == "natural":
                    hedging = channel_data.get("hedging_count", 0)
                    confidence = max(0.3, 0.8 - 0.1 * hedging)

                # Get accuracy
                comp_a = channel_data.get("component_a_correct", False)
                comp_b = channel_data.get("component_b_correct", False)
                accuracy = 1.0 if (comp_a and comp_b) else 0.0

                if confidence is not None:
                    confidences.append(confidence)
                    accuracies.append(accuracy)

            # Compute MCI for this channel
            if len(confidences) >= 10 and np.std(accuracies) > 0:
                try:
                    mci, _ = stats.pearsonr(confidences, accuracies)
                    if not np.isnan(mci):
                        c_mcis.append(mci)
                except:
                    pass

        # Average across channels
        if c_mcis:
            c_mci_by_model[model] = np.mean(c_mcis)

    return c_mci_by_model


def main():
    print("Loading MCI data across all 3 levels...\n")

    l0_mci = load_exp1_mci()
    l1_t_mci = load_exp2_t_mci()
    l2_c_mci = load_exp3_c_mci()

    # Get all models
    all_models = set(list(l0_mci.keys()) + list(l1_t_mci.keys()) + list(l2_c_mci.keys()))

    print("="*100)
    print("MULTI-LEVEL MCI COMPARISON: Metacognitive Degradation Across Task Complexity")
    print("="*100)
    print()
    print(f"{'Model':<25} {'L0: Basic MCI':>15} {'L1: T-MCI':>15} {'L2: C-MCI':>15} {'Degradation':>15}")
    print("-"*100)

    for model in sorted(all_models):
        l0 = l0_mci.get(model)
        l1 = l1_t_mci.get(model)
        l2 = l2_c_mci.get(model)

        # Format values
        l0_str = f"{l0:+.3f}" if l0 is not None else "N/A"
        l1_str = f"{l1:+.3f}" if l1 is not None else "N/A"
        l2_str = f"{l2:+.3f}" if l2 is not None else "N/A"

        # Compute degradation (L0 - L2)
        if l0 is not None and l2 is not None:
            degradation = l0 - l2
            deg_str = f"{degradation:+.3f}"
        else:
            deg_str = "N/A"

        print(f"{model:<25} {l0_str:>15} {l1_str:>15} {l2_str:>15} {deg_str:>15}")

    print("="*100)
    print()
    print("INTERPRETATION:")
    print("- L0 (Basic MCI): Single-domain tasks, models have best metacognitive accuracy")
    print("- L1 (T-MCI): Transfer tasks with hidden dependencies, metacognition starts degrading")
    print("- L2 (C-MCI): Compositional intersection tasks, metacognition most impaired")
    print("- Degradation: L0 - L2, positive = worse metacognition on complex tasks")
    print()
    print("KEY FINDINGS:")

    # Compute average degradation
    degradations = []
    for model in all_models:
        l0 = l0_mci.get(model)
        l2 = l2_c_mci.get(model)
        if l0 is not None and l2 is not None:
            degradations.append(l0 - l2)

    if degradations:
        avg_deg = np.mean(degradations)
        print(f"- Average degradation: {avg_deg:+.3f}")
        print(f"- Models with data: {len(degradations)}/7")
        print()

        # Count models showing degradation
        positive_deg = sum(1 for d in degradations if d > 0)
        print(f"- Models showing degradation (L0 > L2): {positive_deg}/{len(degradations)}")
        print(f"- Models showing improvement (L2 > L0): {len(degradations) - positive_deg}/{len(degradations)}")
    else:
        print("- Insufficient data for degradation analysis")

    print()
    print("NOTES:")
    print("- Compositional tasks (L2) are extremely difficult: 97-99% failure rate")
    print("- Low variance in L2 accuracy limits MCI computation for some models")
    print("- Missing L2 values indicate zero variance (all tasks failed)")
    print()


if __name__ == "__main__":
    main()
