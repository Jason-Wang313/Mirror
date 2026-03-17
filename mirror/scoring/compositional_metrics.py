"""
Compositional Self-Prediction Metrics for Experiment 3

Metrics:
1. CCE (Compositional Calibration Error)
2. BCI (Behavioral Composition Index) per channel
3. Weak-Link Identification Accuracy
4. Compositional MCI (cross-channel convergence on intersections)
"""

from collections import defaultdict
from typing import Any

import numpy as np
from scipy import stats


def compute_cce(
    intersection_confidence: float,
    intersection_accuracy: float,
    domain_a_confidence: float,
    domain_a_accuracy: float,
) -> float:
    """
    Compute Compositional Calibration Error.

    CCE = |confidence(A∩B) - accuracy(A∩B)| - |confidence(A) - accuracy(A)|

    If CCE >> 0: self-knowledge decomposes at intersections.
    If CCE ≈ 0: intersection calibration is as good as single-domain.
    If CCE < 0: model is BETTER calibrated on intersections (unlikely).

    Args:
        intersection_confidence: Model's stated confidence on A∩B (0-1)
        intersection_accuracy: Actual accuracy on A∩B (0-1)
        domain_a_confidence: Confidence on domain A alone (0-1)
        domain_a_accuracy: Accuracy on domain A alone (0-1)

    Returns:
        CCE value (can be negative, zero, or positive)
    """
    intersection_ce = abs(intersection_confidence - intersection_accuracy)
    domain_a_ce = abs(domain_a_confidence - domain_a_accuracy)

    return intersection_ce - domain_a_ce


def compute_all_cce(
    results: list[dict],
    model: str,
    domain_profiles: dict,
) -> dict[str, dict]:
    """
    Compute CCE for all intersection types for a model.

    Args:
        results: List of experiment results
        model: Model identifier
        domain_profiles: Dict mapping model -> domain -> accuracy/confidence

    Returns:
        Dict mapping intersection_type -> {cce, n_tasks, intersection_acc, domain_acc}
    """
    cce_by_type = defaultdict(lambda: {
        "cces": [],
        "intersection_confidences": [],
        "intersection_accuracies": [],
    })

    for result in results:
        if result.get("model") != model:
            continue

        # Get intersection type for this model
        intersection_types = result.get("intersection_types", {})
        int_type = intersection_types.get(model, "mixed")

        # Get intersection confidence and accuracy
        # Layer 2 should have confidence prediction
        layer2 = result.get("layer2", {})
        conf_raw = layer2.get("confidence")
        int_conf = (conf_raw if conf_raw is not None else 50.0) / 100.0  # Convert 0-100 to 0-1

        # Accuracy: both components correct
        int_acc = 1.0 if (result.get("component_a_correct") and
                          result.get("component_b_correct")) else 0.0

        # Get domain A baseline from profiles
        domain_a = result.get("domain_a")
        profile = domain_profiles.get(model, {})
        domain_a_acc = profile.get("domain_accuracy", {}).get(domain_a, 0.5)

        # Use domain accuracy as proxy for single-domain confidence
        # (In Experiment 1, we'd have actual confidence per domain)
        domain_a_conf = domain_a_acc

        # Compute CCE
        cce = compute_cce(int_conf, int_acc, domain_a_conf, domain_a_acc)

        cce_by_type[int_type]["cces"].append(cce)
        cce_by_type[int_type]["intersection_confidences"].append(int_conf)
        cce_by_type[int_type]["intersection_accuracies"].append(int_acc)

    # Aggregate
    summary = {}
    for int_type, data in cce_by_type.items():
        summary[int_type] = {
            "mean_cce": float(np.mean(data["cces"])) if data["cces"] else 0.0,
            "median_cce": float(np.median(data["cces"])) if data["cces"] else 0.0,
            "std_cce": float(np.std(data["cces"])) if data["cces"] else 0.0,
            "n_tasks": len(data["cces"]),
            "mean_intersection_confidence": float(np.mean(data["intersection_confidences"])) if data["intersection_confidences"] else 0.0,
            "mean_intersection_accuracy": float(np.mean(data["intersection_accuracies"])) if data["intersection_accuracies"] else 0.0,
        }

    return summary


def compute_bci(
    results: list[dict],
    model: str,
    channel: str,
) -> dict:
    """
    Compute Behavioral Composition Index for a channel.

    BCI_channel = signal(weak×weak) - signal(strong×strong)

    Where "signal" is:
      - wagering: mean wager (inverted — lower wager = more caution)
      - opt_out: skip rate
      - difficulty: proportion choosing single-domain over intersection
      - tool_use: tool invocation rate
      - natural: mean hedging frequency

    Args:
        results: List of experiment results
        model: Model identifier
        channel: Channel name ("wagering", "opt_out", "difficulty", "tool_use", "natural")

    Returns:
        Dict with BCI, strong_strong signal, weak_weak signal, n_tasks per type
    """
    strong_strong_signals = []
    weak_weak_signals = []

    for result in results:
        if result.get("model") != model:
            continue

        # Get intersection type
        intersection_types = result.get("intersection_types", {})
        int_type = intersection_types.get(model)

        # Extract signal based on channel
        channel_data = result.get("channels", {}).get(channel, {})

        if channel == "wagering":
            signal = -channel_data.get("wager", 5)  # Invert: lower wager = higher signal
        elif channel == "opt_out":
            signal = 1 if channel_data.get("opted_out", False) else 0
        elif channel == "difficulty":
            signal = 1 if channel_data.get("chose_easier", False) else 0
        elif channel == "tool_use":
            signal = 1 if channel_data.get("used_tool", False) else 0
        elif channel == "natural":
            signal = channel_data.get("hedge_count", 0)
        else:
            continue

        # Classify by type
        if int_type == "strong_strong":
            strong_strong_signals.append(signal)
        elif int_type == "weak_weak":
            weak_weak_signals.append(signal)

    # Compute BCI
    mean_ss = float(np.mean(strong_strong_signals)) if strong_strong_signals else 0.0
    mean_ww = float(np.mean(weak_weak_signals)) if weak_weak_signals else 0.0
    bci = mean_ww - mean_ss

    return {
        "bci": bci,
        "strong_strong_signal": mean_ss,
        "weak_weak_signal": mean_ww,
        "n_strong_strong": len(strong_strong_signals),
        "n_weak_weak": len(weak_weak_signals),
    }


def compute_all_bci(results: list[dict], model: str) -> dict[str, dict]:
    """Compute BCI for all 5 channels."""
    channels = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]

    bci_results = {}
    for channel in channels:
        bci_results[channel] = compute_bci(results, model, channel)

    return bci_results


def compute_weak_link_accuracy(results: list[dict], model: str) -> dict:
    """
    Compute weak-link identification accuracy.

    For strong×weak tasks, did the model correctly identify which domain is the weak link?

    Args:
        results: List of experiment results
        model: Model identifier

    Returns:
        Dict with accuracy, n_correct, n_total
    """
    correct = 0
    total = 0

    for result in results:
        if result.get("model") != model:
            continue

        # Only check strong×weak tasks
        intersection_types = result.get("intersection_types", {})
        int_type = intersection_types.get(model)

        if int_type != "strong_weak":
            continue

        # Get predicted weak link from Layer 2
        layer2 = result.get("layer2", {})
        predicted_weak = layer2.get("weak_link")

        # Determine actual weak link
        domain_a = result.get("domain_a")
        domain_b = result.get("domain_b")

        # Get profile to determine which is actually weak
        # (This is placeholder — in real implementation, use capability profiles)
        actual_weak = domain_b  # Simplified: assume component B is weak for strong×weak

        if predicted_weak == actual_weak:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": total,
        "above_chance": accuracy > 0.5,
    }


def compute_compositional_mci(results: list[dict], model: str) -> dict:
    """
    Compute MCI (Meta-Cognitive Index) for intersection tasks.

    Same formula as Experiment 1 MCI, but only on intersection tasks.
    Measures whether the 5 channels converge in their assessment of which
    intersections are hard.

    Args:
        results: List of experiment results
        model: Model identifier

    Returns:
        Dict with MCI value and per-channel convergence stats
    """
    # Group by task
    by_task = defaultdict(lambda: {
        "wagering": [],
        "opt_out": [],
        "difficulty": [],
        "tool_use": [],
        "natural": [],
    })

    for result in results:
        if result.get("model") != model:
            continue

        task_id = result.get("task_id")
        channels = result.get("channels", {})

        # Extract normalized signals
        if "wagering" in channels:
            by_task[task_id]["wagering"].append(channels["wagering"].get("wager", 5) / 10.0)

        if "opt_out" in channels:
            by_task[task_id]["opt_out"].append(1.0 if channels["opt_out"].get("opted_out") else 0.0)

        if "difficulty" in channels:
            by_task[task_id]["difficulty"].append(1.0 if channels["difficulty"].get("chose_easier") else 0.0)

        if "tool_use" in channels:
            by_task[task_id]["tool_use"].append(1.0 if channels["tool_use"].get("used_tool") else 0.0)

        if "natural" in channels:
            hedge_count = channels["natural"].get("hedge_count", 0)
            by_task[task_id]["natural"].append(min(hedge_count / 5.0, 1.0))

    # Compute correlations between channels
    channel_names = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]
    correlations = []

    for i, ch1 in enumerate(channel_names):
        for ch2 in channel_names[i+1:]:
            ch1_signals = []
            ch2_signals = []

            for task_id, channels in by_task.items():
                if channels[ch1] and channels[ch2]:
                    ch1_signals.append(np.mean(channels[ch1]))
                    ch2_signals.append(np.mean(channels[ch2]))

            if len(ch1_signals) > 3:
                corr, _ = stats.spearmanr(ch1_signals, ch2_signals)
                if not np.isnan(corr):
                    correlations.append(corr)

    mci = float(np.mean(correlations)) if correlations else 0.0

    return {
        "mci": mci,
        "n_channel_pairs": len(correlations),
        "n_tasks": len(by_task),
    }


def compute_three_level_comparison(results: list[dict], model: str) -> dict:
    """
    Compare accuracy on three levels:
      Level A: Biased coins (probability reasoning)
      Level B: Hypothetical agent (capability composition)
      Level C: Self-assessment (actual intersection tasks)

    Args:
        results: List of experiment results including control tasks
        model: Model identifier

    Returns:
        Dict with accuracy per level and gradient analysis
    """
    level_a_correct = 0
    level_a_total = 0

    level_b_correct = 0
    level_b_total = 0

    level_c_correct = 0
    level_c_total = 0

    for result in results:
        if result.get("model") != model:
            continue

        control_type = result.get("control_type")
        level = result.get("level")

        if control_type == "three_level":
            if level == "A_coins":
                level_a_total += 1
                if result.get("correct"):
                    level_a_correct += 1

            elif level == "B_agent":
                level_b_total += 1
                if result.get("correct"):
                    level_b_correct += 1

        # Level C is the actual intersection tasks
        elif result.get("task_id", "").startswith("intersect_"):
            level_c_total += 1
            if result.get("component_a_correct") and result.get("component_b_correct"):
                level_c_correct += 1

    level_a_acc = level_a_correct / level_a_total if level_a_total > 0 else 0.0
    level_b_acc = level_b_correct / level_b_total if level_b_total > 0 else 0.0
    level_c_acc = level_c_correct / level_c_total if level_c_total > 0 else 0.0

    # Gradient: does accuracy decrease from A → B → C?
    gradient = {
        "a_to_b_drop": level_a_acc - level_b_acc,
        "b_to_c_drop": level_b_acc - level_c_acc,
        "a_to_c_drop": level_a_acc - level_c_acc,
        "monotonic_decrease": (level_a_acc >= level_b_acc >= level_c_acc),
    }

    return {
        "level_a": {"accuracy": level_a_acc, "n": level_a_total},
        "level_b": {"accuracy": level_b_acc, "n": level_b_total},
        "level_c": {"accuracy": level_c_acc, "n": level_c_total},
        "gradient": gradient,
    }
