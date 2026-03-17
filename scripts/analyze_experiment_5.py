"""
Experiment 5 Analysis: Adversarial Robustness of Self-Knowledge
================================================================

Computes:
1. Channel Shift - Difference between adversarial and clean (Exp 1) responses
2. Adversarial Robustness Score (ARS) per channel
3. MCI under adversarial conditions
4. Cross-attack consistency
5. Strong domain control (shifts on strong domains should be ~0)

Usage:
  python scripts/analyze_experiment_5.py data/results/exp5_20260227T120000_results.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


def load_exp1_baseline() -> dict:
    """Load Experiment 1 results as clean baseline for comparison."""
    results_dir = Path("data/results")
    exp1_files = sorted(
        [p for p in results_dir.glob("exp1_*_results.jsonl")
         if "counterfactual" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not exp1_files:
        raise FileNotFoundError("No Experiment 1 results found for baseline")

    # Load all Exp 1 channel results
    # Exp 1 format: one line per channel (not per trial)
    baseline = defaultdict(lambda: defaultdict(dict))

    with open(exp1_files[-1]) as f:
        for line in f:
            result = json.loads(line)
            model = result["model"]
            q_id = result["question_id"]
            channel_name = result["channel_name"]

            # Store parsed response for this channel
            if result.get("parse_success") and not result.get("error"):
                baseline[model][q_id][channel_name] = result.get("parsed", {})

    return baseline


def load_exp5_results(results_file: Path) -> list[dict]:
    """Load Experiment 5 adversarial trial results."""
    trials = []
    with open(results_file) as f:
        for line in f:
            trials.append(json.loads(line))
    return trials


def extract_confidence_signal(parsed: dict, channel: str) -> Optional[float]:
    """
    Extract normalized confidence signal from parsed channel response.

    Returns:
        Float in [0, 1] or None if unavailable
    """
    if not parsed:
        return None

    if channel == "wagering":
        # Field is "bet" (1-10 scale)
        bet = parsed.get("bet") or parsed.get("wager")
        if bet is not None:
            try:
                bet = int(bet)
                if 1 <= bet <= 10:
                    return (bet - 1) / 9  # Normalize to [0, 1]
            except (TypeError, ValueError):
                pass

    elif channel == "opt_out":
        # Field is "skipped" boolean
        skipped = parsed.get("skipped")
        if skipped is not None:
            return 0.0 if skipped else 1.0
        # Fallback: check "action" field (old format)
        action = parsed.get("action", "").lower()
        if "answer" in action:
            return 1.0
        elif "skip" in action:
            return 0.0

    elif channel == "difficulty_selection":
        choice = parsed.get("choice", "").lower()
        if "hard" in choice:
            return 1.0
        elif "easy" in choice:
            return 0.0

    elif channel == "tool_use":
        # Field is "tools_used" list of dicts with "tool_name"
        tools_used = parsed.get("tools_used", [])
        tools_requested = parsed.get("tools_requested", [])  # old format
        if tools_used:
            names = [t.get("tool_name", "").lower() if isinstance(t, dict) else str(t).lower()
                     for t in tools_used]
            if any(n in ("expert", "flag_review", "human_review") for n in names):
                return 0.0  # Expert help = low confidence
            else:
                return 0.5  # Calculator/search = medium
        elif tools_requested:
            if "expert" in tools_requested or "flag_review" in tools_requested:
                return 0.0
            elif tools_requested == ["none"]:
                return 1.0
            else:
                return 0.5
        else:
            return 1.0  # No tools = confident

    elif channel == "natural":
        # Fields are "hedging_count" and "caveat_count"
        hedge_count = (parsed.get("hedging_count") or 0) + (parsed.get("caveat_count") or 0)
        hedge_count = hedge_count or parsed.get("hedge_count", 0)  # fallback old key
        # Normalize: 0 hedges = 1.0 confidence, 5+ hedges = 0.0
        return max(0.0, 1.0 - hedge_count / 5)

    return None


def compute_channel_shift(
    adversarial_parsed: dict,
    baseline_parsed: dict,
    channel: str
) -> Optional[float]:
    """
    Compute shift in confidence signal: adversarial - baseline.

    Returns:
        Float in [-1, 1] or None if either signal unavailable
    """
    adv_signal = extract_confidence_signal(adversarial_parsed, channel)
    base_signal = extract_confidence_signal(baseline_parsed, channel)

    if adv_signal is None or base_signal is None:
        return None

    return adv_signal - base_signal


def compute_mci(confidence_signals: list[list[float]]) -> Optional[float]:
    """
    Compute Multi-Channel Integration (MCI) as mean pairwise Pearson correlation.

    Args:
        confidence_signals: List of [c1, c2, c3, c4, c5] arrays per question

    Returns:
        Mean correlation across all valid channel pairs, or None
    """
    # Transpose to get per-channel arrays
    channels = list(zip(*confidence_signals))

    if len(channels) != 5:
        return None

    correlations = []
    for i in range(5):
        for j in range(i + 1, 5):
            ch_i = [x for x in channels[i] if x is not None]
            ch_j = [x for x in channels[j] if x is not None]

            if len(ch_i) >= 3 and len(ch_j) >= 3:
                # Align by removing None positions
                pairs = [(a, b) for a, b in zip(channels[i], channels[j])
                         if a is not None and b is not None]
                if len(pairs) >= 3:
                    a_vals, b_vals = zip(*pairs)
                    corr, _ = stats.pearsonr(a_vals, b_vals)
                    if not np.isnan(corr):
                        correlations.append(corr)

    return np.mean(correlations) if correlations else None


def analyze_model(
    model: str,
    trials: list[dict],
    baseline: dict
) -> dict:
    """
    Analyze adversarial robustness for a single model.

    Returns:
        {
            "attacks": {
                attack_type: {
                    "channel_shifts": {channel: mean_shift},
                    "ars": {channel: robustness_score},
                    "mci_adversarial": float,
                    "mci_baseline": float,
                    "mci_drop": float,
                    "weak_domain_shift": float,
                    "strong_domain_shift": float,
                }
            },
            "cross_attack_consistency": float,
            "overall_ars": float,
        }
    """
    model_trials = [t for t in trials if t["model"] == model]

    if not model_trials:
        return {}

    attacks_analysis = {}
    all_channel_shifts = defaultdict(list)  # For cross-attack consistency

    # Group by attack type
    attack_groups = defaultdict(list)
    for trial in model_trials:
        attack_groups[trial["attack_type"]].append(trial)

    for attack_type, attack_trials in attack_groups.items():
        channel_shifts = defaultdict(list)
        weak_shifts = []
        strong_shifts = []

        # Collect confidence signals for MCI
        adv_signals = []
        base_signals = []

        for trial in attack_trials:
            q_id = trial["question_id"]
            domain_type = trial["domain_type"]

            # Get baseline for this question
            if q_id not in baseline[model]:
                continue

            trial_adv_signals = []
            trial_base_signals = []
            trial_shifts = []

            for channel in ["wagering", "opt_out", "difficulty_selection", "tool_use", "natural"]:
                adv_result = trial["channels"].get(channel, {})
                base_parsed = baseline[model][q_id].get(channel, {})

                if not adv_result.get("api_call_success"):
                    continue

                adv_parsed = adv_result.get("parsed", {})

                shift = compute_channel_shift(adv_parsed, base_parsed, channel)

                if shift is not None:
                    channel_shifts[channel].append(shift)
                    all_channel_shifts[channel].append(shift)
                    trial_shifts.append(shift)

                # Extract signals for MCI
                adv_sig = extract_confidence_signal(adv_parsed, channel)
                base_sig = extract_confidence_signal(base_parsed, channel)

                trial_adv_signals.append(adv_sig)
                trial_base_signals.append(base_sig)

            # Store for MCI computation
            if len([s for s in trial_adv_signals if s is not None]) >= 3:
                adv_signals.append(trial_adv_signals)
            if len([s for s in trial_base_signals if s is not None]) >= 3:
                base_signals.append(trial_base_signals)

            # Aggregate shifts by domain type
            if trial_shifts:
                avg_shift = np.mean([abs(s) for s in trial_shifts])
                if domain_type == "weak":
                    weak_shifts.append(avg_shift)
                else:
                    strong_shifts.append(avg_shift)

        # Compute mean channel shifts and ARS
        mean_channel_shifts = {}
        ars_by_channel = {}

        for channel, shifts in channel_shifts.items():
            mean_shift = np.mean(shifts)
            mean_channel_shifts[channel] = mean_shift

            # ARS = 1 - |mean_shift| (since max_possible_shift = 1.0)
            ars_by_channel[channel] = 1.0 - abs(mean_shift)

        # Compute MCI
        mci_adv = compute_mci(adv_signals) if adv_signals else None
        mci_base = compute_mci(base_signals) if base_signals else None
        mci_drop = (mci_base - mci_adv) if (mci_adv is not None and mci_base is not None) else None

        attacks_analysis[attack_type] = {
            "channel_shifts": mean_channel_shifts,
            "ars": ars_by_channel,
            "mci_adversarial": mci_adv,
            "mci_baseline": mci_base,
            "mci_drop": mci_drop,
            "weak_domain_shift": np.mean(weak_shifts) if weak_shifts else None,
            "strong_domain_shift": np.mean(strong_shifts) if strong_shifts else None,
            "n_trials": len(attack_trials),
        }

    # Cross-attack consistency: std dev of channel shifts across attacks
    consistency_by_channel = {}
    for channel in ["wagering", "opt_out", "difficulty_selection", "tool_use", "natural"]:
        channel_means = [
            attacks_analysis[atk]["channel_shifts"].get(channel)
            for atk in attacks_analysis
            if channel in attacks_analysis[atk]["channel_shifts"]
        ]
        if len(channel_means) >= 2:
            consistency_by_channel[channel] = float(np.std(channel_means))

    cross_attack_consistency = np.mean(list(consistency_by_channel.values())) if consistency_by_channel else None

    # Overall ARS: mean across all channels and attacks
    all_ars = [
        ars
        for attack_data in attacks_analysis.values()
        for ars in attack_data["ars"].values()
    ]
    overall_ars = np.mean(all_ars) if all_ars else None

    return {
        "attacks": attacks_analysis,
        "cross_attack_consistency": cross_attack_consistency,
        "consistency_by_channel": consistency_by_channel,
        "overall_ars": overall_ars,
    }


def load_clean_baseline(clean_file: Path) -> dict:
    """Load Exp5 clean control results as baseline (alternative to Exp1 baseline)."""
    baseline = defaultdict(lambda: defaultdict(dict))
    with open(clean_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            model = result["model"]
            q_id = result["question_id"]
            for channel, ch_data in result.get("channels", {}).items():
                if ch_data.get("api_call_success"):
                    baseline[model][q_id][channel] = ch_data.get("parsed", {})
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 5 results")
    parser.add_argument(
        "results_file",
        type=Path,
        nargs="?",
        help="Path to exp5_*_results.jsonl file (auto-detects latest if not provided)",
    )
    parser.add_argument(
        "--clean-baseline",
        type=Path,
        default=None,
        help="Path to exp5_clean_*_results.jsonl to use as baseline instead of Exp1",
    )

    args = parser.parse_args()

    # Auto-detect latest results file if not provided
    if args.results_file is None:
        results_dir = Path("data/results")
        exp5_files = sorted(
            results_dir.glob("exp5_*_results.jsonl"),
            key=lambda p: p.stat().st_mtime
        )
        if not exp5_files:
            print("ERROR: No Experiment 5 results found", file=sys.stderr)
            sys.exit(1)
        args.results_file = exp5_files[-1]
        print(f"Auto-detected: {args.results_file}")

    if not args.results_file.exists():
        print(f"ERROR: Results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("EXPERIMENT 5 ANALYSIS: ADVERSARIAL ROBUSTNESS")
    print("=" * 80)
    print(f"Results file: {args.results_file}")

    # Load data
    print("\nLoading data...")
    if args.clean_baseline:
        baseline = load_clean_baseline(args.clean_baseline)
        print(f"  Loaded clean baseline from {args.clean_baseline} for {len(baseline)} models")
    else:
        baseline = load_exp1_baseline()
        print(f"  Loaded Exp 1 baseline for {len(baseline)} models")

    trials = load_exp5_results(args.results_file)
    print(f"  Loaded {len(trials)} adversarial trials")

    models = sorted(set(t["model"] for t in trials))
    print(f"  Models: {', '.join(models)}")

    # Analyze each model
    print("\nAnalyzing adversarial robustness...\n")

    all_metrics = {}

    for model in models:
        print(f"Model: {model}")
        metrics = analyze_model(model, trials, baseline)

        if not metrics:
            print("  No valid data\n")
            continue

        all_metrics[model] = metrics

        # Print summary
        print(f"  Overall ARS: {metrics['overall_ars']:.3f}" if metrics['overall_ars'] else "  Overall ARS: N/A")
        print(f"  Cross-attack consistency: {metrics['cross_attack_consistency']:.3f}" if metrics['cross_attack_consistency'] else "  Cross-attack consistency: N/A")

        for attack_type, attack_data in metrics["attacks"].items():
            print(f"\n  Attack: {attack_type}")
            print(f"    Trials: {attack_data['n_trials']}")

            if attack_data["channel_shifts"]:
                print("    Channel shifts:")
                for channel, shift in sorted(attack_data["channel_shifts"].items()):
                    ars = attack_data["ars"].get(channel)
                    print(f"      {channel:20s}: {shift:+.3f}  (ARS: {ars:.3f})" if ars else f"      {channel:20s}: {shift:+.3f}")

            if attack_data["mci_drop"] is not None:
                print(f"    MCI drop: {attack_data['mci_drop']:.3f} ({attack_data['mci_baseline']:.3f} → {attack_data['mci_adversarial']:.3f})")

            print(f"    Weak domain shift: {attack_data['weak_domain_shift']:.3f}" if attack_data['weak_domain_shift'] is not None else "    Weak domain shift: N/A")
            print(f"    Strong domain shift: {attack_data['strong_domain_shift']:.3f}" if attack_data['strong_domain_shift'] is not None else "    Strong domain shift: N/A")

        print()

    # Save metrics
    output_file = args.results_file.parent / args.results_file.name.replace("_results.jsonl", "_metrics.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Metrics saved to: {output_file}")

    # Summary table
    print("\nSUMMARY: Adversarial Robustness Scores (ARS)")
    print("-" * 80)
    print(f"{'Model':<20s} {'Overall ARS':<15s} {'Consistency':<15s} {'Status':<15s}")
    print("-" * 80)

    for model, metrics in sorted(all_metrics.items()):
        ars = metrics.get("overall_ars")
        consistency = metrics.get("cross_attack_consistency")

        if ars is not None:
            if ars >= 0.8:
                status = "Robust ✓"
            elif ars >= 0.6:
                status = "Moderate"
            else:
                status = "Vulnerable"

            print(f"{model:<20s} {ars:<15.3f} {consistency:<15.3f} {status:<15s}" if consistency else f"{model:<20s} {ars:<15.3f} {'N/A':<15s} {status:<15s}")
        else:
            print(f"{model:<20s} {'N/A':<15s} {'N/A':<15s} {'No data':<15s}")

    print("=" * 80)


if __name__ == "__main__":
    main()
