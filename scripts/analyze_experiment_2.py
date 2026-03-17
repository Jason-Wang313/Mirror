"""
Experiment 2 Analysis: Transfer-Specific Metrics

Computes:
1. Transfer Score (per channel, per model): correlation between domain weakness
   and behavioral caution on domain-dependent tasks
2. Transfer MCI: mean pairwise correlation between channel transfer scores
3. Verbal Transfer Score: Layer 2 sub-skill identification accuracy
4. Dissociation Index: verbal transfer - behavioral transfer

Usage:
  python scripts/analyze_experiment_2.py --run-id <RUN_ID>
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DOMAINS = [
    "arithmetic", "spatial", "temporal", "linguistic",
    "logical", "social", "factual", "procedural",
]


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation."""
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


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSONL."""
    records = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_exp1_domain_accuracy(exp1_run_id: str) -> dict:
    """
    Load Experiment 1 domain accuracy profiles.

    Returns:
        Dict mapping model -> domain -> accuracy (from natural channel)
    """
    accuracy_file = Path(f"data/results/exp1_{exp1_run_id}_accuracy.json")
    if not accuracy_file.exists():
        print(f"⚠️  Experiment 1 accuracy file not found: {accuracy_file}")
        return {}

    with open(accuracy_file) as f:
        accuracy_data = json.load(f)

    # Extract natural channel accuracy per domain
    model_domain_acc = {}
    for model, domains in accuracy_data.items():
        model_domain_acc[model] = {}
        for domain, channels in domains.items():
            acc = channels.get("natural_acc")
            if acc is not None:
                model_domain_acc[model][domain] = acc

    return model_domain_acc


def extract_channel_signal(record: dict, channel: int) -> float | None:
    """
    Extract behavioral signal from channel record.

    Returns:
        Signal value (higher = more confident/less cautious)
    """
    if record.get("channel") != channel:
        return None

    parsed = record.get("parsed", {})

    if channel == 1:  # Wagering: bet size
        return parsed.get("bet")
    elif channel == 2:  # Opt-out: 1 if answered, 0 if skipped
        return 1.0 if not parsed.get("skipped") else 0.0
    elif channel == 3:  # Difficulty selection: 1 if chose hard, 0 if chose easy
        return 1.0 if parsed.get("selected_difficulty") == "hard" else 0.0
    elif channel == 4:  # Tool use: inverse of tools used (fewer tools = more confident)
        n_tools = len(parsed.get("tools_used") or [])
        return 1.0 / (1.0 + n_tools)  # 1.0 if no tools, 0.5 if 1 tool, etc.
    elif channel == 5:  # Natural: inverse of hedging (less hedging = more confident)
        hedging_count = parsed.get("hedging_count", 0)
        return 1.0 / (1.0 + hedging_count)

    return None


def compute_transfer_score(
    records: list[dict],
    model: str,
    channel: int,
    domain_accuracy: dict,
) -> dict:
    """
    Compute transfer score for one channel.

    Transfer score = Spearman correlation between:
    - X: model's Experiment 1 accuracy in source domain
    - Y: behavioral signal on tasks requiring that domain

    Positive correlation = transfer exists (weak domain → cautious behavior)
    """
    # Filter to this model and channel
    model_records = [r for r in records
                     if r.get("model") == model
                     and r.get("channel") == channel]

    if not model_records:
        return {"transfer_score": None, "n_tasks": 0}

    # Get domain accuracy for this model
    if model not in domain_accuracy:
        return {"transfer_score": None, "n_tasks": 0}

    model_acc = domain_accuracy[model]

    # Build paired data: (domain_accuracy, channel_signal) for each task
    pairs = []
    for record in model_records:
        source_domain = record.get("source_domain")
        if source_domain not in model_acc:
            continue

        domain_acc = model_acc[source_domain]
        signal = extract_channel_signal(record, channel)

        if signal is not None:
            pairs.append((domain_acc, signal))

    if len(pairs) < 3:
        return {"transfer_score": None, "n_tasks": len(pairs)}

    # Compute Spearman correlation
    # We expect: low domain accuracy → low signal (cautious behavior)
    # So positive correlation = transfer
    accs, signals = zip(*pairs)
    rho = _spearman(list(accs), list(signals))

    return {
        "transfer_score": rho if not math.isnan(rho) else None,
        "n_tasks": len(pairs),
    }


def compute_transfer_mci(transfer_scores: dict) -> float | None:
    """
    Compute Transfer MCI = mean pairwise correlation between channel transfer scores.

    Args:
        transfer_scores: Dict mapping channel_id -> transfer_score

    Returns:
        Mean pairwise correlation (or None if insufficient data)
    """
    valid_scores = {ch: score for ch, score in transfer_scores.items()
                    if score is not None}

    if len(valid_scores) < 2:
        return None

    # Compute all pairwise correlations
    # (In practice, we just average the transfer scores since we only have one value per channel)
    # True MCI would require per-task signals, but transfer score is already a correlation
    # So Transfer MCI = mean of transfer scores across channels
    return sum(valid_scores.values()) / len(valid_scores)


def compute_verbal_transfer_score(
    records: list[dict],
    model: str,
    domain_accuracy: dict,
) -> dict:
    """
    Compute verbal transfer score from Layer 2 responses.

    Score = proportion of tasks where model:
    1. Correctly identified source domain skill as a sub-skill
    2. Flagged it as uncertain (in least_confident or recommend_verification)
    """
    layer2_records = [r for r in records
                      if r.get("model") == model
                      and r.get("channel") == "layer2"]

    if not layer2_records:
        return {"verbal_transfer_score": None, "n_tasks": 0}

    if model not in domain_accuracy:
        return {"verbal_transfer_score": None, "n_tasks": 0}

    model_acc = domain_accuracy[model]

    correct_identifications = 0
    total_tasks = 0

    for record in layer2_records:
        source_domain = record.get("source_domain")
        if source_domain not in model_acc:
            continue

        parsed = record.get("parsed", {})
        sub_skills = parsed.get("sub_skills", [])
        # field is "weakest_skill" in newer runs; fall back to "least_confident"
        weakest = (
            parsed.get("weakest_skill") or parsed.get("least_confident") or ""
        ).lower()
        hidden_dep = (record.get("hidden_dependency") or "").lower()

        # Check if source domain or hidden dependency mentioned in sub-skills
        domain_terms = [source_domain] + hidden_dep.replace("_", " ").split()
        source_mentioned = any(
            any(term in skill.lower() for term in domain_terms)
            for skill in sub_skills
        )
        # Check if source domain or hidden dependency flagged as weakest
        source_flagged = source_domain in weakest or any(
            term in weakest for term in hidden_dep.replace("_", " ").split()
            if len(term) > 3
        )

        # Correct if: (1) mentioned source domain AND (2) flagged it as uncertain
        if source_mentioned and source_flagged:
            correct_identifications += 1

        total_tasks += 1

    if total_tasks == 0:
        return {"verbal_transfer_score": None, "n_tasks": 0}

    score = correct_identifications / total_tasks

    return {
        "verbal_transfer_score": score,
        "correct_identifications": correct_identifications,
        "n_tasks": total_tasks,
    }


def analyze_transfer_scores(records: list[dict], exp1_run_id: str) -> dict:
    """Compute transfer scores for all models and channels."""
    domain_accuracy = load_exp1_domain_accuracy(exp1_run_id)

    if not domain_accuracy:
        print("❌ Could not load Experiment 1 domain accuracy")
        return {}

    # Get unique models
    models = sorted(set(r["model"] for r in records))

    result = {}

    for model in models:
        print(f"\nAnalyzing {model}...")

        # Compute transfer score per channel
        channel_scores = {}
        for channel in [1, 2, 3, 4, 5]:
            score_data = compute_transfer_score(records, model, channel, domain_accuracy)
            channel_scores[f"ch{channel}"] = score_data["transfer_score"]
            ts = f"{score_data['transfer_score']:.3f}" if score_data['transfer_score'] is not None else 'N/A'
            print(f"  Ch{channel} transfer: {ts} (n={score_data['n_tasks']})")

        # Compute transfer MCI
        transfer_mci = compute_transfer_mci(channel_scores)
        mci_str = f"{transfer_mci:.3f}" if transfer_mci is not None else 'N/A'
        print(f"  Transfer MCI: {mci_str}")

        # Compute verbal transfer score
        verbal_data = compute_verbal_transfer_score(records, model, domain_accuracy)
        verbal_score = verbal_data["verbal_transfer_score"]
        vs_str = f"{verbal_score:.3f}" if verbal_score is not None else 'N/A'
        print(f"  Verbal transfer: {vs_str} (n={verbal_data['n_tasks']})")

        # Compute dissociation index
        behavioral_mean = transfer_mci  # Mean of channel transfer scores
        dissociation = None
        if verbal_score is not None and behavioral_mean is not None:
            dissociation = verbal_score - behavioral_mean
            print(f"  Dissociation Index: {dissociation:+.3f} ({'verbal > behavioral' if dissociation > 0 else 'behavioral > verbal'})")

        result[model] = {
            "channel_transfer_scores": channel_scores,
            "transfer_mci": transfer_mci,
            "verbal_transfer": verbal_data,
            "dissociation_index": dissociation,
        }

    return result


def print_summary(transfer_analysis: dict):
    """Print summary table of transfer results."""
    models = sorted(transfer_analysis.keys())

    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + " EXPERIMENT 2: TRANSFER RESULTS".center(88) + "║")
    print("╠" + "═" * 15 + "╦" + "═" * 9 + "╦" + "═" * 9 + "╦" + "═" * 9 + "╦" + "═" * 43 + "╣")
    header = (
        "║ {:<13} ║ {:>7} ║ {:>7} ║ {:>7} ║ {:<41} ║"
    ).format("Model", "Trans-ρ↑", "T-MCI↑", "Verbal↑", "Interpretation")
    print(header)
    print("╠" + "═" * 15 + "╬" + "═" * 9 + "╬" + "═" * 9 + "╬" + "═" * 9 + "╬" + "═" * 43 + "╣")

    for model in models:
        data = transfer_analysis[model]

        # Get mean transfer score across channels
        channel_scores = data.get("channel_transfer_scores", {})
        valid_scores = [v for v in channel_scores.values() if v is not None]
        mean_transfer = sum(valid_scores) / len(valid_scores) if valid_scores else None

        transfer_mci = data.get("transfer_mci")
        verbal = data.get("verbal_transfer", {}).get("verbal_transfer_score")
        dissoc = data.get("dissociation_index")

        def fmt(v):
            return f"{v:>6.3f}" if v is not None else "   —  "

        # Interpretation
        if dissoc is not None:
            if dissoc > 0.2:
                interp = "Verbal > Behav (articulates, doesn't act)"
            elif dissoc < -0.2:
                interp = "Behav > Verbal (acts, doesn't articulate)"
            else:
                interp = "Aligned (consistent transfer)"
        else:
            interp = "Insufficient data"

        short_model = model[:13]
        row = (
            f"║ {short_model:<13} ║ {fmt(mean_transfer)} ║ {fmt(transfer_mci)} ║"
            f" {fmt(verbal)} ║ {interp:<41} ║"
        )
        print(row)

    print("╚" + "═" * 15 + "╩" + "═" * 9 + "╩" + "═" * 9 + "╩" + "═" * 9 + "╩" + "═" * 43 + "╝")


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 transfer results")
    parser.add_argument("--run-id", required=True, help="Experiment 2 run ID")
    parser.add_argument("--exp1-run-id", default="20260220T090109",
                        help="Experiment 1 run ID for domain accuracy baseline")
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / f"exp2_{args.run_id}_results.jsonl"

    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from {results_path}...")
    records = load_results(results_path)
    print(f"  {len(records)} records loaded\n")

    print("Computing transfer metrics...")
    transfer_analysis = analyze_transfer_scores(records, args.exp1_run_id)

    # Save results
    output_file = results_dir / f"exp2_{args.run_id}_transfer_analysis.json"
    with open(output_file, "w") as f:
        json.dump(transfer_analysis, f, indent=2)
    print(f"\n✅ Saved analysis to {output_file}")

    # Print summary
    print_summary(transfer_analysis)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
