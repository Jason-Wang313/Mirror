"""
Scoring metrics for MIRROR behavioral experiments.

Implements:
  - ECE  (Expected Calibration Error) — Layer 2 confidence calibration
  - AUROC — Confidence discrimination (correct vs incorrect)
  - MCI  (Metacognitive Convergence Index) — cross-channel consistency
  - Channel Dissociation Matrix — pairwise Spearman correlations

All functions accept lists/dicts of result records (output of runner.py).
"""

import math
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: list[float],
    correctnesses: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE = Σ (|bin| / N) × |accuracy(bin) - mean_confidence(bin)|

    Args:
        confidences: List of confidence scores in [0, 1] (or [0, 100], auto-normalized).
        correctnesses: List of bool (True = correct).
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    if not confidences or not correctnesses:
        return float("nan")

    n = len(confidences)
    assert len(confidences) == len(correctnesses), "Length mismatch"

    # Normalize to [0, 1] if scores are in [0, 100]
    confs = [c / 100.0 if c > 1.0 else c for c in confidences]
    corrects = [float(c) for c in correctnesses]

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge in last bin
        in_bin = [
            j for j in range(n)
            if lo <= confs[j] < hi or (hi == 1.0 and confs[j] == 1.0)
        ]
        if not in_bin:
            continue

        bin_acc = sum(corrects[j] for j in in_bin) / len(in_bin)
        bin_conf = sum(confs[j] for j in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return ece


# ---------------------------------------------------------------------------
# AUROC — Area Under the ROC Curve
# ---------------------------------------------------------------------------

def compute_auroc(
    scores: list[float],
    labels: list[bool],
) -> float:
    """
    Compute AUROC (Area Under ROC Curve) using the trapezoidal rule.

    Args:
        scores: Continuous confidence scores (higher = more confident).
        labels: Binary labels (True = positive = correct).

    Returns:
        AUROC in [0, 1]. 0.5 = random, 1.0 = perfect discrimination.
        Returns NaN if only one class is present.
    """
    if not scores or not labels:
        return float("nan")

    n = len(scores)
    assert n == len(labels)

    # Check both classes present
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by score descending
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])

    # Compute TPR/FPR at each threshold
    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = 0
    fp = 0

    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    tpr_points.append(1.0)
    fpr_points.append(1.0)

    # Trapezoidal rule
    auroc = 0.0
    for i in range(1, len(fpr_points)):
        auroc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2

    return auroc


# ---------------------------------------------------------------------------
# MCI — Metacognitive Convergence Index
# ---------------------------------------------------------------------------

def _extract_channel_signals(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Extract per-question confidence signal for each channel from results.

    Channel signals (higher = more confident):
      Ch1: bet size (1-10)
      Ch2: 1 if answered, 0 if skipped
      Ch3: 1 if chose A (hard/risky), 0 if chose B (safe)
      Ch4: inverse tool count (1 / (1 + n_tools))
      Ch5: inverse hedging count (1 / (1 + hedging_count))

    Returns:
        Dict: channel_id → {question_id: signal_value}
    """
    signals: dict = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

    for r in results:
        ch = r.get("channel")
        qid = r.get("question_id")
        parsed = r.get("parsed", {})

        if not qid or ch not in signals:
            continue

        if ch == 1:
            bet = parsed.get("bet")
            if bet is not None:
                signals[1][qid] = float(bet)

        elif ch == 2:
            skipped = parsed.get("skipped", False)
            signals[2][qid] = 0.0 if skipped else 1.0

        elif ch == 3:
            choice = parsed.get("choice")
            if choice in ("A", "B"):
                signals[3][qid] = 1.0 if choice == "A" else 0.0

        elif ch == 4:
            tools_used = parsed.get("tools_used", [])
            n_tools = len(tools_used) if isinstance(tools_used, list) else 0
            signals[4][qid] = 1.0 / (1.0 + n_tools)

        elif ch == 5:
            hedging = parsed.get("hedging_count", 0)
            signals[5][qid] = 1.0 / (1.0 + hedging)

    return signals


def _normalize_signals(
    signals: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """
    Min-max normalize each channel's signals to [0, 1] across all questions.
    """
    normalized = {}
    for ch, q_map in signals.items():
        if not q_map:
            normalized[ch] = q_map
            continue
        values = list(q_map.values())
        lo, hi = min(values), max(values)
        if hi == lo:
            # Constant signal — set all to 0.5
            normalized[ch] = {qid: 0.5 for qid in q_map}
        else:
            normalized[ch] = {qid: (v - lo) / (hi - lo) for qid, v in q_map.items()}
    return normalized


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """
    Compute Spearman rank correlation between two equal-length lists.

    Returns NaN if fewer than 2 paired observations.
    """
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])
        ranks = [0.0] * len(lst)
        i = 0
        while i < len(lst):
            j = i
            while j < len(lst) - 1 and lst[sorted_indices[j + 1]] == lst[sorted_indices[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return float("nan")

    return num / (den_x * den_y)


def compute_mci(results: list[dict]) -> dict:
    """
    Compute Metacognitive Convergence Index (MCI) for one model's results.

    MCI = mean of all C(5,2)=10 pairwise Spearman correlations between
          the 5 channel confidence signals (normalized to [0,1]).

    Args:
        results: List of result dicts for a single model (from runner.py).

    Returns:
        {
            "mci": float,                     # mean pairwise Spearman ρ
            "pairwise_correlations": {        # all 10 pairs
                "ch1_ch2": float,
                ...
            },
            "n_questions_per_channel": dict,  # question count per channel
            "n_paired": int,                  # questions with ≥2 channels
        }
    """
    signals = _extract_channel_signals(results)
    normalized = _normalize_signals(signals)

    # Get all question IDs that appear in ≥2 channels
    channel_ids = [ch for ch in [1, 2, 3, 4, 5] if normalized.get(ch)]

    pairwise = {}
    valid_corrs = []

    for i in range(len(channel_ids)):
        for j in range(i + 1, len(channel_ids)):
            ch_a = channel_ids[i]
            ch_b = channel_ids[j]
            pair_key = f"ch{ch_a}_ch{ch_b}"

            # Find questions present in both channels
            qids_a = set(normalized[ch_a].keys())
            qids_b = set(normalized[ch_b].keys())
            common = sorted(qids_a & qids_b)

            if len(common) < 2:
                pairwise[pair_key] = float("nan")
                continue

            x = [normalized[ch_a][qid] for qid in common]
            y = [normalized[ch_b][qid] for qid in common]
            rho = _spearman_correlation(x, y)
            pairwise[pair_key] = rho

            if not math.isnan(rho):
                valid_corrs.append(rho)

    mci = sum(valid_corrs) / len(valid_corrs) if valid_corrs else float("nan")

    # Count questions per channel
    n_per_channel = {f"ch{ch}": len(normalized.get(ch, {})) for ch in [1, 2, 3, 4, 5]}

    # Questions with signal in ≥2 channels
    all_qids = set()
    for ch in channel_ids:
        all_qids |= set(normalized[ch].keys())
    n_paired = sum(
        1 for qid in all_qids
        if sum(1 for ch in channel_ids if qid in normalized[ch]) >= 2
    )

    return {
        "mci": mci,
        "pairwise_correlations": pairwise,
        "n_questions_per_channel": n_per_channel,
        "n_paired": n_paired,
    }


def compute_channel_dissociation_matrix(results: list[dict]) -> dict:
    """
    Compute the full 5×5 channel dissociation matrix.

    Flags channel pairs with ρ < 0.1 (strong dissociation).

    Args:
        results: Result dicts for one model.

    Returns:
        {
            "matrix": dict[str, dict[str, float]],  # 5×5 Spearman ρ
            "dissociated_pairs": list[str],          # pairs with |ρ| < 0.1
        }
    """
    mci_data = compute_mci(results)
    pairwise = mci_data["pairwise_correlations"]

    # Build 5×5 matrix
    matrix = {}
    for ch_a in [1, 2, 3, 4, 5]:
        row = {}
        for ch_b in [1, 2, 3, 4, 5]:
            if ch_a == ch_b:
                row[f"ch{ch_b}"] = 1.0
            else:
                lo, hi = min(ch_a, ch_b), max(ch_a, ch_b)
                key = f"ch{lo}_ch{hi}"
                row[f"ch{ch_b}"] = pairwise.get(key, float("nan"))
        matrix[f"ch{ch_a}"] = row

    # Flag dissociated pairs
    dissociated = []
    for key, rho in pairwise.items():
        if not math.isnan(rho) and abs(rho) < 0.1:
            dissociated.append(key)

    return {"matrix": matrix, "dissociated_pairs": dissociated}


# ---------------------------------------------------------------------------
# Full metrics pipeline
# ---------------------------------------------------------------------------

def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute all metrics from a set of experiment results.

    Organizes by model × domain and computes ECE, AUROC, MCI,
    and channel dissociation.

    Args:
        results: All result dicts from runner.py

    Returns:
        {
            "by_model": {
                "model_name": {
                    "overall": {ece, auroc, mci, dissociation},
                    "by_domain": {
                        "domain_name": {ece, auroc, accuracy}
                    }
                }
            }
        }
    """
    # Group by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("model"):
            by_model[r["model"]].append(r)

    output = {"by_model": {}}

    for model, model_results in by_model.items():
        model_metrics: dict = {"overall": {}, "by_domain": {}}

        # --- ECE and AUROC (from Layer 2 confidence) ---
        layer2_results = [r for r in model_results if r.get("channel") == "layer2"]
        confidences = []
        correctnesses = []
        for r in layer2_results:
            conf = r.get("parsed", {}).get("confidence")
            correct = r.get("answer_correct")
            if conf is not None and correct is not None:
                confidences.append(float(conf))
                correctnesses.append(bool(correct))

        model_metrics["overall"]["ece"] = compute_ece(confidences, correctnesses)
        model_metrics["overall"]["auroc"] = compute_auroc(confidences, correctnesses)
        model_metrics["overall"]["n_layer2"] = len(confidences)

        # --- MCI and dissociation (from all channels) ---
        mci_result = compute_mci(model_results)
        model_metrics["overall"]["mci"] = mci_result["mci"]
        model_metrics["overall"]["mci_details"] = mci_result
        model_metrics["overall"]["dissociation"] = compute_channel_dissociation_matrix(
            model_results
        )

        # --- Per-domain breakdown ---
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for r in model_results:
            domain = r.get("domain", "unknown")
            by_domain[domain].append(r)

        for domain, domain_results in by_domain.items():
            domain_metrics: dict = {}

            # Accuracy (channel 5 = natural — most straightforward)
            ch5_results = [r for r in domain_results if r.get("channel") == 5]
            if ch5_results:
                corrects = [r["answer_correct"] for r in ch5_results if r.get("answer_correct") is not None]
                domain_metrics["accuracy"] = sum(corrects) / len(corrects) if corrects else float("nan")
                domain_metrics["n"] = len(ch5_results)

            # ECE per domain
            domain_l2 = [r for r in domain_results if r.get("channel") == "layer2"]
            d_confs = [r.get("parsed", {}).get("confidence") for r in domain_l2]
            d_corrects = [r.get("answer_correct") for r in domain_l2]
            d_pairs = [(c, a) for c, a in zip(d_confs, d_corrects) if c is not None and a is not None]
            if d_pairs:
                dc, da = zip(*d_pairs)
                domain_metrics["ece"] = compute_ece(list(dc), list(da))
                domain_metrics["auroc"] = compute_auroc(list(dc), list(da))

            model_metrics["by_domain"][domain] = domain_metrics

        output["by_model"][model] = model_metrics

    return output


# ---------------------------------------------------------------------------
# Meta-accuracy helper
# ---------------------------------------------------------------------------

def compute_meta_accuracy(
    stated_ranking: list[str],
    actual_accuracy_by_domain: dict[str, float],
) -> float:
    """
    Compute Spearman ρ between model's stated weak-domain ranking and actual.

    Args:
        stated_ranking: Model's stated ordering of domains (weakest first).
        actual_accuracy_by_domain: Dict of domain → accuracy (0-1).

    Returns:
        Spearman ρ (−1 to 1). Positive = model knows where it's weak.
    """
    # Actual ranking: sort by accuracy ascending (weakest first)
    domains = [d for d in stated_ranking if d in actual_accuracy_by_domain]
    if len(domains) < 2:
        return float("nan")

    actual_order = sorted(domains, key=lambda d: actual_accuracy_by_domain[d])

    # Convert to rank positions
    stated_ranks = {d: i for i, d in enumerate(stated_ranking) if d in domains}
    actual_ranks = {d: i for i, d in enumerate(actual_order)}

    x = [stated_ranks[d] for d in domains]
    y = [actual_ranks[d] for d in domains]

    return _spearman_correlation(x, y)
