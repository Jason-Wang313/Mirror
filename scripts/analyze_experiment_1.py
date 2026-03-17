"""
Experiment 1 Analysis
=====================

Reads raw results JSONL, computes all calibration metrics, and produces
seven output JSON files plus a formatted console summary table.

Usage:
  python scripts/analyze_experiment_1.py --run-id <run_id>
  python scripts/analyze_experiment_1.py --run-id <run_id> --tables-only
  python scripts/analyze_experiment_1.py --run-id <run_id> --export-latex
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.scoring.metrics import compute_ece, compute_mci, compute_channel_dissociation_matrix

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

DOMAINS = [
    "arithmetic", "spatial", "temporal", "linguistic",
    "logical", "social", "factual", "procedural",
]

CHANNELS = [
    ("wagering", 1),
    ("opt_out", 2),
    ("difficulty_selection", 3),
    ("tool_use", 4),
    ("natural", 5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> list[dict]:
    records = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation. Returns NaN if degenerate."""
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

    rx, ry = rank(xs), rank(ry_ := ys)  # noqa: assignment below
    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _group_by(records: list[dict], key: str) -> dict:
    out: dict = defaultdict(list)
    for r in records:
        out[r.get(key)].append(r)
    return dict(out)


def _accuracy(records: list[dict]) -> float | None:
    graded = [r for r in records if r.get("answer_correct") is not None]
    if not graded:
        return None
    return sum(1 for r in graded if r["answer_correct"]) / len(graded)


# ---------------------------------------------------------------------------
# Analysis A: Per-model, per-domain accuracy table
# ---------------------------------------------------------------------------

def _auroc_wagering(records: list[dict]) -> float | None:
    """Compute AUROC using wagering bet as confidence score."""
    if roc_auc_score is None:
        return None

    ch_recs = [r for r in records
               if r.get("channel") == 1
               and r.get("parsed")
               and r["parsed"].get("bet") is not None
               and r.get("answer_correct") is not None]

    if len(ch_recs) < 2:
        return None

    y_true = [int(r["answer_correct"]) for r in ch_recs]
    y_score = [r["parsed"]["bet"] for r in ch_recs]

    # Handle edge case: all same label (AUROC undefined)
    if len(set(y_true)) < 2:
        return None

    try:
        return roc_auc_score(y_true, y_score)
    except (ValueError, ZeroDivisionError):
        return None


def analyze_accuracy(records: list[dict]) -> dict:
    """For each (model, domain): accuracy per channel + AUROC."""
    by_model = _group_by(records, "model")
    result = {}
    for model, model_records in by_model.items():
        result[model] = {}
        by_domain = _group_by(model_records, "domain")
        for domain, dom_records in by_domain.items():
            entry = {}
            for ch_name, ch_id in CHANNELS:
                ch_recs = [r for r in dom_records if r.get("channel") == ch_id]
                acc = _accuracy(ch_recs)
                entry[f"{ch_name}_acc"] = acc
            # Add AUROC for wagering channel
            entry["wagering_auroc"] = _auroc_wagering(dom_records)
            result[model][domain] = entry
    return result


# ---------------------------------------------------------------------------
# Analysis B: Channel-specific calibration metrics
# ---------------------------------------------------------------------------

def _ece_for_channel(records: list[dict], ch_id: int) -> float | None:
    ch_recs = [r for r in records if r.get("channel") == ch_id]
    if not ch_recs:
        return None
    if ch_id == 1:  # wagering: bet 1-10
        # Filter once so confidences and correctnesses stay aligned
        graded = [r for r in ch_recs
                  if r.get("parsed")
                  and r["parsed"].get("bet") is not None
                  and r.get("answer_correct") is not None]
        confidences = [r["parsed"]["bet"] * 10 for r in graded]
        correctnesses = [r["answer_correct"] for r in graded]
    else:
        return None
    if not confidences:
        return None
    return compute_ece(confidences, correctnesses)


def _wager_acc_spearman(records: list[dict]) -> float | None:
    ch_recs = [r for r in records
               if r.get("channel") == 1
               and r.get("parsed")
               and r["parsed"].get("bet") is not None
               and r.get("answer_correct") is not None]
    if len(ch_recs) < 3:
        return None
    bets = [r["parsed"]["bet"] for r in ch_recs]
    corrects = [float(r["answer_correct"]) for r in ch_recs]
    return _spearman(bets, corrects)


def _skip_error_alignment(records: list[dict]) -> float | None:
    """Proportion of skips where the model would have erred (estimated from other channels)."""
    ch2_recs = [r for r in records if r.get("channel") == 2]
    if not ch2_recs:
        return None
    skipped_qids = {r["question_id"] for r in ch2_recs if r.get("parsed", {}).get("skipped")}
    if not skipped_qids:
        return None
    # Estimate accuracy on those questions from Channel 5
    ch5_on_skipped = [r for r in records
                      if r.get("channel") == 5
                      and r["question_id"] in skipped_qids
                      and r.get("answer_correct") is not None]
    if not ch5_on_skipped:
        return None
    # Alignment = proportion that model would have got wrong
    would_error = sum(1 for r in ch5_on_skipped if not r["answer_correct"])
    return would_error / len(ch5_on_skipped)


def _tool_accuracy_correlation(records: list[dict]) -> float | None:
    """Spearman between n_tools_used and whether model gets question wrong without tools."""
    ch4_recs = [r for r in records
                if r.get("channel") == 4
                and r.get("parsed") is not None]
    ch5_by_qid = {r["question_id"]: r for r in records if r.get("channel") == 5}
    pairs = []
    for r in ch4_recs:
        ch5 = ch5_by_qid.get(r["question_id"])
        if ch5 is None or ch5.get("answer_correct") is None:
            continue
        n_tools = len(r["parsed"].get("tools_used", []) or [])
        baseline_wrong = float(not ch5["answer_correct"])
        pairs.append((n_tools, baseline_wrong))
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return _spearman(list(xs), list(ys))


def _length_accuracy_spearman(records: list[dict]) -> float | None:
    ch5_recs = [r for r in records
                if r.get("channel") == 5
                and r.get("parsed") is not None
                and r.get("answer_correct") is not None]
    if len(ch5_recs) < 3:
        return None
    lengths = [(r["parsed"].get("response_length") or 0) for r in ch5_recs]
    corrects = [float(r["answer_correct"]) for r in ch5_recs]
    return _spearman(lengths, corrects)


def _hedge_accuracy_spearman(records: list[dict]) -> float | None:
    ch5_recs = [r for r in records
                if r.get("channel") == 5
                and r.get("parsed") is not None
                and r.get("answer_correct") is not None]
    if len(ch5_recs) < 3:
        return None
    hedges = [(r["parsed"].get("hedging_count") or 0) for r in ch5_recs]
    incorrects = [float(not r["answer_correct"]) for r in ch5_recs]
    return _spearman(hedges, incorrects)


def analyze_calibration(records: list[dict]) -> dict:
    """Per-(model, domain) channel-specific calibration metrics."""
    by_model = _group_by(records, "model")
    result = {}
    for model, model_records in by_model.items():
        result[model] = {}
        # Overall (all domains)
        result[model]["overall"] = {
            "wagering_ece": _ece_for_channel(model_records, 1),
            "wagering_spearman": _wager_acc_spearman(model_records),
            "skip_error_alignment": _skip_error_alignment(model_records),
            "tool_accuracy_correlation": _tool_accuracy_correlation(model_records),
            "length_accuracy_spearman": _length_accuracy_spearman(model_records),
            "hedge_accuracy_spearman": _hedge_accuracy_spearman(model_records),
        }
        by_domain = _group_by(model_records, "domain")
        for domain, dom_records in by_domain.items():
            result[model][domain] = {
                "wagering_ece": _ece_for_channel(dom_records, 1),
                "wagering_spearman": _wager_acc_spearman(dom_records),
                "skip_error_alignment": _skip_error_alignment(dom_records),
                "tool_accuracy_correlation": _tool_accuracy_correlation(dom_records),
                "length_accuracy_spearman": _length_accuracy_spearman(dom_records),
                "hedge_accuracy_spearman": _hedge_accuracy_spearman(dom_records),
            }
    return result


# ---------------------------------------------------------------------------
# Analysis C: Metacognitive Convergence Index (MCI)
# ---------------------------------------------------------------------------

def _partial_spearman(xs: list[float], ys: list[float], zs: list[float]) -> float:
    """
    Compute partial Spearman correlation between xs and ys, controlling for zs.

    Formula: ρ_xy|z = (ρ_xy - ρ_xz*ρ_yz) / sqrt((1-ρ_xz²)*(1-ρ_yz²))
    """
    rho_xy = _spearman(xs, ys)
    rho_xz = _spearman(xs, zs)
    rho_yz = _spearman(ys, zs)

    if math.isnan(rho_xy) or math.isnan(rho_xz) or math.isnan(rho_yz):
        return float("nan")

    denominator = math.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denominator == 0:
        return float("nan")

    return (rho_xy - rho_xz * rho_yz) / denominator


def _compute_difficulty_adjusted_mci(model_records: list[dict], q_difficulty_est: dict[str, float]) -> dict:
    """
    Compute difficulty-adjusted MCI using partial correlations.

    For each channel pair, compute partial Spearman controlling for cross-model difficulty.
    """
    # Group by channel
    by_channel: dict[int, list[dict]] = defaultdict(list)
    for r in model_records:
        ch = r.get("channel")
        if ch in [1, 2, 3, 4, 5] and r.get("answer_correct") is not None:
            by_channel[ch].append(r)

    # Build question-level correctness per channel
    ch_correctness: dict[int, dict[str, float]] = {}
    for ch, ch_recs in by_channel.items():
        ch_correctness[ch] = {r["question_id"]: float(r["answer_correct"]) for r in ch_recs}

    # Compute adjusted pairwise correlations
    adjusted_pairwise = {}
    channel_ids = sorted(by_channel.keys())

    for i, ch_a in enumerate(channel_ids):
        for ch_b in channel_ids[i + 1:]:
            # Find paired questions
            qids_a = set(ch_correctness[ch_a].keys())
            qids_b = set(ch_correctness[ch_b].keys())
            paired_qids = sorted(qids_a & qids_b)

            # Filter to questions with difficulty estimate
            paired_qids = [qid for qid in paired_qids if qid in q_difficulty_est]

            if len(paired_qids) < 3:
                continue

            xs = [ch_correctness[ch_a][qid] for qid in paired_qids]
            ys = [ch_correctness[ch_b][qid] for qid in paired_qids]
            zs = [q_difficulty_est[qid] for qid in paired_qids]

            rho_adj = _partial_spearman(xs, ys, zs)
            ch_a_name = [name for name, cid in CHANNELS if cid == ch_a][0] if ch_a in [c[1] for c in CHANNELS] else str(ch_a)
            ch_b_name = [name for name, cid in CHANNELS if cid == ch_b][0] if ch_b in [c[1] for c in CHANNELS] else str(ch_b)

            key = f"{ch_a_name}_{ch_b_name}"
            adjusted_pairwise[key] = rho_adj if not math.isnan(rho_adj) else None

    # Compute adjusted MCI as mean of all pairwise adjusted correlations
    valid_rhos = [v for v in adjusted_pairwise.values() if v is not None]
    mci_adj = sum(valid_rhos) / len(valid_rhos) if valid_rhos else None

    return {
        "mci_difficulty_adjusted": mci_adj,
        "adjusted_pairwise": adjusted_pairwise,
    }


def analyze_mci(records: list[dict]) -> dict:
    """Per-model MCI and pairwise channel correlations."""
    by_model = _group_by(records, "model")
    result = {}

    # Difficulty estimate: cross-model average accuracy per question (from natural channel)
    q_difficulty: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("channel") == 5 and r.get("answer_correct") is not None:
            q_difficulty[r["question_id"]].append(float(r["answer_correct"]))
    q_difficulty_est = {
        qid: sum(vs) / len(vs) for qid, vs in q_difficulty.items() if vs
    }

    for model, model_records in by_model.items():
        mci_data = compute_mci(model_records)
        adjusted_data = _compute_difficulty_adjusted_mci(model_records, q_difficulty_est)

        result[model] = {
            "mci_raw": mci_data.get("mci"),
            "mci_difficulty_adjusted": adjusted_data["mci_difficulty_adjusted"],
            "pairwise": mci_data.get("pairwise_correlations", {}),
            "adjusted_pairwise": adjusted_data["adjusted_pairwise"],
            "n_questions_per_channel": mci_data.get("n_questions_per_channel", {}),
            "n_paired": mci_data.get("n_paired"),
        }
    return result


# ---------------------------------------------------------------------------
# Analysis D: Dissociation matrix
# ---------------------------------------------------------------------------

def analyze_dissociation(records: list[dict]) -> dict:
    """Per-model 5×5 channel dissociation matrix."""
    by_model = _group_by(records, "model")
    result = {}
    for model, model_records in by_model.items():
        matrix_data = compute_channel_dissociation_matrix(model_records)
        result[model] = {
            "matrix": matrix_data.get("matrix", {}),
            "dissociated_pairs": matrix_data.get("dissociated_pairs", []),
        }
    return result


# ---------------------------------------------------------------------------
# Analysis E: Layer 2 vs Layer 1 comparison
# ---------------------------------------------------------------------------

def analyze_layer_comparison(records: list[dict]) -> dict:
    """Compare Layer 2 verbal confidence vs Layer 1 wagering bet calibration."""
    by_model = _group_by(records, "model")
    result = {}
    for model, model_records in by_model.items():
        # Layer 2 records
        l2_recs = [r for r in model_records
                   if r.get("channel") == "layer2" or r.get("channel_name") == "layer2"]
        # Layer 1 wagering records
        ch1_recs = [r for r in model_records if r.get("channel") == 1]

        # Build per-question lookups
        l2_by_qid = {r["question_id"]: r for r in l2_recs}
        ch1_by_qid = {r["question_id"]: r for r in ch1_recs}

        # L2 confidence vs L1 bet (paired by question)
        l2_conf, l1_bet, corrects = [], [], []
        for qid, l2r in l2_by_qid.items():
            conf = l2r.get("parsed", {}).get("confidence")
            ch1r = ch1_by_qid.get(qid)
            if conf is None or ch1r is None:
                continue
            bet = ch1r.get("parsed", {}).get("bet")
            correct = ch1r.get("answer_correct")
            if bet is None or correct is None:
                continue
            l2_conf.append(float(conf))
            l1_bet.append(float(bet))
            corrects.append(float(correct))

        entry: dict = {}
        if len(l2_conf) >= 3:
            entry["l2_vs_l1_spearman"] = _spearman(l2_conf, l1_bet)
            entry["l2_vs_accuracy_spearman"] = _spearman(l2_conf, corrects)
            entry["l1_vs_accuracy_spearman"] = _spearman(l1_bet, corrects)
            l2_better = (
                (entry["l2_vs_accuracy_spearman"] or 0) >
                (entry["l1_vs_accuracy_spearman"] or 0)
            )
            entry["l2_better_calibrated"] = l2_better
            entry["interpretation"] = (
                "verbal confidence (L2) better calibrated than wagering (L1)"
                if l2_better
                else "behavioral wagering (L1) more reliable than verbal self-report (L2)"
            )
        else:
            entry["l2_vs_l1_spearman"] = None
            entry["l2_vs_accuracy_spearman"] = None
            entry["l1_vs_accuracy_spearman"] = None
            entry["l2_better_calibrated"] = None
            entry["interpretation"] = "insufficient paired data"
        entry["n_paired"] = len(l2_conf)
        result[model] = entry

    return result


# ---------------------------------------------------------------------------
# Analysis F: Hallucination wagering analysis
# ---------------------------------------------------------------------------

def analyze_hallucinations(records: list[dict], confidence_threshold: float = 7.0) -> dict:
    """Identify high-confidence errors and analyze their wagering patterns."""
    by_model = _group_by(records, "model")
    result: dict = {"per_model": {}, "global": {}}

    all_hallucination_rates = []

    for model, model_records in by_model.items():
        ch1_recs = [r for r in model_records
                    if r.get("channel") == 1
                    and r.get("parsed") is not None
                    and r.get("answer_correct") is not None]
        if not ch1_recs:
            continue

        hallucinations = [r for r in ch1_recs
                          if not r["answer_correct"]
                          and r["parsed"].get("bet") is not None
                          and r["parsed"]["bet"] >= confidence_threshold]
        # Only include records with a valid bet in mean computations
        correct_with_bet = [r for r in ch1_recs
                            if r["answer_correct"]
                            and r["parsed"].get("bet") is not None]

        hallucination_rate = len(hallucinations) / len(ch1_recs) if ch1_recs else None
        all_hallucination_rates.append((model, hallucination_rate or 0.0))

        hallucination_mean_bet = (
            sum(r["parsed"]["bet"] for r in hallucinations) / len(hallucinations)
            if hallucinations else None
        )
        correct_mean_bet = (
            sum(r["parsed"]["bet"] for r in correct_with_bet) / len(correct_with_bet)
            if correct_with_bet else None
        )

        # Per-domain hallucination counts
        by_domain: dict[str, int] = defaultdict(int)
        for r in hallucinations:
            by_domain[r.get("domain", "unknown")] += 1

        result["per_model"][model] = {
            "n_high_bet_answers": len([r for r in ch1_recs
                                       if r["parsed"].get("bet") is not None
                                       and r["parsed"]["bet"] >= confidence_threshold]),
            "n_hallucinations": len(hallucinations),
            "hallucination_rate": hallucination_rate,
            "hallucination_mean_bet": hallucination_mean_bet,
            "correct_mean_bet": correct_mean_bet,
            "by_domain": dict(by_domain),
        }

    # Cross-model: does higher hallucination rate → lower MCI?
    result["global"]["n_models"] = len(all_hallucination_rates)
    result["global"]["model_ranking"] = sorted(
        all_hallucination_rates, key=lambda t: t[1], reverse=True
    )

    return result


# ---------------------------------------------------------------------------
# Analysis G: Implicit behavioral signals from raw responses
# ---------------------------------------------------------------------------

HEDGE_PHRASES = [
    "I think", "probably", "I'm not sure", "might be", "perhaps",
    "I believe", "not certain", "unclear", "it seems", "possibly",
    "I'm uncertain", "I guess", "maybe", "could be", "appears to",
]

CAVEAT_WORDS = [
    "however", "although", "but note", "caveat", "keep in mind",
    "worth noting", "it's important to", "that said", "on the other hand",
]


def _extract_implicit_signals(raw_response: str) -> dict:
    """Extract implicit behavioral signals from raw response text."""
    if not raw_response:
        return {
            "response_length": 0,
            "hedging_count": 0,
            "caveat_present": False,
        }

    response_lower = raw_response.lower()
    hedging_count = sum(1 for phrase in HEDGE_PHRASES if phrase.lower() in response_lower)
    caveat_present = any(word.lower() in response_lower for word in CAVEAT_WORDS)

    return {
        "response_length": len(raw_response),
        "hedging_count": hedging_count,
        "caveat_present": caveat_present,
    }


def analyze_implicit_signals(records: list[dict]) -> dict:
    """
    Extract implicit behavioral signals from raw responses and compute correlations.

    For each record with raw_response:
    - response_length: len(raw_response)
    - hedging_count: count of hedge phrases
    - caveat_present: boolean for caveat words

    Compute per-model:
    - Spearman(response_length, answer_correct)
    - Spearman(hedging_count, answer_correct)
    - Odds ratio for caveat_present predicting incorrectness
    """
    by_model = _group_by(records, "model")
    result = {}

    for model, model_records in by_model.items():
        # Extract signals from all records with raw_response
        records_with_signals = []
        for r in model_records:
            raw_resp = r.get("raw_response")
            if raw_resp and r.get("answer_correct") is not None:
                signals = _extract_implicit_signals(raw_resp)
                records_with_signals.append({
                    **r,
                    **signals,
                })

        if not records_with_signals:
            result[model] = {
                "n_records": 0,
                "length_accuracy_spearman": None,
                "hedging_accuracy_spearman": None,
                "caveat_odds_ratio": None,
            }
            continue

        # Spearman: response_length vs answer_correct
        lengths = [r["response_length"] for r in records_with_signals]
        corrects = [float(r["answer_correct"]) for r in records_with_signals]
        length_rho = _spearman(lengths, corrects) if len(lengths) >= 3 else None

        # Spearman: hedging_count vs answer_correct
        hedges = [r["hedging_count"] for r in records_with_signals]
        hedging_rho = _spearman(hedges, corrects) if len(hedges) >= 3 else None

        # Odds ratio: caveat_present predicting incorrectness
        caveat_yes_incorrect = sum(1 for r in records_with_signals if r["caveat_present"] and not r["answer_correct"])
        caveat_yes_correct = sum(1 for r in records_with_signals if r["caveat_present"] and r["answer_correct"])
        caveat_no_incorrect = sum(1 for r in records_with_signals if not r["caveat_present"] and not r["answer_correct"])
        caveat_no_correct = sum(1 for r in records_with_signals if not r["caveat_present"] and r["answer_correct"])

        # Odds ratio = (caveat_yes_incorrect / caveat_yes_correct) / (caveat_no_incorrect / caveat_no_correct)
        # Handle zero counts
        if caveat_yes_correct > 0 and caveat_no_correct > 0:
            odds_caveat_yes = caveat_yes_incorrect / caveat_yes_correct if caveat_yes_correct > 0 else float("inf")
            odds_caveat_no = caveat_no_incorrect / caveat_no_correct if caveat_no_correct > 0 else float("inf")
            if odds_caveat_no > 0 and odds_caveat_yes != float("inf"):
                odds_ratio = odds_caveat_yes / odds_caveat_no
            else:
                odds_ratio = None
        else:
            odds_ratio = None

        result[model] = {
            "n_records": len(records_with_signals),
            "length_accuracy_spearman": length_rho if not (length_rho is not None and math.isnan(length_rho)) else None,
            "hedging_accuracy_spearman": hedging_rho if not (hedging_rho is not None and math.isnan(hedging_rho)) else None,
            "caveat_odds_ratio": odds_ratio,
            "caveat_contingency": {
                "caveat_yes_incorrect": caveat_yes_incorrect,
                "caveat_yes_correct": caveat_yes_correct,
                "caveat_no_incorrect": caveat_no_incorrect,
                "caveat_no_correct": caveat_no_correct,
            },
        }

    return result


# ---------------------------------------------------------------------------
# Analysis H: Parse failure report
# ---------------------------------------------------------------------------

def analyze_parse_failures(records: list[dict]) -> dict:
    """Count parse failures and refusals per (model, channel)."""
    result: dict = {"per_model_channel": {}, "flagged": []}

    by_model = _group_by(records, "model")
    for model, model_records in by_model.items():
        result["per_model_channel"][model] = {}
        for ch_name, ch_id in CHANNELS + [("layer2", "layer2")]:
            ch_recs = [r for r in model_records
                       if r.get("channel") == ch_id]
            n = len(ch_recs)
            if n == 0:
                continue
            n_fail = sum(1 for r in ch_recs if not r.get("parse_success", True))
            n_refused = sum(1 for r in ch_recs
                            if r.get("parsed", {}).get("refused", False))
            fail_rate = n_fail / n
            entry = {
                "n_total": n,
                "n_parse_fail": n_fail,
                "n_refused": n_refused,
                "parse_fail_rate": fail_rate,
                "refusal_rate": n_refused / n,
            }
            result["per_model_channel"][model][ch_name] = entry
            if fail_rate > 0.15:
                result["flagged"].append({
                    "model": model,
                    "channel": ch_name,
                    "parse_fail_rate": round(fail_rate, 3),
                    "note": "parse_fail_rate > 15% — manual inspection recommended",
                })

    return result


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary(
    accuracy: dict,
    calibration: dict,
    mci: dict,
    layer_comparison: dict,
    dissociation: dict,
) -> None:
    models = sorted(accuracy.keys())

    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + " EXPERIMENT 1 RESULTS SUMMARY".center(88) + "║")
    print("╠" + "═" * 15 + "╦" + "═" * 7 + "╦" + "═" * 7 + "╦" + "═" * 7 +
          "╦" + "═" * 7 + "╦" + "═" * 7 + "╦" + "═" * 7 + "╦" + "═" * 31 + "╣")
    header = (
        "║ {:<13} ║ {:>5} ║ {:>5} ║ {:>5} ║ {:>5} ║ {:>5} ║ {:>5} ║ {:<29} ║"
    ).format("Model", "Acc↑", "ECE↓", "AUROC↑", "Wag-ρ↑", "MCI↑", "MCI_adj↑", "Layer")
    print(header)
    print("╠" + "═" * 15 + "╬" + "═" * 7 + "╬" + "═" * 7 + "╬" + "═" * 7 +
          "╬" + "═" * 7 + "╬" + "═" * 7 + "╬" + "═" * 7 + "╬" + "═" * 31 + "╣")

    n_l1_better = 0
    for model in models:
        # Natural accuracy (Ch5, all domains)
        ch5_corrects = []
        aurocs = []
        for domain_data in accuracy.get(model, {}).values():
            acc = domain_data.get("natural_acc")
            if acc is not None:
                ch5_corrects.append(acc)
            auroc = domain_data.get("wagering_auroc")
            if auroc is not None:
                aurocs.append(auroc)
        mean_acc = sum(ch5_corrects) / len(ch5_corrects) if ch5_corrects else None
        mean_auroc = sum(aurocs) / len(aurocs) if aurocs else None

        cal = calibration.get(model, {}).get("overall", {})
        ece = cal.get("wagering_ece")
        wag_rho = cal.get("wagering_spearman")
        mci_val = mci.get(model, {}).get("mci_raw")
        mci_adj = mci.get(model, {}).get("mci_difficulty_adjusted")

        lc = layer_comparison.get(model, {})
        l2_better = lc.get("l2_better_calibrated")
        layer_note = "L2 > L1" if l2_better else "L1 > L2" if l2_better is False else "—"
        if l2_better is False:
            n_l1_better += 1

        def fmt(v, decimals=2):
            if v is None:
                return "—"
            return f"{v:.{decimals}f}"

        short_model = model[:13]
        row = (
            f"║ {short_model:<13} ║ {fmt(mean_acc):>5} ║ {fmt(ece):>5} ║"
            f" {fmt(mean_auroc):>5} ║ {fmt(wag_rho):>5} ║ {fmt(mci_val):>5} ║"
            f" {fmt(mci_adj):>5} ║ {layer_note:<29} ║"
        )
        print(row)

    print("╚" + "═" * 15 + "╩" + "═" * 7 + "╩" + "═" * 7 + "╩" + "═" * 7 +
          "╩" + "═" * 7 + "╩" + "═" * 7 + "╩" + "═" * 7 + "╩" + "═" * 31 + "╝")

    # Top dissociations
    print("\nTop dissociations:")
    for model in models:
        diss = dissociation.get(model, {}).get("dissociated_pairs", [])
        mat = dissociation.get(model, {}).get("matrix", {})
        for pair in diss[:2]:
            ch_a, ch_b = pair.split("_", 1)
            rho = mat.get(ch_a, {}).get(ch_b)
            rho_str = f"ρ={rho:.2f}" if rho is not None else "ρ=?"
            print(f"  {model[:20]}: {ch_a} ↔ {ch_b} {rho_str} (DISSOCIATED)")

    # Layer comparison summary
    print(f"\nLayer comparison:")
    n_models = len(models)
    print(f"  {n_l1_better}/{n_models} models: behavioral confidence (L1) better calibrated than verbal (L2)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 Analysis")
    parser.add_argument("--run-id", required=True, help="Run ID to analyse")
    parser.add_argument("--tables-only", action="store_true",
                        help="Skip console output, only write JSON files")
    parser.add_argument("--export-latex", action="store_true",
                        help="Also run export_latex_tables.py after analysis")
    parser.add_argument("--results-dir", default="data/results")
    return parser.parse_args()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  Written: {path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / args.results_dir
    run_id = args.run_id

    results_path = results_dir / f"exp1_{run_id}_results.jsonl"
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}")
        sys.exit(1)

    print(f"\nLoading results from {results_path} …")
    records = load_results(results_path)
    print(f"  {len(records)} records loaded.\n")

    print("Running analyses …")

    # A. Accuracy
    accuracy = analyze_accuracy(records)
    _write_json(results_dir / f"exp1_{run_id}_accuracy.json", accuracy)

    # B. Calibration
    calibration = analyze_calibration(records)
    _write_json(results_dir / f"exp1_{run_id}_calibration.json", calibration)

    # C. MCI
    mci = analyze_mci(records)
    _write_json(results_dir / f"exp1_{run_id}_mci.json", mci)

    # D. Dissociation
    dissociation = analyze_dissociation(records)
    _write_json(results_dir / f"exp1_{run_id}_dissociation.json", dissociation)

    # E. Layer comparison
    layer_comparison = analyze_layer_comparison(records)
    _write_json(results_dir / f"exp1_{run_id}_layer_comparison.json", layer_comparison)

    # F. Hallucination
    hallucination = analyze_hallucinations(records)
    _write_json(results_dir / f"exp1_{run_id}_hallucination.json", hallucination)

    # G. Implicit behavioral signals
    implicit_signals = analyze_implicit_signals(records)
    _write_json(results_dir / f"exp1_{run_id}_implicit_signals.json", implicit_signals)

    # H. Parse failures
    parse_report = analyze_parse_failures(records)
    _write_json(results_dir / f"exp1_{run_id}_parse_report.json", parse_report)

    if not args.tables_only:
        print_summary(accuracy, calibration, mci, layer_comparison, dissociation)

    if args.export_latex:
        import subprocess
        subprocess.run(
            [sys.executable,
             str(repo_root / "scripts" / "export_latex_tables.py"),
             "--run-id", run_id],
            check=False,
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
