"""Tests for Experiment 1 analysis functions (scripts/analyze_experiment_1.py)."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import analysis functions directly
from scripts.analyze_experiment_1 import (
    _accuracy,
    _spearman,
    analyze_accuracy,
    analyze_calibration,
    analyze_hallucinations,
    analyze_layer_comparison,
    analyze_mci,
    analyze_parse_failures,
)
from scripts.export_latex_tables import (
    export_table1_accuracy,
    export_table2_calibration,
    export_table3_dissociation,
    export_table4_layer_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures: mock results
# ---------------------------------------------------------------------------

def _make_record(
    model="llama-3.1-8b",
    question_id="q1",
    channel=1,
    channel_name="wagering",
    domain="arithmetic",
    difficulty="easy",
    answer_correct=True,
    parse_success=True,
    parsed=None,
    layer=1,
):
    """Create a minimal result record."""
    if parsed is None:
        parsed = {"answer": "42", "bet": 7, "parse_success": True, "refused": False}
    return {
        "question_id": question_id,
        "model": model,
        "channel": channel,
        "channel_name": channel_name,
        "layer": layer,
        "domain": domain,
        "difficulty": difficulty,
        "answer_correct": answer_correct,
        "parse_success": parse_success,
        "parsed": parsed,
        "raw_response": "...",
        "latency_ms": 200.0,
    }


def _make_ch1(model, qid, domain, bet, correct):
    return _make_record(
        model=model, question_id=qid, channel=1, channel_name="wagering",
        domain=domain, answer_correct=correct, parse_success=True,
        parsed={"answer": "x", "bet": bet, "parse_success": True, "refused": False},
    )


def _make_ch2(model, qid, domain, skipped, correct):
    return _make_record(
        model=model, question_id=qid, channel=2, channel_name="opt_out",
        domain=domain, answer_correct=None if skipped else correct,
        parsed={"answer": None if skipped else "x", "skipped": skipped,
                "parse_success": True, "refused": False},
    )


def _make_ch5(model, qid, domain, correct, hedges=0, length=100):
    return _make_record(
        model=model, question_id=qid, channel=5, channel_name="natural",
        domain=domain, answer_correct=correct,
        parsed={"answer": "x", "hedging_count": hedges,
                "caveat_count": 0, "response_length": length,
                "parse_success": True, "refused": False},
    )


def _make_l2(model, qid, domain, confidence, correct):
    return _make_record(
        model=model, question_id=qid, channel="layer2", channel_name="layer2",
        domain=domain, layer=2, answer_correct=correct,
        parsed={"answer": "x", "confidence": confidence,
                "parse_success": True, "refused": False},
    )


# ---------------------------------------------------------------------------
# Stratified sampling test (from run_experiment_1 module)
# ---------------------------------------------------------------------------

def test_stratified_sampling():
    from scripts.run_experiment_1 import sample_stratified, DOMAINS

    questions = []
    for i, domain in enumerate(DOMAINS):
        for j in range(10):
            questions.append({
                "question_id": f"{domain}_{j}",
                "domain": domain,
                "difficulty": "easy" if j < 5 else "hard",
                "question_text": "Q",
                "correct_answer": "A",
                "answer_type": "short_text",
            })

    sampled = sample_stratified(questions, 24)
    assert len(sampled) <= 24

    # Each domain should appear
    domains_present = {q["domain"] for q in sampled}
    for d in DOMAINS:
        assert d in domains_present, f"Domain {d} missing from sample"

    # No duplicates
    ids = [q["question_id"] for q in sampled]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# MCI computation
# ---------------------------------------------------------------------------

def test_mci_computation_from_results():
    """With perfectly correlated channels, MCI should be high (> 0.5)."""
    records = []
    # 20 questions: channels 1-5 all signal the same thing
    for i in range(20):
        qid = f"q{i}"
        correct = i % 2 == 0  # alternating
        bet = 8 if correct else 3
        records.append(_make_ch1("model_a", qid, "arithmetic", bet, correct))
        records.append(_make_record(
            model="model_a", question_id=qid, channel=2,
            channel_name="opt_out", domain="arithmetic",
            answer_correct=correct if not (not correct) else None,
            parsed={"answer": "x" if correct else None, "skipped": not correct,
                    "parse_success": True, "refused": False},
        ))
        # Ch3: chose hard (A) when correct=True
        records.append(_make_record(
            model="model_a", question_id=qid, channel=3,
            channel_name="difficulty_selection", domain="arithmetic",
            answer_correct=correct,
            parsed={"choice": "A" if correct else "B", "answer": "x",
                    "parse_success": True, "refused": False},
        ))
        # Ch4: no tools when correct (confident)
        records.append(_make_record(
            model="model_a", question_id=qid, channel=4,
            channel_name="tool_use", domain="arithmetic",
            answer_correct=correct,
            parsed={"answer": "x", "tools_used": [] if correct else [{"tool_name": "calculator"}],
                    "parse_success": True, "refused": False},
        ))
        records.append(_make_ch5("model_a", qid, "arithmetic", correct,
                                  hedges=0 if correct else 3))

    mci_data = analyze_mci(records)
    assert "model_a" in mci_data
    mci_val = mci_data["model_a"]["mci_raw"]
    assert mci_val is not None
    assert mci_val > 0.3, f"Expected MCI > 0.3 for correlated channels, got {mci_val}"


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def test_ece_computation():
    from mirror.scoring.metrics import compute_ece

    # Perfect calibration: bet=10 → always right, bet=1 → always wrong
    confidences = [100.0] * 10 + [10.0] * 10   # already in 0-100 scale
    correctnesses = [True] * 10 + [False] * 10
    ece = compute_ece(confidences, correctnesses)
    assert ece is not None
    assert ece < 0.1, f"Expected near-zero ECE for perfect calibration, got {ece}"

    # Overconfident: always bet=10 but correct only 50% of time
    confidences2 = [100.0] * 20
    correctnesses2 = [True, False] * 10
    ece2 = compute_ece(confidences2, correctnesses2)
    assert ece2 > 0.3, f"Expected high ECE for overconfident model, got {ece2}"


# ---------------------------------------------------------------------------
# Dissociation matrix shape
# ---------------------------------------------------------------------------

def test_dissociation_matrix_shape():
    records = []
    for i in range(15):
        qid = f"q{i}"
        for ch_id, ch_name, parsed in [
            (1, "wagering", {"bet": 5, "answer": "x", "parse_success": True, "refused": False}),
            (2, "opt_out", {"skipped": False, "answer": "x", "parse_success": True, "refused": False}),
            (3, "difficulty_selection", {"choice": "A", "answer": "x", "parse_success": True, "refused": False}),
            (4, "tool_use", {"tools_used": [], "answer": "x", "parse_success": True, "refused": False}),
            (5, "natural", {"hedging_count": 1, "response_length": 50, "answer": "x",
                             "parse_success": True, "refused": False}),
        ]:
            records.append(_make_record(
                model="m", question_id=qid, channel=ch_id, channel_name=ch_name,
                domain="arithmetic", answer_correct=(i % 2 == 0), parsed=parsed,
            ))

    from scripts.analyze_experiment_1 import analyze_dissociation
    result = analyze_dissociation(records)
    assert "m" in result
    matrix = result["m"]["matrix"]
    # Should be a 5-key dict
    assert len(matrix) == 5
    for ch in matrix:
        assert len(matrix[ch]) == 5


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

def test_hallucination_filter():
    records = []
    for i in range(10):
        qid = f"q{i}"
        # High bet, wrong → hallucination
        bet = 8 if i < 4 else 3
        correct = i >= 4  # first 4 are wrong with high bet
        records.append(_make_ch1("model_x", qid, "arithmetic", bet, correct))

    result = analyze_hallucinations(records, confidence_threshold=7.0)
    model_data = result["per_model"]["model_x"]
    assert model_data["n_hallucinations"] == 4
    assert model_data["hallucination_rate"] == pytest.approx(4 / 10)


def test_hallucination_no_high_bets():
    """Model that always bets low — no hallucinations."""
    records = [_make_ch1("model_y", f"q{i}", "arithmetic", 3, i % 2 == 0)
               for i in range(10)]
    result = analyze_hallucinations(records, confidence_threshold=7.0)
    model_data = result["per_model"]["model_y"]
    assert model_data["n_hallucinations"] == 0


# ---------------------------------------------------------------------------
# Layer comparison
# ---------------------------------------------------------------------------

def test_layer_comparison():
    """L1 wagering perfectly correlated with accuracy, L2 anti-correlated → L1 better."""
    records = []
    for i in range(20):
        qid = f"q{i}"
        correct = i % 2 == 0
        # L1: bet high when correct (perfect calibration)
        records.append(_make_ch1("m", qid, "arithmetic", 9 if correct else 2, correct))
        # L2: anti-correlated (high confidence when wrong, low when right)
        records.append(_make_l2("m", qid, "arithmetic", 20 if correct else 80, correct))

    result = analyze_layer_comparison(records)
    assert "m" in result
    entry = result["m"]
    # L1 should correlate better with accuracy than L2 (which is flat at 50)
    l1_rho = entry.get("l1_vs_accuracy_spearman") or 0
    l2_rho = entry.get("l2_vs_accuracy_spearman") or 0
    assert l1_rho > l2_rho, (
        f"Expected L1 (ρ={l1_rho:.2f}) > L2 (ρ={l2_rho:.2f}) for this setup"
    )
    assert entry["l2_better_calibrated"] is False


# ---------------------------------------------------------------------------
# Parse failure report
# ---------------------------------------------------------------------------

def test_parse_failure_report():
    records = []
    # 10 wagering records: 3 fail, 1 refused
    for i in range(10):
        fail = i < 3
        refused = i == 3
        records.append(_make_record(
            model="m", question_id=f"q{i}", channel=1, channel_name="wagering",
            domain="arithmetic", answer_correct=None if fail else True,
            parse_success=not fail,
            parsed={
                "bet": 5, "answer": "x", "refused": refused,
                "parse_success": not fail,
            },
        ))

    result = analyze_parse_failures(records)
    wager_stats = result["per_model_channel"]["m"]["wagering"]
    assert wager_stats["n_total"] == 10
    assert wager_stats["n_parse_fail"] == 3
    assert wager_stats["parse_fail_rate"] == pytest.approx(0.3)

    # 30% fail rate should be flagged
    flagged = result["flagged"]
    assert any(f["model"] == "m" and f["channel"] == "wagering" for f in flagged)


# ---------------------------------------------------------------------------
# LaTeX export format
# ---------------------------------------------------------------------------

def test_latex_export_format(tmp_path):
    """Verify that each table export produces syntactically valid LaTeX snippets."""
    # Minimal accuracy data
    accuracy = {
        "llama-3.1-8b": {
            "arithmetic": {"natural_acc": 0.72, "wagering_acc": 0.70},
            "logical": {"natural_acc": 0.55, "wagering_acc": 0.52},
        }
    }
    calibration = {
        "llama-3.1-8b": {
            "overall": {
                "wagering_ece": 0.12,
                "wagering_spearman": 0.45,
                "skip_error_alignment": 0.60,
            }
        }
    }
    mci = {
        "llama-3.1-8b": {"mci_raw": 0.34, "mci_difficulty_adjusted": 0.21}
    }
    hallucination = {
        "per_model": {
            "llama-3.1-8b": {"hallucination_rate": 0.08}
        },
        "global": {},
    }
    dissociation = {
        "llama-3.1-8b": {
            "dissociated_pairs": ["ch1_vs_ch4"],
            "matrix": {
                "ch1": {"ch1": 1.0, "ch2": 0.45, "ch3": 0.30, "ch4": 0.05, "ch5": 0.40},
                "ch2": {"ch1": 0.45, "ch2": 1.0, "ch3": 0.35, "ch4": 0.20, "ch5": 0.38},
                "ch3": {"ch1": 0.30, "ch2": 0.35, "ch3": 1.0, "ch4": 0.15, "ch5": 0.28},
                "ch4": {"ch1": 0.05, "ch2": 0.20, "ch3": 0.15, "ch4": 1.0, "ch5": 0.10},
                "ch5": {"ch1": 0.40, "ch2": 0.38, "ch3": 0.28, "ch4": 0.10, "ch5": 1.0},
            },
        }
    }
    layer_comparison = {
        "llama-3.1-8b": {
            "l1_vs_accuracy_spearman": 0.45,
            "l2_vs_accuracy_spearman": 0.30,
            "l2_vs_l1_spearman": 0.62,
            "l2_better_calibrated": False,
        }
    }

    table1 = tmp_path / "table1.tex"
    table2 = tmp_path / "table2.tex"
    table3 = tmp_path / "table3.tex"
    table4 = tmp_path / "table4.tex"

    export_table1_accuracy(accuracy, table1)
    export_table2_calibration(calibration, mci, hallucination, table2)
    export_table3_dissociation(dissociation, mci, table3)
    export_table4_layer_comparison(layer_comparison, table4)

    for path in [table1, table2, table3, table4]:
        content = path.read_text(encoding="utf-8")
        # Must be non-empty LaTeX
        assert r"\begin{table}" in content or r"\begin{tabular}" in content, \
            f"{path.name} missing LaTeX table environment"
        assert r"\end{table}" in content or r"\end{tabular}" in content, \
            f"{path.name} missing table end"
        assert r"\toprule" in content, f"{path.name} missing \\toprule"
        assert r"\bottomrule" in content, f"{path.name} missing \\bottomrule"
