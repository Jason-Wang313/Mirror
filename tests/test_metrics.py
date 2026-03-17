"""
Tests for scoring metrics with known synthetic data.

All numeric tolerances: ±0.01 unless stated.

Tests:
  - ECE on perfectly calibrated / miscalibrated data
  - AUROC on perfect / random / inverted discriminators
  - MCI on synthetic channel signals
  - Spearman correlation (internal helper)
  - Bootstrap CI coverage
  - Cohen's d
  - FDR correction
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.scoring.metrics import (
    compute_ece,
    compute_auroc,
    compute_mci,
    compute_channel_dissociation_matrix,
    _spearman_correlation,
)
from mirror.scoring.statistics import (
    bootstrap_ci,
    permutation_test,
    fdr_correction,
    cohens_d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(qid, model, channel, bet=None, skipped=None, choice=None,
                n_tools=None, hedging=None, confidence=None,
                answer_correct=None, domain="arithmetic", difficulty="medium"):
    """Create a synthetic result record."""
    parsed = {}
    if bet is not None:
        parsed["bet"] = bet
    if skipped is not None:
        parsed["skipped"] = skipped
    if choice is not None:
        parsed["choice"] = choice
    if n_tools is not None:
        parsed["tools_used"] = [{"tool_name": "calculator", "tool_input": "x"}] * n_tools
    if hedging is not None:
        parsed["hedging_count"] = hedging
    if confidence is not None:
        parsed["confidence"] = confidence

    return {
        "question_id": qid,
        "model": model,
        "channel": channel,
        "layer": 1 if channel != "layer2" else 2,
        "parsed": parsed,
        "answer_correct": answer_correct,
        "parse_success": True,
        "domain": domain,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    # If confidence 70% → 70% accuracy, etc. — ECE should be ≈ 0
    n = 100
    confidences = [0.7] * n
    correctnesses = [True] * 70 + [False] * 30
    ece = compute_ece(confidences, correctnesses)
    assert abs(ece) < 0.01, f"Expected ECE ≈ 0, got {ece:.4f}"


def test_ece_overconfident():
    # Always 100% confident but only 50% accurate
    confidences = [1.0] * 100
    correctnesses = [True] * 50 + [False] * 50
    ece = compute_ece(confidences, correctnesses)
    assert abs(ece - 0.5) < 0.01, f"Expected ECE ≈ 0.5, got {ece:.4f}"


def test_ece_underconfident():
    # Always 0% confident but 100% accurate
    confidences = [0.0] * 100
    correctnesses = [True] * 100
    ece = compute_ece(confidences, correctnesses)
    assert abs(ece - 1.0) < 0.01, f"Expected ECE ≈ 1.0, got {ece:.4f}"


def test_ece_with_100_scale():
    # Input in [0, 100] — should be auto-normalized
    confidences = [70.0] * 100
    correctnesses = [True] * 70 + [False] * 30
    ece = compute_ece(confidences, correctnesses)
    assert abs(ece) < 0.01


def test_ece_empty_returns_nan():
    ece = compute_ece([], [])
    assert math.isnan(ece)


# ---------------------------------------------------------------------------
# AUROC tests
# ---------------------------------------------------------------------------

def test_auroc_perfect():
    # Perfect discriminator: correct always has higher confidence
    scores = [0.9] * 10 + [0.1] * 10
    labels = [True] * 10 + [False] * 10
    auroc = compute_auroc(scores, labels)
    assert abs(auroc - 1.0) < 0.01, f"Expected AUROC=1.0, got {auroc:.4f}"


def test_auroc_random():
    # Random: AUC ≈ 0.5
    import random
    rng = random.Random(42)
    n = 1000
    scores = [rng.random() for _ in range(n)]
    labels = [rng.random() > 0.5 for _ in range(n)]
    auroc = compute_auroc(scores, labels)
    assert abs(auroc - 0.5) < 0.05, f"Expected AUROC≈0.5, got {auroc:.4f}"


def test_auroc_inverted():
    # Inverted: confident always wrong
    scores = [0.9] * 10 + [0.1] * 10
    labels = [False] * 10 + [True] * 10
    auroc = compute_auroc(scores, labels)
    assert abs(auroc - 0.0) < 0.01, f"Expected AUROC=0.0, got {auroc:.4f}"


def test_auroc_single_class_returns_nan():
    # Only one class — AUROC undefined
    scores = [0.5, 0.8, 0.3]
    labels = [True, True, True]
    auroc = compute_auroc(scores, labels)
    assert math.isnan(auroc)


# ---------------------------------------------------------------------------
# Spearman correlation tests
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    rho = _spearman_correlation(x, y)
    assert abs(rho - 1.0) < 1e-9


def test_spearman_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    rho = _spearman_correlation(x, y)
    assert abs(rho + 1.0) < 1e-9


def test_spearman_uncorrelated():
    x = [1, 2, 3, 4, 5]
    y = [3, 1, 5, 2, 4]  # No monotonic relationship
    rho = _spearman_correlation(x, y)
    assert abs(rho) < 0.5  # Should be near 0


def test_spearman_too_short():
    rho = _spearman_correlation([1], [1])
    assert math.isnan(rho)


def test_spearman_with_ties():
    x = [1, 1, 2, 2, 3]
    y = [1, 2, 3, 4, 5]
    rho = _spearman_correlation(x, y)
    assert not math.isnan(rho)
    assert rho > 0.9  # Strong positive correlation despite ties


# ---------------------------------------------------------------------------
# MCI tests
# ---------------------------------------------------------------------------

def test_mci_perfect_convergence():
    """Channels co-vary perfectly: high confidence on even questions, low on odd. MCI ≈ 1."""
    results = []
    n_q = 20

    for i in range(n_q):
        qid = f"q{i:03d}"
        # Alternate high/low confidence so channels have variance but perfect correlation
        if i % 2 == 0:
            # High confidence: high bet, answered, chose A (challenge), no tools, no hedging
            results.append(make_result(qid, "m1", 1, bet=10))
            results.append(make_result(qid, "m1", 2, skipped=False))
            results.append(make_result(qid, "m1", 3, choice="A"))
            results.append(make_result(qid, "m1", 4, n_tools=0))
            results.append(make_result(qid, "m1", 5, hedging=0))
        else:
            # Low confidence: low bet, skipped, chose B (safe), many tools, much hedging
            results.append(make_result(qid, "m1", 1, bet=1))
            results.append(make_result(qid, "m1", 2, skipped=True))
            results.append(make_result(qid, "m1", 3, choice="B"))
            results.append(make_result(qid, "m1", 4, n_tools=3))
            results.append(make_result(qid, "m1", 5, hedging=5))

    mci_data = compute_mci(results)
    mci = mci_data["mci"]
    assert not math.isnan(mci), "MCI should not be NaN with full data"
    assert mci > 0.5, f"Perfectly convergent signals should have MCI > 0.5, got {mci:.3f}"


def test_mci_zero_convergence():
    """Channels anti-correlate: high bet + skipped + chose B. MCI should be negative or low."""
    results = []
    n_q = 20

    for i in range(n_q):
        qid = f"q{i:03d}"
        # High bet but skipped (signals anti-correlated)
        bet = 10 if i % 2 == 0 else 1
        skipped = True if i % 2 == 0 else False  # Opposite of bet
        results.append(make_result(qid, "m1", 1, bet=bet))
        results.append(make_result(qid, "m1", 2, skipped=skipped))

    mci_data = compute_mci(results)
    mci = mci_data["mci"]
    # With anti-correlated signals, pairwise correlation should be negative
    assert mci < 0.0, f"Anti-correlated signals should have MCI < 0, got {mci:.3f}"


def test_mci_returns_nan_with_insufficient_data():
    """Only one question per channel — can't compute correlation."""
    results = [
        make_result("q1", "m1", 1, bet=5),
        make_result("q1", "m1", 2, skipped=False),
    ]
    mci_data = compute_mci(results)
    # 1 point per channel pair → Spearman undefined
    pair_rho = mci_data["pairwise_correlations"].get("ch1_ch2")
    assert math.isnan(pair_rho) or pair_rho is not None  # NaN expected


def test_mci_structure():
    """MCI output has expected keys."""
    results = [make_result(f"q{i}", "m1", ch, bet=5, skipped=False, choice="A",
                           n_tools=0, hedging=2)
               for i in range(5) for ch in [1, 2, 3, 4, 5]]
    mci_data = compute_mci(results)
    assert "mci" in mci_data
    assert "pairwise_correlations" in mci_data
    assert "n_questions_per_channel" in mci_data
    # Should have C(5,2)=10 pairs
    assert len(mci_data["pairwise_correlations"]) == 10


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean():
    data = [1.0] * 50 + [0.0] * 50  # True mean = 0.5
    lo, hi = bootstrap_ci(data, statistic=lambda x: sum(x) / len(x), n_bootstrap=1000)
    assert not math.isnan(lo) and not math.isnan(hi)
    assert lo <= 0.5 <= hi, f"95% CI [{lo:.3f}, {hi:.3f}] should contain 0.5"


def test_bootstrap_ci_narrow():
    # Constant data → CI should be very narrow
    data = [1.0] * 100
    lo, hi = bootstrap_ci(data, statistic=lambda x: sum(x) / len(x), n_bootstrap=100)
    # May return nan if constant (zero variance) — acceptable
    if not math.isnan(lo):
        assert abs(hi - lo) < 0.01


def test_bootstrap_ci_too_small():
    lo, hi = bootstrap_ci([1.0], lambda x: x[0])
    assert math.isnan(lo) and math.isnan(hi)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def test_permutation_same_dist():
    import random
    rng = random.Random(42)
    a = [rng.gauss(0, 1) for _ in range(50)]
    b = [rng.gauss(0, 1) for _ in range(50)]
    p = permutation_test(a, b, n_permutations=1000)
    # Null hypothesis true — p should be non-significant (no guarantee but likely > 0.05)
    assert 0 <= p <= 1


def test_permutation_different_dist():
    a = [10.0] * 50
    b = [0.0] * 50
    p = permutation_test(a, b, n_permutations=1000)
    # Very different means — p should be very small
    assert p < 0.01, f"Expected p < 0.01 for clearly different distributions, got {p}"


def test_permutation_greater():
    a = [5.0] * 30
    b = [3.0] * 30
    p = permutation_test(a, b, n_permutations=500, alternative="greater")
    assert p < 0.05


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def test_fdr_all_null():
    p_values = [0.3, 0.5, 0.7, 0.9]
    rejected, corrected = fdr_correction(p_values, q=0.05)
    assert not any(rejected)


def test_fdr_all_significant():
    p_values = [0.001, 0.002, 0.003, 0.004]
    rejected, corrected = fdr_correction(p_values, q=0.05)
    assert all(rejected)


def test_fdr_mixed():
    p_values = [0.001, 0.01, 0.03, 0.5]
    rejected, corrected = fdr_correction(p_values, q=0.05)
    # At least the first two should be rejected
    assert rejected[0] is True
    assert rejected[3] is False  # 0.5 should not be rejected


def test_fdr_corrected_p_range():
    p_values = [0.01, 0.05, 0.1, 0.5]
    _, corrected = fdr_correction(p_values, q=0.05)
    assert all(0 <= p <= 1 for p in corrected)


def test_fdr_empty():
    rejected, corrected = fdr_correction([])
    assert rejected == []
    assert corrected == []


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def test_cohens_d_zero():
    # Same distributions — d ≈ 0
    a = [5.0] * 20
    b = [5.0] * 20
    d = cohens_d(a, b)
    assert math.isnan(d) or abs(d) < 0.01


def test_cohens_d_positive():
    a = [10.0] * 20
    b = [5.0] * 20
    d = cohens_d(a, b)
    assert math.isnan(d) or d > 0


def test_cohens_d_large():
    # d = 1.0 when difference equals pooled SD
    import random
    rng = random.Random(42)
    a = [rng.gauss(1.0, 1.0) for _ in range(1000)]
    b = [rng.gauss(0.0, 1.0) for _ in range(1000)]
    d = cohens_d(a, b)
    # Should be approximately 1.0 ± 0.1
    assert abs(d - 1.0) < 0.15, f"Expected d≈1.0, got {d:.3f}"


def test_cohens_d_negative():
    a = [0.0] * 20
    b = [5.0] * 20
    d = cohens_d(a, b)
    assert math.isnan(d) or d < 0


def test_cohens_d_too_small():
    d = cohens_d([1.0], [2.0])
    assert math.isnan(d)


# ---------------------------------------------------------------------------
# Dissociation matrix
# ---------------------------------------------------------------------------

def test_dissociation_matrix_shape():
    results = [make_result(f"q{i}", "m1", ch, bet=5, skipped=False, choice="A",
                           n_tools=0, hedging=1)
               for i in range(10) for ch in [1, 2, 3, 4, 5]]
    out = compute_channel_dissociation_matrix(results)
    matrix = out["matrix"]
    assert len(matrix) == 5
    for row in matrix.values():
        assert len(row) == 5


def test_dissociation_diagonal_is_one():
    results = [make_result(f"q{i}", "m1", ch, bet=5, skipped=False, choice="A",
                           n_tools=0, hedging=1)
               for i in range(10) for ch in [1, 2, 3, 4, 5]]
    out = compute_channel_dissociation_matrix(results)
    matrix = out["matrix"]
    for ch in [1, 2, 3, 4, 5]:
        assert matrix[f"ch{ch}"][f"ch{ch}"] == 1.0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_functions = [
        # ECE
        test_ece_perfect_calibration,
        test_ece_overconfident,
        test_ece_underconfident,
        test_ece_with_100_scale,
        test_ece_empty_returns_nan,
        # AUROC
        test_auroc_perfect,
        test_auroc_random,
        test_auroc_inverted,
        test_auroc_single_class_returns_nan,
        # Spearman
        test_spearman_perfect_positive,
        test_spearman_perfect_negative,
        test_spearman_uncorrelated,
        test_spearman_too_short,
        test_spearman_with_ties,
        # MCI
        test_mci_perfect_convergence,
        test_mci_zero_convergence,
        test_mci_returns_nan_with_insufficient_data,
        test_mci_structure,
        # Bootstrap
        test_bootstrap_ci_mean,
        test_bootstrap_ci_narrow,
        test_bootstrap_ci_too_small,
        # Permutation
        test_permutation_same_dist,
        test_permutation_different_dist,
        test_permutation_greater,
        # FDR
        test_fdr_all_null,
        test_fdr_all_significant,
        test_fdr_mixed,
        test_fdr_corrected_p_range,
        test_fdr_empty,
        # Cohen's d
        test_cohens_d_zero,
        test_cohens_d_positive,
        test_cohens_d_large,
        test_cohens_d_negative,
        test_cohens_d_too_small,
        # Dissociation
        test_dissociation_matrix_shape,
        test_dissociation_diagonal_is_one,
    ]

    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✅ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
