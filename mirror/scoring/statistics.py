"""
Statistical utilities for MIRROR experiment analysis.

Implements:
  - bootstrap_ci         — BCa bootstrap confidence intervals
  - permutation_test     — non-parametric two-group test
  - fdr_correction       — Benjamini-Hochberg FDR at q=0.05
  - cohens_d             — standardized effect size

All functions use only numpy (no scipy dependency required).
BCa acceleration is estimated numerically via jackknife.
"""

import math
import random
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Bootstrap CI (BCa method)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list,
    statistic: Callable,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Args:
        data: Input data (list of numbers or any type statistic accepts).
        statistic: Function that takes a list and returns a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) tuple.
        Returns (NaN, NaN) if statistic cannot be computed.
    """
    rng = random.Random(seed)
    n = len(data)

    if n < 2:
        return float("nan"), float("nan")

    # Observed statistic
    try:
        theta_hat = statistic(data)
    except Exception:
        return float("nan"), float("nan")

    if math.isnan(theta_hat):
        return float("nan"), float("nan")

    # Bootstrap distribution
    boot_stats = []
    for _ in range(n_bootstrap):
        resample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            stat = statistic(resample)
            if not math.isnan(stat):
                boot_stats.append(stat)
        except Exception:
            pass

    if len(boot_stats) < 10:
        return float("nan"), float("nan")

    boot_stats.sort()
    B = len(boot_stats)

    # --- Bias correction (z0) ---
    n_less = sum(1 for s in boot_stats if s < theta_hat)
    prop = n_less / B
    # Clamp to avoid infinity in ppf
    prop = max(1e-10, min(1 - 1e-10, prop))
    z0 = _norm_ppf(prop)

    # --- Acceleration (a) via jackknife ---
    jack_stats = []
    for i in range(n):
        jack_sample = data[:i] + data[i + 1:]
        try:
            jack_stat = statistic(jack_sample)
            if not math.isnan(jack_stat):
                jack_stats.append(jack_stat)
        except Exception:
            pass

    if len(jack_stats) >= 2:
        jack_mean = sum(jack_stats) / len(jack_stats)
        num = sum((jack_mean - s) ** 3 for s in jack_stats)
        den = 6.0 * (sum((jack_mean - s) ** 2 for s in jack_stats) ** 1.5)
        a = num / den if den != 0 else 0.0
    else:
        a = 0.0  # Fall back to BC (no acceleration)

    # --- BCa quantiles ---
    alpha = (1 - ci) / 2.0
    z_lo = _norm_ppf(alpha)
    z_hi = _norm_ppf(1 - alpha)

    def bca_quantile(z_alpha):
        numer = z0 + z_alpha
        adj = z0 + numer / (1 - a * numer)
        return _norm_cdf(adj)

    q_lo = bca_quantile(z_lo)
    q_hi = bca_quantile(z_hi)

    # Clamp quantiles
    q_lo = max(0.0, min(1.0, q_lo))
    q_hi = max(0.0, min(1.0, q_hi))

    idx_lo = int(math.floor(q_lo * B))
    idx_hi = int(math.ceil(q_hi * B)) - 1
    idx_lo = max(0, min(B - 1, idx_lo))
    idx_hi = max(0, min(B - 1, idx_hi))

    return boot_stats[idx_lo], boot_stats[idx_hi]


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    group_a: list[float],
    group_b: list[float],
    statistic: Optional[Callable] = None,
    n_permutations: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> float:
    """
    Non-parametric permutation test for difference between two groups.

    Args:
        group_a: First group values.
        group_b: Second group values.
        statistic: Function mapping (list_a, list_b) → scalar difference.
                   Default: difference of means.
        n_permutations: Number of random permutations.
        seed: Random seed.
        alternative: "two-sided", "greater", "less".

    Returns:
        p-value.
    """
    if statistic is None:
        def statistic(a, b):
            if not a or not b:
                return float("nan")
            return sum(a) / len(a) - sum(b) / len(b)

    rng = random.Random(seed)
    n_a = len(group_a)
    n_b = len(group_b)
    combined = group_a + group_b
    n_total = n_a + n_b

    if n_total < 2:
        return float("nan")

    observed = statistic(group_a, group_b)
    if math.isnan(observed):
        return float("nan")

    count = 0
    for _ in range(n_permutations):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        perm_stat = statistic(perm_a, perm_b)

        if math.isnan(perm_stat):
            continue

        if alternative == "two-sided":
            if abs(perm_stat) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if perm_stat >= observed:
                count += 1
        elif alternative == "less":
            if perm_stat <= observed:
                count += 1

    return count / n_permutations


# ---------------------------------------------------------------------------
# FDR correction — Benjamini-Hochberg
# ---------------------------------------------------------------------------

def fdr_correction(
    p_values: list[float],
    q: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of raw p-values.
        q: Target false discovery rate (default 0.05).

    Returns:
        (rejected, corrected_p_values):
            rejected: List of bool — True if hypothesis rejected after correction.
            corrected_p_values: BH-adjusted p-values (capped at 1.0).
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Sort by p-value ascending, track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    indices, sorted_p = zip(*indexed)

    # BH critical values: p[i] <= q * (i+1) / n
    corrected = [0.0] * n
    # Compute adjusted p-values (Yekutieli step-up)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            corrected[i] = sorted_p[i]
        else:
            corrected[i] = min(corrected[i + 1], sorted_p[i] * n / (i + 1))

    corrected = [min(1.0, c) for c in corrected]

    # Determine rejection
    rejected_sorted = [corrected[i] <= q for i in range(n)]

    # Restore original order
    rejected = [False] * n
    corrected_original = [0.0] * n
    for rank, orig_idx in enumerate(indices):
        rejected[orig_idx] = rejected_sorted[rank]
        corrected_original[orig_idx] = corrected[rank]

    return rejected, corrected_original


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(
    group_a: list[float],
    group_b: list[float],
    pooled: bool = True,
) -> float:
    """
    Compute Cohen's d effect size for two independent groups.

    Args:
        group_a: First group values.
        group_b: Second group values.
        pooled: If True, use pooled standard deviation (default).
                If False, use control group (group_b) SD.

    Returns:
        Cohen's d. Positive = group_a has higher mean.
        Conventions: 0.2 small, 0.5 medium, 0.8 large.
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)

    if pooled:
        sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    else:
        sd = math.sqrt(var_b)

    if sd == 0:
        return float("nan")

    return (mean_a - mean_b) / sd


# ---------------------------------------------------------------------------
# Internal: normal distribution approximations
# ---------------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (Beasley-Springer-Moro approximation).

    Accurate to ~5 decimal places for p in (0.0001, 0.9999).
    """
    p = max(1e-12, min(1 - 1e-12, p))

    # Rational approximation coefficients
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
