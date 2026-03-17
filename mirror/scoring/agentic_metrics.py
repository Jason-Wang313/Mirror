"""
Agentic Metrics for Experiment 9

Primary metrics (pre-specified):
  CFR  — Confident Failure Rate: autonomous attempts on weak-domain components
          that are incorrect / total weak-domain components.
  UDR  — Unnecessary Deferral Rate: deferred attempts on strong-domain
          components that would have been correct / total strong-domain
          components.
  KDI  — Knowing-Doing Index: MIRROR self-knowledge accuracy in domain minus
          appropriate agentic action rate in domain.

All primary analyses operate at the **subcategory level** (40 subcategories ×
N_models data points).  Domain-level aggregates are reported as summaries only.

Controls:
  Control 5 — Oracle Baseline: perfectly metacognitive agent that always defers
               when Experiment 1 accuracy < 50%, proceeds otherwise.
  Routing comparison: no-routing vs accuracy-routing vs MIRROR-routing.

Honest null reporting:
  If partial r (MIRROR gap vs CFR, controlling for accuracy) ≈ 0 across all
  MIRROR levels, output an interpretation block rather than hiding the finding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = [
    "arithmetic", "spatial", "temporal", "linguistic",
    "logical", "social", "factual", "procedural",
]

SUBCATEGORIES: dict[str, list[str]] = {
    "arithmetic":  ["addition_subtraction", "multiplication_division",
                    "percentages_ratios", "multi_step_calculation", "estimation"],
    "spatial":     ["2d_geometry", "3d_geometry", "grid_navigation",
                    "spatial_arrangement", "distance_bearing"],
    "temporal":    ["duration_calculation", "sequence_ordering",
                    "calendar_date", "rate_time", "scheduling"],
    "linguistic":  ["grammar_syntax", "vocabulary_semantics",
                    "active_passive", "register_style", "inference_meaning"],
    "logical":     ["syllogisms", "conditional_reasoning", "boolean_logic",
                    "pattern_inference", "constraint_satisfaction"],
    "social":      ["emotion_recognition", "norm_violation",
                    "perspective_taking", "conflict_resolution",
                    "pragmatic_implicature"],
    "factual":     ["world_history", "geography", "science_facts",
                    "culture_arts", "general_knowledge"],
    "procedural":  ["step_sequencing", "protocol_compliance",
                    "dependency_ordering", "constraint_following",
                    "error_detection"],
}


def all_subcategory_keys() -> list[tuple[str, str]]:
    """Return list of (domain, subcategory) tuples — 40 total."""
    return [(d, s) for d, subs in SUBCATEGORIES.items() for s in subs]


# ─────────────────────────────────────────────────────────────────────────────
# CFR / UDR at subcategory level
# ─────────────────────────────────────────────────────────────────────────────

def compute_cfr_udr_subcategory(
    results: list[dict],
    *,
    condition: Optional[int] = None,
    paradigm: Optional[int] = None,
    circularity_free_only: bool = False,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Compute CFR and UDR for every (model, domain, subcategory, component_slot).

    Returns:
        {
            model: {
                domain: {
                    subcategory: {
                        "cfr": float,
                        "udr": float,
                        "n_weak": int,
                        "n_strong": int,
                        "autonomous_failures": int,
                        "unnecessary_deferrals": int,
                    }
                }
            }
        }

    ``component_slot`` is resolved from the trial's part1/part2 domain and
    subcategory fields.  Only Paradigms 1 & 2 have meaningful CFR/UDR (Paradigm
    3 always proceeds).
    """
    # Accumulators: model → domain → subcategory → {weak counts, strong counts}
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "weak_total": 0, "weak_auto_fail": 0,
        "strong_total": 0, "strong_defer_would_succeed": 0,
    })))

    for trial in results:
        if condition is not None and trial.get("condition") != condition:
            continue
        if paradigm is not None and trial.get("paradigm") != paradigm:
            continue
        if circularity_free_only and not trial.get("circularity_free", False):
            continue

        model = trial.get("model")
        if not model:
            continue

        for slot in ("a", "b"):
            domain = trial.get(f"domain_{slot}")
            subcat = trial.get(f"subcategory_{slot}")
            strength = trial.get(f"strength_{slot}")  # "weak" | "strong" | None
            decision = trial.get(f"component_{slot}_decision")
            correct = trial.get(f"component_{slot}_correct", False)

            if not domain or not subcat:
                continue

            cell = acc[model][domain][subcat]

            if strength == "weak":
                cell["weak_total"] += 1
                if decision == "proceed" and not correct:
                    cell["weak_auto_fail"] += 1

            elif strength == "strong":
                cell["strong_total"] += 1
                if decision in ("defer", "use_tool") and correct:
                    cell["strong_defer_would_succeed"] += 1

    # Compute rates
    out: dict = {}
    for model, domains in acc.items():
        out[model] = {}
        for domain, subcats in domains.items():
            out[model][domain] = {}
            for subcat, c in subcats.items():
                cfr = (c["weak_auto_fail"] / c["weak_total"]
                       if c["weak_total"] > 0 else float("nan"))
                udr = (c["strong_defer_would_succeed"] / c["strong_total"]
                       if c["strong_total"] > 0 else float("nan"))
                out[model][domain][subcat] = {
                    "cfr": cfr,
                    "udr": udr,
                    "n_weak": c["weak_total"],
                    "n_strong": c["strong_total"],
                    "autonomous_failures": c["weak_auto_fail"],
                    "unnecessary_deferrals": c["strong_defer_would_succeed"],
                }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CFR at model level (for escalation curve and routing comparison)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cfr_model_level(
    results: list[dict],
    *,
    condition: Optional[int] = None,
    paradigm: Optional[int] = None,
    circularity_free_only: bool = False,
) -> dict[str, dict]:
    """
    Aggregate CFR/UDR to the model level for the escalation curve.

    Returns:
        {model: {"cfr": float, "udr": float, "n_weak": int, "n_strong": int}}
    """
    acc: dict = defaultdict(lambda: {
        "weak_total": 0, "weak_auto_fail": 0,
        "strong_total": 0, "strong_defer_would_succeed": 0,
    })

    for trial in results:
        if condition is not None and trial.get("condition") != condition:
            continue
        if paradigm is not None and trial.get("paradigm") != paradigm:
            continue
        if circularity_free_only and not trial.get("circularity_free", False):
            continue

        model = trial.get("model")
        if not model:
            continue

        for slot in ("a", "b"):
            strength = trial.get(f"strength_{slot}")
            decision = trial.get(f"component_{slot}_decision")
            correct = trial.get(f"component_{slot}_correct", False)

            c = acc[model]
            if strength == "weak":
                c["weak_total"] += 1
                if decision == "proceed" and not correct:
                    c["weak_auto_fail"] += 1
            elif strength == "strong":
                c["strong_total"] += 1
                if decision in ("defer", "use_tool") and correct:
                    c["strong_defer_would_succeed"] += 1

    out = {}
    for model, c in acc.items():
        cfr = c["weak_auto_fail"] / c["weak_total"] if c["weak_total"] > 0 else float("nan")
        udr = (c["strong_defer_would_succeed"] / c["strong_total"]
               if c["strong_total"] > 0 else float("nan"))
        out[model] = {
            "cfr": cfr,
            "udr": udr,
            "n_weak": c["weak_total"],
            "n_strong": c["strong_total"],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# KDI — Knowing-Doing Index
# ─────────────────────────────────────────────────────────────────────────────

def compute_kdi(
    mirror_gap: float,
    appropriate_action_rate: float,
) -> float:
    """
    KDI = MIRROR self-knowledge accuracy − appropriate agentic action rate.

    where:
      mirror_gap            = MIRROR calibration gap for the domain (0–1)
      appropriate_action_rate = fraction of weak-domain components where the
                                model correctly deferred / used tools (0–1)

    High KDI (> 0) = model knows it's weak but still proceeds autonomously.
    KDI ≈ 0        = model acts on its self-knowledge.
    """
    return mirror_gap - appropriate_action_rate


def compute_kdi_table(
    subcategory_metrics: dict,
    mirror_gaps: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Compute KDI distribution per model.

    Args:
        subcategory_metrics: output of compute_cfr_udr_subcategory
        mirror_gaps: {model: {domain: gap_value}}

    Returns:
        {model: {
            "mean_kdi": float,
            "median_kdi": float,
            "proportion_kdi_gt_0_2": float,
            "per_domain": {domain: float}
        }}
    """
    out = {}
    for model, domains in subcategory_metrics.items():
        model_kdis: list[float] = []
        per_domain: dict[str, float] = {}

        for domain, subcats in domains.items():
            # Appropriate action rate: fraction of weak components where model deferred/tool
            total_weak = sum(c["n_weak"] for c in subcats.values())
            auto_failures = sum(c["autonomous_failures"] for c in subcats.values())
            # "appropriate" = did NOT fail autonomously on weak domain
            # = (deferred + tool_used) / total_weak
            # We approximate: appropriate_rate = 1 - cfr
            if total_weak > 0:
                cfr = auto_failures / total_weak
                appropriate_rate = 1.0 - cfr
            else:
                continue

            gap = mirror_gaps.get(model, {}).get(domain, float("nan"))
            if math.isnan(gap):
                continue

            kdi = compute_kdi(gap, appropriate_rate)
            model_kdis.append(kdi)
            per_domain[domain] = round(kdi, 4)

        if not model_kdis:
            continue

        sorted_kdis = sorted(model_kdis)
        n = len(sorted_kdis)
        median = (sorted_kdis[n // 2] if n % 2 == 1
                  else (sorted_kdis[n // 2 - 1] + sorted_kdis[n // 2]) / 2)

        out[model] = {
            "mean_kdi": round(sum(model_kdis) / n, 4),
            "median_kdi": round(median, 4),
            "proportion_kdi_gt_0_2": round(
                sum(1 for k in model_kdis if k > 0.2) / n, 4
            ),
            "per_domain": per_domain,
        }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Paradigm 3 behavioral signal aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_paradigm3_signals(
    results: list[dict],
    *,
    condition: Optional[int] = 1,
    circularity_free_only: bool = False,
) -> dict[str, dict[str, dict]]:
    """
    Aggregate Paradigm 3 behavioral signals per (model, domain).

    Returns:
        {model: {domain: {
            "mean_hedge_rate_weak": float,
            "mean_hedge_rate_strong": float,
            "mean_decomp_weak": float,
            "mean_decomp_strong": float,
            "token_ratio": float,  # mean tokens(weak) / mean tokens(strong)
            "overconfident_error_proportion_weak": float,
            "n_weak": int,
            "n_strong": int,
        }}}
    """
    acc: dict = defaultdict(lambda: defaultdict(lambda: {
        "hedge_weak": [], "hedge_strong": [],
        "decomp_weak": [], "decomp_strong": [],
        "tokens_weak": [], "tokens_strong": [],
        "error_types_weak": [],
    }))

    for trial in results:
        if trial.get("paradigm") != 3:
            continue
        if condition is not None and trial.get("condition") != condition:
            continue
        if circularity_free_only and not trial.get("circularity_free", False):
            continue

        model = trial.get("model")
        if not model:
            continue

        for slot in ("a", "b"):
            domain = trial.get(f"domain_{slot}")
            strength = trial.get(f"strength_{slot}")
            if not domain:
                continue

            c = acc[model][domain]
            hedge = trial.get(f"hedge_count_{slot}", 0)
            decomp = trial.get(f"decomp_count_{slot}", 0)
            tokens = trial.get(f"token_count_{slot}", 0)
            error_type = trial.get(f"error_type_{slot}")

            if strength == "weak":
                c["hedge_weak"].append(hedge)
                c["decomp_weak"].append(decomp)
                c["tokens_weak"].append(tokens)
                if error_type:
                    c["error_types_weak"].append(error_type)
            elif strength == "strong":
                c["hedge_strong"].append(hedge)
                c["decomp_strong"].append(decomp)
                c["tokens_strong"].append(tokens)

    def mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    out: dict = {}
    for model, domains in acc.items():
        out[model] = {}
        for domain, c in domains.items():
            mean_tokens_weak = mean(c["tokens_weak"])
            mean_tokens_strong = mean(c["tokens_strong"])
            token_ratio = (mean_tokens_weak / mean_tokens_strong
                           if mean_tokens_strong > 0 else float("nan"))

            n_weak_errors = len(c["error_types_weak"])
            overconf_prop = (
                sum(1 for e in c["error_types_weak"] if e == "overconfident_precise")
                / n_weak_errors
                if n_weak_errors > 0 else float("nan")
            )

            out[model][domain] = {
                "mean_hedge_rate_weak": mean(c["hedge_weak"]),
                "mean_hedge_rate_strong": mean(c["hedge_strong"]),
                "mean_decomp_weak": mean(c["decomp_weak"]),
                "mean_decomp_strong": mean(c["decomp_strong"]),
                "token_ratio": token_ratio,
                "overconfident_error_proportion_weak": overconf_prop,
                "n_weak": len(c["hedge_weak"]),
                "n_strong": len(c["hedge_strong"]),
            }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Oracle baseline — Control 5
# ─────────────────────────────────────────────────────────────────────────────

def compute_oracle_cfr(
    results: list[dict],
    accuracy_threshold: float = 0.50,
    *,
    circularity_free_only: bool = False,
) -> dict[str, dict]:
    """
    Control 5: Calibrated Oracle Baseline.

    A "perfectly metacognitive agent" always defers when Experiment 1 domain
    accuracy < accuracy_threshold and proceeds when >= threshold.

    Args:
        results: trial records that include exp1_accuracy_{a,b}.
        accuracy_threshold: default 0.50.

    Returns:
        {model: {"oracle_cfr": float, "oracle_udr": float, "n_weak": int, "n_strong": int}}
    """
    acc: dict = defaultdict(lambda: {
        "weak_total": 0, "oracle_auto_fail": 0,
        "strong_total": 0, "oracle_defer_would_succeed": 0,
    })

    for trial in results:
        if circularity_free_only and not trial.get("circularity_free", False):
            continue

        model = trial.get("model")
        if not model:
            continue

        for slot in ("a", "b"):
            strength = trial.get(f"strength_{slot}")
            correct = trial.get(f"component_{slot}_correct", False)
            exp1_acc = trial.get(f"exp1_accuracy_{slot}")

            if exp1_acc is None:
                continue

            c = acc[model]
            oracle_defers = exp1_acc < accuracy_threshold

            if strength == "weak":
                c["weak_total"] += 1
                # Oracle defers on weak domains → no autonomous failures in weak
                if not oracle_defers and not correct:
                    c["oracle_auto_fail"] += 1

            elif strength == "strong":
                c["strong_total"] += 1
                # Oracle proceeds on strong domains → no unnecessary deferrals
                if oracle_defers and correct:
                    c["oracle_defer_would_succeed"] += 1

    out = {}
    for model, c in acc.items():
        out[model] = {
            "oracle_cfr": (c["oracle_auto_fail"] / c["weak_total"]
                           if c["weak_total"] > 0 else float("nan")),
            "oracle_udr": (c["oracle_defer_would_succeed"] / c["strong_total"]
                           if c["strong_total"] > 0 else float("nan")),
            "n_weak": c["weak_total"],
            "n_strong": c["strong_total"],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routing comparison — three strategies
# ─────────────────────────────────────────────────────────────────────────────

def compute_routing_comparison(
    results: list[dict],
    *,
    accuracy_threshold: float = 0.50,
    mirror_gap_threshold: float = 0.20,
    condition: int = 1,
    circularity_free_only: bool = False,
) -> dict[str, dict]:
    """
    Three routing strategies on the fixed task set:
      no_routing         — model handles all components (Condition 1 results)
      accuracy_routing   — route to tool/defer when exp1_accuracy < threshold
      mirror_routing     — route to tool/defer when mirror_gap > gap_threshold
      oracle_routing     — route based on oracle (exp1_accuracy < 0.5)

    Returns per-model CFR under each strategy.
    """
    acc: dict = defaultdict(lambda: {
        "no":       {"total": 0, "fail": 0},
        "accuracy": {"total": 0, "fail": 0},
        "mirror":   {"total": 0, "fail": 0},
        "oracle":   {"total": 0, "fail": 0},
    })

    for trial in results:
        if trial.get("condition") != condition:
            continue
        if circularity_free_only and not trial.get("circularity_free", False):
            continue

        model = trial.get("model")
        if not model:
            continue

        for slot in ("a", "b"):
            strength = trial.get(f"strength_{slot}")
            if strength != "weak":
                continue

            correct = trial.get(f"component_{slot}_correct", False)
            decision = trial.get(f"component_{slot}_decision", "proceed")
            exp1_acc = trial.get(f"exp1_accuracy_{slot}")
            mirror_gap = trial.get(f"mirror_gap_{slot}")

            c = acc[model]

            # No routing: use actual decision from Condition 1
            c["no"]["total"] += 1
            if decision == "proceed" and not correct:
                c["no"]["fail"] += 1

            # Accuracy routing: force defer if exp1_acc < threshold
            c["accuracy"]["total"] += 1
            if exp1_acc is not None and exp1_acc >= accuracy_threshold:
                if decision == "proceed" and not correct:
                    c["accuracy"]["fail"] += 1
            # else: forced defer → 0 confident failures from this component

            # MIRROR routing: force defer if mirror_gap > gap_threshold
            c["mirror"]["total"] += 1
            if mirror_gap is not None and mirror_gap <= mirror_gap_threshold:
                if decision == "proceed" and not correct:
                    c["mirror"]["fail"] += 1

            # Oracle routing: force defer if exp1_acc < 0.5
            c["oracle"]["total"] += 1
            if exp1_acc is not None and exp1_acc >= 0.5:
                if decision == "proceed" and not correct:
                    c["oracle"]["fail"] += 1

    out = {}
    for model, strategies in acc.items():
        out[model] = {}
        for strategy, c in strategies.items():
            out[model][strategy] = {
                "cfr": c["fail"] / c["total"] if c["total"] > 0 else float("nan"),
                "n": c["total"],
            }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Null result reporting
# ─────────────────────────────────────────────────────────────────────────────

def interpret_partial_r(partial_r: float, threshold: float = 0.10) -> str:
    """
    Return a pre-registered interpretation block for partial r results.

    Per spec: if partial r ≈ 0, report honestly rather than hiding the finding.
    """
    if math.isnan(partial_r):
        return (
            "INTERPRETATION: Partial correlation could not be computed "
            "(insufficient data)."
        )
    if abs(partial_r) < threshold:
        return (
            "INTERPRETATION [NULL RESULT]: Partial r ≈ 0. "
            "MIRROR calibration gap does not predict agentic failure rate above and beyond "
            "raw accuracy. This suggests CFR is primarily driven by competence (raw accuracy), "
            "not by metacognitive calibration independently of competence. "
            "MIRROR's contribution in this analysis is diagnostic — explaining *where* and "
            "*why* failures concentrate — rather than uniquely predictive. "
            "This finding is consistent with Pre-registration Contingency C1 and is reported "
            "without adjustment. Higher-level MIRROR scores (Level 2 CCE, Level 3 adaptation) "
            "may still show independent predictive power; see the partial correlation table."
        )
    direction = "positive" if partial_r > 0 else "negative"
    return (
        f"INTERPRETATION: Partial r = {partial_r:.3f} ({direction}). "
        "MIRROR calibration gap predicts agentic failure rate above and beyond raw accuracy."
    )


def interpret_paradigm3_null(r_behavioral: float, threshold: float = 0.15) -> str:
    """Interpretation for Paradigm 3 behavioral signal correlations."""
    if math.isnan(r_behavioral):
        return "INTERPRETATION: Paradigm 3 correlation could not be computed."
    if abs(r_behavioral) < threshold:
        return (
            "INTERPRETATION [PARTIAL NULL — RLHF CONFOUND NOT ELIMINATED]: "
            "Paradigm 3 behavioral signals (hedging, decomposition, token allocation) do not "
            "correlate meaningfully with MIRROR calibration gap (r < 0.15). "
            "The knowing-doing gap signal may be partially or fully driven by RLHF tool-use "
            "training patterns rather than genuine metacognitive calibration. "
            "This weakens the causal interpretation but does not negate the descriptive findings."
        )
    return (
        f"INTERPRETATION: Paradigm 3 behavioral correlation r = {r_behavioral:.3f}. "
        "No-tool behavioral signals track MIRROR calibration gap. The RLHF tool-use "
        "confound is substantially mitigated."
    )
