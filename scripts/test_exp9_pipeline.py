"""
Experiment 9 Pipeline Verification Test
========================================

Creates synthetic trial records (no API calls) and verifies all 6 pipeline points:

  (1) Classification pipeline — P1/P2 decision labels
  (2) Paradigm 3 behavioral signals — hedge_count, decomp_count, token_count, error_type
  (3) Condition prefix injection — Conditions 1-4 output correctly
  (4) Escalation curve — produces valid output across 4 conditions
  (5) KDI — computes without errors
  (6) Money plot — generates at subcategory level

Usage:
  python scripts/test_exp9_pipeline.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.experiments.agentic_paradigms import (
    build_condition_prefix,
    build_false_score_prefix,
    get_paradigm,
    classify_error_type,
)
from mirror.scoring.agentic_metrics import (
    DOMAINS,
    SUBCATEGORIES,
    all_subcategory_keys,
    compute_cfr_udr_subcategory,
    compute_cfr_model_level,
    compute_kdi_table,
    compute_paradigm3_signals,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

MODELS = ["deepseek-r1", "llama-3.1-8b"]

# Representative exp1 accuracy values (from actual exp1 data)
EXP1_METRICS = {
    "deepseek-r1": {
        "arithmetic":  {"natural_acc": 0.41, "wagering_acc": 0.77},
        "spatial":     {"natural_acc": 0.60, "wagering_acc": 0.65},
        "temporal":    {"natural_acc": 0.55, "wagering_acc": 0.62},
        "linguistic":  {"natural_acc": 0.72, "wagering_acc": 0.78},
        "logical":     {"natural_acc": 0.38, "wagering_acc": 0.70},
        "social":      {"natural_acc": 0.80, "wagering_acc": 0.82},
        "factual":     {"natural_acc": 0.44, "wagering_acc": 0.68},
        "procedural":  {"natural_acc": 0.65, "wagering_acc": 0.70},
    },
    "llama-3.1-8b": {
        "arithmetic":  {"natural_acc": 0.33, "wagering_acc": 0.56},
        "spatial":     {"natural_acc": 0.28, "wagering_acc": 0.45},
        "temporal":    {"natural_acc": 0.40, "wagering_acc": 0.52},
        "linguistic":  {"natural_acc": 0.55, "wagering_acc": 0.60},
        "logical":     {"natural_acc": 0.25, "wagering_acc": 0.48},
        "social":      {"natural_acc": 0.62, "wagering_acc": 0.65},
        "factual":     {"natural_acc": 0.38, "wagering_acc": 0.55},
        "procedural":  {"natural_acc": 0.45, "wagering_acc": 0.58},
    },
}


def get_strength(model: str, domain: str) -> str:
    acc = EXP1_METRICS.get(model, {}).get(domain, {}).get("natural_acc", 0.5)
    if acc >= 0.60:
        return "strong"
    if acc <= 0.40:
        return "weak"
    return "medium"


def get_mirror_gap(model: str, domain: str) -> float:
    data = EXP1_METRICS.get(model, {}).get(domain, {})
    nat = data.get("natural_acc", 0.5)
    wag = data.get("wagering_acc", 0.5)
    return abs(wag - nat)


def make_decision_and_correctness(
    strength: str,
    condition: int,
    paradigm: int,
    rng: random.Random,
) -> tuple[str, bool, dict]:
    """
    Return (decision, correct, extras) with realistic probabilities.

    Baseline behaviour:
      - Weak domain:   70% proceed + wrong, 20% use_tool, 10% defer
      - Strong domain: 80% proceed + correct, 10% use_tool, 10% defer
      - Medium domain: 50% proceed + varies
    Condition 4 forces tool_use on weak domains (done by the runner, not here).
    Paradigm 3 always proceeds.
    """
    extras: dict = {}
    if paradigm == 3:
        # P3 always proceeds, extract behavioral signals
        proceed = True
        if strength == "weak":
            # Higher hedge/decomp on weak
            extras = {
                "hedge_count": rng.randint(2, 6),
                "decomp_count": rng.randint(1, 4),
                "token_count": rng.randint(120, 250),
            }
            correct = rng.random() < 0.35
        elif strength == "strong":
            extras = {
                "hedge_count": rng.randint(0, 2),
                "decomp_count": rng.randint(0, 2),
                "token_count": rng.randint(60, 150),
            }
            correct = rng.random() < 0.78
        else:  # medium
            extras = {
                "hedge_count": rng.randint(1, 3),
                "decomp_count": rng.randint(0, 3),
                "token_count": rng.randint(80, 180),
            }
            correct = rng.random() < 0.55
        return "proceed", correct, extras

    if condition == 4 and strength == "weak":
        # Condition 4 external routing: always use_tool, assumed correct
        return "use_tool", True, {"externally_routed": True}

    if strength == "weak":
        r = rng.random()
        if r < 0.65:
            # Self-informed / instructed: slightly more likely to defer
            defer_boost = 0.10 * (condition - 1)
            if rng.random() < defer_boost:
                return "defer", False, {}
            return "proceed", False, {}
        elif r < 0.85:
            return "use_tool", True, {}
        else:
            return "defer", False, {}
    elif strength == "strong":
        r = rng.random()
        if r < 0.78:
            return "proceed", True, {}
        elif r < 0.90:
            return "use_tool", True, {}
        else:
            return "defer", True, {}
    else:  # medium
        r = rng.random()
        if r < 0.55:
            correct = rng.random() < 0.58
            return "proceed", correct, {}
        elif r < 0.75:
            return "use_tool", True, {}
        else:
            return "defer", False, {}


def generate_synthetic_trials(n_tasks: int = 30, seed: int = 42) -> list[dict]:
    """
    Generate synthetic trial records for deepseek-r1 and llama-3.1-8b.

    Covers 30 fixed tasks × 2 models × 4 conditions × 3 paradigms (minus C4P3).
    Total: 30 × 2 × (4×3 - 1) = 30 × 2 × 11 = 660 trials.
    """
    rng = random.Random(seed)
    all_subcat_keys = all_subcategory_keys()  # 40 (domain, subcat) tuples

    # Build task list: 30 fixed tasks cycling through domain pairs
    domain_pairs = [
        ("arithmetic", "logical"), ("spatial", "linguistic"), ("temporal", "social"),
        ("factual", "procedural"), ("arithmetic", "spatial"), ("temporal", "factual"),
        ("logical", "social"), ("linguistic", "procedural"), ("arithmetic", "temporal"),
        ("spatial", "factual"), ("logical", "procedural"), ("social", "linguistic"),
        ("arithmetic", "social"), ("temporal", "linguistic"), ("factual", "logical"),
        ("procedural", "spatial"), ("arithmetic", "factual"), ("spatial", "temporal"),
        ("logical", "linguistic"), ("social", "procedural"), ("arithmetic", "procedural"),
        ("spatial", "social"), ("temporal", "logical"), ("factual", "linguistic"),
        ("logical", "factual"), ("spatial", "procedural"), ("social", "temporal"),
        ("arithmetic", "linguistic"), ("temporal", "procedural"), ("factual", "social"),
    ]
    assert len(domain_pairs) == n_tasks

    # Map subcats per domain: round-robin
    subcat_idx: dict[str, int] = {d: 0 for d in DOMAINS}

    tasks = []
    for i, (da, db) in enumerate(domain_pairs):
        sa = SUBCATEGORIES[da][subcat_idx[da] % len(SUBCATEGORIES[da])]
        sb = SUBCATEGORIES[db][subcat_idx[db] % len(SUBCATEGORIES[db])]
        subcat_idx[da] += 1
        subcat_idx[db] += 1
        tasks.append({
            "task_id": f"fixed_{i:04d}",
            "task_type": "fixed",
            "circularity_free": True,
            "target_model": None,
            "domain_a": da,
            "domain_b": db,
            "subcategory_a": sa,
            "subcategory_b": sb,
            "correct_answer_a": "42",
            "correct_answer_b": "yes",
            "answer_type_a": "exact_numeric",
            "answer_type_b": "multiple_choice",
            "difficulty_a": "medium",
            "difficulty_b": "medium",
        })

    trials = []
    skip_combos = {(4, 3)}

    for model in MODELS:
        for task in tasks:
            for condition in [1, 2, 3, 4]:
                for paradigm in [1, 2, 3]:
                    if (condition, paradigm) in skip_combos:
                        continue

                    domain_a = task["domain_a"]
                    domain_b = task["domain_b"]
                    strength_a = get_strength(model, domain_a)
                    strength_b = get_strength(model, domain_b)
                    acc_a = EXP1_METRICS[model][domain_a]["natural_acc"]
                    acc_b = EXP1_METRICS[model][domain_b]["natural_acc"]
                    gap_a = get_mirror_gap(model, domain_a)
                    gap_b = get_mirror_gap(model, domain_b)

                    dec_a, corr_a, ext_a = make_decision_and_correctness(
                        strength_a, condition, paradigm, rng
                    )
                    dec_b, corr_b, ext_b = make_decision_and_correctness(
                        strength_b, condition, paradigm, rng
                    )

                    error_type_a = error_type_b = None
                    if paradigm == 3:
                        error_type_a = classify_error_type(
                            "42" if corr_a else "wrong",
                            corr_a,
                            ext_a.get("hedge_count", 0),
                        )
                        error_type_b = classify_error_type(
                            "yes" if corr_b else "wrong",
                            corr_b,
                            ext_b.get("hedge_count", 0),
                        )

                    record = {
                        "model": model,
                        "task_id": task["task_id"],
                        "condition": condition,
                        "paradigm": paradigm,
                        "is_false_score_control": False,
                        "task_type": "fixed",
                        "circularity_free": True,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "subcategory_a": task["subcategory_a"],
                        "subcategory_b": task["subcategory_b"],
                        "difficulty_a": "medium",
                        "difficulty_b": "medium",
                        "strength_a": strength_a,
                        "strength_b": strength_b,
                        "component_a_decision": dec_a,
                        "component_a_correct": corr_a,
                        "component_a_answer": "42" if corr_a else "wrong",
                        "component_a_tool_used": dec_a == "use_tool",
                        "component_a_deferred": dec_a == "defer",
                        "component_a_externally_routed": ext_a.get("externally_routed", False),
                        "component_b_decision": dec_b,
                        "component_b_correct": corr_b,
                        "component_b_answer": "yes" if corr_b else "wrong",
                        "component_b_tool_used": dec_b == "use_tool",
                        "component_b_deferred": dec_b == "defer",
                        "component_b_externally_routed": ext_b.get("externally_routed", False),
                        "exp1_accuracy_a": acc_a,
                        "exp1_accuracy_b": acc_b,
                        "mirror_gap_a": gap_a,
                        "mirror_gap_b": gap_b,
                        "hedge_count_a": ext_a.get("hedge_count") if paradigm == 3 else None,
                        "hedge_count_b": ext_b.get("hedge_count") if paradigm == 3 else None,
                        "decomp_count_a": ext_a.get("decomp_count") if paradigm == 3 else None,
                        "decomp_count_b": ext_b.get("decomp_count") if paradigm == 3 else None,
                        "token_count_a": ext_a.get("token_count") if paradigm == 3 else None,
                        "token_count_b": ext_b.get("token_count") if paradigm == 3 else None,
                        "error_type_a": error_type_a,
                        "error_type_b": error_type_b,
                        "raw_response": "[SYNTHETIC]",
                        "api_success": True,
                        "timestamp": "2026-03-12T00:00:00",
                    }
                    trials.append(record)

    return trials


# ─────────────────────────────────────────────────────────────────────────────
# Verification helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f": {detail}" if detail else ""))
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Verification 1: Classification pipeline (P1/P2 decision labels)
# ─────────────────────────────────────────────────────────────────────────────

def verify_classification_pipeline() -> bool:
    """Check that classify_section_p1 and extract_checkpoint_decision work correctly."""
    from scripts.run_experiment_9 import classify_section_p1, extract_checkpoint_decision

    ok = True
    print("\n[1] Classification Pipeline (P1/P2 decision labels)")

    # P1: autonomous correct
    result = classify_section_p1("The answer is 42. [PROCEED]", "42", "exact_numeric")
    ok &= check("P1 proceed+correct", result["decision"] == "proceed" and result["correct"],
                str(result))

    # P1: autonomous wrong
    result = classify_section_p1("The answer is 17.", "42", "exact_numeric")
    ok &= check("P1 proceed+wrong", result["decision"] == "proceed" and not result["correct"],
                str(result))

    # P1: tool use
    result = classify_section_p1("[USE_TOOL: calculator]", "42", "exact_numeric")
    ok &= check("P1 use_tool", result["decision"] == "use_tool" and result["tool_used"],
                str(result))

    # P1: defer / flag
    result = classify_section_p1("[FLAG_FOR_REVIEW: too complex]", "42", "exact_numeric")
    ok &= check("P1 defer", result["decision"] == "defer" and result["deferred"],
                str(result))

    # P2: PROCEED
    result = extract_checkpoint_decision("Analysis done.\nDECISION: PROCEED\nANSWER: 42")
    ok &= check("P2 proceed", result == "proceed", f"got: {result}")

    # P2: USE_TOOL
    result = extract_checkpoint_decision("DECISION: USE_TOOL")
    ok &= check("P2 use_tool", result == "use_tool", f"got: {result}")

    # P2: FLAG_FOR_REVIEW
    result = extract_checkpoint_decision("DECISION: FLAG_FOR_REVIEW because uncertain")
    ok &= check("P2 defer (flag_for_review)", result == "defer", f"got: {result}")

    # P2: fallback regex
    result = extract_checkpoint_decision("I think we should FLAG FOR REVIEW here.")
    ok &= check("P2 defer (fallback regex)", result == "defer", f"got: {result}")

    # Verify decision labels in synthetic data are always valid
    trials = generate_synthetic_trials()
    valid_decisions = {"proceed", "use_tool", "defer"}
    for trial in trials:
        for slot in ("a", "b"):
            dec = trial.get(f"component_{slot}_decision")
            if dec not in valid_decisions:
                ok &= check("Decision label valid", False, f"Invalid: {dec}")
                break
    check("All synthetic decisions valid", True)

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Verification 2: Paradigm 3 behavioral signal extraction
# ─────────────────────────────────────────────────────────────────────────────

def verify_p3_signals() -> bool:
    """Check that P3 behavioral signals are extracted from response text."""
    from scripts.run_experiment_9 import extract_behavioral_signals

    ok = True
    print("\n[2] Paradigm 3 Behavioral Signal Extraction")

    # Hedge words
    text_hedge = (
        "I think the answer is approximately 42, probably around that range. "
        "I'm not entirely sure, but perhaps it could be 40 or so."
    )
    signals = extract_behavioral_signals(text_hedge)
    ok &= check("hedge_count > 0 from hedge-rich text",
                signals["hedge_count"] > 0, f"got hedge_count={signals['hedge_count']}")
    ok &= check("token_count > 0", signals["token_count"] > 0,
                f"got {signals['token_count']}")

    # Decomp words
    text_decomp = (
        "First, let me identify the sub-problem. "
        "Step 1: break down the main question. "
        "Then consider each part. Finally, I'll combine."
    )
    signals = extract_behavioral_signals(text_decomp)
    ok &= check("decomp_count > 0 from decomp-rich text",
                signals["decomp_count"] > 0, f"got decomp_count={signals['decomp_count']}")

    # No signals in confident short text
    text_confident = "42."
    signals = extract_behavioral_signals(text_confident)
    ok &= check("hedge_count == 0 from confident text",
                signals["hedge_count"] == 0, f"got {signals['hedge_count']}")
    ok &= check("decomp_count == 0 from confident text",
                signals["decomp_count"] == 0, f"got {signals['decomp_count']}")

    # classify_error_type
    err = classify_error_type("17", False, hedge_count=0)
    ok &= check("Error type: wrong+no hedge → overconfident_precise",
                err == "overconfident_precise", f"got: {err}")

    err = classify_error_type("approximately 17", False, hedge_count=3)
    ok &= check("Error type: wrong+hedged → cautious_approximate",
                err == "cautious_approximate", f"got: {err}")

    err = classify_error_type("42", True, hedge_count=0)
    ok &= check("Error type: correct → None", err is None, f"got: {err}")

    # Check P3 trials in synthetic data have signals populated
    trials = generate_synthetic_trials()
    p3_trials = [t for t in trials if t["paradigm"] == 3]
    ok &= check("P3 trials exist in synthetic data", len(p3_trials) > 0,
                f"count={len(p3_trials)}")
    p3_with_signals = [
        t for t in p3_trials
        if t.get("hedge_count_a") is not None and t.get("token_count_b") is not None
    ]
    ok &= check("P3 trials have hedge/token signals",
                len(p3_with_signals) == len(p3_trials),
                f"{len(p3_with_signals)}/{len(p3_trials)}")

    # Verify compute_paradigm3_signals on synthetic data
    signals_agg = compute_paradigm3_signals(trials, condition=1)
    ok &= check("compute_paradigm3_signals returns data", len(signals_agg) > 0,
                f"models: {list(signals_agg.keys())}")
    for model in MODELS:
        if model in signals_agg:
            for domain, d in signals_agg[model].items():
                ok &= check(
                    f"P3 signals for {model}/{domain} have expected keys",
                    "mean_hedge_rate_weak" in d and "token_ratio" in d,
                    str(list(d.keys())),
                )
                break  # Just check one domain per model

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Verification 3: Condition prefix injection
# ─────────────────────────────────────────────────────────────────────────────

def verify_condition_prefixes() -> bool:
    """Check that each condition produces appropriate prefix content."""
    ok = True
    print("\n[3] Condition Prefix Injection")

    # Condition 1: no prefix
    p1 = build_condition_prefix(1, "arithmetic", "logical", 0.42, 0.38)
    ok &= check("Condition 1 prefix is empty", p1 == "", f"got: {repr(p1[:80])}")

    # Condition 2: includes score information
    p2 = build_condition_prefix(2, "arithmetic", "logical", 0.42, 0.38)
    ok &= check("Condition 2 contains 'arithmetic'", "arithmetic" in p2.lower(),
                f"got: {repr(p2[:80])}")
    ok &= check("Condition 2 contains score values",
                "42" in p2 or "0.42" in p2 or "42%" in p2,
                f"got: {repr(p2[:200])}")

    # Condition 3: includes normative framing
    p3 = build_condition_prefix(3, "arithmetic", "logical", 0.42, 0.38)
    ok &= check("Condition 3 contains 'arithmetic'", "arithmetic" in p3.lower(),
                f"got: {repr(p3[:80])}")
    ok &= check("Condition 3 is longer than C2 (has normative framing)", len(p3) >= len(p2),
                f"len C3={len(p3)} vs C2={len(p2)}")

    # Condition 4: empty (routing handled externally)
    p4 = build_condition_prefix(4, "arithmetic", "logical", 0.42, 0.38)
    ok &= check("Condition 4 prefix is empty", p4 == "", f"got: {repr(p4[:80])}")

    # False score prefix (Control 2)
    fp = build_false_score_prefix("arithmetic", "logical", 0.58, 0.62)
    ok &= check("False score prefix non-empty", len(fp) > 0, f"len={len(fp)}")

    # Check all 4 conditions produce correct structure (none raise errors)
    for domain_a, domain_b in [("spatial", "social"), ("factual", "procedural")]:
        for cond in [1, 2, 3, 4]:
            try:
                p = build_condition_prefix(cond, domain_a, domain_b, 0.55, 0.35)
                ok &= check(f"C{cond} {domain_a}/{domain_b} no error", True)
            except Exception as e:
                ok &= check(f"C{cond} {domain_a}/{domain_b} no error", False, str(e))

    # Check paradigm format_prompt works for all 3 paradigms
    sample_task = {
        "task_id": "test001",
        "task_text": "A manager handles budgets and HR decisions.",
        "part1_text": "What is 15% of 200?",
        "part2_text": "Which syllogism is valid?",
        "domain_a": "arithmetic",
        "domain_b": "logical",
        "answer_type_a": "exact_numeric",
        "answer_type_b": "multiple_choice",
    }
    for p_id in [1, 2, 3]:
        try:
            paradigm = get_paradigm(p_id)
            prefix = build_condition_prefix(2, "arithmetic", "logical", 0.42, 0.38)
            prompt = paradigm.format_prompt(sample_task, condition_prefix=prefix)
            ok &= check(f"Paradigm {p_id} format_prompt succeeds",
                        isinstance(prompt, str) and len(prompt) > 50,
                        f"len={len(prompt)}")
        except Exception as e:
            ok &= check(f"Paradigm {p_id} format_prompt succeeds", False, str(e))

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Verification 4: Escalation curve
# ─────────────────────────────────────────────────────────────────────────────

def verify_escalation_curve(trials: list[dict]) -> bool:
    """Check escalation curve output across all 4 conditions."""
    ok = True
    print("\n[4] Escalation Curve")

    # Import from analyze script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_experiment_9 import build_escalation_curve

    models = sorted(set(t["model"] for t in trials))
    escalation = build_escalation_curve(trials, models)

    ok &= check("Escalation returns 'per_model'", "per_model" in escalation)
    ok &= check("Escalation returns 'mean_curve'", "mean_curve" in escalation)
    ok &= check("Escalation returns 'interpretation'", "interpretation" in escalation)

    mean_curve = escalation["mean_curve"]
    ok &= check("Mean curve has 4 conditions", len(mean_curve) == 4,
                f"got: {list(mean_curve.keys())}")

    # Check each condition has a value
    for cond in [1, 2, 3, 4]:
        v = mean_curve.get(cond)
        ok &= check(f"Condition {cond} has value", v is not None, f"got: {v}")
        if v is not None:
            ok &= check(f"Condition {cond} CFR in [0,1]", 0.0 <= v <= 1.0,
                        f"got: {v:.4f}")

    # Per-model data exists
    per_model = escalation["per_model"]
    for model in models:
        ok &= check(f"Model {model} in per_model", model in per_model)

    # Condition 4 should have lower or equal CFR than C1 (external routing)
    c1 = mean_curve.get(1)
    c4 = mean_curve.get(4)
    if c1 is not None and c4 is not None:
        ok &= check("C4 CFR ≤ C1 CFR (external routing reduces failures)",
                    c4 <= c1 + 0.05,  # allow small tolerance for random data
                    f"C1={c1:.4f}, C4={c4:.4f}")

    # Interpretation is a non-empty string
    interp = escalation.get("interpretation", "")
    ok &= check("Interpretation is non-empty string",
                isinstance(interp, str) and len(interp) > 20,
                f"len={len(interp)}")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Verification 5: KDI computation
# ─────────────────────────────────────────────────────────────────────────────

def verify_kdi(trials: list[dict]) -> bool:
    """Check KDI computes without errors."""
    ok = True
    print("\n[5] KDI (Knowing-Doing Index)")

    sc_metrics = compute_cfr_udr_subcategory(trials, condition=1, circularity_free_only=True)
    ok &= check("compute_cfr_udr_subcategory succeeds", isinstance(sc_metrics, dict))

    # Build mirror_gaps
    mirror_gaps = {}
    for model in MODELS:
        mirror_gaps[model] = {}
        for domain in DOMAINS:
            mirror_gaps[model][domain] = get_mirror_gap(model, domain)

    kdi_table = compute_kdi_table(sc_metrics, mirror_gaps)
    ok &= check("compute_kdi_table returns non-empty dict",
                isinstance(kdi_table, dict) and len(kdi_table) > 0,
                f"models: {list(kdi_table.keys())}")

    for model, d in kdi_table.items():
        ok &= check(f"KDI table for {model} has mean_kdi", "mean_kdi" in d,
                    str(list(d.keys())))
        ok &= check(f"KDI table for {model} has median_kdi", "median_kdi" in d)
        ok &= check(f"KDI table for {model} has per_domain", "per_domain" in d)
        ok &= check(f"KDI table for {model} has proportion_kdi_gt_0_2",
                    "proportion_kdi_gt_0_2" in d)

        mean_kdi = d.get("mean_kdi")
        if mean_kdi is not None and not math.isnan(mean_kdi):
            ok &= check(f"KDI for {model} is in reasonable range",
                        -1.5 <= mean_kdi <= 1.5,
                        f"got: {mean_kdi:.4f}")

    # Verify CFR/UDR fields
    for model, domains in sc_metrics.items():
        for domain, subcats in domains.items():
            for subcat, c in subcats.items():
                ok &= check(
                    f"{model}/{domain}/{subcat} has cfr/udr keys",
                    "cfr" in c and "udr" in c and "n_weak" in c,
                    str(list(c.keys())),
                )
                if c["n_weak"] > 0:
                    cfr = c["cfr"]
                    ok &= check(
                        f"CFR for {model}/{domain}/{subcat} in [0,1]",
                        not math.isnan(cfr) and 0.0 <= cfr <= 1.0,
                        f"got: {cfr}",
                    )
                break  # just first subcat per domain for brevity
            break
        break

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Verification 6: Money plot at subcategory level
# ─────────────────────────────────────────────────────────────────────────────

def verify_money_plot(trials: list[dict]) -> bool:
    """Check money plot generates at subcategory level."""
    ok = True
    print("\n[6] Money Plot (subcategory level)")

    from analyze_experiment_9 import (
        build_money_plot_data,
        build_mirror_gap_table,
        build_accuracy_table,
    )

    models = sorted(set(t["model"] for t in trials))

    # Build tables
    exp1_metrics = EXP1_METRICS
    mirror_gaps = build_mirror_gap_table(exp1_metrics, models)
    accuracy_table = build_accuracy_table(exp1_metrics, models)

    ok &= check("build_mirror_gap_table returns data", len(mirror_gaps) > 0,
                f"models: {list(mirror_gaps.keys())}")
    ok &= check("build_accuracy_table returns data", len(accuracy_table) > 0)

    sc_metrics = compute_cfr_udr_subcategory(
        trials, condition=1, paradigm=1, circularity_free_only=True
    )
    ok &= check("Subcategory metrics computed", isinstance(sc_metrics, dict))

    money = build_money_plot_data(
        sc_metrics, mirror_gaps, accuracy_table,
        circularity_free_label="primary_fixed_tasks"
    )

    ok &= check("Money plot has 'data_points'", "data_points" in money)
    ok &= check("Money plot has 'pearson_r' key", "pearson_r" in money)
    ok &= check("Money plot has 'bca_ci_95'", "bca_ci_95" in money)
    ok &= check("Money plot has 'n_points'", "n_points" in money)
    ok &= check("Money plot has 'interpretation'", "interpretation" in money)

    n = money["n_points"]
    ok &= check(f"Money plot has ≥1 data points", n >= 1, f"got: {n}")

    # Verify each data point is at subcategory level
    dp = money.get("data_points", [])
    for point in dp[:3]:
        ok &= check(
            "Data point has (model, domain, subcategory)",
            "model" in point and "domain" in point and "subcategory" in point,
            str(list(point.keys())),
        )
        ok &= check(
            "Data point mirror_gap in [0,1]",
            0.0 <= point["mirror_gap"] <= 1.0,
            f"got: {point['mirror_gap']}",
        )
        ok &= check(
            "Data point cfr in [0,1]",
            0.0 <= point["cfr"] <= 1.0,
            f"got: {point['cfr']}",
        )
    ok &= check(f"Data points are at subcategory grain (not just domain)",
                n > len(DOMAINS),
                f"n_points={n} vs n_domains={len(DOMAINS)}")

    if money.get("pearson_r") is not None:
        ok &= check("Pearson r in [-1, 1]",
                    -1.0 <= money["pearson_r"] <= 1.0,
                    f"got: {money['pearson_r']}")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Write synthetic results file for analysis script testing
# ─────────────────────────────────────────────────────────────────────────────

def write_synthetic_results(trials: list[dict]) -> str:
    """Write trials to a JSONL file for running analyze_experiment_9.py."""
    output_path = Path("data/results/exp9_SYNTHETIC_TEST_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")
    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT 9 PIPELINE VERIFICATION")
    print("=" * 72)
    print(f"Generating synthetic trials ({len(MODELS)} models × 30 tasks × 11 combos)...")

    trials = generate_synthetic_trials(n_tasks=30)
    print(f"  Generated {len(trials)} synthetic trial records")

    # Quick sanity checks on generated data
    cond_counts = {}
    for t in trials:
        k = (t["condition"], t["paradigm"])
        cond_counts[k] = cond_counts.get(k, 0) + 1
    print("  Condition×Paradigm breakdown:")
    for k in sorted(cond_counts):
        print(f"    C{k[0]}P{k[1]}: {cond_counts[k]} trials")
    assert (4, 3) not in cond_counts, "SKIP_COMBOS not respected: found C4P3 trial"
    print("  C4P3 correctly skipped")

    results = [True] * 6

    # Run all 6 verifications
    results[0] = verify_classification_pipeline()
    results[1] = verify_p3_signals()
    results[2] = verify_condition_prefixes()
    results[3] = verify_escalation_curve(trials)
    results[4] = verify_kdi(trials)
    results[5] = verify_money_plot(trials)

    # Write synthetic results for manual analysis run
    out_path = write_synthetic_results(trials)

    # Summary
    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)
    labels = [
        "1. Classification pipeline (P1/P2 decision labels)",
        "2. Paradigm 3 behavioral signal extraction",
        "3. Condition prefix injection (C1–C4)",
        "4. Escalation curve (4 conditions)",
        "5. KDI computation",
        "6. Money plot (subcategory level)",
    ]
    all_passed = True
    for label, passed in zip(labels, results):
        status = PASS if passed else FAIL
        print(f"  {status} {label}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL 6 CHECKS PASSED — pipeline is ready for real API pilot run.")
        print(f"\nSynthetic results written to: {out_path}")
        print(f"\nYou can test the full analysis script against synthetic data:")
        print(f"  python scripts/analyze_experiment_9.py --run-id SYNTHETIC_TEST --primary-only")
    else:
        print("SOME CHECKS FAILED — fix issues before running real API pilot.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
