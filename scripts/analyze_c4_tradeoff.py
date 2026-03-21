"""
C4 Escalation Trade-off Analysis for Experiment 9
==================================================

Condition 4 (Constrained) forces tool-use on weak-domain components via external
routing. This script computes the trade-off between correct escalations (failures
prevented) and false escalations (successes blocked) by comparing C4 decisions
against what would have happened under C1 (Uninformed / autonomous).

For each model with both C1 and C4 data:
  - Identifies weak-domain task components under C4
  - Computes escalation rate (fraction where decision != "proceed")
  - Cross-references C1 outcomes for the SAME task+slot to classify:
      correct_escalation = C1 was wrong (failure prevented)
      false_escalation   = C1 was right (success blocked)
  - Computes non-escalated success rate, autonomous completion rate
  - Net utility = (failures_prevented - successes_blocked) per 100 weak-domain tasks

Also analyzes strong-domain components under C4 to check for collateral UDR.

Output: paper/tables/c4_tradeoff_results.json + printed summary table.
"""

from __future__ import annotations

import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_exp9_records(run_id: str = "20260312T140842") -> list[dict]:
    """Load all JSONL records from the main results file and all shard files."""
    pattern = f"data/results/exp9_{run_id}*.jsonl"
    files = sorted(glob.glob(pattern))

    # Deduplicate by (model, task_id, condition, paradigm, is_false_score_control)
    seen = set()
    records = []

    for fpath in files:
        # Skip synthetic test file
        if "SYNTHETIC" in fpath:
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip failed API calls
                if not r.get("api_success", True):
                    continue

                key = (
                    r.get("model"),
                    r.get("task_id"),
                    r.get("condition"),
                    r.get("paradigm"),
                    r.get("is_false_score_control", False),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(r)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# C4 Escalation Trade-off Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_c4_tradeoff(records: list[dict]) -> dict:
    """
    Compute C4 escalation trade-off metrics for each model.

    For each model, we look at:
    1. Weak-domain components under C4 (Paradigms 1 & 2):
       - These are externally routed (decision="use_tool", externally_routed=True)
       - Compare against the SAME task+slot under C1 to see if C1 was correct
    2. Strong-domain components under C4:
       - These are NOT externally routed (model decides freely)
       - Check for unnecessary deferrals / tool use
    3. Non-weak components under C4 that were NOT externally routed:
       - Autonomous success rate
    """

    # Step 1: Build lookup for C1 outcomes: (model, task_id, paradigm, slot) -> correct
    c1_outcomes = {}  # key -> bool (correct)
    c1_decisions = {}  # key -> str (decision)

    for r in records:
        if r.get("condition") != 1:
            continue
        if r.get("is_false_score_control"):
            continue
        model = r.get("model")
        task_id = r.get("task_id")
        paradigm = r.get("paradigm")

        for slot in ("a", "b"):
            key = (model, task_id, paradigm, slot)
            c1_outcomes[key] = r.get(f"component_{slot}_correct", False)
            c1_decisions[key] = r.get(f"component_{slot}_decision", "proceed")

    # Step 2: Analyze C4 components
    # Per model per paradigm accumulators
    model_data = defaultdict(lambda: {
        # Weak domain (escalated by C4)
        "weak_total": 0,
        "weak_escalated": 0,          # externally routed (decision != proceed)
        "weak_not_escalated": 0,      # not routed (decision == proceed)
        "weak_escalated_c1_wrong": 0, # correct escalation: C1 would have failed
        "weak_escalated_c1_right": 0, # false escalation: C1 would have succeeded
        "weak_escalated_c1_unknown": 0,
        "weak_escalated_correct": 0,  # C4 got it right (oracle: always True for ext_routed)
        "weak_not_escalated_correct": 0,

        # Strong domain (not externally routed)
        "strong_total": 0,
        "strong_proceed": 0,
        "strong_use_tool": 0,
        "strong_defer": 0,
        "strong_proceed_correct": 0,
        "strong_use_tool_or_defer_would_succeed": 0,  # UDR under C4
        "strong_correct": 0,

        # Medium domain
        "medium_total": 0,
        "medium_correct": 0,

        # All non-externally-routed
        "autonomous_total": 0,
        "autonomous_correct": 0,

        # Paradigm breakdown
        "paradigm_counts": defaultdict(int),

        # Per-domain breakdown for weak
        "weak_domain_breakdown": defaultdict(lambda: {
            "total": 0, "escalated": 0, "c1_wrong": 0, "c1_right": 0,
        }),
    })

    for r in records:
        if r.get("condition") != 4:
            continue
        if r.get("is_false_score_control"):
            continue

        model = r.get("model")
        task_id = r.get("task_id")
        paradigm = r.get("paradigm")

        if not model or not task_id:
            continue

        md = model_data[model]
        md["paradigm_counts"][paradigm] += 1

        for slot in ("a", "b"):
            strength = r.get(f"strength_{slot}", "unknown")
            decision = r.get(f"component_{slot}_decision", "proceed")
            correct = r.get(f"component_{slot}_correct", False)
            ext_routed = r.get(f"component_{slot}_externally_routed", False)
            domain = r.get(f"domain_{slot}", "unknown")

            # Skip unknown strength (missing exp1 metrics)
            if strength == "unknown":
                continue

            if strength == "weak":
                md["weak_total"] += 1

                # Is this component escalated (externally routed)?
                is_escalated = ext_routed or decision != "proceed"

                if is_escalated:
                    md["weak_escalated"] += 1
                    if correct:
                        md["weak_escalated_correct"] += 1

                    # Cross-reference C1 for trade-off analysis
                    c1_key = (model, task_id, paradigm, slot)
                    if c1_key in c1_outcomes:
                        c1_correct = c1_outcomes[c1_key]
                        if c1_correct:
                            md["weak_escalated_c1_right"] += 1  # false escalation
                        else:
                            md["weak_escalated_c1_wrong"] += 1  # correct escalation
                    else:
                        md["weak_escalated_c1_unknown"] += 1

                    # Domain breakdown
                    db = md["weak_domain_breakdown"][domain]
                    db["total"] += 1
                    db["escalated"] += 1
                    if c1_key in c1_outcomes:
                        if c1_outcomes[c1_key]:
                            db["c1_right"] += 1
                        else:
                            db["c1_wrong"] += 1
                else:
                    md["weak_not_escalated"] += 1
                    if correct:
                        md["weak_not_escalated_correct"] += 1
                    md["autonomous_total"] += 1
                    if correct:
                        md["autonomous_correct"] += 1
                    db = md["weak_domain_breakdown"][domain]
                    db["total"] += 1

            elif strength == "strong":
                md["strong_total"] += 1
                if correct:
                    md["strong_correct"] += 1

                if decision == "proceed":
                    md["strong_proceed"] += 1
                    if correct:
                        md["strong_proceed_correct"] += 1
                elif decision == "use_tool":
                    md["strong_use_tool"] += 1
                    # Check if C1 would have been correct (for UDR)
                    c1_key = (model, task_id, paradigm, slot)
                    if c1_key in c1_outcomes and c1_outcomes[c1_key]:
                        md["strong_use_tool_or_defer_would_succeed"] += 1
                elif decision == "defer":
                    md["strong_defer"] += 1
                    c1_key = (model, task_id, paradigm, slot)
                    if c1_key in c1_outcomes and c1_outcomes[c1_key]:
                        md["strong_use_tool_or_defer_would_succeed"] += 1

                if not ext_routed:
                    md["autonomous_total"] += 1
                    if correct:
                        md["autonomous_correct"] += 1

            elif strength == "medium":
                md["medium_total"] += 1
                if correct:
                    md["medium_correct"] += 1
                if not ext_routed:
                    md["autonomous_total"] += 1
                    if correct:
                        md["autonomous_correct"] += 1

    # Step 3: Compute per-model metrics
    results = {}

    for model in sorted(model_data.keys()):
        md = model_data[model]

        # Skip models with no weak-domain data
        if md["weak_total"] == 0:
            continue

        weak_total = md["weak_total"]
        weak_escalated = md["weak_escalated"]
        weak_not_escalated = md["weak_not_escalated"]

        # Escalation rate
        escalation_rate = weak_escalated / weak_total if weak_total > 0 else 0

        # Among escalated: correct vs false escalation
        escalated_matched = md["weak_escalated_c1_wrong"] + md["weak_escalated_c1_right"]
        correct_escalation_rate = (
            md["weak_escalated_c1_wrong"] / escalated_matched
            if escalated_matched > 0 else float("nan")
        )
        false_escalation_rate = (
            md["weak_escalated_c1_right"] / escalated_matched
            if escalated_matched > 0 else float("nan")
        )

        # C1 accuracy on weak-domain (counterfactual: what would have happened)
        c1_weak_correct = md["weak_escalated_c1_right"]  # these were right under C1
        c1_weak_wrong = md["weak_escalated_c1_wrong"]    # these were wrong under C1
        c1_weak_total = c1_weak_correct + c1_weak_wrong
        c1_weak_accuracy = c1_weak_correct / c1_weak_total if c1_weak_total > 0 else float("nan")

        # Non-escalated success rate (C4 weak tasks NOT externally routed)
        non_escalated_success = (
            md["weak_not_escalated_correct"] / weak_not_escalated
            if weak_not_escalated > 0 else float("nan")
        )

        # Overall C4 weak-domain accuracy (escalated always correct, non-escalated may vary)
        c4_weak_correct = md["weak_escalated_correct"] + md["weak_not_escalated_correct"]
        c4_weak_accuracy = c4_weak_correct / weak_total if weak_total > 0 else float("nan")

        # Autonomous completion rate under C4 (fraction not externally routed)
        autonomous_completion_rate = weak_not_escalated / weak_total if weak_total > 0 else 0

        # Net utility per 100 weak-domain tasks
        failures_prevented_per_100 = (md["weak_escalated_c1_wrong"] / weak_total * 100) if weak_total > 0 else 0
        successes_blocked_per_100 = (md["weak_escalated_c1_right"] / weak_total * 100) if weak_total > 0 else 0
        net_utility_per_100 = failures_prevented_per_100 - successes_blocked_per_100

        # Strong-domain UDR under C4
        strong_udr = (
            md["strong_use_tool_or_defer_would_succeed"] / md["strong_total"]
            if md["strong_total"] > 0 else float("nan")
        )
        strong_accuracy = (
            md["strong_correct"] / md["strong_total"]
            if md["strong_total"] > 0 else float("nan")
        )

        # Autonomous accuracy (all non-routed components)
        autonomous_accuracy = (
            md["autonomous_correct"] / md["autonomous_total"]
            if md["autonomous_total"] > 0 else float("nan")
        )

        # Accuracy improvement: C4 weak accuracy vs C1 weak accuracy
        accuracy_improvement = (
            (c4_weak_accuracy - (1 - c1_weak_accuracy))
            if not (c1_weak_accuracy != c1_weak_accuracy) else float("nan")  # nan check
        )
        # More precise: c4 gets all escalated correct via oracle, plus non-escalated
        # vs C1 where the model decided on its own
        # accuracy_delta = c4_weak_accuracy - c1_weak_accuracy  (but note c1_weak_accuracy is from matched subset)

        # Domain breakdown for weak
        domain_breakdown = {}
        for domain, db in md["weak_domain_breakdown"].items():
            domain_breakdown[domain] = {
                "total": db["total"],
                "escalated": db["escalated"],
                "c1_would_fail": db["c1_wrong"],
                "c1_would_succeed": db["c1_right"],
                "escalation_rate": db["escalated"] / db["total"] if db["total"] > 0 else 0,
            }

        results[model] = {
            # Sample sizes
            "n_weak_total": weak_total,
            "n_weak_escalated": weak_escalated,
            "n_weak_not_escalated": weak_not_escalated,
            "n_escalated_matched_to_c1": escalated_matched,
            "n_escalated_c1_unknown": md["weak_escalated_c1_unknown"],
            "n_strong_total": md["strong_total"],
            "n_medium_total": md["medium_total"],

            # Primary metrics
            "escalation_rate": round(escalation_rate, 4),
            "correct_escalation_rate": round(correct_escalation_rate, 4) if correct_escalation_rate == correct_escalation_rate else None,
            "false_escalation_rate": round(false_escalation_rate, 4) if false_escalation_rate == false_escalation_rate else None,

            # Counterfactual
            "c1_weak_accuracy": round(c1_weak_accuracy, 4) if c1_weak_accuracy == c1_weak_accuracy else None,
            "c4_weak_accuracy": round(c4_weak_accuracy, 4) if c4_weak_accuracy == c4_weak_accuracy else None,

            # Non-escalated
            "non_escalated_success_rate": round(non_escalated_success, 4) if non_escalated_success == non_escalated_success else None,
            "autonomous_completion_rate": round(autonomous_completion_rate, 4),

            # Net utility
            "failures_prevented_per_100": round(failures_prevented_per_100, 2),
            "successes_blocked_per_100": round(successes_blocked_per_100, 2),
            "net_utility_per_100": round(net_utility_per_100, 2),

            # Strong-domain impact
            "strong_udr": round(strong_udr, 4) if strong_udr == strong_udr else None,
            "strong_accuracy": round(strong_accuracy, 4) if strong_accuracy == strong_accuracy else None,

            # Autonomous accuracy (all non-routed)
            "autonomous_accuracy": round(autonomous_accuracy, 4) if autonomous_accuracy == autonomous_accuracy else None,

            # Raw counts
            "weak_escalated_c1_wrong": md["weak_escalated_c1_wrong"],
            "weak_escalated_c1_right": md["weak_escalated_c1_right"],

            # Paradigm breakdown
            "paradigm_counts": dict(md["paradigm_counts"]),

            # Domain breakdown
            "weak_domain_breakdown": domain_breakdown,
        }

    return results


def compute_summary_stats(results: dict) -> dict:
    """Compute cross-model summary statistics."""
    if not results:
        return {}

    models = list(results.keys())

    def safe_values(key):
        vals = []
        for m in models:
            v = results[m].get(key)
            if v is not None and v == v:  # not NaN
                vals.append(v)
        return vals

    def stats(vals):
        if not vals:
            return {"mean": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n": len(vals),
        }

    return {
        "n_models": len(models),
        "models": models,
        "escalation_rate": stats(safe_values("escalation_rate")),
        "correct_escalation_rate": stats(safe_values("correct_escalation_rate")),
        "false_escalation_rate": stats(safe_values("false_escalation_rate")),
        "c1_weak_accuracy": stats(safe_values("c1_weak_accuracy")),
        "c4_weak_accuracy": stats(safe_values("c4_weak_accuracy")),
        "net_utility_per_100": stats(safe_values("net_utility_per_100")),
        "failures_prevented_per_100": stats(safe_values("failures_prevented_per_100")),
        "successes_blocked_per_100": stats(safe_values("successes_blocked_per_100")),
        "strong_udr": stats(safe_values("strong_udr")),
        "autonomous_completion_rate": stats(safe_values("autonomous_completion_rate")),
        "autonomous_accuracy": stats(safe_values("autonomous_accuracy")),
    }


def print_summary_table(results: dict, summary: dict) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 120)
    print("C4 ESCALATION TRADE-OFF ANALYSIS — EXPERIMENT 9")
    print("=" * 120)

    # Header
    hdr = (
        f"{'Model':25s} │ {'N_weak':>6s} {'Esc%':>6s} │ "
        f"{'CorrectEsc':>10s} {'FalseEsc':>10s} │ "
        f"{'C1_Acc':>6s} {'C4_Acc':>6s} │ "
        f"{'Prev/100':>8s} {'Block/100':>9s} {'NetUtil':>8s} │ "
        f"{'S_UDR':>6s}"
    )
    print(hdr)
    print("─" * 120)

    for model in sorted(results.keys()):
        r = results[model]

        def fmt(v, pct=False):
            if v is None:
                return "  N/A "
            if pct:
                return f"{v*100:5.1f}%"
            return f"{v:6.2f}"

        row = (
            f"{model:25s} │ "
            f"{r['n_weak_total']:6d} "
            f"{fmt(r['escalation_rate'], True)} │ "
            f"{fmt(r['correct_escalation_rate'], True):>10s} "
            f"{fmt(r['false_escalation_rate'], True):>10s} │ "
            f"{fmt(r['c1_weak_accuracy'], True)} "
            f"{fmt(r['c4_weak_accuracy'], True)} │ "
            f"{r['failures_prevented_per_100']:8.2f} "
            f"{r['successes_blocked_per_100']:9.2f} "
            f"{r['net_utility_per_100']:8.2f} │ "
            f"{fmt(r['strong_udr'], True)}"
        )
        print(row)

    print("─" * 120)

    # Summary row
    s = summary
    def sfmt(d, pct=False):
        if not d or d.get("mean") is None:
            return "  N/A "
        if pct:
            return f"{d['mean']*100:5.1f}%"
        return f"{d['mean']:6.2f}"

    print(
        f"{'MEAN':25s} │ "
        f"{'':>6s} "
        f"{sfmt(s['escalation_rate'], True)} │ "
        f"{sfmt(s['correct_escalation_rate'], True):>10s} "
        f"{sfmt(s['false_escalation_rate'], True):>10s} │ "
        f"{sfmt(s['c1_weak_accuracy'], True)} "
        f"{sfmt(s['c4_weak_accuracy'], True)} │ "
        f"{s['failures_prevented_per_100']['mean']:8.2f} "
        f"{s['successes_blocked_per_100']['mean']:9.2f} "
        f"{s['net_utility_per_100']['mean']:8.2f} │ "
        f"{sfmt(s['strong_udr'], True)}"
    )

    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)

    net_util = s["net_utility_per_100"]
    if net_util and net_util["mean"] is not None:
        mean_nu = net_util["mean"]
        if mean_nu > 5:
            print(f"  Net utility is POSITIVE ({mean_nu:.1f} per 100 tasks): C4 routing prevents")
            print(f"  more failures than it blocks successes. External metacognitive routing adds value.")
        elif mean_nu < -5:
            print(f"  Net utility is NEGATIVE ({mean_nu:.1f} per 100 tasks): C4 routing blocks")
            print(f"  more successes than it prevents failures. External routing is counterproductive.")
        else:
            print(f"  Net utility is NEAR ZERO ({mean_nu:.1f} per 100 tasks): C4 routing is")
            print(f"  approximately neutral — failures prevented roughly equal successes blocked.")

    prev = s["failures_prevented_per_100"]
    block = s["successes_blocked_per_100"]
    if prev and block and prev["mean"] is not None and block["mean"] is not None:
        print(f"\n  Failures prevented: {prev['mean']:.1f}/100 (range: {prev['min']:.1f}–{prev['max']:.1f})")
        print(f"  Successes blocked:  {block['mean']:.1f}/100 (range: {block['min']:.1f}–{block['max']:.1f})")

    c1_acc = s["c1_weak_accuracy"]
    c4_acc = s["c4_weak_accuracy"]
    if c1_acc and c4_acc and c1_acc["mean"] is not None and c4_acc["mean"] is not None:
        delta = c4_acc["mean"] - c1_acc["mean"]
        print(f"\n  C1 weak-domain accuracy: {c1_acc['mean']*100:.1f}% (range: {c1_acc['min']*100:.1f}–{c1_acc['max']*100:.1f}%)")
        print(f"  C4 weak-domain accuracy: {c4_acc['mean']*100:.1f}% (range: {c4_acc['min']*100:.1f}–{c4_acc['max']*100:.1f}%)")
        print(f"  Accuracy improvement:    {delta*100:+.1f} pp")

    esc = s["escalation_rate"]
    if esc and esc["mean"] is not None:
        print(f"\n  Escalation rate: {esc['mean']*100:.1f}% (range: {esc['min']*100:.1f}–{esc['max']*100:.1f}%)")

    sudr = s["strong_udr"]
    if sudr and sudr["mean"] is not None:
        print(f"  Strong-domain UDR under C4: {sudr['mean']*100:.1f}% (range: {sudr['min']*100:.1f}–{sudr['max']*100:.1f}%)")

    corr_esc = s["correct_escalation_rate"]
    if corr_esc and corr_esc["mean"] is not None:
        print(f"\n  Among escalated weak-domain tasks:")
        print(f"    Correct escalation (C1 would fail):  {corr_esc['mean']*100:.1f}%")
        false_esc = s["false_escalation_rate"]
        if false_esc and false_esc["mean"] is not None:
            print(f"    False escalation (C1 would succeed): {false_esc['mean']*100:.1f}%")

    print()

    # Per-model detail
    print("\n" + "=" * 120)
    print("PER-MODEL WEAK-DOMAIN BREAKDOWN")
    print("=" * 120)

    for model in sorted(results.keys()):
        r = results[model]
        print(f"\n  {model}:")
        print(f"    Weak tasks: {r['n_weak_total']}, Escalated: {r['n_weak_escalated']}, "
              f"Not escalated: {r['n_weak_not_escalated']}")
        print(f"    Matched to C1: {r['n_escalated_matched_to_c1']}, "
              f"Unmatched: {r['n_escalated_c1_unknown']}")

        if r.get("weak_domain_breakdown"):
            print(f"    Domain breakdown:")
            for domain in sorted(r["weak_domain_breakdown"].keys()):
                db = r["weak_domain_breakdown"][domain]
                print(f"      {domain:15s}: total={db['total']:3d}, escalated={db['escalated']:3d}, "
                      f"C1_fail={db['c1_would_fail']:3d}, C1_succeed={db['c1_would_succeed']:3d}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading Exp9 data...")
    records = load_all_exp9_records()

    n_c1 = sum(1 for r in records if r.get("condition") == 1)
    n_c4 = sum(1 for r in records if r.get("condition") == 4)
    print(f"  Total records: {len(records)}")
    print(f"  C1 records: {n_c1}")
    print(f"  C4 records: {n_c4}")

    print("\nComputing C4 escalation trade-off...")
    results = analyze_c4_tradeoff(records)
    print(f"  Models with C4 weak-domain data: {len(results)}")

    summary = compute_summary_stats(results)

    print_summary_table(results, summary)

    # Save results
    output_path = Path("paper/tables/c4_tradeoff_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "C4 Escalation Trade-off Analysis for Experiment 9",
        "methodology": {
            "c4_mechanism": "External routing forces tool-use (oracle: always correct) on weak-domain "
                           "components where MIRROR domain accuracy < 50%.",
            "escalation_definition": "A weak-domain component is 'escalated' if it was externally "
                                     "routed (decision=use_tool, externally_routed=True).",
            "correct_escalation": "C1 outcome for same task+slot was WRONG (failure prevented).",
            "false_escalation": "C1 outcome for same task+slot was RIGHT (success blocked).",
            "net_utility": "(failures_prevented - successes_blocked) per 100 weak-domain tasks.",
            "weak_threshold": "exp1 natural_acc <= 0.40",
            "strong_threshold": "exp1 natural_acc >= 0.60",
            "paradigms_analyzed": "1 (Autonomous) and 2 (Checkpoint) — P3 is skipped under C4.",
        },
        "per_model": results,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
