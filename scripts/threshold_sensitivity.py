"""
Threshold Sensitivity Analysis for Experiment 9 CFR
====================================================

Computes CFR under three alternative weak-domain definitions:
  1. Median: domain is weak if model's natural_acc < model's median across 8 domains
  2. Bottom-3: domain is weak if it's among the model's 3 lowest-accuracy domains
  3. Absolute <0.40: domain is weak if natural_acc < 0.40

For each definition, computes CFR_C1, CFR_C4, and reduction.

Note: C4 in the actual experiment used a FIXED absolute threshold of 0.50
(see apply_condition4_routing in run_experiment_9.py). Therefore C4 results
are the same regardless of our post-hoc "weak" re-definition -- the routing
decision was baked in at run time. What changes across definitions is WHICH
components we count as "weak" for the CFR denominator/numerator.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("data/results")
PAPER_DIR = Path("paper/tables")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Exp1 accuracy data (merge all files, newer overrides)
# ─────────────────────────────────────────────────────────────────────────────

def load_exp1_metrics() -> dict:
    exp1_files = sorted(
        [p for p in RESULTS_DIR.glob("exp1_*_accuracy.json") if "meta" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not exp1_files:
        raise FileNotFoundError("No Experiment 1 accuracy metrics found")
    merged = {}
    for p in exp1_files:
        with open(p) as f:
            data = json.load(f)
        merged.update(data)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Exp9 results (run_id 20260312T140842)
# ─────────────────────────────────────────────────────────────────────────────

def load_exp9_results() -> list[dict]:
    """Load the main results file plus all shard files for this run_id."""
    run_id = "20260312T140842"
    all_files = sorted(RESULTS_DIR.glob(f"exp9_{run_id}_*"))
    jsonl_files = [f for f in all_files if f.suffix == ".jsonl"]

    results = []
    seen_keys = set()

    for fpath in jsonl_files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Deduplicate by (model, task_id, condition, paradigm, is_false_score_control)
                    key = (
                        rec.get("model"),
                        rec.get("task_id"),
                        rec.get("condition"),
                        rec.get("paradigm"),
                        rec.get("is_false_score_control", False),
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append(rec)
                except json.JSONDecodeError:
                    pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build weak-domain classifiers
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = [
    "arithmetic", "factual", "linguistic", "logical",
    "procedural", "social", "spatial", "scientific",
]


def build_weak_sets(exp1: dict) -> dict:
    """
    For each model, build 3 sets of weak domains under 3 definitions.
    Returns: {definition_name: {model: set_of_weak_domains}}
    """
    definitions = {
        "Median": {},
        "Bottom-3": {},
        "Absolute <0.40": {},
    }

    for model, domains_data in exp1.items():
        # Collect natural_acc per domain
        accs = {}
        for d in DOMAINS:
            dd = domains_data.get(d)
            if isinstance(dd, dict) and dd.get("natural_acc") is not None:
                accs[d] = dd["natural_acc"]

        if not accs:
            continue

        # Definition 1: Median
        vals = sorted(accs.values())
        median_val = vals[len(vals) // 2] if len(vals) % 2 == 1 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        definitions["Median"][model] = {d for d, a in accs.items() if a < median_val}

        # Definition 2: Bottom-3
        sorted_domains = sorted(accs.items(), key=lambda x: x[1])
        definitions["Bottom-3"][model] = {d for d, _ in sorted_domains[:3]}

        # Definition 3: Absolute <0.40
        definitions["Absolute <0.40"][model] = {d for d, a in accs.items() if a < 0.40}

    return definitions


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute CFR for a given condition and weak-domain definition
# ─────────────────────────────────────────────────────────────────────────────

def compute_cfr(
    results: list[dict],
    condition: int,
    weak_sets: dict,  # {model: set_of_weak_domains}
    exp1: dict,
) -> dict:
    """
    CFR = (weak-domain components where model proceeded and got it wrong)
        / (total weak-domain components)

    Only paradigms 1 and 2 (tool-use paradigms).
    Exclude false score control records and records without api_success.
    """
    weak_total = 0
    auto_fail = 0

    for rec in results:
        if rec.get("condition") != condition:
            continue
        if rec.get("paradigm") not in (1, 2):
            continue
        if rec.get("is_false_score_control", False):
            continue
        if not rec.get("api_success", True):
            continue

        model = rec.get("model")
        if model not in weak_sets:
            continue

        model_weak = weak_sets[model]

        for slot in ("a", "b"):
            domain = rec.get(f"domain_{slot}")
            if domain in model_weak:
                weak_total += 1
                decision = rec.get(f"component_{slot}_decision")
                correct = rec.get(f"component_{slot}_correct", False)
                if decision == "proceed" and not correct:
                    auto_fail += 1

    cfr = auto_fail / weak_total if weak_total > 0 else None
    return {
        "cfr": cfr,
        "auto_fail": auto_fail,
        "weak_total": weak_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Also compute CFR using the ORIGINAL strength_{a,b} field (as stored)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cfr_original_strength(results: list[dict], condition: int) -> dict:
    """
    Use the strength_a/strength_b fields as stored in the records.
    The original code uses: weak = natural_acc <= 0.40, strong = natural_acc >= 0.60.
    """
    weak_total = 0
    auto_fail = 0

    for rec in results:
        if rec.get("condition") != condition:
            continue
        if rec.get("paradigm") not in (1, 2):
            continue
        if rec.get("is_false_score_control", False):
            continue
        if not rec.get("api_success", True):
            continue

        for slot in ("a", "b"):
            if rec.get(f"strength_{slot}") == "weak":
                weak_total += 1
                decision = rec.get(f"component_{slot}_decision")
                correct = rec.get(f"component_{slot}_correct", False)
                if decision == "proceed" and not correct:
                    auto_fail += 1

    cfr = auto_fail / weak_total if weak_total > 0 else None
    return {
        "cfr": cfr,
        "auto_fail": auto_fail,
        "weak_total": weak_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS — Experiment 9 CFR")
    print("=" * 80)

    # Load data
    print("\nLoading Exp1 accuracy data...")
    exp1 = load_exp1_metrics()
    print(f"  {len(exp1)} models in Exp1 data")

    print("Loading Exp9 results (run_id 20260312T140842)...")
    results = load_exp9_results()
    print(f"  {len(results)} total records loaded")

    # Filter to non-control, paradigm 1+2 records only
    relevant = [
        r for r in results
        if r.get("paradigm") in (1, 2)
        and not r.get("is_false_score_control", False)
        and r.get("api_success", True)
    ]
    models_in_data = sorted(set(r["model"] for r in relevant if r.get("model")))
    print(f"  {len(relevant)} relevant records (P1+P2, non-control, api_success)")
    print(f"  Models: {', '.join(models_in_data)}")

    # Build weak-domain sets
    weak_defs = build_weak_sets(exp1)

    # Print weak-domain assignments per model per definition
    print("\n" + "─" * 80)
    print("WEAK DOMAIN ASSIGNMENTS BY DEFINITION")
    print("─" * 80)
    for defn_name, model_sets in weak_defs.items():
        print(f"\n  {defn_name}:")
        for model in models_in_data:
            if model in model_sets:
                weak_doms = sorted(model_sets[model])
                print(f"    {model:<25}: {', '.join(weak_doms) if weak_doms else '(none)'}")
            else:
                print(f"    {model:<25}: (no Exp1 data)")

    # ── Note about C4 routing threshold ──────────────────────────────────────
    print("\n" + "─" * 80)
    print("NOTE ON C4 ROUTING")
    print("─" * 80)
    print("  C4 uses a FIXED absolute threshold of 0.50 (natural_acc < 0.50 → force tool use).")
    print("  The routing decision was made at run time and is baked into the data.")
    print("  Changing our 'weak' definition post-hoc changes WHICH components we")
    print("  COUNT as weak, but does NOT change whether C4 routed them or not.")
    print("  Therefore C4 CFR changes across definitions because the denominator")
    print("  and numerator set changes, not because the routing changed.")

    # ── Compute CFR for each definition ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    output = {}

    # Also include the original strength-based definition for reference
    print(f"\n{'Definition':<20} | {'C1 CFR':>10} | {'C4 CFR':>10} | {'Reduction':>10} | {'N_weak_C1':>10} | {'N_weak_C4':>10}")
    print(f"{'─'*20}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}")

    # Original definition (strength_a/b == "weak", i.e., natural_acc <= 0.40)
    c1_orig = compute_cfr_original_strength(results, condition=1)
    c4_orig = compute_cfr_original_strength(results, condition=4)
    reduction_orig = ((c1_orig["cfr"] - c4_orig["cfr"]) / c1_orig["cfr"]
                      if c1_orig["cfr"] and c4_orig["cfr"] and c1_orig["cfr"] > 0
                      else None)
    red_str = f"{reduction_orig:.1%}" if reduction_orig is not None else "N/A"
    c1_str = f"{c1_orig['cfr']:.4f}" if c1_orig["cfr"] is not None else "N/A"
    c4_str = f"{c4_orig['cfr']:.4f}" if c4_orig["cfr"] is not None else "N/A"
    print(f"{'Original (<=0.40)':<20} | {c1_str:>10} | {c4_str:>10} | {red_str:>10} | {c1_orig['weak_total']:>10} | {c4_orig['weak_total']:>10}")

    output["Original (<=0.40)"] = {
        "C1_CFR": c1_orig["cfr"],
        "C4_CFR": c4_orig["cfr"],
        "reduction": reduction_orig,
        "N_weak_C1": c1_orig["weak_total"],
        "N_weak_C4": c4_orig["weak_total"],
        "C1_auto_fail": c1_orig["auto_fail"],
        "C4_auto_fail": c4_orig["auto_fail"],
    }

    # Three alternative definitions
    for defn_name in ["Median", "Bottom-3", "Absolute <0.40"]:
        model_weak = weak_defs[defn_name]

        c1 = compute_cfr(results, condition=1, weak_sets=model_weak, exp1=exp1)
        c4 = compute_cfr(results, condition=4, weak_sets=model_weak, exp1=exp1)

        reduction = ((c1["cfr"] - c4["cfr"]) / c1["cfr"]
                     if c1["cfr"] and c4["cfr"] and c1["cfr"] > 0
                     else None)

        red_str = f"{reduction:.1%}" if reduction is not None else "N/A"
        c1_str = f"{c1['cfr']:.4f}" if c1["cfr"] is not None else "N/A"
        c4_str = f"{c4['cfr']:.4f}" if c4["cfr"] is not None else "N/A"

        print(f"{defn_name:<20} | {c1_str:>10} | {c4_str:>10} | {red_str:>10} | {c1['weak_total']:>10} | {c4['weak_total']:>10}")

        output[defn_name] = {
            "C1_CFR": c1["cfr"],
            "C4_CFR": c4["cfr"],
            "reduction": reduction,
            "N_weak_C1": c1["weak_total"],
            "N_weak_C4": c4["weak_total"],
            "C1_auto_fail": c1["auto_fail"],
            "C4_auto_fail": c4["auto_fail"],
        }

    # ── Per-model breakdown for each definition ──────────────────────────────
    print("\n" + "─" * 80)
    print("PER-MODEL C1 CFR BY DEFINITION")
    print("─" * 80)

    header = f"{'Model':<25}"
    for defn_name in ["Original (<=0.40)", "Median", "Bottom-3", "Absolute <0.40"]:
        header += f" | {defn_name:>18}"
    print(header)
    print("─" * (25 + 4 * 21))

    per_model_output = {}

    for model in models_in_data:
        row = f"{model:<25}"
        per_model_output[model] = {}

        # Original
        model_results = [r for r in results if r.get("model") == model]

        for defn_name in ["Original (<=0.40)", "Median", "Bottom-3", "Absolute <0.40"]:
            if defn_name == "Original (<=0.40)":
                cfr_data = compute_cfr_original_strength(
                    [r for r in results if r.get("model") == model], condition=1
                )
            else:
                # Build single-model weak set
                model_weak = weak_defs[defn_name]
                single_weak = {model: model_weak.get(model, set())}
                cfr_data = compute_cfr(
                    [r for r in results if r.get("model") == model],
                    condition=1, weak_sets=single_weak, exp1=exp1
                )

            if cfr_data["cfr"] is not None:
                row += f" | {cfr_data['cfr']:>10.4f} ({cfr_data['weak_total']:>4})"
            else:
                row += f" | {'N/A':>18}"

            per_model_output[model][defn_name] = {
                "cfr": cfr_data["cfr"],
                "weak_total": cfr_data["weak_total"],
                "auto_fail": cfr_data["auto_fail"],
            }

        print(row)

    # ── Save output ──────────────────────────────────────────────────────────
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_DIR / "threshold_sensitivity.json"

    save_data = {
        "description": "Threshold sensitivity analysis for Exp9 CFR under 4 weak-domain definitions",
        "c4_routing_note": (
            "C4 used a FIXED absolute threshold of natural_acc < 0.50 at run time. "
            "Alternative definitions change which components are counted as 'weak' "
            "in the CFR calculation, not the routing decision itself."
        ),
        "definitions": {
            "Original (<=0.40)": "Domain is weak if model's natural_acc <= 0.40 (run-time strength field)",
            "Median": "Domain is weak if model's natural_acc < model's median across all domains",
            "Bottom-3": "Domain is weak if it's among the model's 3 lowest-accuracy domains",
            "Absolute <0.40": "Domain is weak if natural_acc < 0.40 (strict less-than, recalculated)",
        },
        "aggregate": {k: {kk: (round(vv, 6) if isinstance(vv, float) and vv is not None else vv)
                          for kk, vv in v.items()}
                      for k, v in output.items()},
        "per_model_C1_CFR": per_model_output,
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
