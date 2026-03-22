#!/usr/bin/env python3
"""
Exp9 Permutation Sensitivity Test

Simulates the effect of label disagreement (17/50 audit disagreement rate)
on the Condition 1 CFR baseline. For 1000 iterations, randomly flips 17
component-level correctness labels across the fixed-task C1 P1+P2 dataset
and recomputes per-model CFR and mean CFR across models.

Reports:
  - Original C1 CFR (mean +/- SD across models)
  - Permuted distribution (mean +/- SD of mean-CFR)
  - Whether the 62% reduction (C1->C3 or C1->C4) is preserved under noise

Saves results to paper/supplementary/exp9_permutation_sensitivity.json
"""

import json
import glob
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = Path("C:/Users/wangz/MIRROR")
RESULTS_DIR = BASE / "data" / "results"
OUTPUT_PATH = BASE / "paper" / "supplementary" / "exp9_permutation_sensitivity.json"

N_ITERATIONS = 1000
N_FLIPS = 17  # simulated audit disagreement rate: 17/50
SEED = 42


def load_exp9_data():
    """Load all exp9 run data from JSONL files."""
    pattern = str(RESULTS_DIR / "exp9_20260312T140842*.jsonl")
    files = sorted(glob.glob(pattern))
    records = []
    seen_keys = set()
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Dedup by (model, task_id, condition, paradigm)
                key = (rec.get("model"), rec.get("task_id"),
                       rec.get("condition"), rec.get("paradigm"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    records.append(rec)
    return records


def filter_c1_fixed_p1p2(records):
    """Filter to condition=1, circularity_free=True, paradigm 1 or 2."""
    filtered = []
    for r in records:
        if r.get("condition") != 1:
            continue
        if not r.get("circularity_free", False):
            continue
        if r.get("paradigm") not in (1, 2):
            continue
        if not r.get("api_success", True):
            continue
        # Exclude qwen-3-235b (100% API failure)
        if r.get("model") == "qwen-3-235b":
            continue
        filtered.append(r)
    return filtered


def extract_components(records):
    """
    Extract individual component-level observations from trials.
    Each trial has components a and b; we extract them as separate entries
    for the purpose of CFR computation.

    Returns list of dicts with keys:
        model, domain, subcategory, strength, decision, correct, trial_idx, slot
    """
    components = []
    for i, trial in enumerate(records):
        model = trial.get("model")
        for slot in ("a", "b"):
            domain = trial.get(f"domain_{slot}")
            subcat = trial.get(f"subcategory_{slot}")
            strength = trial.get(f"strength_{slot}")
            decision = trial.get(f"component_{slot}_decision")
            correct = trial.get(f"component_{slot}_correct", False)
            if domain and subcat:
                components.append({
                    "model": model,
                    "domain": domain,
                    "subcategory": subcat,
                    "strength": strength,
                    "decision": decision,
                    "correct": correct,
                    "trial_idx": i,
                    "slot": slot,
                })
    return components


def compute_per_model_cfr(components):
    """
    Compute CFR per model from component-level observations.
    CFR = (weak-domain components where decision=proceed AND incorrect) / (total weak-domain components)
    """
    acc = defaultdict(lambda: {"weak_total": 0, "weak_auto_fail": 0})
    for c in components:
        if c["strength"] == "weak":
            acc[c["model"]]["weak_total"] += 1
            if c["decision"] == "proceed" and not c["correct"]:
                acc[c["model"]]["weak_auto_fail"] += 1

    cfr_per_model = {}
    for model, counts in acc.items():
        if counts["weak_total"] > 0:
            cfr_per_model[model] = counts["weak_auto_fail"] / counts["weak_total"]
        else:
            cfr_per_model[model] = float("nan")
    return cfr_per_model


def compute_mean_cfr(cfr_per_model):
    """Mean CFR across models (excluding NaN)."""
    vals = [v for v in cfr_per_model.values() if not np.isnan(v)]
    if vals:
        return float(np.mean(vals))
    return float("nan")


def run_permutation_test(components, n_iterations=N_ITERATIONS, n_flips=N_FLIPS, seed=SEED):
    """
    For each iteration, randomly select n_flips weak-domain component indices
    and flip their 'correct' label. Recompute mean CFR across models.
    """
    rng = np.random.RandomState(seed)

    # Identify indices of weak-domain components (only these matter for CFR)
    weak_indices = [i for i, c in enumerate(components) if c["strength"] == "weak"]
    print(f"  Total weak-domain components: {len(weak_indices)}")

    # Also find all component indices (we flip from all to simulate label noise)
    all_indices = list(range(len(components)))
    print(f"  Total components: {len(all_indices)}")

    # Original CFR
    original_cfr = compute_per_model_cfr(components)
    original_mean = compute_mean_cfr(original_cfr)

    permuted_means = []
    for iteration in range(n_iterations):
        # Randomly pick n_flips indices from ALL components to flip
        flip_indices = set(rng.choice(len(components), size=n_flips, replace=False))

        # Create modified components
        modified = []
        for i, c in enumerate(components):
            if i in flip_indices:
                modified.append({**c, "correct": not c["correct"]})
            else:
                modified.append(c)

        perm_cfr = compute_per_model_cfr(modified)
        perm_mean = compute_mean_cfr(perm_cfr)
        permuted_means.append(perm_mean)

    return original_cfr, original_mean, permuted_means


def main():
    print("Loading Exp9 data...")
    records = load_exp9_data()
    print(f"  Total records loaded: {len(records)}")

    print("Filtering to C1, fixed tasks, P1+P2...")
    filtered = filter_c1_fixed_p1p2(records)
    print(f"  Filtered records: {len(filtered)}")

    models = sorted(set(r.get("model") for r in filtered))
    print(f"  Models: {models}")

    print("Extracting components...")
    components = extract_components(filtered)
    print(f"  Total components: {len(components)}")

    print(f"Running permutation test ({N_ITERATIONS} iterations, {N_FLIPS} flips)...")
    original_cfr, original_mean, permuted_means = run_permutation_test(components)

    permuted_arr = np.array(permuted_means)
    perm_mean = float(np.mean(permuted_arr))
    perm_sd = float(np.std(permuted_arr))

    # Compute the 62% reduction check:
    # The claim is CFR drops 62% from C1 to C3/C4.
    # We check: what fraction of permuted means still show > 50% of the original mean?
    # (i.e., is the baseline stable enough that a 62% reduction would be detectable?)
    threshold_62pct = original_mean * (1 - 0.62)  # target after 62% reduction
    fraction_above_threshold = float(np.mean(permuted_arr > threshold_62pct))

    # What's the max perturbation as % of original?
    max_shift = float(np.max(np.abs(permuted_arr - original_mean)))
    max_shift_pct = max_shift / original_mean * 100 if original_mean > 0 else 0

    # 95% CI of permuted means
    ci_lower = float(np.percentile(permuted_arr, 2.5))
    ci_upper = float(np.percentile(permuted_arr, 97.5))

    # Per-model original CFR
    per_model_cfr = {m: round(v, 4) for m, v in sorted(original_cfr.items())}
    model_cfr_values = [v for v in original_cfr.values() if not np.isnan(v)]
    original_sd = float(np.std(model_cfr_values)) if model_cfr_values else 0.0

    result = {
        "description": "Permutation sensitivity test for Exp9 C1 CFR baseline",
        "method": (
            f"For {N_ITERATIONS} iterations, randomly flipped {N_FLIPS} component-level "
            f"correctness labels (simulating 17/50 audit disagreement rate) and recomputed "
            f"mean CFR across models. Fixed tasks only, C1, P1+P2."
        ),
        "n_records": len(filtered),
        "n_components": len(components),
        "n_models": len(models),
        "models": models,
        "n_iterations": N_ITERATIONS,
        "n_flips_per_iteration": N_FLIPS,
        "original_c1_cfr": {
            "per_model": per_model_cfr,
            "mean": round(original_mean, 4),
            "sd": round(original_sd, 4),
        },
        "permuted_distribution": {
            "mean_of_means": round(perm_mean, 4),
            "sd_of_means": round(perm_sd, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "max_absolute_shift": round(max_shift, 4),
            "max_shift_pct_of_original": round(max_shift_pct, 2),
        },
        "reduction_62pct_preserved": {
            "description": (
                "Fraction of permuted iterations where C1 CFR remains above the "
                "threshold that a 62% reduction would need to exceed"
            ),
            "original_cfr_mean": round(original_mean, 4),
            "target_after_62pct_reduction": round(threshold_62pct, 4),
            "fraction_permuted_above_target": round(fraction_above_threshold, 4),
            "reduction_preserved": fraction_above_threshold > 0.95,
            "interpretation": (
                "The 62% CFR reduction IS robust to label noise"
                if fraction_above_threshold > 0.95
                else "The 62% CFR reduction may NOT be robust to label noise"
            ),
        },
        "conclusion": "",  # filled below
    }

    # Build conclusion
    if perm_sd / original_mean < 0.05:
        stability = "highly stable"
    elif perm_sd / original_mean < 0.10:
        stability = "stable"
    else:
        stability = "moderately sensitive"

    result["conclusion"] = (
        f"The C1 baseline CFR (mean={original_mean:.4f}, SD={original_sd:.4f} across "
        f"{len(models)} models) is {stability} under simulated label noise "
        f"(permuted mean={perm_mean:.4f}, SD={perm_sd:.4f}, 95% CI=[{ci_lower:.4f}, "
        f"{ci_upper:.4f}]). Maximum shift from label flipping is "
        f"{max_shift_pct:.1f}% of the original mean. "
        f"The claimed 62% reduction is {'preserved' if fraction_above_threshold > 0.95 else 'NOT preserved'} "
        f"in {fraction_above_threshold*100:.1f}% of permutations."
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOriginal C1 CFR:")
    for m, v in sorted(original_cfr.items()):
        print(f"  {m:25s}: {v:.4f}")
    print(f"  {'Mean':25s}: {original_mean:.4f} +/- {original_sd:.4f}")

    print(f"\nPermuted distribution ({N_ITERATIONS} iterations, {N_FLIPS} flips):")
    print(f"  Mean of means: {perm_mean:.4f}")
    print(f"  SD of means:   {perm_sd:.4f}")
    print(f"  95% CI:        [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Max shift:     {max_shift:.4f} ({max_shift_pct:.1f}% of original)")

    print(f"\n62% reduction preserved: {result['reduction_62pct_preserved']['reduction_preserved']}")
    print(f"  ({fraction_above_threshold*100:.1f}% of permutations above target threshold)")

    print(f"\nConclusion: {result['conclusion']}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
