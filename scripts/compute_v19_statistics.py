"""
Compute missing statistics for MIRROR v19 paper:
  (a) Exp3 v2 CCE bootstrap 95% CIs per model
  (b) Exp9 escalation curve full stats (11 models including gemini)

Outputs: data/results/v19_statistics.json

Usage:
    cd C:/Users/wangz/MIRROR
    python scripts/compute_v19_statistics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EXP3_V2 = ROOT / "data" / "results" / "exp3_v2_expanded_results.jsonl"
EXP9_ANALYSIS = ROOT / "data" / "results" / "exp9_20260312T140842_analysis" / "analysis.json"
EXP9_MAIN = ROOT / "data" / "results" / "exp9_20260312T140842_results.jsonl"
EXP9_GEMINI = ROOT / "data" / "results" / "exp9_20260323_gemini_results.jsonl"
OUTPUT = ROOT / "data" / "results" / "v19_statistics.json"

N_BOOTSTRAP = 10_000
SEED = 42
EXCLUDE_MODELS = {"qwen-3-235b", "qwen3-235b-nim", "command-r-plus"}


# ---------------------------------------------------------------------------
# Bootstrap BCa (reused pattern from generate_escalation_figures.py)
# ---------------------------------------------------------------------------

def bootstrap_bca_mean(values: list[float], n: int = N_BOOTSTRAP,
                       seed: int = SEED) -> tuple[float, float]:
    """BCa bootstrap 95% CI on mean."""
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    n_pts = len(arr)
    if n_pts < 3:
        return float("nan"), float("nan")
    obs = float(np.mean(arr))
    boot = np.array([np.mean(rng.choice(arr, size=n_pts, replace=True))
                     for _ in range(n)])
    prop_less = np.clip(np.mean(boot < obs), 1e-6, 1 - 1e-6)
    z0 = sp_stats.norm.ppf(prop_less)
    jack = np.array([np.mean(np.delete(arr, i)) for i in range(n_pts)])
    jack_mean = np.mean(jack)
    numer = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = numer / denom if abs(denom) > 1e-12 else 0.0

    def adj(z_a):
        z_adj = z0 + (z0 + z_a) / (1 - a * (z0 + z_a))
        return float(sp_stats.norm.cdf(z_adj))

    lo = float(np.percentile(boot, adj(sp_stats.norm.ppf(0.025)) * 100))
    hi = float(np.percentile(boot, adj(sp_stats.norm.ppf(0.975)) * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# Part A: Exp3 v2 CCE Bootstrap CIs
# ---------------------------------------------------------------------------

def compute_exp3_cce_cis() -> list[dict]:
    """Compute per-model CCE with BCa 95% CIs from expanded v2 data."""
    records = []
    with open(EXP3_V2, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter to layer2 channel (has explicit confidence values)
    layer2 = [r for r in records if r.get("channel") == "layer2"]

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for r in layer2:
        by_model.setdefault(r["model"], []).append(r)

    results = []
    for model in sorted(by_model):
        items = by_model[model]
        errors = []
        for item in items:
            conf = item.get("confidence")
            correct = item.get("answer_correct", False)
            if conf is not None:
                errors.append(abs(conf - (1.0 if correct else 0.0)))
        if not errors:
            continue
        mean_cce = float(np.mean(errors))
        ci_lo, ci_hi = bootstrap_bca_mean(errors)
        results.append({
            "model": model,
            "cce": round(mean_cce, 3),
            "ci_lower": round(ci_lo, 3),
            "ci_upper": round(ci_hi, 3),
            "n": len(errors),
            "ci_excludes_zero": ci_lo > 0,
        })
    return results


# ---------------------------------------------------------------------------
# Part B: Exp9 Escalation Curve Full Stats (11 models)
# ---------------------------------------------------------------------------

def load_exp9_results(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("model") not in EXCLUDE_MODELS:
                    results.append(r)
    return results


def patch_strengths(results: list[dict]) -> None:
    """Retroactively classify strength fields from exp1 accuracy files."""
    import glob as _glob
    merged: dict = {}
    for fp in sorted(_glob.glob(str(ROOT / "data" / "results" / "exp1_*_accuracy.json")),
                     key=lambda x: Path(x).stat().st_mtime):
        merged.update(json.loads(Path(fp).read_text(encoding="utf-8")))
    for r in results:
        model = r.get("model", "")
        if model not in merged:
            continue
        for slot in ("a", "b"):
            if r.get(f"strength_{slot}") not in ("unknown", None, ""):
                continue
            domain = r.get(f"domain_{slot}")
            if not domain:
                continue
            nat = merged[model].get(domain, {}).get("natural_acc")
            if nat is None:
                continue
            if nat >= 0.60:
                r[f"strength_{slot}"] = "strong"
            elif nat <= 0.40:
                r[f"strength_{slot}"] = "weak"
            else:
                r[f"strength_{slot}"] = "medium"


def compute_cfr_single_model(results: list[dict], model: str,
                              condition: int, paradigms: list[int]) -> float | None:
    """Compute CFR for a single model at a given condition."""
    weak_total = 0
    auto_fail = 0
    for r in results:
        if (r.get("model") != model
                or r.get("condition") != condition
                or r.get("paradigm") not in paradigms
                or r.get("is_false_score_control", False)):
            continue
        for slot in ("a", "b"):
            if r.get(f"strength_{slot}") == "weak":
                weak_total += 1
                if (r.get(f"component_{slot}_decision") == "proceed"
                        and not r.get(f"component_{slot}_correct", False)):
                    auto_fail += 1
    return auto_fail / weak_total if weak_total > 0 else None


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def cohens_d_paired(x: list[float], y: list[float]) -> float:
    """Paired Cohen's d (using SD of differences)."""
    diff = np.array(x) - np.array(y)
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return float("inf") if abs(np.mean(diff)) > 1e-12 else 0.0
    return float(np.mean(diff) / sd)


def compute_escalation_stats() -> dict:
    """Compute full escalation curve stats with 11 models (including gemini)."""
    # Load existing 10-model data
    with open(EXP9_ANALYSIS, encoding="utf-8") as f:
        analysis = json.load(f)
    per_model_10 = analysis["escalation_curve"]["per_model"]

    # Load and process gemini data
    gemini_results = load_exp9_results(EXP9_GEMINI)
    patch_strengths(gemini_results)

    # Compute gemini CFR per condition
    gemini_cfr = {}
    for cond in [1, 2, 3, 4]:
        cfr = compute_cfr_single_model(gemini_results, "gemini-2.5-pro",
                                        cond, [1, 2, 3])
        gemini_cfr[str(cond)] = cfr

    # Merge into 11-model dataset
    per_model = dict(per_model_10)
    per_model["gemini-2.5-pro"] = gemini_cfr

    # Compute per-condition stats
    conditions = ["1", "2", "3", "4"]
    cond_labels = ["C1 (Uninformed)", "C2 (Self-informed)",
                   "C3 (Instructed)", "C4 (Constrained)"]
    stats_by_cond = {}
    for cond, label in zip(conditions, cond_labels):
        vals = [per_model[m][cond] for m in per_model
                if per_model[m].get(cond) is not None]
        if vals:
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            ci_lo, ci_hi = bootstrap_bca_mean(vals)
            stats_by_cond[cond] = {
                "label": label,
                "n": len(vals),
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
            }
        else:
            stats_by_cond[cond] = {"label": label, "n": 0}

    # Pairwise comparisons
    comparisons = [("1", "2"), ("2", "3"), ("3", "4")]
    comp_labels = ["C1→C2", "C2→C3", "C3→C4"]
    raw_ps = []
    comp_results = []

    for (ca, cb), clabel in zip(comparisons, comp_labels):
        models_both = [m for m in per_model
                       if per_model[m].get(ca) is not None
                       and per_model[m].get(cb) is not None]
        x = [per_model[m][ca] for m in models_both]
        y = [per_model[m][cb] for m in models_both]

        d = cohens_d_paired(x, y)

        # Wilcoxon signed-rank test
        diff = np.array(x) - np.array(y)
        if np.all(diff == 0):
            p = 1.0
        else:
            try:
                result = sp_stats.wilcoxon(diff)
                p = float(result.pvalue)
            except Exception:
                _, p = sp_stats.ttest_rel(x, y)
                p = float(p)

        raw_ps.append(p)
        comp_results.append({
            "comparison": clabel,
            "n_pairs": len(models_both),
            "cohens_d": round(d, 3),
            "p_raw": round(p, 4),
        })

    # Holm-Bonferroni correction
    adjusted_ps = holm_bonferroni(raw_ps)
    for i, cr in enumerate(comp_results):
        cr["p_holm"] = round(adjusted_ps[i], 4)
        cr["significant_holm"] = adjusted_ps[i] < 0.05

    # Per-model values for reference
    model_table = {}
    for m in sorted(per_model):
        model_table[m] = {
            c: round(per_model[m][c], 4) if per_model[m].get(c) is not None else None
            for c in conditions
        }

    return {
        "n_models": len(per_model),
        "models": sorted(per_model.keys()),
        "by_condition": stats_by_cond,
        "comparisons": comp_results,
        "per_model": model_table,
        "gemini_cfr": {k: round(v, 4) if v is not None else None
                       for k, v in gemini_cfr.items()},
    }


# ---------------------------------------------------------------------------
# Part C: Exp9 Model Exclusion Verification
# ---------------------------------------------------------------------------

def verify_exp9_exclusions() -> dict:
    all_models = [
        "deepseek-r1", "deepseek-v3", "gemini-2.5-pro", "gemma-3-12b",
        "gemma-3-27b", "gpt-oss-120b", "kimi-k2", "llama-3.1-405b",
        "llama-3.1-70b", "llama-3.1-8b", "llama-3.2-3b", "llama-3.3-70b",
        "mistral-large", "mixtral-8x22b", "phi-4", "qwen3-next-80b",
    ]
    exp9_models = [
        "deepseek-r1", "gemini-2.5-pro", "gemma-3-27b", "gpt-oss-120b",
        "kimi-k2", "llama-3.1-405b", "llama-3.1-70b", "llama-3.1-8b",
        "llama-3.3-70b", "mistral-large", "phi-4",
    ]
    missing = sorted(set(all_models) - set(exp9_models))
    return {
        "total_models": len(all_models),
        "exp9_models": len(exp9_models),
        "missing_from_exp9": missing,
        "reason": "Lacked complete Experiment 1 calibration profiles needed for "
                  "tailored task generation (run with fewer channels or added after "
                  "Experiment 9 task bank was generated).",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MIRROR v19 Statistics Computation")
    print("=" * 60)

    # Part A: Exp3 CCE CIs
    print("\n--- Part A: Exp3 v2 CCE Bootstrap CIs ---")
    exp3_results = compute_exp3_cce_cis()
    all_exclude_zero = all(r["ci_excludes_zero"] for r in exp3_results)
    print(f"  Models: {len(exp3_results)}")
    print(f"  CCE range: {min(r['cce'] for r in exp3_results):.3f} – "
          f"{max(r['cce'] for r in exp3_results):.3f}")
    print(f"  All CIs exclude zero: {all_exclude_zero}")
    for r in exp3_results:
        print(f"    {r['model']:20s}  CCE={r['cce']:.3f}  "
              f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]  n={r['n']}")

    # Part B: Exp9 Escalation Stats
    print("\n--- Part B: Exp9 Escalation Curve (11 models) ---")
    esc_results = compute_escalation_stats()
    print(f"  Models: {esc_results['n_models']} ({', '.join(esc_results['models'])})")
    print(f"\n  Per-condition:")
    for cond in ["1", "2", "3", "4"]:
        s = esc_results["by_condition"][cond]
        print(f"    {s['label']:25s}  n={s['n']:2d}  mean={s['mean']:.4f}  "
              f"SD={s['sd']:.4f}  95% CI [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
    print(f"\n  Pairwise comparisons:")
    for c in esc_results["comparisons"]:
        sig = "*" if c["significant_holm"] else "ns"
        print(f"    {c['comparison']:8s}  d={c['cohens_d']:+.3f}  "
              f"p_raw={c['p_raw']:.4f}  p_holm={c['p_holm']:.4f}  [{sig}]")

    # Part C: Exclusions
    print("\n--- Part C: Exp9 Model Exclusions ---")
    exclusions = verify_exp9_exclusions()
    print(f"  Missing from Exp9: {exclusions['missing_from_exp9']}")

    # Save all results
    output = {
        "exp3_cce_cis": exp3_results,
        "exp9_escalation": esc_results,
        "exp9_exclusions": exclusions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
