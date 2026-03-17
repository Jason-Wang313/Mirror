"""
Export a clean, paper-integration-ready JSON summary from Exp9 analysis.

Produces paper/paper_summary.json with:
  - Primary statistics (r, p, CI, partial_r, β)
  - Per-model CFR, UDR, KDI
  - Escalation curve (conditions 1-4, mean + per-model)
  - C4 breakdown (which models still fail and why)
  - MIRROR gap table (per new model per domain)
  - Control results (C2, C3, C4, C5)
  - Data quality flags
  - Formatted inline-citation strings (ready to paste into LaTeX)

Usage:
    python scripts/export_paper_summary.py --run-id 20260312T140842
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"
PAPER_DIR = REPO_ROOT / "paper"


def load_analysis(run_id: str) -> dict:
    path = RESULTS_DIR / f"exp9_{run_id}_analysis" / "analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_exp1_metrics() -> dict:
    """Merge all exp1 accuracy files; newest overrides for same model."""
    files = sorted(
        [p for p in RESULTS_DIR.glob("exp1_*_accuracy.json") if "meta" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    merged: dict = {}
    for f in files:
        try:
            merged.update(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return merged


def _fmt(v, decimals: int = 4) -> str | None:
    """Format float to string, None-safe."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), decimals)


def build_primary_stats(analysis: dict) -> dict:
    mp = analysis.get("money_plot_primary", {})
    partial = analysis.get("partial_correlation_table", {})
    composite = partial.get("MIRROR_composite", {})
    bca = mp.get("bca_ci_95") or []
    bca_lo = bca[0] if len(bca) >= 2 else None
    bca_hi = bca[1] if len(bca) >= 2 else None
    partial_r = mp.get("partial_r_ctrl_accuracy")
    me = mp.get("mixed_effects") or {}
    return {
        "n_data_points": mp.get("n_points"),
        "pearson_r": _fmt(mp.get("pearson_r"), 4),
        "pearson_p": _fmt(mp.get("pearson_p"), 4),
        "spearman_rho": _fmt(mp.get("spearman_r"), 4),
        "bca_ci_95_low": _fmt(bca_lo, 4),
        "bca_ci_95_high": _fmt(bca_hi, 4),
        "partial_r_controlling_accuracy": _fmt(partial_r, 4),
        "mixed_effects_beta": _fmt(me.get("beta"), 4),
        "mixed_effects_p": _fmt(me.get("p_value"), 4),
        "partial_r_composite": _fmt(composite.get("partial_r"), 4),
        "q_value_fdr": _fmt(composite.get("q_value"), 4),
        "interpretation": mp.get("interpretation", ""),
        "result_type": "NULL_RESULT" if abs(mp.get("pearson_r") or 0) < 0.2 else "POSITIVE",
        "citation_string": (
            f"r = {_fmt(mp.get('pearson_r'), 3)}, "
            f"p = {_fmt(mp.get('pearson_p'), 3)}, "
            f"BCa 95\\% CI [{_fmt(bca_lo, 3)}, {_fmt(bca_hi, 3)}], "
            f"partial r = {_fmt(partial_r, 3)}"
        ),
    }


def build_escalation_stats(analysis: dict) -> dict:
    esc = analysis.get("escalation_curve", {})
    mean = esc.get("mean_curve", {})
    per_model = esc.get("per_model", {})

    # Keys may be ints or strings depending on serialization
    def _get(d, k):
        return d.get(k) if d.get(k) is not None else d.get(str(k))

    c1, c2, c3, c4 = _get(mean, 1), _get(mean, 2), _get(mean, 3), _get(mean, 4)
    drop_total = _fmt(c1 - c4, 4) if c1 and c4 else None
    pct_reduction = _fmt((c1 - c4) / c1 * 100, 1) if c1 and c4 else None

    return {
        "mean_cfr_c1": _fmt(c1, 4),
        "mean_cfr_c2": _fmt(c2, 4),
        "mean_cfr_c3": _fmt(c3, 4),
        "mean_cfr_c4": _fmt(c4, 4),
        "drop_c1_c2": _fmt(c1 - c2, 4) if c1 and c2 else None,
        "drop_c2_c3": _fmt(c2 - c3, 4) if c2 and c3 else None,
        "drop_c3_c4": _fmt(c3 - c4, 4) if c3 and c4 else None,
        "total_drop_c1_c4": drop_total,
        "pct_reduction_c1_to_c4": pct_reduction,
        "per_model_cfr": {
            m: {str(c): _fmt(cvals.get(c) if cvals.get(c) is not None else cvals.get(str(c)), 4)
                for c in [1, 2, 3, 4]}
            for m, cvals in per_model.items()
        },
        "interpretation": esc.get("interpretation", ""),
        "citation_string": (
            f"CFR: C1={_fmt(c1,3)}, C2={_fmt(c2,3)}, "
            f"C3={_fmt(c3,3)}, C4={_fmt(c4,3)}; "
            f"total reduction {pct_reduction}\\%"
        ) if all(v is not None for v in [c1, c2, c3, c4]) else "",
    }


def build_per_model_table(analysis: dict) -> dict:
    cfr_udr = analysis.get("cfr_udr_condition1", {})
    kdi_raw = analysis.get("kdi_table", {})

    # Flatten KDI to mean/median per model
    # Actual structure: {model: {"mean_kdi": float, "per_domain": {domain: float}}}
    kdi_summary: dict = {}
    for model, model_data in kdi_raw.items():
        if not isinstance(model_data, dict):
            continue
        # Use pre-computed mean/median if available
        mean_kdi = model_data.get("mean_kdi")
        median_kdi = model_data.get("median_kdi")
        per_domain = model_data.get("per_domain", {})
        vals = [v for v in per_domain.values() if v is not None and isinstance(v, (int, float))]
        if mean_kdi is None and vals:
            mean_kdi = sum(vals) / len(vals)
        if median_kdi is None and vals:
            median_kdi = sorted(vals)[len(vals) // 2]
        if mean_kdi is not None:
            kdi_summary[model] = {
                "mean_kdi": _fmt(mean_kdi, 4),
                "median_kdi": _fmt(median_kdi, 4),
                "n_subcategories": len(vals),
                "pct_above_0.2": _fmt(sum(1 for v in vals if v > 0.2) / len(vals) * 100, 1) if vals else None,
                "by_domain": {d: _fmt(v, 4) for d, v in per_domain.items()},
            }

    # Combine
    models = sorted(set(list(cfr_udr.keys()) + list(kdi_summary.keys())))
    table: dict = {}
    for model in models:
        cfu = cfr_udr.get(model, {})
        kdi = kdi_summary.get(model, {})
        table[model] = {
            "cfr_c1": _fmt(cfu.get("cfr"), 4),
            "udr_c1": _fmt(cfu.get("udr"), 4),
            "n_weak": cfu.get("n_weak"),
            "n_strong": cfu.get("n_strong"),
            "mean_kdi": kdi.get("mean_kdi"),
            "median_kdi": kdi.get("median_kdi"),
            "n_subcategories_kdi": kdi.get("n_subcategories"),
            "kdi_by_domain": kdi.get("by_domain", {}),
        }
    return table


def build_mirror_gap_table(exp1_metrics: dict, exclude: set) -> dict:
    DOMAINS = ["arithmetic", "spatial", "temporal", "linguistic",
               "logical", "social", "factual", "procedural"]
    table: dict = {}
    for model, model_data in exp1_metrics.items():
        if model in exclude:
            continue
        table[model] = {}
        for domain in DOMAINS:
            d = model_data.get(domain, {})
            nat = d.get("natural_acc")
            wag = d.get("wagering_acc")
            gap = abs(wag - nat) if nat is not None and wag is not None else None
            table[model][domain] = {
                "natural_acc": _fmt(nat, 4),
                "wagering_acc": _fmt(wag, 4),
                "mirror_gap": _fmt(gap, 4),
                "strength": (
                    "strong" if nat is not None and nat >= 0.60 else
                    "weak" if nat is not None and nat <= 0.40 else
                    "medium" if nat is not None else "unknown"
                ),
            }
    return table


def build_c4_breakdown(analysis: dict) -> dict:
    c4 = analysis.get("c4_cfr_breakdown", {})
    if not c4:
        return {}
    out: dict = {}
    for model, d in c4.items():
        cfr = d.get("cfr")
        out[model] = {
            "cfr_c4": _fmt(cfr, 4),
            "n_failures": d.get("n_failures"),
            "n_weak_trials": d.get("n_weak_trials"),
            "still_fails": cfr is not None and cfr > 0,
            "top_failing_domain": (
                max(d.get("by_domain", {}).items(),
                    key=lambda x: x[1].get("cfr") or 0,
                    default=(None, {}))[0]
            ),
            "domain_cfr": {dom: _fmt(dv.get("cfr"), 4)
                           for dom, dv in d.get("by_domain", {}).items()},
            "error_types": d.get("error_type_distribution", {}),
        }
    return out


def build_controls_summary(analysis: dict) -> dict:
    ctrl2 = analysis.get("control2_false_scores", {})
    ctrl3 = analysis.get("control3_dissociation", {})
    ctrl4 = analysis.get("control4_breakdown", {})
    cohens = analysis.get("cohens_d_strong_weak", {})
    oracle = analysis.get("oracle_baseline", {})

    return {
        "control2_false_scores": {
            "per_model": {m: d.get("finding") for m, d in ctrl2.items()
                          if isinstance(d, dict) and "finding" in d},
            "summary": ctrl2.get("summary", {}),
        },
        "control3_dissociation": {
            "n_pairs": ctrl3.get("n_dissociation_pairs"),
            "permutation_p": _fmt(ctrl3.get("permutation_p"), 4),
            "interpretation": ctrl3.get("interpretation", ""),
        },
        "control4_cohens_d": {
            "cohens_d": _fmt(cohens.get("cohens_d"), 4),
            "mean_weak_deferral": _fmt(cohens.get("mean_deferral_weak"), 4),
            "mean_strong_deferral": _fmt(cohens.get("mean_deferral_strong"), 4),
            "interpretation": cohens.get("interpretation", ""),
            "n_models_inverted": ctrl4.get("interpretation", "").split("/")[0].split(" ")[-1]
                                 if ctrl4.get("interpretation") else None,
        },
        "control5_oracle": {
            m: {"real_cfr_c1": _fmt(d.get("cfr"), 4),
                "oracle_cfr": _fmt(d.get("oracle_cfr"), 4),
                "gap": _fmt((d.get("cfr") or 0) - (d.get("oracle_cfr") or 0), 4)}
            for m, d in oracle.items()
            if isinstance(d, dict) and d.get("cfr") is not None
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Export paper summary JSON")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exclude-models",
                        default="qwen-3-235b,qwen3-235b-nim,command-r-plus",
                        help="Comma-separated models to exclude")
    args = parser.parse_args()

    exclude = {m.strip() for m in args.exclude_models.split(",") if m.strip()}

    print(f"\n{'='*60}")
    print(f"MIRROR EXP9 PAPER SUMMARY EXPORT  run_id={args.run_id}")
    print(f"Excluding: {sorted(exclude)}")
    print(f"{'='*60}\n")

    analysis = load_analysis(args.run_id)
    exp1_metrics = load_exp1_metrics()

    print(f"  Loaded analysis: {len(analysis.get('models', []))} models")
    print(f"  Loaded exp1 metrics: {len(exp1_metrics)} models\n")

    summary = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "run_id": args.run_id,
            "n_trials": analysis.get("n_trials"),
            "n_trials_total": analysis.get("n_trials_total_including_excluded"),
            "models_included": analysis.get("models", []),
            "models_excluded": sorted(analysis.get("excluded_models", [])),
            "exp1_models": sorted(exp1_metrics.keys()),
        },
        "primary_result": build_primary_stats(analysis),
        "escalation_curve": build_escalation_stats(analysis),
        "per_model_metrics": build_per_model_table(analysis),
        "mirror_gap_table": build_mirror_gap_table(exp1_metrics, exclude),
        "c4_cfr_breakdown": build_c4_breakdown(analysis),
        "controls": build_controls_summary(analysis),
        "data_quality": analysis.get("data_quality", {}),
    }

    # Print key numbers
    pr = summary["primary_result"]
    esc = summary["escalation_curve"]
    print("─" * 50)
    print("PRIMARY RESULT:")
    print(f"  r = {pr['pearson_r']}, p = {pr['pearson_p']}")
    print(f"  BCa CI: [{pr['bca_ci_95_low']}, {pr['bca_ci_95_high']}]")
    print(f"  partial r = {pr['partial_r_controlling_accuracy']}")
    print(f"  → {pr['result_type']}")
    print()
    print("ESCALATION CURVE:")
    print(f"  C1={esc['mean_cfr_c1']}  C2={esc['mean_cfr_c2']}  "
          f"C3={esc['mean_cfr_c3']}  C4={esc['mean_cfr_c4']}")
    print(f"  Total drop: {esc['total_drop_c1_c4']} ({esc['pct_reduction_c1_to_c4']}%)")
    print()
    print("PER-MODEL CFR/KDI (C1):")
    pm = summary["per_model_metrics"]
    for model in sorted(pm.keys()):
        d = pm[model]
        print(f"  {model:<25} CFR={d['cfr_c1']}  KDI={d['mean_kdi']}")
    print()
    print("C4 REMAINING FAILURES:")
    c4b = summary["c4_cfr_breakdown"]
    failures = [(m, d['cfr_c4'], d['n_failures']) for m, d in c4b.items()
                if d.get('still_fails')]
    if failures:
        for m, cfr, n in sorted(failures, key=lambda x: x[1] or 0, reverse=True):
            top = c4b[m].get('top_failing_domain', '?')
            print(f"  {m:<25} C4 CFR={cfr}  n_fail={n}  top_domain={top}")
    else:
        print("  No models fail under C4.")
    print("─" * 50)

    # Write
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAPER_DIR / "paper_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExported: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
