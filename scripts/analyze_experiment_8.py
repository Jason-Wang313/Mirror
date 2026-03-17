"""
Experiment 8 Analysis: Scaling Analysis
=========================================

Analyzes how MIRROR metacognitive calibration metrics scale with model size.

Produces:
  1. Scaling regression for each metric vs log2(params)
     Metrics: natural_acc, wagering_acc, mirror_gap, mci, ece, ars, tri, ehs
  2. Llama-family scaling curve (primary 3-point series: 8B, 70B, 405B)
  3. Cross-family comparison (Llama vs Phi vs Gemma vs others)
  4. "Hero figure" data: 4 MIRROR levels × model size
     Level 1 (natural_acc), Level 2 (wagering_acc), Level 3 (mci), Level 4 (mirror_gap)
  5. Generation comparison: Llama 3.1 70B vs Llama 3.3 70B (same scale, different generation)

Usage:
  python scripts/analyze_experiment_8.py                         # auto-find latest
  python scripts/analyze_experiment_8.py data/results/exp8_*_scaling_data.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Model Metadata ─────────────────────────────────────────────────────────────

SCALING_MODELS_META = {
    "llama-3.1-8b":   {"params_b": 8,   "family": "llama-3.1", "gen": 1, "primary": True},
    "llama-3.1-70b":  {"params_b": 70,  "family": "llama-3.1", "gen": 1, "primary": True},
    "llama-3.1-405b": {"params_b": 405, "family": "llama-3.1", "gen": 1, "primary": True},
    "llama-3.3-70b":  {"params_b": 70,  "family": "llama-3.3", "gen": 2, "primary": False},
    "phi-4":          {"params_b": 4,   "family": "phi",       "gen": 1, "primary": False},
    "gemma-3-27b":    {"params_b": 27,  "family": "gemma-3",   "gen": 1, "primary": False},
    "mistral-large":  {"params_b": 675, "family": "mistral",   "gen": 1, "primary": False},
    "deepseek-r1":    {"params_b": 671, "family": "deepseek",  "gen": 1, "primary": False},
    "deepseek-v3":    {"params_b": 671, "family": "deepseek",  "gen": 2, "primary": False},
    "gpt-oss-120b":   {"params_b": 120, "family": "openai",    "gen": 1, "primary": False},
}

METRICS = ["natural_acc", "wagering_acc", "mirror_gap", "mci", "ece",
           "adversarial_ars", "trust_robustness_index", "epistemic_hygiene_score"]

METRIC_LABELS = {
    "natural_acc": "Natural Accuracy",
    "wagering_acc": "Wagering Accuracy",
    "mirror_gap": "MIRROR Gap (|wager−natural|)",
    "mci": "Metacognitive Convergence Index",
    "ece": "Expected Calibration Error",
    "adversarial_ars": "Adversarial Robustness Score",
    "trust_robustness_index": "Trust Robustness Index (6a)",
    "epistemic_hygiene_score": "Epistemic Hygiene Score (6b)",
}


# ── Regression ─────────────────────────────────────────────────────────────────

def ols_regression(x: list, y: list) -> dict:
    """OLS regression with 95% CI and significance test."""
    if len(x) < 2:
        return {"warning": f"Insufficient points ({len(x)}) for regression"}
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
    n = len(x_arr)
    df = n - 2

    # 95% CI for slope
    t_crit = stats.t.ppf(0.975, df)
    ci_lo = slope - t_crit * std_err
    ci_hi = slope + t_crit * std_err

    # Predicted values
    y_pred = intercept + slope * x_arr
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "n": n,
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "r": round(float(r_value), 4),
        "r_squared": round(float(r_squared), 4),
        "p_value": round(float(p_value), 6),
        "std_err": round(float(std_err), 6),
        "slope_95ci": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
        "significant_p05": p_value < 0.05,
        "significant_p01": p_value < 0.01,
        "direction": "positive" if slope > 0 else "negative",
    }


def bootstrap_regression_ci(x: list, y: list, n_boot: int = 2000) -> dict:
    """Bootstrap confidence interval on slope."""
    if len(x) < 3:
        return {}
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    slopes = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x_arr), size=len(x_arr), replace=True)
        if len(np.unique(idx)) < 2:
            continue
        try:
            s, *_ = stats.linregress(x_arr[idx], y_arr[idx])
            slopes.append(s)
        except Exception:
            pass
    if not slopes:
        return {}
    return {
        "bootstrap_slope_95ci": [
            round(float(np.percentile(slopes, 2.5)), 6),
            round(float(np.percentile(slopes, 97.5)), 6),
        ],
        "n_boot": len(slopes),
    }


# ── Analysis Functions ─────────────────────────────────────────────────────────

def analyze_scaling_regressions(scaling_data: dict) -> dict:
    """Fit scaling regressions for each metric, Llama-primary series first."""
    results = {}

    for metric in METRICS:
        metric_result = {
            "metric": metric,
            "label": METRIC_LABELS.get(metric, metric),
        }

        # Primary: Llama 3.1 family (8B, 70B, 405B)
        primary_models = [m for m, meta in SCALING_MODELS_META.items() if meta["primary"]]
        x_primary, y_primary, labels_primary = [], [], []
        for m in primary_models:
            rec = scaling_data.get(m, {})
            val = rec.get(metric)
            if val is not None:
                log2p = float(np.log2(SCALING_MODELS_META[m]["params_b"]))
                x_primary.append(log2p)
                y_primary.append(val)
                labels_primary.append(m)

        metric_result["llama31_series"] = {
            "models": labels_primary,
            "log2_params": x_primary,
            "values": y_primary,
        }
        if len(x_primary) >= 2:
            reg = ols_regression(x_primary, y_primary)
            boot = bootstrap_regression_ci(x_primary, y_primary)
            reg.update(boot)
            metric_result["llama31_regression"] = reg
        else:
            metric_result["llama31_regression"] = {"warning": f"Only {len(x_primary)} Llama 3.1 data points"}

        # All models with data
        x_all, y_all, labels_all = [], [], []
        for m, rec in scaling_data.items():
            val = rec.get(metric)
            if val is not None and m in SCALING_MODELS_META:
                log2p = float(np.log2(SCALING_MODELS_META[m]["params_b"]))
                x_all.append(log2p)
                y_all.append(val)
                labels_all.append(m)

        metric_result["all_models_series"] = {
            "models": labels_all,
            "log2_params": x_all,
            "values": y_all,
        }
        if len(x_all) >= 3:
            reg_all = ols_regression(x_all, y_all)
            boot_all = bootstrap_regression_ci(x_all, y_all)
            reg_all.update(boot_all)
            metric_result["all_models_regression"] = reg_all
        else:
            metric_result["all_models_regression"] = {"warning": f"Only {len(x_all)} data points"}

        results[metric] = metric_result

    return results


def load_cross_experiment_metrics() -> dict:
    """
    Load per-model metrics from Exp2/3/4 to enrich scaling analysis.
    Returns dict: {model: {transfer_mci, cce_mean_inverted, adaptation_index}}
    """
    results_dir = Path("data/results")
    out = {}

    # Exp2: transfer_mci (Level 1)
    exp2_files = sorted(results_dir.glob("exp2_*_transfer_analysis.json"),
                        key=lambda p: p.stat().st_mtime)
    if exp2_files:
        try:
            with open(exp2_files[-1]) as f:
                e2 = json.load(f)
            for model, data in e2.items():
                tm = data.get("transfer_mci")
                if tm is not None:
                    out.setdefault(model, {})["transfer_mci"] = float(tm)
        except Exception as e:
            print(f"  [WARN] Could not load Exp2 transfer data: {e}")

    # Exp3: mean CCE inverted (Level 2) — mean CCE across intersection types
    exp3_files = sorted(results_dir.glob("exp3_*_metrics.json"),
                        key=lambda p: p.stat().st_mtime)
    if exp3_files:
        try:
            with open(exp3_files[-1]) as f:
                e3 = json.load(f)
            for model, data in e3.items():
                cce_dict = data.get("cce", {})
                cce_vals = [
                    v["mean_cce"] for v in cce_dict.values()
                    if isinstance(v, dict) and v.get("mean_cce") is not None
                ]
                if cce_vals:
                    mean_cce = float(np.mean(cce_vals))
                    # Invert so higher = better calibration (CCE is error-like)
                    cce_inv = float(1.0 - mean_cce)
                    out.setdefault(model, {})["cce_mean"] = mean_cce
                    out.setdefault(model, {})["cce_mean_inverted"] = cce_inv
        except Exception as e:
            print(f"  [WARN] Could not load Exp3 CCE data: {e}")

    # Exp4: mean_ai (Adaptation Index, Level 3)
    exp4_files = sorted(results_dir.glob("exp4_*_metrics.json"),
                        key=lambda p: p.stat().st_mtime)
    if exp4_files:
        try:
            with open(exp4_files[-1]) as f:
                e4 = json.load(f)
            for model, data in e4.items():
                ai = data.get("mean_ai")
                if ai is not None:
                    out.setdefault(model, {})["adaptation_index"] = float(ai)
        except Exception as e:
            print(f"  [WARN] Could not load Exp4 adaptation data: {e}")

    return out


def analyze_hero_figure(scaling_data: dict) -> dict:
    """
    Build 'hero figure' data: 4 MIRROR levels × model size for Llama 3.1 family.

    Per phase2 spec:
      Level 0: natural_acc (Self-Knowledge Atlas, Exp1 — ECE/AUROC proxy)
      Level 1: transfer_mci (Knowledge Transfer, Exp2)
      Level 2: cce_mean_inverted (Compositional Self-Knowledge, Exp3)
      Level 3: adaptation_index (Adaptive Self-Regulation, Exp4)
    """
    hero_metrics = ["natural_acc", "transfer_mci", "cce_mean_inverted", "adaptation_index"]
    hero_labels = {
        "natural_acc": "L0: Self-Knowledge (Natural Accuracy, Exp1)",
        "transfer_mci": "L1: Knowledge Transfer (Transfer MCI, Exp2)",
        "cce_mean_inverted": "L2: Compositional Self-Knowledge (1−CCE, Exp3)",
        "adaptation_index": "L3: Adaptive Self-Regulation (Adaptation Index, Exp4)",
    }

    primary_models = sorted(
        [m for m, meta in SCALING_MODELS_META.items() if meta["primary"]],
        key=lambda m: SCALING_MODELS_META[m]["params_b"]
    )

    hero_data = {}
    for metric in hero_metrics:
        series = {}
        for m in primary_models:
            val = scaling_data.get(m, {}).get(metric)
            if val is not None:
                series[m] = {
                    "params_b": SCALING_MODELS_META[m]["params_b"],
                    "value": val,
                }
        hero_data[metric] = {
            "label": hero_labels[metric],
            "series": series,
        }

    # Compute scaling score: does metric improve consistently with scale?
    hero_summary = {}
    for metric, data in hero_data.items():
        vals_ordered = [data["series"][m]["value"] for m in primary_models if m in data["series"]]
        if len(vals_ordered) >= 2:
            # Spearman correlation with rank order
            rho, p = stats.spearmanr(range(len(vals_ordered)), vals_ordered)
            # For natural_acc and wagering_acc, positive = good. For mirror_gap, negative = good.
            improves_with_scale = rho > 0.5 if metric not in ["mirror_gap", "ece"] else rho < -0.5
            hero_summary[metric] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p), 4),
                "improves_with_scale": improves_with_scale,
                "interpretation": (
                    f"{'Increases' if rho > 0 else 'Decreases'} with scale (ρ={rho:.3f}, p={p:.3f})"
                ),
            }

    return {
        "levels": hero_data,
        "summary": hero_summary,
        "models_on_curve": primary_models,
    }


def analyze_generation_comparison(scaling_data: dict) -> dict:
    """
    Compare Llama 3.1 70B vs Llama 3.3 70B — same scale, different generation.
    Isolates generation/training improvements from scaling effects.
    """
    m31 = "llama-3.1-70b"
    m33 = "llama-3.3-70b"

    rec31 = scaling_data.get(m31, {})
    rec33 = scaling_data.get(m33, {})

    if not rec31.get("natural_acc") and not rec33.get("natural_acc"):
        return {"warning": "No data for either Llama 70B generation"}

    comparison = {"model_31": m31, "model_33": m33}
    for metric in METRICS:
        v31 = rec31.get(metric)
        v33 = rec33.get(metric)
        if v31 is not None and v33 is not None:
            delta = round(v33 - v31, 4)
            comparison[metric] = {
                "llama_31": round(v31, 4),
                "llama_33": round(v33, 4),
                "delta_33_minus_31": delta,
                "direction": "improved" if delta > 0 else "regressed",
            }

    return comparison


def generate_hero_figure(hero: dict, run_id: str):
    """Generate publication-ready hero figure: 4 MIRROR levels × model size."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping hero figure")
        return None

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    level_colors = {
        "natural_acc": "#1f77b4",
        "transfer_mci": "#ff7f0e",
        "cce_mean_inverted": "#2ca02c",
        "adaptation_index": "#d62728",
    }
    level_markers = {
        "natural_acc": "o",
        "transfer_mci": "s",
        "cce_mean_inverted": "^",
        "adaptation_index": "D",
    }

    # Normalize adaptation_index to [0,1] range for visualization (divide by 100)
    adapt_scale = 100.0

    plotted = False
    for metric, data in hero["levels"].items():
        series = data["series"]
        if not series:
            continue
        pts = sorted(series.items(), key=lambda x: x[1]["params_b"])
        xs = [np.log2(p["params_b"]) for _, p in pts]
        ys_raw = [p["value"] for _, p in pts]

        # Normalize adaptation_index for display
        if metric == "adaptation_index":
            ys = [y / adapt_scale for y in ys_raw]
            label_suffix = " (÷100)"
        else:
            ys = ys_raw
            label_suffix = ""

        short_label = data["label"].split("(")[0].strip()
        ax.plot(xs, ys, marker=level_markers[metric], color=level_colors[metric],
                linewidth=2, markersize=8, label=f"{short_label}{label_suffix}")
        plotted = True

    if not plotted:
        print("  No data to plot for hero figure")
        plt.close(fig)
        return None

    # X-axis ticks: log2 param sizes → actual labels
    llama_points = [(8, "8B"), (70, "70B"), (405, "405B")]
    xticks = [np.log2(p) for p, _ in llama_points]
    xlabels = [lbl for _, lbl in llama_points]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlabel("Model Size (Llama 3.1 family)", fontsize=13)
    ax.set_ylabel("Metric Value (normalized)", fontsize=13)
    ax.set_title("Experiment 8: MIRROR Level Performance vs Scale\n"
                 "(L0=Self-Knowledge, L1=Transfer, L2=Compositional, L3=Adaptation)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    # Annotate the predicted pattern
    ax.text(0.98, 0.02,
            "Predicted: L0 rises, L1-L3 plateau near zero",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray", style="italic")

    fig.tight_layout()

    out_pdf = figures_dir / f"exp8_{run_id}_hero_figure.pdf"
    out_png = figures_dir / f"exp8_{run_id}_hero_figure.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Hero figure: {out_pdf}  {out_png}")
    return str(out_pdf)


def analyze_cross_family(scaling_data: dict) -> dict:
    """
    Compare models at similar scales across different families.
    Groups: ~8B tier, ~27B tier, ~70B tier, ~120B+ tier.
    """
    tiers = {
        "~4-8B": [m for m, meta in SCALING_MODELS_META.items() if 3 <= meta["params_b"] <= 12],
        "~27B": [m for m, meta in SCALING_MODELS_META.items() if 15 <= meta["params_b"] <= 40],
        "~70B": [m for m, meta in SCALING_MODELS_META.items() if 50 <= meta["params_b"] <= 100],
        "~120B+": [m for m, meta in SCALING_MODELS_META.items() if meta["params_b"] > 100],
    }

    result = {}
    for tier_label, tier_models in tiers.items():
        tier_data = {}
        for m in tier_models:
            rec = scaling_data.get(m, {})
            if rec.get("natural_acc") is not None:
                tier_data[m] = {
                    "params_b": SCALING_MODELS_META[m]["params_b"],
                    "family": SCALING_MODELS_META[m]["family"],
                    "natural_acc": rec.get("natural_acc"),
                    "mirror_gap": rec.get("mirror_gap"),
                    "mci": rec.get("mci"),
                }
        if tier_data:
            result[tier_label] = {
                "models": tier_data,
                "best_natural_acc": max(tier_data, key=lambda m: tier_data[m]["natural_acc"] or 0),
                "best_mci": max(tier_data, key=lambda m: tier_data[m]["mci"] or 0) if any(
                    tier_data[m]["mci"] for m in tier_data) else None,
            }

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 8: Scaling Analysis")
    parser.add_argument("scaling_data_file", nargs="?", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Resolve input
    if args.latest or args.scaling_data_file is None:
        results_dir = Path("data/results")
        candidates = sorted(
            results_dir.glob("exp8_*_scaling_data.json"),
            key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            print("ERROR: No exp8 scaling data found. Run `python scripts/run_experiment_8.py --extract-only` first.")
            sys.exit(1)
        data_path = candidates[-1]
        print(f"Using: {data_path}")
    else:
        data_path = Path(args.scaling_data_file)

    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        scaling_data = json.load(f)

    n_models = len([m for m, rec in scaling_data.items() if rec.get("natural_acc") is not None])
    print(f"Loaded scaling data: {len(scaling_data)} models ({n_models} with complete data)\n")

    # ── Enrich with Exp2/3/4 MIRROR level metrics ──
    print("Loading cross-experiment MIRROR level metrics (Exp2/3/4)...")
    cross_exp = load_cross_experiment_metrics()
    injected = 0
    for model, extras in cross_exp.items():
        if model in scaling_data:
            scaling_data[model].update(extras)
            injected += 1
    print(f"  Injected Exp2/3/4 metrics for {injected} models")
    for m in ["llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b"]:
        if m in cross_exp:
            e = cross_exp[m]
            def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {m}: transfer_mci={_fmt(e.get('transfer_mci','N/A'))}  "
                  f"cce_inv={_fmt(e.get('cce_mean_inverted','N/A'))}  "
                  f"adaptation_index={_fmt(e.get('adaptation_index','N/A'))}")

    # ── Analyses ──
    print("Running scaling regressions...")
    regressions = analyze_scaling_regressions(scaling_data)

    print("Building hero figure...")
    hero = analyze_hero_figure(scaling_data)

    print("Generation comparison (Llama 3.1 vs 3.3 at 70B)...")
    gen_comparison = analyze_generation_comparison(scaling_data)

    print("Cross-family tier analysis...")
    cross_family = analyze_cross_family(scaling_data)

    # ── Compile output ──
    output = {
        "experiment": "Experiment 8: Scaling Analysis",
        "source_file": str(data_path),
        "n_models_total": len(scaling_data),
        "n_models_with_data": n_models,
        "scaling_regressions": regressions,
        "hero_figure": hero,
        "generation_comparison_70b": gen_comparison,
        "cross_family_tiers": cross_family,
        "raw_scaling_data": scaling_data,
    }

    # ── Save ──
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = data_path.parent / data_path.name.replace("_scaling_data.json", "_analysis.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\nAnalysis saved: {out_path}")

    # ── Print summary ──
    print("\n── Scaling Regression Summary (Llama 3.1 family) ──")
    print(f"{'Metric':<40} {'slope':>10} {'R²':>8} {'p':>10} {'sig':>6}")
    print("-" * 78)
    for metric, data in regressions.items():
        reg = data.get("llama31_regression", {})
        if "warning" in reg:
            print(f"{METRIC_LABELS.get(metric, metric):<40} {'N/A':>10}")
            continue
        slope = reg.get("slope", "—")
        r2 = reg.get("r_squared", "—")
        p = reg.get("p_value", "—")
        sig = "***" if reg.get("significant_p01") else ("*" if reg.get("significant_p05") else "ns")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{METRIC_LABELS.get(metric, metric):<40} {fmt(slope):>10} {fmt(r2):>8} {fmt(p):>10} {sig:>6}")

    # ── Hero figure ──
    run_id = data_path.stem.replace("exp8_", "").replace("_scaling_data", "")
    print("\nGenerating hero figure...")
    generate_hero_figure(hero, run_id)

    print("\n── Hero Figure: 4 MIRROR Levels × Scale ──")
    for metric, data in hero["levels"].items():
        series_str = " → ".join(
            f"{m.split('-')[1][:3]}:{v['value']:.3f}"
            for m, v in sorted(data["series"].items(), key=lambda x: x[1]["params_b"])
        )
        summary = hero["summary"].get(metric, {})
        improves = "↑ with scale" if summary.get("improves_with_scale") else "↓ with scale"
        print(f"  {data['label'][:45]:<45} {series_str}  [{improves}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
