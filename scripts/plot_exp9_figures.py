"""
Publication-ready figures for MIRROR Experiment 9.

Generates 3 figures:
  Fig 1 — Escalation Curve (CFR across 4 conditions, per-model + mean)
  Fig 2 — Money Plot (MIRROR calibration gap vs CFR, model × subcategory)
  Fig 3 — KDI Distribution (per-model violin/strip plot)

Usage:
    python scripts/plot_exp9_figures.py --run-id 20260312T140842
    python scripts/plot_exp9_figures.py --run-id 20260312T140842 --format pdf,png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "data" / "results"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Model colour palette (consistent across figures)
MODEL_COLORS = {
    "llama-3.1-8b":    "#E07B54",
    "llama-3.1-70b":   "#C04A2F",
    "llama-3.1-405b":  "#8B1A0E",
    "llama-3.3-70b":   "#FF9F40",
    "mistral-large":   "#5B8DB8",
    "deepseek-r1":     "#4CAF75",
    "deepseek-v3":     "#2E7D52",
    "gpt-oss-120b":    "#9C6EC7",
    "gemma-3-27b":     "#F5B800",
    "kimi-k2":         "#00ACC1",
    "phi-4":           "#8D6E63",
    "gemini-2.5-pro":  "#34A853",
}
DEFAULT_COLOR = "#888888"

CONDITION_LABELS = {1: "C1\nUninformed", 2: "C2\nSelf-\ninformed",
                    3: "C3\nInstructed", 4: "C4\nConstrained"}
CONDITION_COLORS = ["#D32F2F", "#F57C00", "#1976D2", "#388E3C"]


def load_analysis(run_id: str) -> dict:
    path = RESULTS_DIR / f"exp9_{run_id}_analysis" / "analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Escalation Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_escalation_curve(analysis: dict, formats: list[str]) -> Path:
    escalation = analysis.get("escalation_curve", {})
    per_model = escalation.get("per_model", {})
    mean_curve = escalation.get("mean_curve", {})
    models = [m for m in per_model if any(
        per_model[m].get(c) is not None for c in [1, 2, 3, 4]
    )]

    conditions = [1, 2, 3, 4]
    cond_x = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Per-model thin lines (muted)
    for model in sorted(models):
        vals = [per_model[model].get(c) for c in conditions]
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)
        if any(v is not None for v in vals):
            xs = [x for x, v in zip(cond_x, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.0,
                    marker="o", markersize=3, zorder=2)

    # Mean curve (bold)
    mean_vals = [mean_curve.get(c) for c in conditions]
    xs_mean = [x for x, v in zip(cond_x, mean_vals) if v is not None]
    ys_mean = [v for v in mean_vals if v is not None]
    ax.plot(xs_mean, ys_mean, color="#1A1A1A", linewidth=2.8,
            marker="D", markersize=7, zorder=5, label="Mean (all models)")

    # Annotate mean values
    for x, y in zip(xs_mean, ys_mean):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold",
                    color="#1A1A1A")

    # Shade background regions
    for i, (x, color) in enumerate(zip(cond_x, CONDITION_COLORS)):
        ax.axvspan(x - 0.4, x + 0.4, alpha=0.06, color=color, zorder=0)

    # Model legend (small, right side)
    handles = [mpatches.Patch(color=MODEL_COLORS.get(m, DEFAULT_COLOR), label=m, alpha=0.7)
               for m in sorted(models)]
    handles.append(plt.Line2D([0], [0], color="#1A1A1A", linewidth=2.5,
                               marker="D", markersize=6, label="Mean"))
    ax.legend(handles=handles, loc="upper right", fontsize=7.5,
              framealpha=0.85, ncol=2, columnspacing=0.8, handlelength=1.2)

    ax.set_xticks(cond_x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Confident Failure Rate (CFR)", fontsize=11)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.5, 3.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Escalation Curve: CFR Across Metacognitive Conditions", pad=8)
    ax.axhline(0, color="#CCCCCC", linewidth=0.7, zorder=0)

    # Drop annotation arrow between C1 and C4
    c1_y = mean_curve.get(1, 0)
    c4_y = mean_curve.get(4, 0)
    if c1_y and c4_y:
        drop_pct = (c1_y - c4_y) / c1_y * 100
        ax.annotate(
            f"−{drop_pct:.0f}%\n(C1→C4)",
            xy=(3, c4_y), xytext=(2.4, (c1_y + c4_y) / 2),
            fontsize=8.5, color="#388E3C", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#388E3C", lw=1.2),
            ha="center",
        )

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = FIGURES_DIR / f"fig1_escalation_curve.{fmt}"
        fig.savefig(p, format=fmt)
        saved.append(p)
        print(f"  Saved: {p}")
    plt.close(fig)
    return saved[0]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Money Plot (MIRROR gap vs CFR)
# ─────────────────────────────────────────────────────────────────────────────

def plot_money_plot(analysis: dict, formats: list[str]) -> Path:
    money = analysis.get("money_plot_primary", {})
    data_points = money.get("data_points", [])

    if not data_points:
        # Fallback: try supplementary
        money = analysis.get("money_plot_supplementary", {})
        data_points = money.get("data_points", [])

    if not data_points:
        print("  WARNING: No money plot data points available — skipping Fig 2")
        return None

    xs = np.array([p["mirror_gap"] for p in data_points])
    ys = np.array([p["cfr"] for p in data_points])
    models_pts = [p.get("model", "") for p in data_points]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Per-model scatter
    for model in sorted(set(models_pts)):
        idx = [i for i, m in enumerate(models_pts) if m == model]
        ax.scatter(xs[idx], ys[idx],
                   color=MODEL_COLORS.get(model, DEFAULT_COLOR),
                   alpha=0.72, s=42, zorder=3,
                   label=model, edgecolors="white", linewidths=0.4)

    # Regression line + CI
    if len(xs) >= 5:
        slope, intercept, r_val, p_val, se = stats.linregress(xs, ys)
        x_line = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#333333", linewidth=1.6,
                linestyle="--", zorder=4, label=f"OLS (r={r_val:.3f}, p={p_val:.3f})")

        # 95% CI band
        n = len(xs)
        x_mean = xs.mean()
        se_line = se * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ((xs - x_mean) ** 2).sum())
        t_crit = stats.t.ppf(0.975, df=n - 2)
        ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                        alpha=0.12, color="#555555", zorder=2)

        # Stats annotation
        bca_lo = money.get("bca_ci_low")
        bca_hi = money.get("bca_ci_high")
        ci_str = f"BCa 95% CI [{bca_lo:.3f}, {bca_hi:.3f}]" if bca_lo is not None else ""
        partial_r = money.get("partial_r")
        partial_str = f"partial r = {partial_r:.3f}" if partial_r is not None else ""
        stats_text = f"Pearson r = {r_val:.3f}  (p = {p_val:.3f})\n{ci_str}\n{partial_str}"
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xlabel("MIRROR Calibration Gap  |wagering_acc − natural_acc|", fontsize=11)
    ax.set_ylabel("Confident Failure Rate (CFR)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(-0.02, max(xs.max() + 0.05, 0.6))
    ax.set_ylim(-0.02, min(ys.max() + 0.08, 1.05))
    ax.set_title("Money Plot: MIRROR Calibration Gap vs Agentic Failure Rate\n"
                 "(subcategory level, fixed tasks, Condition 1, Paradigm 1)", pad=8)
    ax.axhline(0, color="#DDDDDD", linewidth=0.7)
    ax.axvline(0, color="#DDDDDD", linewidth=0.7)

    # Legend (compact)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=7.5,
              framealpha=0.85, ncol=2, columnspacing=0.8, handlelength=1.0)

    # Null result stamp
    n_pts = len(data_points)
    ax.text(0.97, 0.03, f"NULL RESULT\n(N = {n_pts} subcategory-model pairs)",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, color="#C62828", style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFEBEE",
                      edgecolor="#EF9A9A", alpha=0.9))

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = FIGURES_DIR / f"fig2_money_plot.{fmt}"
        fig.savefig(p, format=fmt)
        saved.append(p)
        print(f"  Saved: {p}")
    plt.close(fig)
    return saved[0]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: KDI Distribution (per-model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kdi_distribution(analysis: dict, formats: list[str]) -> Path:
    kdi_table = analysis.get("kdi_table", {})
    if not kdi_table:
        print("  WARNING: No KDI data available — skipping Fig 3")
        return None

    # Rebuild per-subcategory KDI values per model from table
    model_kdi: dict[str, list[float]] = {}
    for model, domains in kdi_table.items():
        if not isinstance(domains, dict):
            continue
        vals = []
        per_domain = domains.get("per_domain", {})
        for domain, kdi_val in per_domain.items():
            if kdi_val is not None and isinstance(kdi_val, (int, float)):
                vals.append(float(kdi_val))
        if vals:
            model_kdi[model] = vals

    if not model_kdi:
        print("  WARNING: KDI table empty — skipping Fig 3")
        return None

    # Sort models by median KDI (ascending = worst first)
    sorted_models = sorted(model_kdi.keys(), key=lambda m: float(np.median(model_kdi[m])))
    n_models = len(sorted_models)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    positions = list(range(n_models))

    for i, model in enumerate(sorted_models):
        vals = model_kdi[model]
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        # Violin
        if len(vals) >= 3:
            try:
                vp = ax.violinplot([vals], positions=[i], widths=0.55,
                                   showmedians=False, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                    pc.set_edgecolor(color)
                    pc.set_linewidth(0.8)
            except Exception:
                pass

        # Strip plot (jittered)
        jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, alpha=0.75, s=28, zorder=4,
                   edgecolors="white", linewidths=0.3)

        # Median line
        med = np.median(vals)
        ax.hlines(med, i - 0.22, i + 0.22, colors=color,
                  linewidths=2.2, zorder=5)

        # Annotate median value
        ax.text(i, med - 0.03, f"{med:.3f}", ha="center", va="top",
                fontsize=7.5, color=color, fontweight="bold")

    # Zero reference line
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=1, label="KDI = 0 (ideal)")

    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("llama-3.1-", "llama\n3.1-").replace("llama-3.3-", "llama\n3.3-")
                        for m in sorted_models],
                       fontsize=8.5, rotation=0)
    ax.set_ylabel("Knowing-Doing Index (KDI)", fontsize=11)
    ax.set_title("KDI Distribution per Model\n"
                 "(KDI = MIRROR gap − appropriate action rate; lower = worse)", pad=8)
    ax.set_xlim(-0.6, n_models - 0.4)

    # Shading: negative = bad
    y_min, y_max = ax.get_ylim()
    ax.axhspan(y_min, 0, alpha=0.04, color="#D32F2F", zorder=0)
    ax.axhspan(0, max(y_max, 0.05), alpha=0.04, color="#388E3C", zorder=0)
    ax.text(n_models - 0.55, -0.01, "All models below 0\n(fail to act on knowledge)",
            ha="right", va="top", fontsize=8, color="#C62828", style="italic")

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = FIGURES_DIR / f"fig3_kdi_distribution.{fmt}"
        fig.savefig(p, format=fmt)
        saved.append(p)
        print(f"  Saved: {p}")
    plt.close(fig)
    return saved[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Exp9 publication figures")
    parser.add_argument("--run-id", required=True, help="Experiment 9 run ID")
    parser.add_argument("--format", default="png,pdf",
                        help="Comma-separated output formats (default: png,pdf)")
    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]

    print(f"\n{'='*60}")
    print(f"MIRROR EXP9 FIGURES  run_id={args.run_id}")
    print(f"Formats: {formats}")
    print(f"Output: {FIGURES_DIR}/")
    print(f"{'='*60}\n")

    analysis = load_analysis(args.run_id)
    print(f"Loaded analysis: {len(analysis.get('models', []))} models\n")

    print("[Fig 1] Escalation Curve...")
    plot_escalation_curve(analysis, formats)

    print("\n[Fig 2] Money Plot...")
    plot_money_plot(analysis, formats)

    print("\n[Fig 3] KDI Distribution...")
    plot_kdi_distribution(analysis, formats)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
