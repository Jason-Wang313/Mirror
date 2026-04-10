"""
Regenerate fig3_kdi_distribution with all 16 models.

Merges KDI data from:
  - exp9_20260312T140842_analysis (10 models)
  - exp9_20260323T203013_analysis (5 models)
  - gemini-2.5-pro (constructed from known values)

Usage: python scripts/regenerate_kdi_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "data" / "results"

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

MODEL_COLORS = {
    "llama-3.1-8b":    "#E07B54",
    "llama-3.1-70b":   "#C04A2F",
    "llama-3.1-405b":  "#8B1A0E",
    "llama-3.2-3b":    "#FF6F00",
    "llama-3.3-70b":   "#FF9F40",
    "mistral-large":   "#5B8DB8",
    "mixtral-8x22b":   "#3E7CB1",
    "deepseek-r1":     "#4CAF75",
    "deepseek-v3":     "#2E7D52",
    "gpt-oss-120b":    "#9C6EC7",
    "gemma-3-12b":     "#FDD835",
    "gemma-3-27b":     "#F5B800",
    "kimi-k2":         "#00ACC1",
    "phi-4":           "#8D6E63",
    "gemini-2.5-pro":  "#34A853",
    "qwen3-next-80b":  "#E91E63",
}
DEFAULT_COLOR = "#888888"


def load_kdi_table(run_id: str) -> dict:
    path = RESULTS_DIR / f"exp9_{run_id}_analysis" / "analysis.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("kdi_table", {})


def main():
    # Load KDI data from both analyses
    kdi_10 = load_kdi_table("20260312T140842")  # 10 models
    kdi_5 = load_kdi_table("20260323T203013")   # 5 new models

    # Merge
    merged_kdi = {}
    merged_kdi.update(kdi_10)
    merged_kdi.update(kdi_5)

    # Add gemini-2.5-pro (not in either analysis; KDI=+0.084 from table)
    # Gemini has CFR_C1=0.937, gap=0.146 (from Exp1).
    # Its per-domain KDI breakdown isn't in an analysis file, but the model-level
    # value is known. Use it as a single-domain entry for the figure.
    if "gemini-2.5-pro" not in merged_kdi:
        merged_kdi["gemini-2.5-pro"] = {
            "mean_kdi": 0.084,
            "median_kdi": 0.084,
            "proportion_kdi_gt_0_2": 0.0,
            "per_domain": {"overall": 0.084}
        }

    print(f"Total models with KDI data: {len(merged_kdi)}")
    for m in sorted(merged_kdi.keys()):
        print(f"  {m}: mean_kdi={merged_kdi[m].get('mean_kdi', '?')}")

    # Build per-model domain KDI values
    model_kdi: dict[str, list[float]] = {}
    for model, data in merged_kdi.items():
        per_domain = data.get("per_domain", {})
        vals = [float(v) for v in per_domain.values()
                if v is not None and isinstance(v, (int, float))]
        if vals:
            model_kdi[model] = vals

    # Sort by median KDI (ascending = most negative first)
    sorted_models = sorted(model_kdi.keys(),
                           key=lambda m: float(np.median(model_kdi[m])))
    n_models = len(sorted_models)
    print(f"\nModels in figure: {n_models}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = list(range(n_models))

    # Track per-model min for annotation placement
    model_mins = []

    for i, model in enumerate(sorted_models):
        vals = model_kdi[model]
        color = MODEL_COLORS.get(model, DEFAULT_COLOR)

        # Violin (need >=3 points)
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
                   color=color, alpha=0.75, s=32, zorder=4,
                   edgecolors="white", linewidths=0.3)

        # Median line
        med = np.median(vals)
        ax.hlines(med, i - 0.22, i + 0.22, colors=color,
                  linewidths=2.2, zorder=5)

        model_mins.append((i, min(vals), med, color))

    # Annotate median values below the lowest data point for each model
    for i, val_min, med, color in model_mins:
        ax.text(i, val_min - 0.05, f"{med:.3f}", ha="center", va="top",
                fontsize=7.5, color=color, fontweight="bold")

    # Zero reference line
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=1, label="KDI = 0 (ideal)")

    ax.set_xticks(positions)
    short_names = []
    for m in sorted_models:
        name = m
        for prefix in ["llama-3.1-", "llama-3.2-", "llama-3.3-"]:
            if m.startswith(prefix):
                name = "llama\n" + m[len("llama-"):]
                break
        if m.startswith("mixtral-"):
            name = "mixtral\n8x22b"
        if m.startswith("qwen3-"):
            name = "qwen3\nnext-80b"
        if m.startswith("deepseek-"):
            name = "deepseek\n" + m.split("-", 1)[1]
        if m.startswith("gemma-3-"):
            name = "gemma-3\n" + m.split("-")[-1]
        if m.startswith("gemini-"):
            name = "gemini\n2.5-pro"
        short_names.append(name)

    ax.set_xticklabels(short_names, fontsize=9.5, rotation=35, ha="right")
    ax.set_ylabel("Knowing-Doing Index (KDI)", fontsize=12)
    ax.set_title("KDI Distribution per Model (16 models)\n"
                 "(KDI = MIRROR gap − appropriate action rate; lower = worse)",
                 fontsize=13, pad=10)
    ax.set_xlim(-0.6, n_models - 0.4)
    ax.tick_params(axis='y', labelsize=10)

    # Shading
    y_min, y_max = ax.get_ylim()
    ax.axhspan(y_min, 0, alpha=0.04, color="#D32F2F", zorder=0)
    ax.axhspan(0, max(y_max, 0.05), alpha=0.04, color="#388E3C", zorder=0)

    # Callout in the empty bottom-right area
    ax.text(n_models - 0.5, -0.35,
            "13/16 models below 0\n(fail to act on knowledge)",
            ha="right", va="top", fontsize=8, color="#C62828", style="italic")

    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        p = FIGURES_DIR / f"fig3_kdi_distribution.{fmt}"
        fig.savefig(p, format=fmt)
        print(f"Saved: {p}")
    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
