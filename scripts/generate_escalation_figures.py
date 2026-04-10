"""
Generate Exp9 escalation curve figures:
  1. Main escalation curve with BCa 95% CI and significance brackets
  2. Per-paradigm breakdown (P1, P2, P3)

Outputs:
  figures/exp9_escalation_curve_with_ci.pdf / .png
  figures/exp9_escalation_per_paradigm.pdf / .png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_FILE = Path("data/results/exp9_20260312T140842_results.jsonl")
EXCLUDE_MODELS = {"qwen-3-235b", "qwen3-235b-nim", "command-r-plus"}
N_BOOTSTRAP = 10_000
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("model") not in EXCLUDE_MODELS:
                results.append(r)
    return results


def patch_strengths(results: list[dict]) -> None:
    """Retroactively classify strength fields from exp1 accuracy files."""
    import glob as _glob
    merged: dict = {}
    for fp in sorted(_glob.glob("data/results/exp1_*_accuracy.json"),
                     key=lambda x: Path(x).stat().st_mtime):
        merged.update(json.loads(Path(fp).read_text(encoding="utf-8")))

    for r in results:
        model = r.get("model", "")
        if model not in merged:
            continue
        for slot in ("a", "b"):
            if r.get(f"strength_{slot}") != "unknown":
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


# ── CFR computation ───────────────────────────────────────────────────────────

def compute_cfr_per_model(results: list[dict], condition: int,
                           paradigms: list[int]) -> dict[str, float | None]:
    models = sorted(set(r["model"] for r in results))
    out: dict[str, float | None] = {}
    for model in models:
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
        out[model] = auto_fail / weak_total if weak_total > 0 else None
    return out


# ── Bootstrap BCa ─────────────────────────────────────────────────────────────

def bootstrap_bca_mean(values: list[float], n: int = N_BOOTSTRAP,
                        seed: int = 42) -> tuple[float, float]:
    arr = np.array(values)
    rng = np.random.default_rng(seed)
    n_pts = len(arr)
    if n_pts < 3:
        return float("nan"), float("nan")
    obs = float(np.mean(arr))
    boot = np.array([np.mean(rng.choice(arr, size=n_pts, replace=True)) for _ in range(n)])
    prop_less = np.clip(np.mean(boot < obs), 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    jack = np.array([np.mean(np.delete(arr, i)) for i in range(n_pts)])
    jack_mean = np.mean(jack)
    numer = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = numer / denom if abs(denom) > 1e-12 else 0.0
    def adj(z_a):
        z_adj = z0 + (z0 + z_a) / (1 - a * (z0 + z_a))
        return float(stats.norm.cdf(z_adj))
    lo = float(np.percentile(boot, adj(stats.norm.ppf(0.025)) * 100))
    hi = float(np.percentile(boot, adj(stats.norm.ppf(0.975)) * 100))
    return lo, hi


# ── Significance tests (Wilcoxon signed-rank) ────────────────────────────────

def significance_label(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pairwise_wilcoxon(cfr_by_cond: dict[int, dict[str, float | None]],
                      cond_a: int, cond_b: int) -> tuple[float, str]:
    models = sorted(cfr_by_cond[cond_a].keys())
    pairs = [(cfr_by_cond[cond_a][m], cfr_by_cond[cond_b][m])
             for m in models
             if cfr_by_cond[cond_a].get(m) is not None
             and cfr_by_cond[cond_b].get(m) is not None]
    if len(pairs) < 3:
        return float("nan"), "n/a"
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    diff = x - y
    if np.all(diff == 0):
        return 1.0, "ns"
    try:
        result = stats.wilcoxon(diff)
        p = float(result.pvalue)
    except Exception:
        _, p = stats.ttest_rel(x, y)
        p = float(p)
    return p, significance_label(p)


# ── Main figure generation ────────────────────────────────────────────────────

def build_escalation_stats(results: list[dict],
                            paradigms: list[int]) -> dict:
    """Compute mean CFR, BCa CI, and per-model values for conditions 1–4."""
    cond_data: dict[int, dict[str, float | None]] = {}
    for cond in [1, 2, 3, 4]:
        cond_data[cond] = compute_cfr_per_model(results, cond, paradigms)

    stats_out: dict[int, dict] = {}
    for cond in [1, 2, 3, 4]:
        vals = [v for v in cond_data[cond].values() if v is not None]
        if not vals:
            stats_out[cond] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0, "values": []}
            continue
        mean = float(np.mean(vals))
        lo, hi = bootstrap_bca_mean(vals)
        stats_out[cond] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                           "n": len(vals), "values": vals}

    # Pairwise significance
    sig_pairs = {}
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        p, label = pairwise_wilcoxon(cond_data, a, b)
        sig_pairs[(a, b)] = {"p": p, "label": label}

    return {"by_condition": stats_out, "significance": sig_pairs,
            "per_model_by_cond": cond_data}


def plot_escalation(stats: dict, title: str, out_stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    by_cond = stats["by_condition"]
    sig = stats["significance"]

    conditions = [1, 2, 3, 4]
    labels = ["C1\nUninformed", "C2\nSelf-informed", "C3\nInstructed", "C4\nConstrained"]
    means = [by_cond[c]["mean"] for c in conditions]
    lo = [by_cond[c]["ci_lo"] for c in conditions]
    hi = [by_cond[c]["ci_hi"] for c in conditions]

    # Error bar format: (lo_err, hi_err) = (mean - lo, hi - mean)
    yerr_lo = [m - l if m is not None and l is not None else 0
               for m, l in zip(means, lo)]
    yerr_hi = [h - m if m is not None and h is not None else 0
               for m, h in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Colorblind-safe escalation gradient: bad (coral) → good (teal)
    colors = ["#E8655A", "#EDA247", "#A3C96B", "#2A9D8F"]
    x_pos = np.arange(len(conditions))

    bars = ax.bar(x_pos, means, color=colors, alpha=0.90, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.errorbar(x_pos, means,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=5, capthick=1.5,
                linewidth=1.5, zorder=4)

    # Overlay individual model points
    per_model = stats.get("per_model_by_cond", {})
    for ci, cond in enumerate(conditions):
        model_vals = [v for v in per_model.get(cond, {}).values() if v is not None]
        if model_vals:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(model_vals))
            ax.scatter(ci + jitter, model_vals, color="black", alpha=0.45,
                       s=22, zorder=5)

    # Significance brackets
    sig_pairs_ordered = [(1, 2), (2, 3), (3, 4)]
    bracket_heights = [0.70, 0.76, 0.82]
    for (a, b), bh in zip(sig_pairs_ordered, bracket_heights):
        label = sig[(a, b)]["label"]
        xi = conditions.index(a)
        xj = conditions.index(b)
        ax.plot([xi, xi, xj, xj],
                [bh - 0.01, bh, bh, bh - 0.01],
                lw=1.2, color="black")
        ax.text((xi + xj) / 2, bh + 0.005, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Confident Failure Rate (CFR)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylim(0, 0.92)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # CI legend
    ci_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                               label="95% BCa bootstrap CI")
    ax.legend(handles=[ci_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_per_paradigm(all_stats: dict[int, dict], out_stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paradigm_labels = {1: "P1: Autonomous Tool Use",
                       2: "P2: Checkpoint Decisions",
                       3: "P3: No-Tool Behavioral"}
    # Colorblind-safe escalation gradient matching Figure 2
    colors = ["#E8655A", "#EDA247", "#A3C96B", "#2A9D8F"]
    conditions = [1, 2, 3, 4]
    x_labels = ["C1", "C2", "C3", "C4"]
    x_pos = np.arange(4)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5), sharey=True)

    for ax, (pid, s) in zip(axes, sorted(all_stats.items())):
        by_cond = s["by_condition"]
        means_raw = [by_cond[c]["mean"] for c in conditions]
        means = [m if m is not None else 0.0 for m in means_raw]
        lo = [by_cond[c]["ci_lo"] for c in conditions]
        hi = [by_cond[c]["ci_hi"] for c in conditions]
        yerr_lo = [m - l if means_raw[i] is not None and l is not None else 0
                   for i, (m, l) in enumerate(zip(means, lo))]
        yerr_hi = [h - m if means_raw[i] is not None and h is not None else 0
                   for i, (m, h) in enumerate(zip(means, hi))]

        ax.bar(x_pos, means, color=colors, alpha=0.85, width=0.55, zorder=3,
               edgecolor="white", linewidth=0.8)
        ax.errorbar(x_pos, means,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

        # Per-model dots
        per_model = s.get("per_model_by_cond", {})
        for ci, cond in enumerate(conditions):
            model_vals = [v for v in per_model.get(cond, {}).values() if v is not None]
            if model_vals:
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(model_vals))
                ax.scatter(ci + jitter, model_vals, color="black", alpha=0.4,
                           s=28, zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(paradigm_labels[pid], fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='y', labelsize=9)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if pid == 1:
            ax.set_ylabel("Confident Failure Rate (CFR)", fontsize=10)

    fig.suptitle("Escalation Curve: CFR by Condition, Per Paradigm", fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def print_paradigm_table(all_stats: dict[int, dict]) -> None:
    print("\n| Paradigm | C1 CFR | C2 CFR | C3 CFR | C4 CFR | C1→C4 reduction |")
    print("|----------|--------|--------|--------|--------|-----------------|")
    labels_map = {1: "P1 Autonomous   ",
                  2: "P2 Checkpoint   ",
                  3: "P3 Behavioral   ",
                  0: "All (P1+P2)     "}
    for pid in sorted(k for k in all_stats.keys() if k != 0):
        by_cond = all_stats[pid]["by_condition"]
        means = {c: by_cond[c]["mean"] for c in [1, 2, 3, 4]}
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A "
        c1 = means[1]
        c4 = means[4]
        red = f"{(c1 - c4) / c1 * 100:.1f}%" if c1 and c4 else "N/A"
        label = labels_map[pid]
        print(f"| {label} | {fmt(means[1])} | {fmt(means[2])} | {fmt(means[3])} | {fmt(means[4])} | {red:>15} |")
    # All combined
    all_combined_stats = all_stats.get(0)
    if all_combined_stats:
        by_cond = all_combined_stats["by_condition"]
        means = {c: by_cond[c]["mean"] for c in [1, 2, 3, 4]}
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A "
        c1 = means[1]
        c4 = means[4]
        red = f"{(c1 - c4) / c1 * 100:.1f}%" if c1 and c4 else "N/A"
        print(f"| {labels_map[0]} | {fmt(means[1])} | {fmt(means[2])} | {fmt(means[3])} | {fmt(means[4])} | {red:>15} |")


def main():
    print("Loading results...")
    results = load_results()
    print(f"  {len(results)} records (excluding {EXCLUDE_MODELS})")
    patch_strengths(results)
    print("  Strength fields patched.")

    # ── 2a: Main escalation curve (P1+P2) ─────────────────────────────────
    print("\n[2a] Main escalation curve (Paradigms 1+2)...")
    main_stats = build_escalation_stats(results, paradigms=[1, 2])
    print("\n  Per-condition summary:")
    for cond in [1, 2, 3, 4]:
        s = main_stats["by_condition"][cond]
        m, lo, hi = s["mean"], s["ci_lo"], s["ci_hi"]
        m_s = f"{m:.4f}" if m is not None else "N/A"
        ci_s = (f"[{lo:.4f}, {hi:.4f}]"
                if lo is not None and hi is not None else "N/A")
        print(f"  C{cond}: mean={m_s}  95% BCa CI={ci_s}  n={s['n']}")
    print("\n  Adjacent-condition significance (Wilcoxon signed-rank):")
    for (a, b), v in main_stats["significance"].items():
        print(f"  C{a}→C{b}: p={v['p']:.4f}  {v['label']}")
    plot_escalation(main_stats,
                    title="Exp9: Escalation Curve — CFR Across 4 Conditions\n(Paradigms 1+2, n=10 models, BCa 95% CI)",
                    out_stem="exp9_escalation_curve_with_ci")

    # ── 2b: Per-paradigm breakdown ─────────────────────────────────────────
    print("\n[2b] Per-paradigm escalation curves...")
    paradigm_stats: dict[int, dict] = {}
    for pid in [1, 2, 3]:
        pname = {1: "Autonomous Tool Use", 2: "Checkpoint Decisions",
                 3: "No-Tool Behavioral"}[pid]
        print(f"\n  Paradigm {pid} ({pname}):")
        ps = build_escalation_stats(results, paradigms=[pid])
        paradigm_stats[pid] = ps
        for cond in [1, 2, 3, 4]:
            s = ps["by_condition"][cond]
            m = s["mean"]
            m_s = f"{m:.4f}" if m is not None else "N/A"
            lo, hi = s["ci_lo"], s["ci_hi"]
            ci_s = (f"[{lo:.4f}, {hi:.4f}]"
                    if lo is not None and hi is not None else "N/A")
            print(f"    C{cond}: {m_s}  CI={ci_s}")

    paradigm_stats[0] = main_stats  # combined
    print_paradigm_table(paradigm_stats)
    plot_per_paradigm({pid: paradigm_stats[pid] for pid in [1, 2, 3]},
                      out_stem="exp9_escalation_per_paradigm")

    print("\nDone.")


if __name__ == "__main__":
    main()
