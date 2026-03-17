"""
LaTeX Table Export
==================

Generates publication-ready LaTeX tables from Experiment 1 analysis JSONs.
Tables go directly into the paper.

Usage:
  python scripts/export_latex_tables.py --run-id <run_id>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DOMAINS_SHORT = {
    "arithmetic":   "Arith",
    "spatial":      "Spatial",
    "temporal":     "Temporal",
    "linguistic":   "Ling",
    "logical":      "Logic",
    "social":       "Social",
    "factual":      "Factual",
    "procedural":   "Proced",
}

DOMAIN_ORDER = list(DOMAINS_SHORT.keys())

MODEL_LABELS = {
    "llama-3.1-8b":   "Llama 8B",
    "llama-3.1-70b":  "Llama 70B",
    "llama-3.1-405b": "Llama 405B",
    "mistral-large":  "Mistral-L",
    "qwen-3-235b":    "Qwen-235B",
    "gpt-oss-120b":   "GPT-OSS-120B",
    "deepseek-r1":    "DeepSeek-R1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
        return "—"
    return f"{v:.{decimals}f}"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model[:14])


def _write_table(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Table 1: Model × Domain Accuracy Matrix (Channel 5 natural)
# ---------------------------------------------------------------------------

def export_table1_accuracy(accuracy: dict, out_path: Path) -> None:
    models = sorted(accuracy.keys())
    domains_present = [d for d in DOMAIN_ORDER if any(d in accuracy.get(m, {}) for m in models)]

    col_spec = "l" + "c" * len(domains_present)
    domain_headers = " & ".join(DOMAINS_SHORT.get(d, d) for d in domains_present)

    rows = []
    for model in models:
        label = _model_label(model)
        cells = []
        for domain in domains_present:
            v = accuracy.get(model, {}).get(domain, {}).get("natural_acc")
            cells.append(_fmt(v))
        rows.append(f"        {label} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    tex = rf"""% Table 1: Baseline accuracy by model and domain (Channel 5, natural framing)
\begin{{table}}[ht]
\centering
\caption{{Baseline accuracy by model and domain (Channel 5, no metacognitive framing).
  Values are proportion correct. Dashes indicate insufficient data.}}
\label{{tab:accuracy}}
\begin{{tabular}}{{{col_spec}}}
\toprule
Model & {domain_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write_table(out_path, tex)


# ---------------------------------------------------------------------------
# Table 2: Metacognitive Calibration Summary
# ---------------------------------------------------------------------------

def export_table2_calibration(
    calibration: dict,
    mci: dict,
    hallucination: dict,
    out_path: Path,
) -> None:
    models = sorted(calibration.keys())

    rows = []
    for model in models:
        label = _model_label(model)
        cal = calibration.get(model, {}).get("overall", {})
        ece = _fmt(cal.get("wagering_ece"))
        wag_rho = _fmt(cal.get("wagering_spearman"))
        skip_align = _fmt(cal.get("skip_error_alignment"))
        mci_raw = _fmt(mci.get(model, {}).get("mci_raw"))
        mci_adj = _fmt(mci.get(model, {}).get("mci_difficulty_adjusted"))
        hall_rate = _fmt(
            hallucination.get("per_model", {}).get(model, {}).get("hallucination_rate")
        )
        rows.append(
            f"        {label} & {ece} & {wag_rho} & {skip_align}"
            f" & {mci_raw} & {mci_adj} & {hall_rate}" + r" \\"
        )

    body = "\n".join(rows)
    tex = rf"""% Table 2: Metacognitive calibration summary across models
\begin{{table}}[ht]
\centering
\caption{{Metacognitive calibration summary. ECE: Expected Calibration Error
  (lower is better). Wager-Acc~$\rho$: Spearman correlation between bet size
  and accuracy (higher is better). Skip Align: proportion of skipped questions
  where the model would have erred. MCI: Metacognitive Convergence Index.
  Hall. Rate: proportion of high-confidence ($\geq$7) answers that are wrong.}}
\label{{tab:calibration}}
\begin{{tabular}}{{lcccccc}}
\toprule
Model & ECE$\downarrow$ & Wager-Acc $\rho\uparrow$ & Skip Align$\uparrow$ & MCI Raw$\uparrow$ & MCI Adj$\uparrow$ & Hall. Rate$\downarrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write_table(out_path, tex)


# ---------------------------------------------------------------------------
# Table 3: MCI Dissociation Matrix (top 3 models)
# ---------------------------------------------------------------------------

CHANNEL_LABELS = {
    "ch1": "Wager",
    "ch2": "Opt-Out",
    "ch3": "Diff-Sel",
    "ch4": "Tool-Use",
    "ch5": "Natural",
}

CHANNEL_ORDER = ["ch1", "ch2", "ch3", "ch4", "ch5"]


def export_table3_dissociation(dissociation: dict, mci: dict, out_path: Path) -> None:
    # Pick top 3 models: best MCI, worst MCI, most dissociated
    mci_vals = [(m, v.get("mci_raw") or 0.0) for m, v in mci.items()]
    mci_vals.sort(key=lambda t: t[1])

    selected: list[str] = []
    if mci_vals:
        selected.append(mci_vals[-1][0])   # best MCI
    if len(mci_vals) > 1:
        selected.append(mci_vals[0][0])    # worst MCI
    # Most dissociated = highest number of dissociated pairs
    n_diss = [(m, len(v.get("dissociated_pairs", []))) for m, v in dissociation.items()]
    n_diss.sort(key=lambda t: -t[1])
    for m, _ in n_diss:
        if m not in selected:
            selected.append(m)
            break

    tables = []
    for model in selected:
        label = _model_label(model)
        mat = dissociation.get(model, {}).get("matrix", {})
        col_spec = "l" + "c" * len(CHANNEL_ORDER)
        col_headers = " & ".join(CHANNEL_LABELS[c] for c in CHANNEL_ORDER)
        rows = []
        for row_ch in CHANNEL_ORDER:
            row_label = CHANNEL_LABELS[row_ch]
            cells = []
            for col_ch in CHANNEL_ORDER:
                v = mat.get(row_ch, {}).get(col_ch)
                if row_ch == col_ch:
                    cells.append("1.00")
                elif v is None:
                    cells.append("—")
                else:
                    cell = _fmt(v)
                    if abs(v) < 0.1:
                        cell = r"\textbf{" + cell + "}"   # bold dissociated pairs
                    cells.append(cell)
            rows.append(f"        {row_label} & " + " & ".join(cells) + r" \\")
        body = "\n".join(rows)
        subtable = rf"""
\paragraph{{{label}}}
\begin{{tabular}}{{{col_spec}}}
\toprule
 & {col_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""
        tables.append(subtable)

    all_tables = "\n\n".join(tables)
    tex = rf"""% Table 3: MCI dissociation matrices for selected models
\begin{{table}}[ht]
\centering
\caption{{Channel dissociation matrices (Spearman~$\rho$ between channel confidence
  signals) for the best-calibrated model, worst-calibrated model, and most
  dissociated model. Bold entries indicate $|\rho| < 0.1$ (dissociated channel
  pair).}}
\label{{tab:dissociation}}
{all_tables}
\end{{table}}
"""
    _write_table(out_path, tex)


# ---------------------------------------------------------------------------
# Table 4: Layer 1 vs Layer 2 comparison
# ---------------------------------------------------------------------------

def export_table4_layer_comparison(layer_comparison: dict, out_path: Path) -> None:
    models = sorted(layer_comparison.keys())
    rows = []
    for model in models:
        label = _model_label(model)
        lc = layer_comparison.get(model, {})
        l1_rho = _fmt(lc.get("l1_vs_accuracy_spearman"))
        l2_rho = _fmt(lc.get("l2_vs_accuracy_spearman"))
        l2_l1_rho = _fmt(lc.get("l2_vs_l1_spearman"))
        l2_better = lc.get("l2_better_calibrated")
        better = "L2" if l2_better else "L1" if l2_better is False else "—"
        rows.append(
            f"        {label} & {l1_rho} & {l2_rho} & {l2_l1_rho} & {better}" + r" \\"
        )

    body = "\n".join(rows)
    tex = rf"""% Table 4: Layer 1 (behavioral) vs Layer 2 (verbal) calibration comparison
\begin{{table}}[ht]
\centering
\caption{{Layer 1 (behavioral wagering) vs Layer 2 (verbal self-report) calibration.
  L1-Acc~$\rho$: Spearman correlation between wagering bet and accuracy.
  L2-Acc~$\rho$: correlation between stated confidence and accuracy.
  L2--L1~$\rho$: correlation between L2 confidence and L1 bet.
  Better: which layer is better calibrated with actual accuracy.}}
\label{{tab:layers}}
\begin{{tabular}}{{lcccc}}
\toprule
Model & L1-Acc $\rho\uparrow$ & L2-Acc $\rho\uparrow$ & L2--L1 $\rho$ & Better \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write_table(out_path, tex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX tables for Experiment 1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="data/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / args.results_dir
    tables_dir = repo_root / "paper" / "tables"
    run_id = args.run_id

    def load(suffix: str) -> dict:
        p = results_dir / f"exp1_{run_id}_{suffix}.json"
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping dependent tables.")
            return {}
        return _load_json(p)

    print(f"\nExporting LaTeX tables for run {run_id} …")
    accuracy = load("accuracy")
    calibration = load("calibration")
    mci = load("mci")
    dissociation = load("dissociation")
    layer_comparison = load("layer_comparison")
    hallucination = load("hallucination")

    export_table1_accuracy(accuracy, tables_dir / "table1_accuracy.tex")
    export_table2_calibration(calibration, mci, hallucination,
                              tables_dir / "table2_calibration.tex")
    export_table3_dissociation(dissociation, mci, tables_dir / "table3_dissociation.tex")
    export_table4_layer_comparison(layer_comparison, tables_dir / "table4_layers.tex")

    print(f"\nAll tables written to {tables_dir}/")


if __name__ == "__main__":
    main()
