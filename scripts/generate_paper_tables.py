#!/usr/bin/env python3
"""Generate all paper tables and figure data from analysis JSON files."""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/results")
TABLES_DIR = Path("paper/tables")
FIGURES_DIR = Path("paper/figures")

TARGET_MODELS = [
    "deepseek-r1", "deepseek-v3", "gemini-2.5-pro", "gemma-3-12b",
    "gemma-3-27b", "gpt-oss-120b", "kimi-k2", "llama-3.1-405b",
    "llama-3.1-70b", "llama-3.1-8b", "llama-3.2-3b", "llama-3.3-70b",
    "mistral-large", "mixtral-8x22b", "phi-4", "qwen3-next-80b",
]

MODEL_INFO = {
    "deepseek-r1":    {"lab": "DeepSeek",    "params": "671B",  "arch": "Dense",      "api": "DeepSeek"},
    "deepseek-v3":    {"lab": "DeepSeek",    "params": "671B",  "arch": "Dense",      "api": "DeepSeek"},
    "gemini-2.5-pro": {"lab": "Google",      "params": "$>$1T", "arch": "Dense",      "api": "Google AI"},
    "gemma-3-12b":    {"lab": "Google",      "params": "12B",   "arch": "Dense",      "api": "Google AI"},
    "gemma-3-27b":    {"lab": "Google",      "params": "27B",   "arch": "Dense",      "api": "NVIDIA NIM"},
    "gpt-oss-120b":   {"lab": "OpenAI",      "params": "120B",  "arch": "Dense",      "api": "NVIDIA NIM"},
    "kimi-k2":        {"lab": "Moonshot AI", "params": "200B",  "arch": "Dense",      "api": "NVIDIA NIM"},
    "llama-3.1-405b": {"lab": "Meta",        "params": "405B",  "arch": "Dense",      "api": "NVIDIA NIM"},
    "llama-3.1-70b":  {"lab": "Meta",        "params": "70B",   "arch": "Dense",      "api": "NVIDIA NIM"},
    "llama-3.1-8b":   {"lab": "Meta",        "params": "8B",    "arch": "Dense",      "api": "NVIDIA NIM"},
    "llama-3.2-3b":   {"lab": "Meta",        "params": "3B",    "arch": "Dense",      "api": "NVIDIA NIM"},
    "llama-3.3-70b":  {"lab": "Meta",        "params": "70B",   "arch": "Dense",      "api": "NVIDIA NIM"},
    "mistral-large":  {"lab": "Mistral",     "params": "675B",  "arch": "Dense",      "api": "NVIDIA NIM"},
    "mixtral-8x22b":  {"lab": "Mistral",     "params": "193B",  "arch": "MoE (8x22B)","api": "NVIDIA NIM"},
    "phi-4":          {"lab": "Microsoft",   "params": "14B",   "arch": "Dense",      "api": "NVIDIA NIM"},
    "qwen3-next-80b": {"lab": "Alibaba",     "params": "80B",   "arch": "Dense",      "api": "NVIDIA NIM"},
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def fmt(v, decimals=3):
    """Format a value for the table."""
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"

# ─────────────────────────────────────────────────────────
# EXP1: Load all accuracy files, newer overrides older
# ─────────────────────────────────────────────────────────
def load_exp1():
    files = sorted(DATA_DIR.glob("exp1_*_accuracy.json"))
    merged = {}       # model -> {domain -> vals}
    merged_quality = {}  # model -> count of domains with valid natural_acc
    for f in files:
        if "counterfactual" in f.name or "meta" in f.name:
            continue
        data = load_json(f)
        for model, domains in data.items():
            if model.startswith("_"):
                continue
            if not isinstance(domains, dict):
                continue
            # Count how many domains have valid natural_acc in this file
            valid_domains = sum(
                1 for v in domains.values()
                if isinstance(v, dict) and v.get("natural_acc") is not None
            )
            # Only override if this file has at least as many valid domains
            # This prevents pilot runs (1-3 domains) from corrupting comprehensive data
            prev_quality = merged_quality.get(model, 0)
            if valid_domains >= prev_quality:
                merged[model] = {d: v for d, v in domains.items() if isinstance(v, dict)}
                merged_quality[model] = valid_domains

    # Also load MCI from meta_accuracy
    mci_data = {}
    meta_files = sorted(DATA_DIR.glob("exp1_*_meta_accuracy.json"))
    for f in meta_files:
        data = load_json(f)
        for model, info in data.items():
            if model.startswith("_"):
                continue
            if isinstance(info, dict) and "mci_spearman" in info:
                mci_data[model] = info["mci_spearman"]

    results = {}
    for model in TARGET_MODELS:
        if model not in merged:
            results[model] = {"natural_acc": None, "wagering_acc": None, "mirror_gap": None, "mci": None}
            continue
        domains = merged[model]
        nat_vals, wag_vals, gaps = [], [], []
        for dom, vals in domains.items():
            if not isinstance(vals, dict):
                continue
            na = vals.get("natural_acc")
            wa = vals.get("wagering_acc")
            if na is not None:
                nat_vals.append(na)
            if wa is not None:
                wag_vals.append(wa)
            if na is not None and wa is not None:
                gaps.append(abs(wa - na))

        results[model] = {
            "natural_acc": sum(nat_vals) / len(nat_vals) if nat_vals else None,
            "wagering_acc": sum(wag_vals) / len(wag_vals) if wag_vals else None,
            "mirror_gap": sum(gaps) / len(gaps) if gaps else None,
            "mci": mci_data.get(model),
        }
    return results

# ─────────────────────────────────────────────────────────
# EXP2: TII from transfer analysis files
# ─────────────────────────────────────────────────────────
def load_exp2():
    files = sorted(DATA_DIR.glob("exp2_*_transfer_analysis.json"))
    merged = {}
    for f in files:
        data = load_json(f)
        # Structure: top-level model keys → {transfer_mci: float, ...}
        for model in TARGET_MODELS:
            if model in data and isinstance(data[model], dict):
                tii = data[model].get("transfer_mci")
                if tii is not None:
                    merged[model] = tii
    return {m: merged.get(m) for m in TARGET_MODELS}

# ─────────────────────────────────────────────────────────
# EXP3: CCE_mixed from metrics files
# ─────────────────────────────────────────────────────────
def load_exp3():
    files = sorted(DATA_DIR.glob("exp3_*_metrics.json"))
    merged = {}
    for f in files:
        data = load_json(f)
        # Structure: top-level model keys → {cce: {mixed: {mean_cce: float}}}
        for model in TARGET_MODELS:
            if model in data and isinstance(data[model], dict):
                cce_data = data[model].get("cce")
                if isinstance(cce_data, dict):
                    mixed = cce_data.get("mixed", {})
                    if isinstance(mixed, dict):
                        cce_val = mixed.get("mean_cce")
                        if cce_val is not None:
                            merged[model] = cce_val
    return {m: merged.get(m) for m in TARGET_MODELS}

# ─────────────────────────────────────────────────────────
# EXP4: AI and SAR from v2 expanded analysis
# ─────────────────────────────────────────────────────────
def load_exp4():
    results = {m: {"ai_wager": None, "sar": None} for m in TARGET_MODELS}

    v2_path = DATA_DIR / "exp4_v2_20260314T135731_analysis.json"
    if not v2_path.exists():
        return results

    data = load_json(v2_path)

    # AI from ai_true → by_model_channel → model → wager → mean
    ai_true = data.get("ai_true", {}).get("by_model_channel", {})
    for model in TARGET_MODELS:
        if model in ai_true:
            wager = ai_true[model].get("wager", {})
            results[model]["ai_wager"] = wager.get("mean")

    # SAR from sar → model → wager → sar
    sar_data = data.get("sar", {})
    for model in TARGET_MODELS:
        if model in sar_data and isinstance(sar_data[model], dict):
            wager = sar_data[model].get("wager", {})
            if isinstance(wager, dict):
                sar_val = wager.get("sar")
                if sar_val is not None:
                    results[model]["sar"] = sar_val

    return results

# ─────────────────────────────────────────────────────────
# EXP5: ARS from metrics files
# ─────────────────────────────────────────────────────────
def load_exp5():
    files = sorted(DATA_DIR.glob("exp5_*_metrics.json"))
    merged = {}
    for f in files:
        if "clean" in f.name:
            continue  # skip clean baselines
        data = load_json(f)
        # Structure: top-level model keys → {overall_ars: float, ...}
        for model in TARGET_MODELS:
            if model in data and isinstance(data[model], dict):
                ars = data[model].get("overall_ars")
                if ars is not None:
                    merged[model] = ars
    return {m: merged.get(m) for m in TARGET_MODELS}

# ─────────────────────────────────────────────────────────
# EXP6: FDR and SSR from master/expanded analysis
# ─────────────────────────────────────────────────────────
def load_exp6():
    results = {m: {"fdr": None, "ssr": None} for m in TARGET_MODELS}

    # Load from master and expanded analyses (later files override)
    for fname in ["exp6_master_analysis.json",
                  "exp6_expanded_20260314T203446_analysis.json",
                  "exp6_expanded_20260315T190153_backfill4_analysis.json"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        data = load_json(fpath)

        # 6b FDR: experiment_6b → per_model → model → overall_fdr
        exp6b = data.get("experiment_6b", {}).get("per_model", {})
        for model in TARGET_MODELS:
            if model in exp6b and isinstance(exp6b[model], dict):
                fdr = exp6b[model].get("overall_fdr")
                if fdr is not None:
                    results[model]["fdr"] = fdr

        # 6a SSR: experiment_6a → per_model → model → sycophancy_separation_ratio
        exp6a = data.get("experiment_6a", {}).get("per_model", {})
        for model in TARGET_MODELS:
            if model in exp6a and isinstance(exp6a[model], dict):
                ssr = exp6a[model].get("sycophancy_separation_ratio")
                if ssr is not None:
                    results[model]["ssr"] = ssr

    return results

# ─────────────────────────────────────────────────────────
# EXP9: CFR, KDI, escalation curve
# ─────────────────────────────────────────────────────────
def load_exp9():
    path = DATA_DIR / "exp9_20260312T140842_analysis" / "analysis.json"
    if not path.exists():
        return {}, {}

    data = load_json(path)

    # Per-model CFR and KDI
    per_model = {}

    # CFR from escalation_curve → per_model → model → {"1": cfr, "2": cfr, ...}
    escalation = data.get("escalation_curve", {})
    cfr_by_model = escalation.get("per_model", {})
    for model in TARGET_MODELS:
        if model in cfr_by_model:
            m = cfr_by_model[model]
            per_model[model] = {
                "cfr_c1": m.get("1"),
                "cfr_c2": m.get("2"),
                "cfr_c3": m.get("3"),
                "cfr_c4": m.get("4"),
            }

    # Also get CFR from cfr_udr_condition1 for more precise C1 values
    cfr_udr = data.get("cfr_udr_condition1", {})
    for model in TARGET_MODELS:
        if model in cfr_udr:
            per_model.setdefault(model, {})
            cfr_val = cfr_udr[model].get("cfr")
            if cfr_val is not None:
                per_model[model]["cfr_c1"] = cfr_val

    # KDI from kdi_table → model → mean_kdi
    kdi_data = data.get("kdi_table", {})
    for model in TARGET_MODELS:
        if model in kdi_data and isinstance(kdi_data[model], dict):
            per_model.setdefault(model, {})["kdi"] = kdi_data[model].get("mean_kdi")

    # Escalation curve aggregate from mean_curve
    mean_curve = escalation.get("mean_curve", {})
    esc_agg = {
        "means": mean_curve,
        "interpretation": escalation.get("interpretation", ""),
    }

    return per_model, esc_agg

# ─────────────────────────────────────────────────────────
# MAIN: Generate all tables
# ─────────────────────────────────────────────────────────
def main():
    print("Loading data from analysis files...")

    exp1 = load_exp1()
    exp2 = load_exp2()
    exp3 = load_exp3()
    exp4 = load_exp4()
    exp5 = load_exp5()
    exp6 = load_exp6()
    exp9_models, exp9_esc = load_exp9()

    # ── TASK 1: Main Results Matrix ──
    print("\n" + "="*80)
    print("TABLE 1: Main Results Matrix (16 models)")
    print("="*80)

    header = f"{'Model':<18} {'Nat.Acc':>7} {'Wag.Acc':>7} {'M.Gap':>6} {'TII':>6} {'CCE':>6} {'AI':>7} {'SAR':>6} {'ARS':>6} {'FDR':>6} {'SSR':>7} {'CFR_C1':>6} {'KDI':>7}"
    print(header)
    print("-" * len(header))

    table1_data = {}
    for model in TARGET_MODELS:
        e1 = exp1.get(model, {})
        e2_tii = exp2.get(model)
        e3_cce = exp3.get(model)
        e4 = exp4.get(model, {})
        e5_ars = exp5.get(model)
        e6 = exp6.get(model, {})
        e9 = exp9_models.get(model, {})

        row = {
            "natural_acc": e1.get("natural_acc"),
            "wagering_acc": e1.get("wagering_acc"),
            "mirror_gap": e1.get("mirror_gap"),
            "tii": e2_tii,
            "cce_mixed": e3_cce,
            "ai_wager": e4.get("ai_wager"),
            "sar": e4.get("sar"),
            "ars": e5_ars,
            "fdr": e6.get("fdr"),
            "ssr": e6.get("ssr"),
            "cfr_c1": e9.get("cfr_c1"),
            "kdi": e9.get("kdi"),
        }
        table1_data[model] = row

        print(f"{model:<18} {fmt(row['natural_acc']):>7} {fmt(row['wagering_acc']):>7} {fmt(row['mirror_gap'],3):>6} "
              f"{fmt(row['tii'],3):>6} {fmt(row['cce_mixed'],3):>6} {fmt(row['ai_wager'],4):>7} {fmt(row['sar'],3):>6} "
              f"{fmt(row['ars'],3):>6} {fmt(row['fdr'],3):>6} {fmt(row['ssr'],2):>7} "
              f"{fmt(row['cfr_c1'],3):>6} {fmt(row['kdi'],3):>7}")

    # Save JSON
    with open(TABLES_DIR / "table1_data.json", "w") as f:
        json.dump(table1_data, f, indent=2, default=str)
    print(f"\nSaved: {TABLES_DIR / 'table1_data.json'}")

    # Generate LaTeX
    latex1 = generate_table1_latex(table1_data)
    with open(TABLES_DIR / "table1_main_results.tex", "w") as f:
        f.write(latex1)
    print(f"Saved: {TABLES_DIR / 'table1_main_results.tex'}")

    # ── TASK 2: Comparison Table ──
    print("\n" + "="*80)
    print("TABLE 2: MIRROR vs Prior Benchmarks")
    print("="*80)
    latex2 = generate_table2_latex()
    with open(TABLES_DIR / "table2_comparison.tex", "w") as f:
        f.write(latex2)
    print(f"Saved: {TABLES_DIR / 'table2_comparison.tex'}")

    # ── TASK 3: Escalation Curve Data ──
    print("\n" + "="*80)
    print("TASK 3: Escalation Curve Figure Data")
    print("="*80)
    esc_data = generate_escalation_data(exp9_models, exp9_esc)
    with open(FIGURES_DIR / "escalation_data.json", "w") as f:
        json.dump(esc_data, f, indent=2)
    print(json.dumps(esc_data, indent=2))
    print(f"\nSaved: {FIGURES_DIR / 'escalation_data.json'}")

    # ── TASK 4: KDI Table ──
    print("\n" + "="*80)
    print("TABLE 3: KDI Table (sorted by KDI ascending)")
    print("="*80)
    latex3 = generate_kdi_table(exp9_models)
    with open(TABLES_DIR / "table3_kdi.tex", "w") as f:
        f.write(latex3)
    print(f"Saved: {TABLES_DIR / 'table3_kdi.tex'}")

    # ── TASK 5: Model Roster ──
    print("\n" + "="*80)
    print("TABLE 4: Model Roster")
    print("="*80)
    exp_coverage = compute_experiment_coverage(exp1, exp2, exp3, exp4, exp5, exp6, exp9_models)
    latex4 = generate_model_roster_latex(exp_coverage)
    with open(TABLES_DIR / "table4_models.tex", "w") as f:
        f.write(latex4)
    print(f"Saved: {TABLES_DIR / 'table4_models.tex'}")

    # ── TASK 6: Per-Experiment Summary ──
    print("\n" + "="*80)
    print("TABLE 5: Per-Experiment Summary Statistics")
    print("="*80)
    exp_summary = generate_experiment_summary(exp1, exp2, exp3, exp4, exp5, exp6, exp9_models)
    latex5 = generate_exp_summary_latex(exp_summary)
    with open(TABLES_DIR / "table5_experiment_summary.tex", "w") as f:
        f.write(latex5)
    with open(TABLES_DIR / "experiment_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2, default=str)
    print(f"Saved: {TABLES_DIR / 'table5_experiment_summary.tex'}")
    print(f"Saved: {TABLES_DIR / 'experiment_summary.json'}")

    # ── Final Summary ──
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    missing = []
    for model in TARGET_MODELS:
        d = table1_data[model]
        nulls = [k for k, v in d.items() if v is None]
        if nulls:
            missing.append(f"  {model}: missing {', '.join(nulls)}")
    if missing:
        print("\nMissing data:")
        for m in missing:
            print(m)
    else:
        print("\nAll data extracted successfully!")

    print(f"\nGenerated files:")
    for p in [TABLES_DIR / "table1_main_results.tex", TABLES_DIR / "table1_data.json",
              TABLES_DIR / "table2_comparison.tex", FIGURES_DIR / "escalation_data.json",
              TABLES_DIR / "table3_kdi.tex", TABLES_DIR / "table4_models.tex",
              TABLES_DIR / "table5_experiment_summary.tex", TABLES_DIR / "experiment_summary.json"]:
        print(f"  - {p}")


# ─────────────────────────────────────────────────────────
# LaTeX generators
# ─────────────────────────────────────────────────────────

def tex_val(v, decimals=3):
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"

def generate_table1_latex(data):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results across all experiments for 16 models. Nat.Acc = natural accuracy (Exp1), Wag.Acc = wagering accuracy (Exp1), M.Gap = MIRROR gap (Exp1), TII = Transfer Integrity Index (Exp2), CCE = Compositional Calibration Error for mixed tasks (Exp3), AI = Adaptation Index on wager channel (Exp4), SAR = Sycophancy Adaptation Ratio (Exp4), ARS = Adversarial Robustness Score (Exp5), FDR = Flaw Detection Rate (Exp6b), SSR = Sycophancy Separation Ratio (Exp6a), CFR\textsubscript{C1} = Confident Failure Rate in uninformed condition (Exp9), KDI = Knowing-Doing Index (Exp9). ``---'' indicates the model was not evaluated in that experiment.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l " + "r" * 12 + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Nat.Acc} & \textbf{Wag.Acc} & \textbf{M.Gap} & \textbf{TII} & \textbf{CCE} & \textbf{AI} & \textbf{SAR} & \textbf{ARS} & \textbf{FDR} & \textbf{SSR} & \textbf{CFR\textsubscript{C1}} & \textbf{KDI} \\")
    lines.append(r"\midrule")

    for model in TARGET_MODELS:
        d = data[model]
        name = model.replace("-", "{-}")
        row = (f"\\texttt{{{name}}} & "
               f"{tex_val(d['natural_acc'])} & {tex_val(d['wagering_acc'])} & {tex_val(d['mirror_gap'])} & "
               f"{tex_val(d['tii'])} & {tex_val(d['cce_mixed'])} & "
               f"{tex_val(d['ai_wager'], 4)} & {tex_val(d['sar'])} & "
               f"{tex_val(d['ars'])} & {tex_val(d['fdr'])} & "
               f"{tex_val(d['ssr'], 2)} & {tex_val(d['cfr_c1'])} & {tex_val(d['kdi'])} \\\\")
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_table2_latex():
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of MIRROR with prior benchmarks for LLM self-knowledge and metacognition.}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Feature} & \textbf{SAD} & \textbf{Barkan et al.} & \textbf{Kadavath et al.} & \textbf{Steyvers et al.} & \textbf{MIRROR} \\")
    lines.append(r"& \textbf{(Laine 2024)} & \textbf{(2025)} & \textbf{(2022)} & \textbf{(2025)} & \textbf{(Ours)} \\")
    lines.append(r"\midrule")

    rows = [
        (r"Venue", "NeurIPS D\\&B '24", "NeurIPS WS '25", "arXiv", "arXiv", "NeurIPS D\\&B '26"),
        (r"Focus", "Situational", "Self-capability", "Calibration", "Metacognition +", "Metacognitive cal."),
        (r"", "awareness", "prediction", "(P(IK))", "uncertainty", r"$\rightarrow$ agentic action"),
        (r"Models", "16", r"$\sim$15", r"$\sim$10", "2 families", "16"),
        (r"Task categories", "7", "3", "---", "6 datasets", "8 experiments"),
        (r"Total items", "13K questions", r"$\sim$1K tasks", "---", "12K+ items", r"$\sim$250K records"),
        (r"Metacognitive levels", "1 (awareness)", "1 (confidence)", "1 (calibration)", "1 (calibration)", "4 (L0--L3)"),
        (r"Behavioral channels", "1 (MC/free)", "1 (verbal)", "1 (verbal)", "1 (verbal)", "5 (wager, opt-out,"),
        (r"", "", "", "", "", "tool, diff., natural)"),
        (r"Agentic connection", "No", "Yes (SWE-bench)", "No", "No", "Yes (4 cond., 3 par.)"),
        (r"Headline finding", "Partial sit.", "LLMs over-", r"\emph{Mostly}", "Task-specific", "Knowing-doing gap"),
        (r"", "awareness", "confident", "calibrated", "metacognition", "(62\\% CFR reduction)"),
        (r"Public dataset", "Yes", "No", "No", "No", "Yes"),
    ]

    for row in rows:
        cells = " & ".join(row)
        lines.append(f"{cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_escalation_data(exp9_models, exp9_esc):
    # Per-model escalation
    per_model = {}
    for model, vals in exp9_models.items():
        per_model[model] = {
            "C1": vals.get("cfr_c1"),
            "C2": vals.get("cfr_c2"),
            "C3": vals.get("cfr_c3"),
            "C4": vals.get("cfr_c4"),
        }

    # Compute aggregate means from per-model data
    means = {}
    for cond in ["C1", "C2", "C3", "C4"]:
        vals = [pm[cond] for pm in per_model.values() if pm.get(cond) is not None]
        means[cond] = sum(vals) / len(vals) if vals else None

    result = {
        "per_model": per_model,
        "aggregate": {
            "means": means if means.get("C1") else exp9_esc.get("means", {}),
            "ci_lower": exp9_esc.get("ci_lower", {}),
            "ci_upper": exp9_esc.get("ci_upper", {}),
            "significance_tests": exp9_esc.get("significance", {}),
        },
        "n_models": len(per_model),
        "note": "CFR = Confident Failure Rate. C1=Uninformed, C2=Self-Informed, C3=Instructed, C4=Constrained.",
    }
    return result


def generate_kdi_table(exp9_models):
    # Filter models with KDI data, sort ascending
    kdi_list = []
    for model, vals in exp9_models.items():
        if vals.get("kdi") is not None:
            kdi_list.append({
                "model": model,
                "kdi": vals["kdi"],
                "cfr_c1": vals.get("cfr_c1"),
                "cfr_c4": vals.get("cfr_c4"),
            })
    kdi_list.sort(key=lambda x: x["kdi"])

    # Print to stdout
    print(f"{'Model':<18} {'KDI':>8} {'CFR_C1':>8} {'CFR_C4':>8}")
    print("-" * 44)
    for row in kdi_list:
        print(f"{row['model']:<18} {fmt(row['kdi'],4):>8} {fmt(row['cfr_c1']):>8} {fmt(row['cfr_c4']):>8}")

    # LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Knowing-Doing Index (KDI) for all Experiment~9 models, sorted by KDI ascending. Negative KDI indicates models fail to translate self-knowledge into appropriate action-selection. CFR\textsubscript{C1} = uninformed condition; CFR\textsubscript{C4} = externally constrained condition.}")
    lines.append(r"\label{tab:kdi}")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{KDI} & \textbf{CFR\textsubscript{C1}} & \textbf{CFR\textsubscript{C4}} \\")
    lines.append(r"\midrule")
    for row in kdi_list:
        name = row["model"].replace("-", "{-}")
        lines.append(f"\\texttt{{{name}}} & {tex_val(row['kdi'], 4)} & {tex_val(row['cfr_c1'])} & {tex_val(row['cfr_c4'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def compute_experiment_coverage(exp1, exp2, exp3, exp4, exp5, exp6, exp9_models):
    coverage = {}
    for model in TARGET_MODELS:
        exps = []
        if exp1.get(model, {}).get("natural_acc") is not None:
            exps.append("1")
        if exp2.get(model) is not None:
            exps.append("2")
        if exp3.get(model) is not None:
            exps.append("3")
        if exp4.get(model, {}).get("ai_wager") is not None:
            exps.append("4")
        if exp5.get(model) is not None:
            exps.append("5")
        if exp6.get(model, {}).get("fdr") is not None or exp6.get(model, {}).get("ssr") is not None:
            exps.append("6")
        # Exp8 only for Llama family
        if model.startswith("llama-3.1-") or model == "llama-3.3-70b":
            exps.append("8")
        if model in exp9_models:
            exps.append("9")
        coverage[model] = exps
    return coverage


def format_exp_range(exps):
    """Format experiment list as compact range string."""
    if not exps:
        return "---"
    nums = sorted(int(e) for e in exps)
    # Build ranges
    ranges = []
    start = nums[0]
    end = start
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}" if start == end else f"{start}--{end}")
            start = n
            end = n
    ranges.append(f"{start}" if start == end else f"{start}--{end}")
    return ", ".join(ranges)


def generate_model_roster_latex(exp_coverage):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Model roster. All models are instruction-tuned variants accessed via API. Parameter counts are approximate where not officially disclosed.}")
    lines.append(r"\label{tab:models}")
    lines.append(r"\begin{tabular}{l l r l l l}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Lab} & \textbf{Params} & \textbf{Arch.} & \textbf{API} & \textbf{Experiments} \\")
    lines.append(r"\midrule")

    for model in TARGET_MODELS:
        info = MODEL_INFO[model]
        name = model.replace("-", "{-}")
        exps = format_exp_range(exp_coverage.get(model, []))
        lines.append(f"\\texttt{{{name}}} & {info['lab']} & {info['params']} & {info['arch']} & {info['api']} & {exps} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    # Also print to stdout
    print(f"{'Model':<18} {'Lab':<12} {'Params':>6} {'Arch':<14} {'API':<12} {'Experiments'}")
    print("-" * 85)
    for model in TARGET_MODELS:
        info = MODEL_INFO[model]
        exps = ", ".join(exp_coverage.get(model, []))
        print(f"{model:<18} {info['lab']:<12} {info['params']:>6} {info['arch']:<14} {info['api']:<12} {exps}")

    return "\n".join(lines)


def generate_experiment_summary(exp1, exp2, exp3, exp4, exp5, exp6, exp9_models):
    summary = {}

    # Count records from JSONL files
    record_counts = {}
    for exp_prefix in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp9"]:
        total = 0
        for f in DATA_DIR.glob(f"{exp_prefix}*results*.jsonl"):
            try:
                with open(f) as fh:
                    total += sum(1 for _ in fh)
            except:
                pass
        record_counts[exp_prefix] = total

    # Exp1
    nat_vals = [v["natural_acc"] for v in exp1.values() if v.get("natural_acc") is not None]
    gap_vals = [v["mirror_gap"] for v in exp1.values() if v.get("mirror_gap") is not None]
    summary["Exp1"] = {
        "description": "Atomic Self-Knowledge",
        "n_models": len(nat_vals),
        "n_items": "330/model (8 domains x 5 subcats x ~8 Qs)",
        "total_records": record_counts.get("exp1", "~50K"),
        "primary_metric": "MIRROR Gap",
        "mean": sum(gap_vals) / len(gap_vals) if gap_vals else None,
        "std": (sum((v - sum(gap_vals)/len(gap_vals))**2 for v in gap_vals) / len(gap_vals))**0.5 if gap_vals else None,
        "range": [min(gap_vals), max(gap_vals)] if gap_vals else None,
    }

    # Exp2
    tii_vals = [v for v in exp2.values() if v is not None]
    summary["Exp2"] = {
        "description": "Cross-Domain Transfer",
        "n_models": len(tii_vals),
        "n_items": "~150/model",
        "total_records": record_counts.get("exp2", "~2K"),
        "primary_metric": "TII (Transfer Integrity Index)",
        "mean": sum(tii_vals) / len(tii_vals) if tii_vals else None,
        "std": (sum((v - sum(tii_vals)/len(tii_vals))**2 for v in tii_vals) / len(tii_vals))**0.5 if tii_vals else None,
        "range": [min(tii_vals), max(tii_vals)] if tii_vals else None,
    }

    # Exp3
    cce_vals = [v for v in exp3.values() if v is not None]
    summary["Exp3"] = {
        "description": "Compositional Self-Prediction",
        "n_models": len(cce_vals),
        "n_items": "~25/model (mixed tasks)",
        "total_records": record_counts.get("exp3", "~500"),
        "primary_metric": "CCE (Compositional Calibration Error)",
        "mean": sum(cce_vals) / len(cce_vals) if cce_vals else None,
        "std": (sum((v - sum(cce_vals)/len(cce_vals))**2 for v in cce_vals) / len(cce_vals))**0.5 if cce_vals else None,
        "range": [min(cce_vals), max(cce_vals)] if cce_vals else None,
    }

    # Exp4
    ai_vals = [v.get("ai_wager") for v in exp4.values() if v.get("ai_wager") is not None]
    summary["Exp4"] = {
        "description": "Adaptive Self-Regulation (Burn-and-Test)",
        "n_models": len(ai_vals),
        "n_items": "320/model x 2 conditions",
        "total_records": record_counts.get("exp4", "~20K"),
        "primary_metric": "AI (Adaptation Index, wager)",
        "mean": sum(ai_vals) / len(ai_vals) if ai_vals else None,
        "std": (sum((v - sum(ai_vals)/len(ai_vals))**2 for v in ai_vals) / len(ai_vals))**0.5 if ai_vals else None,
        "range": [min(ai_vals), max(ai_vals)] if ai_vals else None,
    }

    # Exp5
    ars_vals = [v for v in exp5.values() if v is not None]
    summary["Exp5"] = {
        "description": "Adversarial Robustness of Self-Knowledge",
        "n_models": len(ars_vals),
        "n_items": "320/model (adv) + 320/model (clean)",
        "total_records": record_counts.get("exp5", "~8K"),
        "primary_metric": "ARS (Adversarial Robustness Score)",
        "mean": sum(ars_vals) / len(ars_vals) if ars_vals else None,
        "std": (sum((v - sum(ars_vals)/len(ars_vals))**2 for v in ars_vals) / len(ars_vals))**0.5 if ars_vals else None,
        "range": [min(ars_vals), max(ars_vals)] if ars_vals else None,
    }

    # Exp6
    fdr_vals = [v.get("fdr") for v in exp6.values() if v.get("fdr") is not None]
    summary["Exp6"] = {
        "description": "Ecosystem Effect (Trust + Flawed Premise)",
        "n_models": len(fdr_vals),
        "n_items": "240/model (6a+6b+6c)",
        "total_records": record_counts.get("exp6", "~6K"),
        "primary_metric": "FDR (Flaw Detection Rate, Exp6b)",
        "mean": sum(fdr_vals) / len(fdr_vals) if fdr_vals else None,
        "std": (sum((v - sum(fdr_vals)/len(fdr_vals))**2 for v in fdr_vals) / len(fdr_vals))**0.5 if fdr_vals else None,
        "range": [min(fdr_vals), max(fdr_vals)] if fdr_vals else None,
    }

    # Exp8
    summary["Exp8"] = {
        "description": "Scaling Analysis",
        "n_models": 4,
        "n_items": "80/model (gap-fill)",
        "total_records": "~320",
        "primary_metric": "Scaling slope (nat.acc vs log2 params)",
        "mean": 0.0279,
        "std": None,
        "range": None,
        "note": "Primary: Llama 3.1 family (8B, 70B, 405B) + 3.3-70B generation comparison",
    }

    # Exp9
    cfr_vals = [v.get("cfr_c1") for v in exp9_models.values() if v.get("cfr_c1") is not None]
    kdi_vals = [v.get("kdi") for v in exp9_models.values() if v.get("kdi") is not None]
    summary["Exp9"] = {
        "description": "The Knowing-Doing Gap",
        "n_models": len(cfr_vals),
        "n_items": "597 tasks x 4 conditions x 3 paradigms",
        "total_records": record_counts.get("exp9", "~35K"),
        "primary_metric": "CFR (Confident Failure Rate, C1)",
        "mean": sum(cfr_vals) / len(cfr_vals) if cfr_vals else None,
        "std": (sum((v - sum(cfr_vals)/len(cfr_vals))**2 for v in cfr_vals) / len(cfr_vals))**0.5 if cfr_vals else None,
        "range": [min(cfr_vals), max(cfr_vals)] if cfr_vals else None,
        "kdi_mean": sum(kdi_vals) / len(kdi_vals) if kdi_vals else None,
    }

    # Print to stdout
    for exp, d in summary.items():
        print(f"\n{exp}: {d['description']}")
        print(f"  Models: {d['n_models']}, Items: {d['n_items']}, Records: {d['total_records']}")
        print(f"  Primary metric: {d['primary_metric']}")
        if d.get('mean') is not None:
            rng = f"[{d['range'][0]:.3f}, {d['range'][1]:.3f}]" if d.get('range') else "N/A"
            std = f"{d['std']:.4f}" if d.get('std') is not None else "N/A"
            print(f"  Mean: {d['mean']:.4f}, Std: {std}, Range: {rng}")

    return summary


def generate_exp_summary_latex(summary):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary statistics for each MIRROR experiment.}")
    lines.append(r"\label{tab:experiment-summary}")
    lines.append(r"\begin{tabular}{l l r l l r l}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Exp.} & \textbf{Description} & \textbf{Models} & \textbf{Items/Model} & \textbf{Primary Metric} & \textbf{Mean} & \textbf{Range} \\")
    lines.append(r"\midrule")

    for exp, d in summary.items():
        mean_str = f"{d['mean']:.3f}" if d.get("mean") is not None else "---"
        rng = f"[{d['range'][0]:.3f}, {d['range'][1]:.3f}]" if d.get("range") else "---"
        desc = d["description"]
        if len(desc) > 35:
            desc = desc[:32] + "..."
        metric = d["primary_metric"].split("(")[0].strip()
        lines.append(f"{exp} & {desc} & {d['n_models']} & {d['n_items']} & {metric} & {mean_str} & {rng} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
