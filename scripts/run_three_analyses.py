#!/usr/bin/env python3
"""
Three supplementary analyses for MIRROR paper:
  A1: Kendall's tau MCI robustness check
  A2: Provider-specific CFR analysis
  A3: End-to-end utility table from Exp9
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "data" / "results"
SUPP = PROJECT / "paper" / "supplementary"
SUPP.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Utility: Kendall's tau (pure Python, no scipy)
# =========================================================================

def kendall_tau(x, y):
    """Compute Kendall's tau-b between two equal-length sequences."""
    n = len(x)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    tied_xy = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            if dx == 0 and dy == 0:
                tied_xy += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    n_pairs = n * (n - 1) / 2
    denom_x = math.sqrt((n_pairs - tied_x - tied_xy) * (n_pairs - tied_y - tied_xy))
    if denom_x == 0:
        return float("nan")

    return (concordant - discordant) / denom_x


def spearman_rho(x, y):
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])
        ranks = [0.0] * len(lst)
        i = 0
        while i < len(lst):
            j = i
            while j < len(lst) - 1 and lst[sorted_indices[j + 1]] == lst[sorted_indices[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def mann_whitney_u(x, y):
    """Compute Mann-Whitney U test statistic and approximate p-value (two-sided)."""
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return float("nan"), float("nan")

    # Rank all values
    combined = [(v, 'x') for v in x] + [(v, 'y') for v in y]
    combined.sort(key=lambda t: t[0])
    n = nx + ny

    # Assign ranks with ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and combined[j + 1][0] == combined[j][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # Sum of ranks for group x
    r_x = sum(ranks[i] for i in range(n) if combined[i][1] == 'x')

    u_x = r_x - nx * (nx + 1) / 2
    u_y = nx * ny - u_x
    u = min(u_x, u_y)

    # Normal approximation for p-value
    mu = nx * ny / 2
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12)
    if sigma == 0:
        return u, float("nan")
    z = (u - mu) / sigma
    # Two-sided p-value using error function approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return u, p


# =========================================================================
# A1: MCI Robustness (Kendall's tau vs Spearman)
# =========================================================================

def analysis_1_mci_robustness():
    print("=" * 70)
    print("A1: MCI ROBUSTNESS CHECK — Kendall's tau vs Spearman's rho")
    print("=" * 70)

    # Find the largest Exp1 JSONL file
    exp1_files = sorted(RESULTS.glob("exp1_*_results.jsonl"), key=lambda f: f.stat().st_size)
    if not exp1_files:
        print("ERROR: No exp1 JSONL files found")
        return {}

    # Use largest file
    exp1_file = exp1_files[-1]
    print(f"Loading: {exp1_file.name} ({exp1_file.stat().st_size / 1e6:.1f} MB)")

    records = []
    with open(exp1_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records)} records")

    # Group by model
    by_model = defaultdict(list)
    for r in records:
        m = r.get("model")
        if m:
            by_model[m].append(r)

    results = {}

    for model, model_records in sorted(by_model.items()):
        # Extract channel signals per question (same logic as metrics.py)
        signals = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

        for r in model_records:
            ch = r.get("channel")
            qid = r.get("question_id")
            parsed = r.get("parsed", {})
            if not parsed:
                parsed = {}

            if not qid or ch not in signals:
                continue

            if ch == 1:
                bet = parsed.get("bet")
                if bet is not None:
                    try:
                        signals[1][qid] = float(bet)
                    except (ValueError, TypeError):
                        pass

            elif ch == 2:
                skipped = parsed.get("skipped", False)
                signals[2][qid] = 0.0 if skipped else 1.0

            elif ch == 3:
                choice = parsed.get("choice")
                if choice in ("A", "B"):
                    signals[3][qid] = 1.0 if choice == "A" else 0.0

            elif ch == 4:
                tools_used = parsed.get("tools_used", [])
                n_tools = len(tools_used) if isinstance(tools_used, list) else 0
                signals[4][qid] = 1.0 / (1.0 + n_tools)

            elif ch == 5:
                hedging = parsed.get("hedging_count", 0)
                if hedging is None:
                    hedging = 0
                signals[5][qid] = 1.0 / (1.0 + hedging)

        # Normalize each channel to [0,1]
        normalized = {}
        for ch, q_map in signals.items():
            if not q_map:
                normalized[ch] = q_map
                continue
            values = list(q_map.values())
            lo, hi = min(values), max(values)
            if hi == lo:
                normalized[ch] = {qid: 0.5 for qid in q_map}
            else:
                normalized[ch] = {qid: (v - lo) / (hi - lo) for qid, v in q_map.items()}

        # Compute pairwise correlations (both Spearman and Kendall)
        channel_ids = [ch for ch in [1, 2, 3, 4, 5] if normalized.get(ch)]
        spearman_corrs = []
        kendall_corrs = []
        pairwise_detail = {}

        for i in range(len(channel_ids)):
            for j in range(i + 1, len(channel_ids)):
                ch_a = channel_ids[i]
                ch_b = channel_ids[j]
                pair_key = f"ch{ch_a}_ch{ch_b}"

                qids_a = set(normalized[ch_a].keys())
                qids_b = set(normalized[ch_b].keys())
                common = sorted(qids_a & qids_b)

                if len(common) < 2:
                    pairwise_detail[pair_key] = {"spearman": float("nan"), "kendall": float("nan"), "n": len(common)}
                    continue

                x = [normalized[ch_a][qid] for qid in common]
                y = [normalized[ch_b][qid] for qid in common]

                rho = spearman_rho(x, y)
                tau = kendall_tau(x, y)

                if not math.isnan(rho):
                    spearman_corrs.append(rho)
                if not math.isnan(tau):
                    kendall_corrs.append(tau)

                pairwise_detail[pair_key] = {
                    "spearman": round(rho, 6) if not math.isnan(rho) else None,
                    "kendall": round(tau, 6) if not math.isnan(tau) else None,
                    "n": len(common)
                }

        mci_spearman = sum(spearman_corrs) / len(spearman_corrs) if spearman_corrs else float("nan")
        mci_kendall = sum(kendall_corrs) / len(kendall_corrs) if kendall_corrs else float("nan")

        n_per_ch = {f"ch{ch}": len(signals[ch]) for ch in [1, 2, 3, 4, 5]}

        results[model] = {
            "mci_spearman": round(mci_spearman, 6) if not math.isnan(mci_spearman) else None,
            "mci_kendall": round(mci_kendall, 6) if not math.isnan(mci_kendall) else None,
            "n_valid_pairs_spearman": len(spearman_corrs),
            "n_valid_pairs_kendall": len(kendall_corrs),
            "n_questions_per_channel": n_per_ch,
            "pairwise_detail": pairwise_detail,
        }

    # Print summary table
    print()
    print(f"{'Model':<25} {'MCI_Spearman':>14} {'MCI_Kendall':>14} {'#Pairs':>8} {'Ch1':>5} {'Ch2':>5} {'Ch3':>5} {'Ch4':>5} {'Ch5':>5}")
    print("-" * 100)

    for model in sorted(results.keys()):
        r = results[model]
        mci_s = f"{r['mci_spearman']:.4f}" if r['mci_spearman'] is not None else "N/A"
        mci_k = f"{r['mci_kendall']:.4f}" if r['mci_kendall'] is not None else "N/A"
        nch = r["n_questions_per_channel"]
        print(f"{model:<25} {mci_s:>14} {mci_k:>14} {r['n_valid_pairs_kendall']:>8} "
              f"{nch.get('ch1', 0):>5} {nch.get('ch2', 0):>5} {nch.get('ch3', 0):>5} "
              f"{nch.get('ch4', 0):>5} {nch.get('ch5', 0):>5}")

    # Also report the Ch1-Ch5 (wagering vs natural) pair specifically
    print()
    print("Focused comparison — Ch1 (wagering) vs Ch5 (natural):")
    print(f"{'Model':<25} {'Spearman':>10} {'Kendall':>10} {'N':>6}")
    print("-" * 55)
    for model in sorted(results.keys()):
        pd = results[model]["pairwise_detail"]
        ch15 = pd.get("ch1_ch5", {})
        s = f"{ch15.get('spearman', 'N/A')}" if ch15.get('spearman') is not None else "N/A"
        k = f"{ch15.get('kendall', 'N/A')}" if ch15.get('kendall') is not None else "N/A"
        n = ch15.get("n", 0)
        # Format numbers if they're floats
        if isinstance(ch15.get('spearman'), (int, float)):
            s = f"{ch15['spearman']:.4f}"
        if isinstance(ch15.get('kendall'), (int, float)):
            k = f"{ch15['kendall']:.4f}"
        print(f"{model:<25} {s:>10} {k:>10} {n:>6}")

    out_path = SUPP / "mci_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


# =========================================================================
# A2: Provider-specific CFR analysis
# =========================================================================

def analysis_2_provider():
    print()
    print("=" * 70)
    print("A2: PROVIDER-SPECIFIC CFR ANALYSIS")
    print("=" * 70)

    table1_path = PROJECT / "paper" / "tables" / "table1_data.json"
    with open(table1_path, "r", encoding="utf-8") as f:
        table1 = json.load(f)

    # Define families (only models with cfr_c1 data)
    meta_models = ["llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b", "llama-3.3-70b"]
    non_meta_nim = ["gemma-3-27b", "gpt-oss-120b", "kimi-k2", "mistral-large", "phi-4"]
    deepseek_models = ["deepseek-r1"]

    # Collect CFR values per group
    groups = {
        "Meta (LLaMA)": meta_models,
        "Non-Meta NIM": non_meta_nim,
        "DeepSeek": deepseek_models,
    }

    group_data = {}
    for group_name, model_list in groups.items():
        cfr_values = []
        model_details = {}
        for m in model_list:
            if m in table1 and table1[m].get("cfr_c1") is not None:
                cfr = table1[m]["cfr_c1"]
                cfr_values.append(cfr)
                model_details[m] = round(cfr, 4)
            else:
                model_details[m] = None

        mean_cfr = sum(cfr_values) / len(cfr_values) if cfr_values else None
        group_data[group_name] = {
            "models": model_details,
            "cfr_values": [round(v, 4) for v in cfr_values],
            "n": len(cfr_values),
            "mean_cfr": round(mean_cfr, 4) if mean_cfr is not None else None,
        }

    # Print summary
    print()
    print(f"{'Group':<20} {'N':>4} {'Mean CFR':>10} {'Models with CFR':>50}")
    print("-" * 90)
    for gname, gd in group_data.items():
        models_str = ", ".join(f"{m}={v}" for m, v in gd["models"].items() if v is not None)
        mean_str = f"{gd['mean_cfr']:.4f}" if gd['mean_cfr'] is not None else "N/A"
        print(f"{gname:<20} {gd['n']:>4} {mean_str:>10} {models_str:>50}")

    # Mann-Whitney U test: Meta vs Non-Meta
    meta_cfrs = group_data["Meta (LLaMA)"]["cfr_values"]
    non_meta_cfrs = group_data["Non-Meta NIM"]["cfr_values"]

    test_result = {}
    if len(meta_cfrs) >= 3 and len(non_meta_cfrs) >= 3:
        u_stat, p_val = mann_whitney_u(meta_cfrs, non_meta_cfrs)
        test_result = {
            "test": "Mann-Whitney U",
            "U_statistic": round(u_stat, 4),
            "p_value": round(p_val, 4),
            "meta_n": len(meta_cfrs),
            "non_meta_n": len(non_meta_cfrs),
            "significant_at_05": p_val < 0.05,
            "interpretation": (
                "Provider family significantly affects CFR (p < 0.05)"
                if p_val < 0.05
                else "No significant difference in CFR between Meta and Non-Meta groups (p >= 0.05)"
            )
        }
        print(f"\nMann-Whitney U test (Meta vs Non-Meta NIM):")
        print(f"  Meta CFRs:     {meta_cfrs} (mean={group_data['Meta (LLaMA)']['mean_cfr']:.4f})")
        print(f"  Non-Meta CFRs: {non_meta_cfrs} (mean={group_data['Non-Meta NIM']['mean_cfr']:.4f})")
        print(f"  U = {u_stat:.4f}, p = {p_val:.4f}")
        print(f"  Conclusion: {test_result['interpretation']}")
    else:
        print(f"\nMann-Whitney U test: SKIPPED (Meta n={len(meta_cfrs)}, Non-Meta n={len(non_meta_cfrs)}; need >=3 each)")
        test_result = {
            "test": "Mann-Whitney U",
            "skipped": True,
            "reason": f"Meta n={len(meta_cfrs)}, Non-Meta n={len(non_meta_cfrs)}; need >=3 each"
        }

    output = {
        "groups": group_data,
        "mann_whitney_test": test_result,
    }

    out_path = SUPP / "provider_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    return output


# =========================================================================
# A3: End-to-end utility table
# =========================================================================

def analysis_3_utility():
    print()
    print("=" * 70)
    print("A3: END-TO-END UTILITY TABLE (Exp9)")
    print("=" * 70)

    exp9_file = RESULTS / "exp9_20260312T140842_results.jsonl"
    if not exp9_file.exists():
        print(f"ERROR: {exp9_file} not found")
        return {}

    print(f"Loading: {exp9_file.name}")
    records = []
    with open(exp9_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records)} records")

    # Exclude qwen-3-235b (100% API failure) per memory
    excluded_models = {"qwen-3-235b", "qwen3-235b-nim", "qwen-3-235b-nim"}

    # Each record has two components (a and b). Process each separately.
    # We only look at Paradigm 1 (autonomous) for decision classification and
    # conditions C1-C4.
    # A "component" is one half of a task record.

    # Structure: per model, per condition
    # For each component: count total, autonomous (proceed), autonomous_correct,
    # autonomous_incorrect, escalated (use_tool or defer)

    model_condition_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "autonomous": 0,
        "autonomous_correct": 0,
        "autonomous_incorrect": 0,
        "escalated": 0,
    }))

    for r in records:
        model = r.get("model", "")
        if model in excluded_models:
            continue
        if not r.get("api_success", True):
            continue

        condition = r.get("condition")
        if condition not in [1, 2, 3, 4]:
            continue

        # Process component a
        for comp in ["a", "b"]:
            decision = r.get(f"component_{comp}_decision")
            correct = r.get(f"component_{comp}_correct")

            if decision is None:
                continue

            stats = model_condition_stats[model][condition]
            stats["total"] += 1

            if decision == "proceed":
                stats["autonomous"] += 1
                if correct is True:
                    stats["autonomous_correct"] += 1
                else:
                    stats["autonomous_incorrect"] += 1
            else:
                # use_tool, defer, or any other decision
                stats["escalated"] += 1

    models = sorted([m for m in model_condition_stats.keys()])
    print(f"\nModels with data: {len(models)}: {', '.join(models)}")

    # Compute per-condition aggregates (mean across models)
    condition_agg = {}
    for cond in [1, 2, 3, 4]:
        agg = {
            "total": 0,
            "autonomous": 0,
            "autonomous_correct": 0,
            "autonomous_incorrect": 0,
            "escalated": 0,
            "n_models": 0,
            "per_model": {},
        }

        model_autonomy_rates = []
        model_success_rates = []
        model_cfrs = []

        for model in models:
            stats = model_condition_stats[model].get(cond)
            if stats is None or stats["total"] == 0:
                continue

            agg["n_models"] += 1
            agg["total"] += stats["total"]
            agg["autonomous"] += stats["autonomous"]
            agg["autonomous_correct"] += stats["autonomous_correct"]
            agg["autonomous_incorrect"] += stats["autonomous_incorrect"]
            agg["escalated"] += stats["escalated"]

            autonomy_rate = stats["autonomous"] / stats["total"] if stats["total"] > 0 else 0
            # System success = autonomous_correct + escalated (assuming escalated resolved)
            system_success = (stats["autonomous_correct"] + stats["escalated"]) / stats["total"] if stats["total"] > 0 else 0
            # CFR = autonomous_incorrect / total (or more precisely, on weak-domain components)
            cfr = stats["autonomous_incorrect"] / stats["total"] if stats["total"] > 0 else 0

            model_autonomy_rates.append(autonomy_rate)
            model_success_rates.append(system_success)
            model_cfrs.append(cfr)

            agg["per_model"][model] = {
                "total": stats["total"],
                "autonomous": stats["autonomous"],
                "autonomous_correct": stats["autonomous_correct"],
                "autonomous_incorrect": stats["autonomous_incorrect"],
                "escalated": stats["escalated"],
                "autonomy_rate": round(autonomy_rate, 4),
                "system_success_rate": round(system_success, 4),
                "cfr": round(cfr, 4),
            }

        # Means
        if model_autonomy_rates:
            agg["mean_autonomy_rate"] = round(sum(model_autonomy_rates) / len(model_autonomy_rates), 4)
            agg["mean_system_success_rate"] = round(sum(model_success_rates) / len(model_success_rates), 4)
            agg["mean_cfr"] = round(sum(model_cfrs) / len(model_cfrs), 4)
        else:
            agg["mean_autonomy_rate"] = None
            agg["mean_system_success_rate"] = None
            agg["mean_cfr"] = None

        # Overall rates
        if agg["total"] > 0:
            agg["overall_autonomy_rate"] = round(agg["autonomous"] / agg["total"], 4)
            agg["overall_system_success_rate"] = round(
                (agg["autonomous_correct"] + agg["escalated"]) / agg["total"], 4)
            agg["overall_cfr"] = round(agg["autonomous_incorrect"] / agg["total"], 4)
        else:
            agg["overall_autonomy_rate"] = None
            agg["overall_system_success_rate"] = None
            agg["overall_cfr"] = None

        condition_agg[f"C{cond}"] = agg

    # Print table
    print()
    print(f"{'Condition':<12} {'Total':>8} {'Auton':>8} {'Auton_OK':>10} {'Auton_Wrong':>12} {'Escalated':>10} {'Auton%':>8} {'SysSucc%':>10} {'CFR':>8} {'#Models':>8}")
    print("-" * 100)

    for cond_key in ["C1", "C2", "C3", "C4"]:
        a = condition_agg[cond_key]
        print(f"{cond_key:<12} {a['total']:>8} {a['autonomous']:>8} {a['autonomous_correct']:>10} "
              f"{a['autonomous_incorrect']:>12} {a['escalated']:>10} "
              f"{a['overall_autonomy_rate']*100 if a['overall_autonomy_rate'] is not None else 0:>7.1f}% "
              f"{a['overall_system_success_rate']*100 if a['overall_system_success_rate'] is not None else 0:>9.1f}% "
              f"{a['overall_cfr'] if a['overall_cfr'] is not None else 0:>7.4f} "
              f"{a['n_models']:>8}")

    # Mean across models
    print()
    print("Mean across models:")
    print(f"{'Condition':<12} {'Mean Auton%':>12} {'Mean SysSucc%':>14} {'Mean CFR':>10}")
    print("-" * 50)
    for cond_key in ["C1", "C2", "C3", "C4"]:
        a = condition_agg[cond_key]
        ma = f"{a['mean_autonomy_rate']*100:.1f}%" if a['mean_autonomy_rate'] is not None else "N/A"
        ms = f"{a['mean_system_success_rate']*100:.1f}%" if a['mean_system_success_rate'] is not None else "N/A"
        mc = f"{a['mean_cfr']:.4f}" if a['mean_cfr'] is not None else "N/A"
        print(f"{cond_key:<12} {ma:>12} {ms:>14} {mc:>10}")

    # Per-model breakdown for C1 (the main condition)
    print()
    print("Per-model detail (C1 — Uninformed):")
    print(f"{'Model':<25} {'Total':>7} {'Auton':>7} {'OK':>5} {'Wrong':>7} {'Esc':>5} {'Auton%':>8} {'SysSucc%':>10} {'CFR':>8}")
    print("-" * 90)
    c1 = condition_agg["C1"]
    for model in sorted(c1["per_model"].keys()):
        pm = c1["per_model"][model]
        print(f"{model:<25} {pm['total']:>7} {pm['autonomous']:>7} {pm['autonomous_correct']:>5} "
              f"{pm['autonomous_incorrect']:>7} {pm['escalated']:>5} "
              f"{pm['autonomy_rate']*100:>7.1f}% {pm['system_success_rate']*100:>9.1f}% {pm['cfr']:>7.4f}")

    # Clean output for JSON (remove per_model detail to keep it manageable; include it)
    out_path = SUPP / "end_to_end_utility.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(condition_agg, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return condition_agg


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("MIRROR Supplementary Analyses")
    print("=" * 70)
    print()

    a1 = analysis_1_mci_robustness()
    a2 = analysis_2_provider()
    a3 = analysis_3_utility()

    print()
    print("=" * 70)
    print("ALL THREE ANALYSES COMPLETE")
    print("=" * 70)
