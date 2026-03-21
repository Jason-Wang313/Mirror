"""
Compute ECE, Brier Score, and Overconfidence for all Exp1 models.

Loads all exp1 JSONL files, filters to Channel 1 (wagering) records,
extracts (bet, correct) pairs, normalizes bet to [0,1], and computes
per-model calibration metrics.

Also correlates ECE with MIRROR Gap from paper/tables/table1_data.json.
"""

import json
import math
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_all_exp1_records(results_dir: str) -> list[dict]:
    """Load all exp1 JSONL files from results_dir."""
    patterns = [
        os.path.join(results_dir, "exp1_*_results.jsonl"),
        os.path.join(results_dir, "exp1_*_shard.jsonl"),
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))

    # Exclude counterfactual files
    all_files = [f for f in all_files if "counterfactual" not in os.path.basename(f)]

    records = []
    file_counts = {}
    for fpath in sorted(all_files):
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                        count += 1
                    except json.JSONDecodeError:
                        pass
        file_counts[os.path.basename(fpath)] = count

    print(f"Loaded {len(records)} total records from {len(all_files)} files")
    print(f"\nPer-file record counts:")
    for fname, cnt in sorted(file_counts.items()):
        print(f"  {fname}: {cnt}")

    return records


def filter_wagering(records: list[dict]) -> list[dict]:
    """Filter to Channel 1 (wagering) records with valid bet and correctness."""
    filtered = []
    for r in records:
        if r.get("channel") != 1:
            continue
        parsed = r.get("parsed")
        if not parsed:
            continue
        bet = parsed.get("bet")
        if bet is None:
            continue
        correct = r.get("answer_correct")
        if correct is None:
            continue
        filtered.append(r)
    return filtered


def compute_ece(confidences: list[float], correctnesses: list[float], n_bins: int = 10) -> float:
    """ECE with equal-width bins. Confidences in [0,1]."""
    if not confidences:
        return float("nan")
    n = len(confidences)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            j for j in range(n)
            if lo <= confidences[j] < hi or (hi == 1.0 and confidences[j] == 1.0)
        ]
        if not in_bin:
            continue
        bin_acc = sum(correctnesses[j] for j in in_bin) / len(in_bin)
        bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return ece


def compute_brier(confidences: list[float], correctnesses: list[float]) -> float:
    """Brier Score = mean((p_hat - y)^2)."""
    if not confidences:
        return float("nan")
    n = len(confidences)
    return sum((confidences[i] - correctnesses[i]) ** 2 for i in range(n)) / n


def compute_overconfidence(confidences: list[float], correctnesses: list[float]) -> float:
    """Overconfidence = mean(p_hat) - mean(y)."""
    if not confidences:
        return float("nan")
    return sum(confidences) / len(confidences) - sum(correctnesses) / len(correctnesses)


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation and approximate p-value."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")

    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan"), float("nan")
    rho = num / (den_x * den_y)

    # Approximate p-value using t-distribution approximation
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        # Two-tailed p-value approximation using normal distribution for large n
        # For small n, this is approximate
        from math import erfc
        p_value = erfc(abs(t_stat) / math.sqrt(2))

    return rho, p_value


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = str(repo_root / "data" / "results")
    table1_path = repo_root / "paper" / "tables" / "table1_data.json"
    output_path = repo_root / "paper" / "tables" / "ece_brier_results.json"

    # Step 1: Load all records
    print("=" * 70)
    print("STEP 1: Loading all Exp1 JSONL files")
    print("=" * 70)
    records = load_all_exp1_records(results_dir)

    # Step 2: Filter to wagering
    print(f"\n{'=' * 70}")
    print("STEP 2: Filtering to Channel 1 (wagering) records")
    print("=" * 70)
    wagering = filter_wagering(records)
    print(f"  Wagering records with valid bet + correctness: {len(wagering)}")

    # Group by model
    by_model = defaultdict(list)
    for r in wagering:
        by_model[r["model"]].append(r)

    print(f"  Models found: {len(by_model)}")
    for model in sorted(by_model.keys()):
        print(f"    {model}: {len(by_model[model])} records")

    # Step 3: Load MIRROR Gap from table1_data.json
    print(f"\n{'=' * 70}")
    print("STEP 3: Loading MIRROR Gap from table1_data.json")
    print("=" * 70)
    with open(table1_path, "r", encoding="utf-8") as f:
        table1 = json.load(f)
    print(f"  Models in table1: {len(table1)}")

    # Step 4: Compute metrics per model
    print(f"\n{'=' * 70}")
    print("STEP 4: Computing ECE, Brier Score, Overconfidence per model")
    print("=" * 70)

    results = {}
    for model in sorted(by_model.keys()):
        recs = by_model[model]
        # Extract (bet, correct) pairs
        # Bet is 1-10, normalize to [0,1] by dividing by 10
        confidences = [r["parsed"]["bet"] / 10.0 for r in recs]
        correctnesses = [float(r["answer_correct"]) for r in recs]

        ece = compute_ece(confidences, correctnesses)
        brier = compute_brier(confidences, correctnesses)
        overconf = compute_overconfidence(confidences, correctnesses)

        # Get MIRROR gap
        mirror_gap = None
        if model in table1:
            mirror_gap = table1[model].get("mirror_gap")

        # Also compute mean confidence and mean accuracy for context
        mean_conf = sum(confidences) / len(confidences)
        mean_acc = sum(correctnesses) / len(correctnesses)

        results[model] = {
            "n_records": len(recs),
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "overconfidence": round(overconf, 4),
            "mean_confidence": round(mean_conf, 4),
            "mean_accuracy": round(mean_acc, 4),
            "mirror_gap": round(mirror_gap, 4) if mirror_gap is not None else None,
        }

    # Step 5: Print table
    print(f"\n{'=' * 70}")
    print("RESULTS TABLE")
    print("=" * 70)

    header = f"{'Model':<22} | {'N':>5} | {'ECE':>6} | {'Brier':>6} | {'Overconf':>8} | {'M.Gap':>6} | {'MeanConf':>8} | {'MeanAcc':>7}"
    print(header)
    print("-" * len(header))

    # Collect for Spearman correlation
    models_with_both = []
    ece_vals = []
    mgap_vals = []
    brier_vals = []

    for model in sorted(results.keys()):
        r = results[model]
        mgap_str = f"{r['mirror_gap']:.4f}" if r['mirror_gap'] is not None else "   N/A"
        print(
            f"{model:<22} | {r['n_records']:>5} | {r['ece']:>6.4f} | {r['brier_score']:>6.4f} | "
            f"{r['overconfidence']:>8.4f} | {mgap_str:>6} | {r['mean_confidence']:>8.4f} | {r['mean_accuracy']:>7.4f}"
        )
        if r["mirror_gap"] is not None:
            models_with_both.append(model)
            ece_vals.append(r["ece"])
            mgap_vals.append(r["mirror_gap"])
            brier_vals.append(r["brier_score"])

    # Step 6: Spearman correlation between MIRROR Gap and ECE
    print(f"\n{'=' * 70}")
    print("STEP 6: Spearman correlations (across models)")
    print("=" * 70)
    print(f"  N models with both ECE and MIRROR Gap: {len(models_with_both)}")

    rho_ece, p_ece = spearman(mgap_vals, ece_vals)
    rho_brier, p_brier = spearman(mgap_vals, brier_vals)
    overconf_vals = [results[m]["overconfidence"] for m in models_with_both]
    rho_overconf, p_overconf = spearman(mgap_vals, overconf_vals)

    print(f"\n  MIRROR Gap vs ECE:            rho = {rho_ece:+.4f}, p = {p_ece:.4f}")
    print(f"  MIRROR Gap vs Brier Score:    rho = {rho_brier:+.4f}, p = {p_brier:.4f}")
    print(f"  MIRROR Gap vs Overconfidence: rho = {rho_overconf:+.4f}, p = {p_overconf:.4f}")

    # Save results
    output = {
        "per_model": results,
        "correlations": {
            "mirror_gap_vs_ece": {
                "spearman_rho": round(rho_ece, 4) if not math.isnan(rho_ece) else None,
                "p_value": round(p_ece, 4) if not math.isnan(p_ece) else None,
                "n_models": len(models_with_both),
            },
            "mirror_gap_vs_brier": {
                "spearman_rho": round(rho_brier, 4) if not math.isnan(rho_brier) else None,
                "p_value": round(p_brier, 4) if not math.isnan(p_brier) else None,
                "n_models": len(models_with_both),
            },
            "mirror_gap_vs_overconfidence": {
                "spearman_rho": round(rho_overconf, 4) if not math.isnan(rho_overconf) else None,
                "p_value": round(p_overconf, 4) if not math.isnan(p_overconf) else None,
                "n_models": len(models_with_both),
            },
        },
        "methodology": {
            "wager_scale": "1-10 (normalized to 0.1-1.0 by dividing by 10)",
            "ece_bins": 10,
            "ece_formula": "sum((|bin|/N) * |accuracy(bin) - mean_confidence(bin)|)",
            "brier_formula": "mean((p_hat - y)^2)",
            "overconfidence_formula": "mean(p_hat) - mean(y)",
            "filter": "channel==1, parsed.bet is not None, answer_correct is not None",
            "source_files": "all exp1_*_results.jsonl and exp1_*_shard.jsonl (excluding counterfactual)",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
