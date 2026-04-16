"""
Reconcile the real human audit against automated labels.
Computes: agreement rate, Cohen's kappa, error breakdown, and LaTeX table.

Usage:
    python audit/reconcile_real_audit.py
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
AUDIT_CSV = AUDIT_DIR / "real_human_audit_100.csv"
SOURCE_CSV = Path.home() / "Downloads" / "human_audit_protocol_run" / "human_audit_items.csv"
OUTPUT_JSON = AUDIT_DIR / "real_human_audit_metrics.json"
OUTPUT_LATEX = AUDIT_DIR / "real_human_audit_table.tex"


def load_auto_labels() -> dict[str, str]:
    """Load automated labels from the original unblinded CSV."""
    labels = {}
    with open(SOURCE_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            iid = row.get("item_id", "").strip()
            auto = row.get("auto_label", "").strip().lower()
            if iid and auto:
                labels[iid] = auto
    return labels


def cohens_kappa(n_agree: int, n_total: int,
                 p_yes_human: float, p_yes_auto: float) -> float:
    """Compute Cohen's kappa from marginal proportions."""
    if n_total == 0:
        return 0.0
    p_o = n_agree / n_total
    p_e = p_yes_human * p_yes_auto + (1 - p_yes_human) * (1 - p_yes_auto)
    if abs(1 - p_e) < 1e-10:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def main():
    # Load human audit
    with open(AUDIT_CSV, encoding="utf-8", newline="") as f:
        audit_rows = list(csv.DictReader(f))

    # Check completeness
    empty = [r for r in audit_rows if not r.get("human_label", "").strip()]
    if empty:
        print(f"WARNING: {len(empty)} items have no human_label. Skipping them.")
        audit_rows = [r for r in audit_rows if r.get("human_label", "").strip()]

    if not audit_rows:
        print("ERROR: No labeled items found. Fill in human_label column first.")
        return

    auto_labels = load_auto_labels()

    # Reconcile
    by_exp = defaultdict(lambda: {
        "n": 0, "agree": 0, "disagree": 0,
        "human_correct": 0, "auto_correct": 0,
        "error_types": Counter(),
    })

    for row in audit_rows:
        iid = row["item_id"].strip()
        exp = row["experiment"].strip()
        human = row["human_label"].strip().lower()
        auto = auto_labels.get(iid, "").lower()

        if human == "ambiguous":
            by_exp[exp]["n"] += 1
            by_exp[exp]["error_types"]["ambiguous"] += 1
            # Count ambiguous as disagreement for kappa purposes
            by_exp[exp]["disagree"] += 1
            continue

        by_exp[exp]["n"] += 1
        if human == "correct":
            by_exp[exp]["human_correct"] += 1
        if auto == "correct":
            by_exp[exp]["auto_correct"] += 1

        if human == auto:
            by_exp[exp]["agree"] += 1
        else:
            by_exp[exp]["disagree"] += 1
            # Classify disagreement
            notes = row.get("notes", "").strip().lower()
            if "parse" in notes or "parsing" in notes:
                by_exp[exp]["error_types"]["parsing"] += 1
            elif "ambig" in notes or "edge" in notes:
                by_exp[exp]["error_types"]["ambiguous"] += 1
            elif human != auto:
                # Default: genuine error in automated label
                by_exp[exp]["error_types"]["genuine"] += 1

    # Compute metrics
    results = {}
    total_n = 0
    total_agree = 0
    total_genuine = 0

    for exp in sorted(by_exp):
        d = by_exp[exp]
        n = d["n"]
        agree = d["agree"]
        rate = agree / n if n else 0
        ci_lo, ci_hi = wilson_ci(rate, n)

        p_h = d["human_correct"] / n if n else 0
        p_a = d["auto_correct"] / n if n else 0
        kappa = cohens_kappa(agree, n, p_h, p_a)

        genuine = d["error_types"].get("genuine", 0)

        results[exp] = {
            "n": n,
            "agreement": round(rate, 3),
            "agreement_pct": f"{rate*100:.1f}%",
            "ci95": [round(ci_lo, 3), round(ci_hi, 3)],
            "kappa": round(kappa, 3),
            "genuine_errors": genuine,
            "ambiguous": d["error_types"].get("ambiguous", 0),
            "parsing": d["error_types"].get("parsing", 0),
        }
        total_n += n
        total_agree += agree
        total_genuine += genuine

    overall_rate = total_agree / total_n if total_n else 0
    overall_ci = wilson_ci(overall_rate, total_n)

    output = {
        "per_experiment": results,
        "overall": {
            "n": total_n,
            "agreement": round(overall_rate, 3),
            "agreement_pct": f"{overall_rate*100:.1f}%",
            "ci95": [round(overall_ci[0], 3), round(overall_ci[1], 3)],
            "genuine_errors": total_genuine,
            "error_rate": round(total_genuine / total_n, 4) if total_n else 0,
        }
    }

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Generate LaTeX table
    exp_labels = {
        "exp1": "Exp 1",
        "exp3": "Exp 3",
        "exp5": "Exp 5",
        "exp6b": "Exp 6b",
        "exp9": "Exp 9",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Experiment & $N$ & Agreement & $\kappa$ & Amb & Parse & Gen \\",
        r"\midrule",
    ]
    for exp in ["exp1", "exp3", "exp5", "exp6b", "exp9"]:
        if exp not in results:
            continue
        r = results[exp]
        lines.append(
            f"{exp_labels.get(exp, exp)} & {r['n']} & {r['agreement_pct']} & "
            f"{r['kappa']:.3f} & {r['ambiguous']} & {r['parsing']} & {r['genuine_errors']} \\\\"
        )
    lines.append(r"\midrule")
    o = output["overall"]
    lines.append(
        f"\\textbf{{Overall}} & \\textbf{{{o['n']}}} & \\textbf{{{o['agreement_pct']}}} & "
        f"--- & --- & --- & \\textbf{{{o['genuine_errors']}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(OUTPUT_LATEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Print results
    print("=" * 60)
    print("REAL HUMAN AUDIT RESULTS")
    print("=" * 60)
    for exp in sorted(results):
        r = results[exp]
        print(f"  {exp:6s}: n={r['n']:3d}  agreement={r['agreement_pct']:6s}  "
              f"κ={r['kappa']:.3f}  genuine_errors={r['genuine_errors']}")
    print(f"\n  Overall: n={o['n']}  agreement={o['agreement_pct']}  "
          f"genuine_errors={o['genuine_errors']}  error_rate={o['error_rate']:.2%}")
    print(f"\n  Saved: {OUTPUT_JSON}")
    print(f"  LaTeX: {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
