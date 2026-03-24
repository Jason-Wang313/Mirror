"""
Score multi-rater agreement for the 100-item human audit packet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VALID_LABELS = {"correct", "incorrect", "ambiguous"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_label(v: str) -> str:
    return (v or "").strip().lower()


def cohen_kappa(a: list[str], b: list[str], categories: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = 0.0
    for c in categories:
        pa = sum(x == c for x in a) / n
        pb = sum(y == c for y in b) / n
        pe += pa * pb
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(label_matrix: list[list[str]], categories: list[str]) -> float:
    if not label_matrix:
        return 0.0
    n_items = len(label_matrix)
    n_raters = len(label_matrix[0]) if label_matrix[0] else 0
    if n_raters <= 1:
        return 0.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = [[0] * len(categories) for _ in range(n_items)]
    for i, row in enumerate(label_matrix):
        for lbl in row:
            counts[i][cat_to_idx[lbl]] += 1

    p_i = []
    for row in counts:
        numer = sum(v * (v - 1) for v in row)
        p_i.append(numer / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / n_items

    p_j = []
    for j in range(len(categories)):
        p_j.append(sum(counts[i][j] for i in range(n_items)) / (n_items * n_raters))
    p_e = sum(v * v for v in p_j)
    if p_e >= 1.0:
        return 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def format_table_md(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        line = "| " + " | ".join(r[c].ljust(widths[c]) for c in range(len(r))) + " |"
        out.append(line)
        if i == 0:
            out.append("| " + " | ".join("-" * widths[c] for c in range(len(r))) + " |")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score 3-rater agreement on 100-item human audit.")
    parser.add_argument(
        "--rater-files",
        nargs="+",
        type=Path,
        default=[
            ROOT / "audit" / "real_human_audit_100_R1.csv",
            ROOT / "audit" / "real_human_audit_100_R2.csv",
            ROOT / "audit" / "real_human_audit_100_R3.csv",
        ],
    )
    parser.add_argument("--expected-rows", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "audit" / "human_baseline_packet" / "results")
    parser.add_argument("--summary-stem", type=str, default="human_audit_multirater_summary")
    args = parser.parse_args()

    if len(args.rater_files) < 2:
        raise ValueError("Need at least two rater files.")

    rater_rows = []
    rater_ids = []
    validation = []
    for idx, p in enumerate(args.rater_files, start=1):
        if not p.exists():
            raise FileNotFoundError(f"Missing rater file: {p}")
        rows = read_csv(p)
        missing_label = sum(1 for r in rows if not normalize_label(r.get("human_label", "")))
        invalid_label = sum(
            1 for r in rows if normalize_label(r.get("human_label", "")) not in VALID_LABELS
        )
        missing_conf = sum(1 for r in rows if not (r.get("confidence") or "").strip())
        validation.append(
            {
                "rater": f"R{idx}",
                "path": str(p),
                "row_count": len(rows),
                "expected_rows": args.expected_rows,
                "row_count_ok": len(rows) == args.expected_rows,
                "blank_human_label": missing_label,
                "invalid_human_label": invalid_label,
                "blank_confidence": missing_conf,
            }
        )
        rater_rows.append(rows)
        rater_ids.append(f"R{idx}")

    if not all(v["row_count_ok"] and v["blank_human_label"] == 0 and v["invalid_human_label"] == 0 for v in validation):
        out = {
            "status": "failed_validation",
            "validation": validation,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_dir / f"{args.summary_stem}.json"
        out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("Validation failed. See summary JSON for details.")

    item_sets = [set(r["item_id"] for r in rows) for rows in rater_rows]
    shared_ids = sorted(set.intersection(*item_sets))
    if not shared_ids:
        raise ValueError("No overlapping item_ids across raters.")

    by_rater = []
    exp_by_item = {}
    for rows in rater_rows:
        lookup = {}
        for r in rows:
            iid = r["item_id"]
            lookup[iid] = normalize_label(r.get("human_label", ""))
            exp_by_item[iid] = r.get("experiment", "")
        by_rater.append(lookup)

    label_matrix = [[lookup[iid] for lookup in by_rater] for iid in shared_ids]
    categories = sorted(VALID_LABELS)

    pairwise = []
    for i in range(len(by_rater)):
        for j in range(i + 1, len(by_rater)):
            a = [by_rater[i][iid] for iid in shared_ids]
            b = [by_rater[j][iid] for iid in shared_ids]
            raw = sum(x == y for x, y in zip(a, b)) / len(shared_ids)
            kap = cohen_kappa(a, b, categories)
            pairwise.append(
                {
                    "pair": f"{rater_ids[i]}-{rater_ids[j]}",
                    "raw_agreement": raw,
                    "cohen_kappa": kap,
                }
            )

    unanimous = sum(1 for row in label_matrix if len(set(row)) == 1)
    unanimous_rate = unanimous / len(label_matrix)
    fleiss = fleiss_kappa(label_matrix, categories)

    label_counts = {}
    for rid, lookup in zip(rater_ids, by_rater):
        cnt = Counter(lookup[iid] for iid in shared_ids)
        label_counts[rid] = dict(sorted(cnt.items()))

    majority_counts = Counter()
    per_experiment = defaultdict(lambda: {"n": 0, "unanimous": 0})
    for iid, labels in zip(shared_ids, label_matrix):
        cnt = Counter(labels)
        majority = cnt.most_common(1)[0][0]
        majority_counts[majority] += 1
        exp = exp_by_item.get(iid, "")
        per_experiment[exp]["n"] += 1
        if len(set(labels)) == 1:
            per_experiment[exp]["unanimous"] += 1

    per_exp_summary = {
        exp: {
            "n": v["n"],
            "unanimous_n": v["unanimous"],
            "unanimous_rate": (v["unanimous"] / v["n"]) if v["n"] else 0.0,
        }
        for exp, v in sorted(per_experiment.items())
    }

    summary = {
        "status": "complete",
        "n_raters": len(rater_ids),
        "n_shared_items": len(shared_ids),
        "rater_ids": rater_ids,
        "validation": validation,
        "agreement": {
            "unanimous_rate": unanimous_rate,
            "unanimous_n": unanimous,
            "fleiss_kappa": fleiss,
            "pairwise": pairwise,
        },
        "label_counts": label_counts,
        "majority_label_counts": dict(sorted(majority_counts.items())),
        "per_experiment": per_exp_summary,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.summary_stem}.json"
    out_md = args.out_dir / f"{args.summary_stem}.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pair_rows = [["Pair", "Raw Agreement", "Cohen's kappa"]]
    for p in pairwise:
        pair_rows.append([p["pair"], f"{p['raw_agreement']:.3f}", f"{p['cohen_kappa']:.3f}"])
    exp_rows = [["Experiment", "N", "Unanimous N", "Unanimous Rate"]]
    for exp, v in per_exp_summary.items():
        exp_rows.append([exp, str(v["n"]), str(v["unanimous_n"]), f"{v['unanimous_rate']:.3f}"])

    md_lines = [
        "# Multi-Rater Human Audit Summary",
        "",
        "## Core Agreement",
        "",
        f"- Shared items: {len(shared_ids)}",
        f"- Raters: {', '.join(rater_ids)}",
        f"- Unanimous agreement: {unanimous_rate:.3f} ({unanimous}/{len(shared_ids)})",
        f"- Fleiss' kappa: {fleiss:.3f}",
        "",
        "## Pairwise Agreement",
        "",
        *format_table_md(pair_rows),
        "",
        "## Label Distribution",
        "",
    ]
    for rid in rater_ids:
        md_lines.append(f"- {rid}: {label_counts[rid]}")
    md_lines.extend(
        [
            "",
            f"- Majority-label counts: {dict(sorted(majority_counts.items()))}",
            "",
            "## Per-Experiment Unanimity",
            "",
            *format_table_md(exp_rows),
            "",
            "## Manuscript Insert (Draft)",
            "",
            (
                f"A 3-rater audit on 100 items yielded unanimous agreement on {unanimous}/{len(shared_ids)} items "
                f"({unanimous_rate:.1%}) with Fleiss' $\\kappa={fleiss:.3f}$. Pairwise Cohen's $\\kappa$ ranged from "
                f"{min(p['cohen_kappa'] for p in pairwise):.3f} to {max(p['cohen_kappa'] for p in pairwise):.3f}."
            ),
            "",
        ]
    )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
