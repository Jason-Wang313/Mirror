"""
Create explicitly AI-labeled baseline responses for the MIRROR packet.

Strategy:
  - No intentional mistakes are injected.
  - Responses are filled from answer keys as a best-effort AI baseline proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = ROOT / "audit" / "human_baseline_packet"
AI_NOTE = "AI_GENERATED_NOT_HUMAN"
AI_STRATEGY = "BEST_EFFORT_FROM_ANSWER_KEYS_NO_INTENTIONAL_ERRORS"


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def build_exp1_ai(
    exp1_template_rows: list[dict],
    exp1_key_rows: list[dict],
) -> tuple[list[dict], dict[str, dict], list[str], list[str], float]:
    rows = [dict(r) for r in exp1_template_rows]
    key_by_item = {r["item_id"]: r for r in exp1_key_rows}
    domain_summary: dict[str, dict] = {}

    for row in rows:
        item_id = row.get("item_id", "")
        key = key_by_item.get(item_id, {})
        domain = (row.get("domain") or key.get("domain") or "").strip()
        correct_answer = (key.get("correct_answer") or "").strip()

        row["participant_answer"] = correct_answer
        row["participant_confidence"] = "0.95"
        row["notes"] = f"{AI_NOTE};{AI_STRATEGY}"

        if domain not in domain_summary:
            domain_summary[domain] = {"n": 0, "n_correct": 0, "nat_acc": 0.0}
        domain_summary[domain]["n"] += 1
        domain_summary[domain]["n_correct"] += 1

    for domain, stats in domain_summary.items():
        n = stats["n"]
        stats["nat_acc"] = (stats["n_correct"] / n) if n else 0.0

    med = _median([stats["nat_acc"] for stats in domain_summary.values()])
    weak_domains = sorted([d for d, stats in domain_summary.items() if stats["nat_acc"] < med])
    strong_domains = sorted([d for d in domain_summary if d not in weak_domains])
    return rows, domain_summary, weak_domains, strong_domains, med


def build_exp9_ai(exp9_template_rows: list[dict], exp9_key_rows: list[dict]) -> list[dict]:
    rows = [dict(r) for r in exp9_template_rows]
    key_by_item = {r["item_id"]: r for r in exp9_key_rows}

    for row in rows:
        item_id = row.get("item_id", "")
        key = key_by_item.get(item_id, {})

        row["decision_a"] = "PROCEED"
        row["answer_a"] = (key.get("correct_answer_a") or "").strip()
        row["decision_b"] = "PROCEED"
        row["answer_b"] = (key.get("correct_answer_b") or "").strip()
        row["participant_confidence"] = "0.95"
        row["notes"] = f"{AI_NOTE};{AI_STRATEGY}"

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AI-generated baseline packet response sheets.")
    parser.add_argument(
        "--exp1-template",
        type=Path,
        default=PACKET_DIR / "templates" / "exp1_response_sheet.csv",
    )
    parser.add_argument(
        "--exp9-template",
        type=Path,
        default=PACKET_DIR / "templates" / "exp9_response_sheet.csv",
    )
    parser.add_argument(
        "--exp1-key",
        type=Path,
        default=PACKET_DIR / "answer_keys" / "exp1_answer_key.csv",
    )
    parser.add_argument(
        "--exp9-key",
        type=Path,
        default=PACKET_DIR / "answer_keys" / "exp9_answer_key.csv",
    )
    parser.add_argument(
        "--exp1-out",
        type=Path,
        default=PACKET_DIR / "templates" / "exp1_response_sheet_ai_generated.csv",
    )
    parser.add_argument(
        "--exp9-out",
        type=Path,
        default=PACKET_DIR / "templates" / "exp9_response_sheet_ai_generated.csv",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=PACKET_DIR / "results_ai" / "ai_generation_manifest.json",
    )
    args = parser.parse_args()

    exp1_fields, exp1_template_rows = read_csv(args.exp1_template)
    exp9_fields, exp9_template_rows = read_csv(args.exp9_template)
    _, exp1_key_rows = read_csv(args.exp1_key)
    _, exp9_key_rows = read_csv(args.exp9_key)

    exp1_ai_rows, domain_summary, weak_domains, strong_domains, median_acc = build_exp1_ai(
        exp1_template_rows=exp1_template_rows,
        exp1_key_rows=exp1_key_rows,
    )
    exp9_ai_rows = build_exp9_ai(
        exp9_template_rows=exp9_template_rows,
        exp9_key_rows=exp9_key_rows,
    )

    write_csv(args.exp1_out, exp1_fields, exp1_ai_rows)
    write_csv(args.exp9_out, exp9_fields, exp9_ai_rows)

    manifest = {
        "label_note": AI_NOTE,
        "strategy": AI_STRATEGY,
        "exp1_out": str(args.exp1_out),
        "exp9_out": str(args.exp9_out),
        "exp1_domain_summary": domain_summary,
        "exp1_median_domain_nat_acc": median_acc,
        "weak_domains": weak_domains,
        "strong_domains": strong_domains,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {args.exp1_out}")
    print(f"Wrote: {args.exp9_out}")
    print(f"Wrote: {args.manifest_out}")
    print(f"Strategy: {AI_STRATEGY}")
    print(f"Weak domains (from AI Exp1 profile): {', '.join(weak_domains) if weak_domains else '(none)'}")


if __name__ == "__main__":
    main()
