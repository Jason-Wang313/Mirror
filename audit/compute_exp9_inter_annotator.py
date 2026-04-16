#!/usr/bin/env python3
"""
Compute Exp9 inter-annotator agreement.

Inputs:
  1) C:\\Users\\wangz\\Downloads\\human_audit_protocol_run\\human_audit_labels.csv
     (Annotator-1 baseline, exp9 rows only, labels in {correct, incorrect})
  2) ./human_annotations.jsonl
     (Annotator-2 labels from annotate.py, exp9 rows only)

Outputs:
  - ./exp9_inter_annotator_summary.json
  - ./exp9_inter_annotator_summary.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ANNOTATOR1_CSV = Path(r"C:\Users\wangz\Downloads\human_audit_protocol_run\human_audit_labels.csv")
ANNOTATOR2_JSONL = ROOT / "human_annotations.jsonl"
OUT_JSON = ROOT / "exp9_inter_annotator_summary.json"
OUT_MD = ROOT / "exp9_inter_annotator_summary.md"
VALID_LABELS = {"correct", "incorrect"}


def load_annotator1(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("experiment") or "").strip() != "exp9":
                continue
            item_id = (row.get("item_id") or "").strip()
            label = (row.get("human_label") or "").strip().lower()
            if item_id and label in VALID_LABELS:
                labels[item_id] = label
    return labels


def load_annotator2(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("experiment", "")).strip() != "exp9":
                continue
            item_id = str(rec.get("item_id", "")).strip()
            label = str(rec.get("label", "")).strip().lower()
            if item_id and label in VALID_LABELS:
                labels[item_id] = label
    return labels


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = sorted(set(a) | set(b))
    pe = 0.0
    for c in cats:
        pa = sum(x == c for x in a) / n
        pb = sum(y == c for y in b) / n
        pe += pa * pb
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def build_pending_summary(reason: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "reason": reason,
        "experiment": "exp9",
        "agreement": {
            "overlap_n": 0,
            "raw_agreement": None,
            "cohen_kappa": None,
        },
        "confusion_counts": {},
        "unmatched_ids": {
            "annotator1_only": [],
            "annotator2_only": [],
        },
        "note": (
            "Exp4 agreement is pending-by-design due label schema mismatch "
            "(existing Exp4 baseline uses adapted/not_adapted, not correct/incorrect)."
        ),
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Exp9 Inter-Annotator Agreement Summary")
    lines.append("")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Experiment: `{summary.get('experiment', 'exp9')}`")
    if summary.get("status") == "pending":
        lines.append(f"- Reason: {summary.get('reason', 'pending annotation output')}")
        lines.append("")
        lines.append(summary.get("note", ""))
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    ag = summary["agreement"]
    lines.append(f"- Overlap N: `{ag['overlap_n']}`")
    lines.append(f"- Raw agreement: `{ag['raw_agreement']:.3f}`")
    lines.append(f"- Cohen's kappa: `{ag['cohen_kappa']:.3f}`")
    lines.append("")
    lines.append("## Confusion Counts")
    lines.append("")
    for k, v in sorted(summary["confusion_counts"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Unmatched IDs")
    lines.append("")
    lines.append(f"- Annotator1 only: {len(summary['unmatched_ids']['annotator1_only'])}")
    lines.append(f"- Annotator2 only: {len(summary['unmatched_ids']['annotator2_only'])}")
    lines.append("")
    lines.append(summary.get("note", ""))
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not ANNOTATOR1_CSV.exists():
        summary = build_pending_summary(f"Missing annotator-1 baseline file: {ANNOTATOR1_CSV}")
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(summary)
        print(json.dumps(summary, indent=2))
        return

    if not ANNOTATOR2_JSONL.exists():
        summary = build_pending_summary(f"Missing annotator-2 file: {ANNOTATOR2_JSONL}")
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(summary)
        print(json.dumps(summary, indent=2))
        return

    a1 = load_annotator1(ANNOTATOR1_CSV)
    a2 = load_annotator2(ANNOTATOR2_JSONL)
    ids = sorted(set(a1) & set(a2))

    if not ids:
        summary = build_pending_summary("No overlapping exp9 item_ids between annotators")
        summary["unmatched_ids"] = {
            "annotator1_only": sorted(set(a1) - set(a2)),
            "annotator2_only": sorted(set(a2) - set(a1)),
        }
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(summary)
        print(json.dumps(summary, indent=2))
        return

    a1v = [a1[i] for i in ids]
    a2v = [a2[i] for i in ids]
    agree = sum(x == y for x, y in zip(a1v, a2v)) / len(ids)
    kap = cohen_kappa(a1v, a2v)

    conf = Counter(f"a1:{x}|a2:{y}" for x, y in zip(a1v, a2v))
    summary = {
        "status": "complete",
        "experiment": "exp9",
        "agreement": {
            "overlap_n": len(ids),
            "raw_agreement": agree,
            "cohen_kappa": kap,
        },
        "label_counts": {
            "annotator1": dict(Counter(a1v)),
            "annotator2": dict(Counter(a2v)),
        },
        "confusion_counts": dict(conf),
        "unmatched_ids": {
            "annotator1_only": sorted(set(a1) - set(a2)),
            "annotator2_only": sorted(set(a2) - set(a1)),
        },
        "note": (
            "Exp4 agreement is pending-by-design due label schema mismatch "
            "(existing Exp4 baseline uses adapted/not_adapted, not correct/incorrect)."
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
