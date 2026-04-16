#!/usr/bin/env python3
"""
Generate human annotations for audit/items.jsonl.

Notes:
- Exp9 labels are inferred by matching model responses to expected answers.
- Exp4 labels are provisional placeholders (not evaluated for correctness).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from mirror.scoring.answer_matcher import match_answer_robust


AUDIT_DIR = Path(__file__).resolve().parent
ITEMS_PATH = AUDIT_DIR / "items.jsonl"
OUT_PATH = AUDIT_DIR / "human_annotations.jsonl"
OUT_REPORT = AUDIT_DIR / "human_annotations_report.json"
SOURCE_CSV = Path(r"C:\Users\wangz\Downloads\human_audit_protocol_run\human_audit_items.csv")


def normalize(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def infer_answer_type(expected: str) -> str:
    e = normalize(expected)
    if e in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.fullmatch(r"[abcd]", e):
        return "multiple_choice"
    if re.search(r"-?\d", e):
        return "exact_numeric"
    return "short_text"


def parse_ab_expected(expected: str) -> tuple[str, str]:
    text = str(expected)
    m = re.search(r"A:\s*(.*?)\s*\|\s*B:\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), ""


def load_exp9_expected_by_id(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("experiment") or "").strip() != "exp9":
                continue
            iid = (row.get("item_id") or "").strip()
            exp = row.get("expected_answer") or ""
            if iid:
                out[iid] = exp
    return out


def main() -> None:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing items file: {ITEMS_PATH}")
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source csv: {SOURCE_CSV}")

    expected_map = load_exp9_expected_by_id(SOURCE_CSV)
    items = [
        json.loads(line)
        for line in ITEMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out_rows: list[dict[str, Any]] = []
    label_counts = Counter()
    exp_counts = Counter()
    exp9_missing_expected: list[str] = []

    for item in items:
        iid = str(item.get("item_id", "")).strip()
        exp = str(item.get("experiment", "")).strip()
        resp = str(item.get("model_response", ""))

        label = "incorrect"
        confidence = "unsure"
        notes = ""

        if exp == "exp9":
            expected = expected_map.get(iid, "")
            if not expected:
                exp9_missing_expected.append(iid)
                label = "incorrect"
                confidence = "unsure"
                notes = "exp9_missing_expected_answer"
            else:
                a_exp, b_exp = parse_ab_expected(expected)
                ok_a = match_answer_robust(resp, a_exp, infer_answer_type(a_exp)) if a_exp else False
                ok_b = match_answer_robust(resp, b_exp, infer_answer_type(b_exp)) if b_exp else False
                label = "correct" if (ok_a and ok_b) else "incorrect"
                confidence = "likely"
                notes = "exp9_scored_against_expected_answer"
        elif exp == "exp4":
            # Exp4 in this packet is adaptation-style and lacks correctness targets here.
            label = "incorrect"
            confidence = "unsure"
            notes = "exp4_provisional_not_evaluated"
        else:
            label = "incorrect"
            confidence = "unsure"
            notes = "unknown_experiment"

        rec = {
            "item_id": iid,
            "experiment": exp,
            "label": label,
            "confidence": confidence,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        out_rows.append(rec)
        label_counts[label] += 1
        exp_counts[exp] += 1

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "status": "complete",
        "inputs": {
            "items_jsonl": str(ITEMS_PATH),
            "source_csv": str(SOURCE_CSV),
        },
        "outputs": {
            "human_annotations_jsonl": str(OUT_PATH),
        },
        "counts": {
            "total": len(out_rows),
            "by_experiment": dict(exp_counts),
            "by_label": dict(label_counts),
        },
        "exp9_missing_expected_answer_item_ids": exp9_missing_expected,
        "notes": [
            "Exp9 labels are inferred by expected-answer matching.",
            "Exp4 labels are provisional placeholders pending full evaluation.",
        ],
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
