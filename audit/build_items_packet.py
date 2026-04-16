#!/usr/bin/env python3
"""
Build blinded 75-item packet for second annotator.

Source:
  C:\\Users\\wangz\\Downloads\\human_audit_protocol_run\\human_audit_items.csv

Outputs:
  - ./items.jsonl
  - ./items_build_report.json
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


SOURCE = Path(r"C:\Users\wangz\Downloads\human_audit_protocol_run\human_audit_items.csv")
OUT_DIR = Path(__file__).resolve().parent
OUT_ITEMS = OUT_DIR / "items.jsonl"
OUT_REPORT = OUT_DIR / "items_build_report.json"
SEED = 42


def main() -> None:
    rows: list[dict[str, str]] = []
    with SOURCE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp = (r.get("experiment") or "").strip()
            if exp not in {"exp4", "exp9"}:
                continue
            rows.append(
                {
                    "item_id": (r.get("item_id") or "").strip(),
                    "experiment": exp,
                    "question": (r.get("question_or_task_text") or "").strip(),
                    "model_response": r.get("model_response") or "",
                }
            )

    exp4_n = sum(1 for r in rows if r["experiment"] == "exp4")
    exp9_n = sum(1 for r in rows if r["experiment"] == "exp9")
    total_n = len(rows)
    if not (exp4_n == 25 and exp9_n == 50 and total_n == 75):
        raise SystemExit(
            f"Count check failed: expected exp4=25 exp9=50 total=75, "
            f"got exp4={exp4_n} exp9={exp9_n} total={total_n}"
        )

    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate item_id detected in packet source")

    random.seed(SEED)
    random.shuffle(rows)

    with OUT_ITEMS.open("w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty_resp_ids = [r["item_id"] for r in rows if not (r["model_response"] or "").strip()]
    report = {
        "source": str(SOURCE),
        "output": str(OUT_ITEMS),
        "seed": SEED,
        "counts": {
            "exp4": exp4_n,
            "exp9": exp9_n,
            "total": total_n,
        },
        "unique_item_ids": len(set(ids)),
        "empty_model_response_item_ids": empty_resp_ids,
        "empty_model_response_count": len(empty_resp_ids),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
