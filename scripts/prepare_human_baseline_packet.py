"""
Prepare a concrete human-baseline run packet for MIRROR v20.

Outputs (under audit/human_baseline_packet):
  - tasks/exp1_human_baseline_tasks.csv
  - tasks/exp9_human_baseline_tasks.csv
  - templates/exp1_response_sheet.csv
  - templates/exp9_response_sheet.csv
  - answer_keys/exp1_answer_key.csv
  - answer_keys/exp9_answer_key.csv
  - packet_manifest.json

Design:
  - Exp1 subset: stratified by domain (default 10 per domain = 80 total)
  - Exp9 subset: fixed circularity-free tasks, stratified by domain pair
    (default 2 per pair = 56 total)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "audit" / "human_baseline_packet"
DIFFICULTY_SCORE = {"easy": 0, "medium": 1, "hard": 2}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _difficulty_score(value: str) -> int:
    return DIFFICULTY_SCORE.get(str(value or "").strip().lower(), -1)


def _apply_difficulty_priority(pool: list[dict], rng: random.Random, hard_priority: bool) -> list[dict]:
    ordered = list(pool)
    rng.shuffle(ordered)
    if hard_priority:
        # Stable sort keeps random order for equal-difficulty items.
        ordered.sort(key=lambda x: _difficulty_score(x.get("difficulty", "")), reverse=True)
    return ordered


def _task_difficulty_min(task: dict) -> int:
    da = _difficulty_score(task.get("difficulty_a", ""))
    db = _difficulty_score(task.get("difficulty_b", ""))
    return min(da, db)


def _task_difficulty_sum(task: dict) -> int:
    da = _difficulty_score(task.get("difficulty_a", ""))
    db = _difficulty_score(task.get("difficulty_b", ""))
    return da + db


def sample_exp1(
    questions: list[dict],
    per_domain: int,
    seed: int,
    hard_priority: bool = False,
    min_difficulty: str = "easy",
) -> list[dict]:
    rng = random.Random(seed)
    min_score = DIFFICULTY_SCORE[min_difficulty]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        domain = str(q.get("domain", "")).strip()
        if domain:
            by_domain[domain].append(q)

    selected: list[dict] = []
    for domain in sorted(by_domain):
        pool = [q for q in by_domain[domain] if _difficulty_score(q.get("difficulty", "")) >= min_score]
        if len(pool) < per_domain:
            # Fallback to full pool if filtered items are insufficient.
            pool = by_domain[domain]
        by_subcat: dict[str, list[dict]] = defaultdict(list)
        for q in pool:
            sub = str(q.get("subcategory", "unknown"))
            by_subcat[sub].append(q)
        subcats = sorted(by_subcat)
        for sub in subcats:
            by_subcat[sub] = _apply_difficulty_priority(by_subcat[sub], rng, hard_priority)

        # Round-robin by subcategory for broad coverage.
        picks: list[dict] = []
        idx = 0
        while len(picks) < per_domain:
            progressed = False
            for sub in subcats:
                bucket = by_subcat[sub]
                if idx < len(bucket):
                    picks.append(bucket[idx])
                    progressed = True
                    if len(picks) >= per_domain:
                        break
            if not progressed:
                break
            idx += 1
        selected.extend(picks)
    return selected


def _pair_key(task: dict) -> str:
    a = str(task.get("domain_a", "")).strip()
    b = str(task.get("domain_b", "")).strip()
    lo, hi = sorted([a, b])
    return f"{lo}+{hi}"


def sample_exp9(
    tasks: list[dict],
    per_pair: int,
    seed: int,
    hard_priority: bool = False,
    min_difficulty: str = "easy",
) -> list[dict]:
    rng = random.Random(seed)
    min_score = DIFFICULTY_SCORE[min_difficulty]
    fixed = [
        t for t in tasks
        if t.get("task_type") == "fixed" and bool(t.get("circularity_free", False))
    ]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for t in fixed:
        by_pair[_pair_key(t)].append(t)

    selected: list[dict] = []
    for pair in sorted(by_pair):
        full_pool = list(by_pair[pair])
        pool = [t for t in full_pool if _task_difficulty_min(t) >= min_score]
        if len(pool) < per_pair:
            # Fallback to full pair pool if strict filtering is too narrow.
            pool = full_pool
        pool = list(pool)
        rng.shuffle(pool)
        if hard_priority:
            pool.sort(key=_task_difficulty_sum, reverse=True)
        selected.extend(pool[:per_pair])
    return selected


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_packet(
    exp1_per_domain: int,
    exp9_per_pair: int,
    seed: int,
    exp1_hard_priority: bool = False,
    exp9_hard_priority: bool = False,
    exp1_min_difficulty: str = "easy",
    exp9_min_difficulty: str = "easy",
) -> dict:
    questions = load_jsonl(DATA_DIR / "questions.jsonl")
    exp9_tasks = load_jsonl(DATA_DIR / "exp9_tasks.jsonl")

    exp1 = sample_exp1(
        questions,
        exp1_per_domain,
        seed,
        hard_priority=exp1_hard_priority,
        min_difficulty=exp1_min_difficulty,
    )
    exp9 = sample_exp9(
        exp9_tasks,
        exp9_per_pair,
        seed,
        hard_priority=exp9_hard_priority,
        min_difficulty=exp9_min_difficulty,
    )

    exp1_tasks_rows: list[dict] = []
    exp1_response_rows: list[dict] = []
    exp1_key_rows: list[dict] = []
    for i, q in enumerate(exp1, start=1):
        item_id = f"HB_EXP1_{i:04d}"
        qid = str(q.get("question_id") or q.get("source_id") or f"q{i:04d}")
        base = {
            "item_id": item_id,
            "question_id": qid,
            "domain": q.get("domain", ""),
            "subcategory": q.get("subcategory", ""),
            "difficulty": q.get("difficulty", ""),
            "answer_type": q.get("answer_type", "short_text"),
            "question_text": q.get("question_text", ""),
        }
        exp1_tasks_rows.append(base)
        exp1_response_rows.append(
            {
                **base,
                "participant_answer": "",
                "participant_confidence": "",
                "notes": "",
            }
        )
        exp1_key_rows.append(
            {
                "item_id": item_id,
                "question_id": qid,
                "correct_answer": q.get("correct_answer", ""),
                "answer_type": q.get("answer_type", "short_text"),
                "domain": q.get("domain", ""),
            }
        )

    exp9_tasks_rows: list[dict] = []
    exp9_response_rows: list[dict] = []
    exp9_key_rows: list[dict] = []
    for i, t in enumerate(exp9, start=1):
        item_id = f"HB_EXP9_{i:04d}"
        base = {
            "item_id": item_id,
            "task_id": t.get("task_id", ""),
            "domain_pair": _pair_key(t),
            "domain_a": t.get("domain_a", ""),
            "domain_b": t.get("domain_b", ""),
            "subcategory_a": t.get("subcategory_a", ""),
            "subcategory_b": t.get("subcategory_b", ""),
            "difficulty_a": t.get("difficulty_a", ""),
            "difficulty_b": t.get("difficulty_b", ""),
            "task_text": t.get("task_text", ""),
            "part1_text": t.get("part1_text", ""),
            "part2_text": t.get("part2_text", ""),
            "decision_options": "PROCEED | USE_TOOL | FLAG_FOR_REVIEW",
        }
        exp9_tasks_rows.append(base)
        exp9_response_rows.append(
            {
                **base,
                "decision_a": "",
                "answer_a": "",
                "decision_b": "",
                "answer_b": "",
                "participant_confidence": "",
                "notes": "",
            }
        )
        exp9_key_rows.append(
            {
                "item_id": item_id,
                "task_id": t.get("task_id", ""),
                "domain_a": t.get("domain_a", ""),
                "domain_b": t.get("domain_b", ""),
                "correct_answer_a": t.get("correct_answer_a", ""),
                "correct_answer_b": t.get("correct_answer_b", ""),
                "answer_type_a": t.get("answer_type_a", "short_text"),
                "answer_type_b": t.get("answer_type_b", "short_text"),
            }
        )

    write_csv(
        OUT_DIR / "tasks" / "exp1_human_baseline_tasks.csv",
        [
            "item_id",
            "question_id",
            "domain",
            "subcategory",
            "difficulty",
            "answer_type",
            "question_text",
        ],
        exp1_tasks_rows,
    )
    write_csv(
        OUT_DIR / "tasks" / "exp9_human_baseline_tasks.csv",
        [
            "item_id",
            "task_id",
            "domain_pair",
            "domain_a",
            "domain_b",
            "subcategory_a",
            "subcategory_b",
            "difficulty_a",
            "difficulty_b",
            "task_text",
            "part1_text",
            "part2_text",
            "decision_options",
        ],
        exp9_tasks_rows,
    )
    write_csv(
        OUT_DIR / "templates" / "exp1_response_sheet.csv",
        [
            "item_id",
            "question_id",
            "domain",
            "subcategory",
            "difficulty",
            "answer_type",
            "question_text",
            "participant_answer",
            "participant_confidence",
            "notes",
        ],
        exp1_response_rows,
    )
    write_csv(
        OUT_DIR / "templates" / "exp9_response_sheet.csv",
        [
            "item_id",
            "task_id",
            "domain_pair",
            "domain_a",
            "domain_b",
            "subcategory_a",
            "subcategory_b",
            "difficulty_a",
            "difficulty_b",
            "task_text",
            "part1_text",
            "part2_text",
            "decision_options",
            "decision_a",
            "answer_a",
            "decision_b",
            "answer_b",
            "participant_confidence",
            "notes",
        ],
        exp9_response_rows,
    )
    write_csv(
        OUT_DIR / "answer_keys" / "exp1_answer_key.csv",
        ["item_id", "question_id", "correct_answer", "answer_type", "domain"],
        exp1_key_rows,
    )
    write_csv(
        OUT_DIR / "answer_keys" / "exp9_answer_key.csv",
        [
            "item_id",
            "task_id",
            "domain_a",
            "domain_b",
            "correct_answer_a",
            "correct_answer_b",
            "answer_type_a",
            "answer_type_b",
        ],
        exp9_key_rows,
    )

    manifest = {
        "seed": seed,
        "exp1_per_domain": exp1_per_domain,
        "exp9_per_pair": exp9_per_pair,
        "exp1_hard_priority": exp1_hard_priority,
        "exp9_hard_priority": exp9_hard_priority,
        "exp1_min_difficulty": exp1_min_difficulty,
        "exp9_min_difficulty": exp9_min_difficulty,
        "exp1_items_total": len(exp1_tasks_rows),
        "exp9_items_total": len(exp9_tasks_rows),
        "exp1_domains": sorted({r["domain"] for r in exp1_tasks_rows}),
        "exp9_domain_pairs": sorted({r["domain_pair"] for r in exp9_tasks_rows}),
    }
    (OUT_DIR / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIRROR human-baseline packet.")
    parser.add_argument("--exp1-per-domain", type=int, default=10)
    parser.add_argument("--exp9-per-pair", type=int, default=2)
    parser.add_argument("--exp1-hard-priority", action="store_true")
    parser.add_argument("--exp9-hard-priority", action="store_true")
    parser.add_argument("--exp1-min-difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--exp9-min-difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_packet(
        exp1_per_domain=args.exp1_per_domain,
        exp9_per_pair=args.exp9_per_pair,
        seed=args.seed,
        exp1_hard_priority=args.exp1_hard_priority,
        exp9_hard_priority=args.exp9_hard_priority,
        exp1_min_difficulty=args.exp1_min_difficulty,
        exp9_min_difficulty=args.exp9_min_difficulty,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Packet written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
