"""
Prepare a concrete human-baseline run packet for MIRROR v20.

Outputs (under audit/human_baseline_packet by default):
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

Hard-packet v2 additions:
  - Exp1 empirical-hardness mode from historical Exp1 results.
  - Exp9 bank growth to hit stable per-pair quotas (synthetic recomposition from
    verified fixed circularity-free components when needed).
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
DEFAULT_OUT_DIR = ROOT / "audit" / "human_baseline_packet"
DIFFICULTY_SCORE = {"easy": 0, "medium": 1, "hard": 2}
DOMAINS = ["arithmetic", "factual", "linguistic", "logical", "procedural", "social", "spatial", "temporal"]


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


def _derive_exp1_empirical_hardness(
    exp1_result_files: list[Path],
    channels: set[str],
) -> dict[str, dict]:
    """
    Build per-question empirical hardness from historical Exp1 runs.

    Hardness = 1 - empirical accuracy, computed over rows with non-null answer_correct.
    """
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "correct": 0})

    for fp in exp1_result_files:
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                channel_name = str(row.get("channel_name") or "").strip().lower()
                if channels and channel_name not in channels:
                    continue
                qid = str(row.get("question_id") or "").strip()
                if not qid:
                    continue
                answer_correct = row.get("answer_correct")
                if answer_correct is None:
                    continue
                stats[qid]["n"] += 1
                stats[qid]["correct"] += 1 if bool(answer_correct) else 0

    out: dict[str, dict] = {}
    for qid, s in stats.items():
        n = int(s["n"])
        if n <= 0:
            continue
        acc = float(s["correct"]) / float(n)
        out[qid] = {
            "n_obs": n,
            "empirical_acc": acc,
            "empirical_hardness": 1.0 - acc,
        }
    return out


def _attach_exp1_hardness(questions: list[dict], empirical: dict[str, dict]) -> list[dict]:
    enriched: list[dict] = []
    for q in questions:
        qid = str(q.get("question_id") or q.get("source_id") or "").strip()
        e = empirical.get(qid, {})
        hardness = e.get("empirical_hardness")
        if hardness is None:
            # Conservative fallback when historical hardness is unavailable.
            hardness = 0.5
        row = dict(q)
        row["empirical_hardness"] = float(hardness)
        row["empirical_hardness_obs"] = int(e.get("n_obs", 0))
        row["empirical_accuracy"] = e.get("empirical_acc")
        enriched.append(row)
    return enriched


def _hardest_quintile(pool: list[dict], quantile: float) -> list[dict]:
    if not pool:
        return []
    quantile = max(0.0, min(1.0, float(quantile)))
    scores = sorted(float(x.get("empirical_hardness", 0.5)) for x in pool)
    cutoff_idx = int((len(scores) - 1) * quantile)
    cutoff = scores[cutoff_idx]
    keep = [x for x in pool if float(x.get("empirical_hardness", 0.5)) >= cutoff]
    return keep if keep else pool


def sample_exp1(
    questions: list[dict],
    per_domain: int,
    seed: int,
    hard_priority: bool = False,
    min_difficulty: str = "easy",
    empirical_hardness_mode: bool = False,
    empirical_hardness_quantile: float = 0.80,
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

        if empirical_hardness_mode:
            hard_pool = _hardest_quintile(pool, empirical_hardness_quantile)
            if len(hard_pool) >= per_domain:
                pool = hard_pool
            else:
                pool = sorted(pool, key=lambda x: float(x.get("empirical_hardness", 0.5)), reverse=True)

        by_subcat: dict[str, list[dict]] = defaultdict(list)
        for q in pool:
            sub = str(q.get("subcategory", "unknown"))
            by_subcat[sub].append(q)
        subcats = sorted(by_subcat)
        for sub in subcats:
            by_subcat[sub] = _apply_difficulty_priority(by_subcat[sub], rng, hard_priority)
            if empirical_hardness_mode:
                by_subcat[sub].sort(key=lambda x: float(x.get("empirical_hardness", 0.5)), reverse=True)

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


def _extract_component(task: dict, slot: str) -> dict:
    return {
        "domain": str(task.get(f"domain_{slot}", "")).strip(),
        "subcategory": str(task.get(f"subcategory_{slot}", "")).strip(),
        "difficulty": str(task.get(f"difficulty_{slot}", "")).strip(),
        "text": str(task.get(f"part{1 if slot == 'a' else 2}_text", "")).strip(),
        "correct_answer": str(task.get(f"correct_answer_{slot}", "")).strip(),
        "answer_type": str(task.get(f"answer_type_{slot}", "short_text")).strip(),
    }


def _task_signature(task: dict) -> tuple[str, str, str, str]:
    return (
        str(task.get("domain_a", "")),
        str(task.get("domain_b", "")),
        str(task.get("part1_text", "")),
        str(task.get("part2_text", "")),
    )


def _build_augmented_exp9_bank(tasks: list[dict], per_pair: int, seed: int) -> tuple[list[dict], dict]:
    """
    Expand fixed circularity-free Exp9 bank to per-pair quota by deterministic
    component recomposition.
    """
    rng = random.Random(seed)

    fixed = [t for t in tasks if t.get("task_type") == "fixed" and bool(t.get("circularity_free", False))]

    all_pairs = []
    for i, d1 in enumerate(DOMAINS):
        for d2 in DOMAINS[i + 1:]:
            all_pairs.append(f"{d1}+{d2}")

    by_pair: dict[str, list[dict]] = defaultdict(list)
    for t in fixed:
        by_pair[_pair_key(t)].append(t)

    # Global domain component pools for missing-pair synthesis.
    comps_by_domain: dict[str, list[dict]] = defaultdict(list)
    for t in fixed:
        for slot in ("a", "b"):
            c = _extract_component(t, slot)
            if c["domain"] and c["text"]:
                comps_by_domain[c["domain"]].append(c)

    out_bank: list[dict] = []
    augmentation_counts: dict[str, int] = {}

    for pair in sorted(all_pairs):
        d1, d2 = pair.split("+", 1)
        base = list(by_pair.get(pair, []))
        seen = {_task_signature(t) for t in base}
        generated: list[dict] = []

        pool_a = [_extract_component(t, "a") for t in base if str(t.get("domain_a", "")).strip() == d1]
        pool_b = [_extract_component(t, "b") for t in base if str(t.get("domain_b", "")).strip() == d2]
        if not pool_a:
            pool_a = [_extract_component(t, "b") for t in base if str(t.get("domain_b", "")).strip() == d1]
        if not pool_b:
            pool_b = [_extract_component(t, "a") for t in base if str(t.get("domain_a", "")).strip() == d2]

        if not pool_a:
            pool_a = comps_by_domain.get(d1, [])
        if not pool_b:
            pool_b = comps_by_domain.get(d2, [])

        candidate_pairs = [(a, b) for a in pool_a for b in pool_b if a["text"] and b["text"]]
        rng.shuffle(candidate_pairs)

        needed = max(0, per_pair - len(base))
        aug_idx = 0
        for comp_a, comp_b in candidate_pairs:
            if needed <= 0:
                break
            task = {
                "task_id": f"fixed_{d1[:3]}_{d2[:3]}_aug_{aug_idx:03d}",
                "task_type": "fixed",
                "circularity_free": True,
                "target_model": None,
                "domain_a": d1,
                "domain_b": d2,
                "subcategory_a": comp_a["subcategory"] or "mixed",
                "subcategory_b": comp_b["subcategory"] or "mixed",
                "difficulty_a": comp_a["difficulty"] or "medium",
                "difficulty_b": comp_b["difficulty"] or "medium",
                "task_text": f"Composite fixed-task augmentation for {d1}+{d2} (variant {aug_idx + 1}).",
                "part1_text": comp_a["text"],
                "part2_text": comp_b["text"],
                "correct_answer_a": comp_a["correct_answer"],
                "correct_answer_b": comp_b["correct_answer"],
                "answer_type_a": comp_a["answer_type"] or "short_text",
                "answer_type_b": comp_b["answer_type"] or "short_text",
                "augmentation_type": "component_recomposition",
                "augmentation_parent_pair": pair,
            }
            sig = _task_signature(task)
            aug_idx += 1
            if sig in seen:
                continue
            generated.append(task)
            seen.add(sig)
            needed -= 1

        merged = base + generated
        if len(merged) < per_pair:
            # Last resort: deterministic repeats (new IDs) to satisfy strict quotas.
            i = 0
            while len(merged) < per_pair and merged:
                src = dict(merged[i % len(merged)])
                src["task_id"] = f"{src.get('task_id', 'fixed')}_rep_{i:03d}"
                src["augmentation_type"] = src.get("augmentation_type", "deterministic_repeat")
                merged.append(src)
                i += 1

        out_bank.extend(merged)
        augmentation_counts[pair] = max(0, len(merged) - len(base))

    summary = {
        "pairs_total": len(all_pairs),
        "target_per_pair": per_pair,
        "total_rows": len(out_bank),
        "augmentation_counts_by_pair": augmentation_counts,
        "n_pairs_augmented": sum(1 for v in augmentation_counts.values() if v > 0),
    }
    return out_bank, summary


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
    out_dir: Path,
    exp1_per_domain: int,
    exp9_per_pair: int,
    seed: int,
    exp1_hard_priority: bool = False,
    exp9_hard_priority: bool = False,
    exp1_min_difficulty: str = "easy",
    exp9_min_difficulty: str = "easy",
    exp1_empirical_hardness_mode: bool = False,
    exp1_empirical_hardness_quantile: float = 0.80,
    exp1_hardness_result_files: list[Path] | None = None,
    exp1_hardness_channels: set[str] | None = None,
    exp9_expand_bank: bool = False,
) -> dict:
    questions = load_jsonl(DATA_DIR / "questions.jsonl")
    exp9_tasks = load_jsonl(DATA_DIR / "exp9_tasks.jsonl")

    empirical_summary: dict | None = None
    if exp1_empirical_hardness_mode:
        result_files = exp1_hardness_result_files or sorted((ROOT / "data" / "results").glob("exp1_*_results.jsonl"))
        channels = exp1_hardness_channels or {"natural"}
        empirical = _derive_exp1_empirical_hardness(result_files, channels=channels)
        questions = _attach_exp1_hardness(questions, empirical)
        empirical_summary = {
            "enabled": True,
            "result_files_used": [str(p) for p in result_files if p.exists()],
            "channels": sorted(channels),
            "n_questions_with_empirical_hardness": len(empirical),
            "hardness_quantile": exp1_empirical_hardness_quantile,
        }

    bank_summary: dict | None = None
    if exp9_expand_bank:
        exp9_tasks, bank_summary = _build_augmented_exp9_bank(exp9_tasks, per_pair=exp9_per_pair, seed=seed)

    exp1 = sample_exp1(
        questions,
        exp1_per_domain,
        seed,
        hard_priority=exp1_hard_priority,
        min_difficulty=exp1_min_difficulty,
        empirical_hardness_mode=exp1_empirical_hardness_mode,
        empirical_hardness_quantile=exp1_empirical_hardness_quantile,
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
            "empirical_hardness": q.get("empirical_hardness", ""),
            "empirical_hardness_obs": q.get("empirical_hardness_obs", ""),
            "empirical_accuracy": q.get("empirical_accuracy", ""),
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
            "augmentation_type": t.get("augmentation_type", ""),
            "augmentation_parent_pair": t.get("augmentation_parent_pair", ""),
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
        out_dir / "tasks" / "exp1_human_baseline_tasks.csv",
        [
            "item_id",
            "question_id",
            "domain",
            "subcategory",
            "difficulty",
            "empirical_hardness",
            "empirical_hardness_obs",
            "empirical_accuracy",
            "answer_type",
            "question_text",
        ],
        exp1_tasks_rows,
    )
    write_csv(
        out_dir / "tasks" / "exp9_human_baseline_tasks.csv",
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
            "augmentation_type",
            "augmentation_parent_pair",
        ],
        exp9_tasks_rows,
    )
    write_csv(
        out_dir / "templates" / "exp1_response_sheet.csv",
        [
            "item_id",
            "question_id",
            "domain",
            "subcategory",
            "difficulty",
            "empirical_hardness",
            "empirical_hardness_obs",
            "empirical_accuracy",
            "answer_type",
            "question_text",
            "participant_answer",
            "participant_confidence",
            "notes",
        ],
        exp1_response_rows,
    )
    write_csv(
        out_dir / "templates" / "exp9_response_sheet.csv",
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
            "augmentation_type",
            "augmentation_parent_pair",
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
        out_dir / "answer_keys" / "exp1_answer_key.csv",
        ["item_id", "question_id", "correct_answer", "answer_type", "domain"],
        exp1_key_rows,
    )
    write_csv(
        out_dir / "answer_keys" / "exp9_answer_key.csv",
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
        "exp1_empirical_hardness": empirical_summary,
        "exp9_bank_expansion": bank_summary,
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIRROR human-baseline packet.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exp1-per-domain", type=int, default=10)
    parser.add_argument("--exp9-per-pair", type=int, default=2)
    parser.add_argument("--exp1-hard-priority", action="store_true")
    parser.add_argument("--exp9-hard-priority", action="store_true")
    parser.add_argument("--exp1-min-difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--exp9-min-difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--exp1-empirical-hardness-mode", action="store_true")
    parser.add_argument("--exp1-hardness-quantile", type=float, default=0.80)
    parser.add_argument("--exp1-hardness-files", nargs="+", type=Path, default=None)
    parser.add_argument("--exp1-hardness-channels", nargs="+", type=str, default=["natural"])
    parser.add_argument("--exp9-expand-bank", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_packet(
        out_dir=args.out_dir,
        exp1_per_domain=args.exp1_per_domain,
        exp9_per_pair=args.exp9_per_pair,
        seed=args.seed,
        exp1_hard_priority=args.exp1_hard_priority,
        exp9_hard_priority=args.exp9_hard_priority,
        exp1_min_difficulty=args.exp1_min_difficulty,
        exp9_min_difficulty=args.exp9_min_difficulty,
        exp1_empirical_hardness_mode=args.exp1_empirical_hardness_mode,
        exp1_empirical_hardness_quantile=args.exp1_hardness_quantile,
        exp1_hardness_result_files=args.exp1_hardness_files,
        exp1_hardness_channels={str(x).strip().lower() for x in args.exp1_hardness_channels if str(x).strip()},
        exp9_expand_bank=bool(args.exp9_expand_bank),
    )
    print(json.dumps(manifest, indent=2))
    print(f"Packet written to: {args.out_dir}")


if __name__ == "__main__":
    main()
