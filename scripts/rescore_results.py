"""
Re-score experiment results using the fixed answer matcher.

Usage:
    python scripts/rescore_results.py [results_jsonl_path]

Defaults to data/results/exp1_20260217T210412_results.jsonl.

For each record:
  1. Look up correct_answer and answer_type from data/questions.jsonl
  2. Re-run extract_answer_from_response + match_answer_robust with fixed logic
  3. Update answer_correct in place
  4. Write results back to the same file (atomically via temp file)

Prints a before/after summary of True/False/None counts and the flip breakdown.
"""

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.scoring.answer_matcher import (
    extract_answer_from_response,
    match_answer_robust,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = Path("data/results/exp1_20260217T210412_results.jsonl")
QUESTIONS_FILE = Path("data/questions.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> dict:
    """Return {source_id: {correct_answer, answer_type, ...}} mapping."""
    qmap = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            sid = q.get("source_id") or q.get("question_id")
            if sid:
                qmap[sid] = q
    return qmap


def rescore_record(record: dict, qmap: dict) -> tuple[dict, str]:
    """
    Re-score a single result record.

    Returns:
        (updated_record, flip_label)
        flip_label is one of: 'F→T', 'T→F', 'N→T', 'N→F', 'no_change', 'no_question'
    """
    qid = record.get("question_id")
    question = qmap.get(qid)

    if question is None:
        return record, "no_question"

    correct_answer = question.get("correct_answer")
    answer_type = question.get("answer_type", "short_text")

    if correct_answer is None:
        return record, "no_question"

    # Get the predicted answer from parsed field, or re-extract from raw_response
    parsed = record.get("parsed") or {}
    predicted = parsed.get("answer") or ""

    # If predicted is empty but raw_response exists, attempt extraction
    raw = record.get("raw_response") or ""
    if (not predicted or not str(predicted).strip()) and raw:
        predicted = extract_answer_from_response(raw, answer_type=answer_type) or ""

    # Skip records where the question was skipped (Channel 2 SKIP)
    if parsed.get("skipped"):
        new_correct = None
        flip = "no_change" if record.get("answer_correct") is None else (
            "T→N" if record["answer_correct"] is True else
            "F→N" if record["answer_correct"] is False else "no_change"
        )
        record = dict(record)
        record["answer_correct"] = new_correct
        return record, flip

    # Skip refused responses
    if parsed.get("refused"):
        old = record.get("answer_correct")
        record = dict(record)
        record["answer_correct"] = False
        if old is False:
            return record, "no_change"
        return record, f"{old}→F"

    # Re-run the fixed matcher
    try:
        new_correct = match_answer_robust(
            predicted=str(predicted).strip(),
            correct=str(correct_answer).strip(),
            answer_type=answer_type,
            metadata=question.get("metadata") or {},
        )
    except Exception as exc:
        print(f"  WARN: match_answer_robust failed for {qid}: {exc}")
        new_correct = None

    old_correct = record.get("answer_correct")

    if old_correct == new_correct:
        flip = "no_change"
    elif old_correct is False and new_correct is True:
        flip = "F→T"
    elif old_correct is True and new_correct is False:
        flip = "T→F"
    elif old_correct is None and new_correct is True:
        flip = "N→T"
    elif old_correct is None and new_correct is False:
        flip = "N→F"
    else:
        flip = f"{old_correct}→{new_correct}"

    record = dict(record)
    record["answer_correct"] = new_correct
    return record, flip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading questions from {QUESTIONS_FILE} ...")
    qmap = load_questions(QUESTIONS_FILE)
    print(f"  {len(qmap):,} questions loaded")

    print(f"\nRescoring {results_path} ...")
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records):,} records loaded")

    # Track before distribution
    before = Counter(r.get("answer_correct") for r in records)

    # Rescore all records
    flips = Counter()
    updated = []
    missing_qids = set()

    for record in records:
        new_record, flip = rescore_record(record, qmap)
        updated.append(new_record)
        flips[flip] += 1
        if flip == "no_question":
            missing_qids.add(record.get("question_id"))

    after = Counter(r.get("answer_correct") for r in updated)

    # Write back atomically
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=results_path.parent, prefix=".rescore_tmp_", suffix=".jsonl"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for record in updated:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, results_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Report
    print(f"\n{'─' * 55}")
    print(f"{'RESCORE SUMMARY':^55}")
    print(f"{'─' * 55}")
    print(f"  {'Status':<20} {'Before':>8} {'After':>8} {'Δ':>7}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*7}")
    for key in [True, False, None]:
        label = {True: "Correct", False: "Wrong", None: "No score"}[key]
        b = before.get(key, 0)
        a = after.get(key, 0)
        print(f"  {label:<20} {b:>8,} {a:>8,} {a-b:>+7,}")
    print(f"{'─' * 55}")

    total = len(records)
    acc_before = before.get(True, 0) / max(before.get(True, 0) + before.get(False, 0), 1)
    scored_after = after.get(True, 0) + after.get(False, 0)
    acc_after = after.get(True, 0) / max(scored_after, 1)
    print(f"\n  Overall accuracy: {acc_before:.1%} → {acc_after:.1%}")
    print(f"  (scored records: {before.get(True,0)+before.get(False,0):,} → {scored_after:,})")

    print(f"\n  Flip breakdown:")
    for flip_type, count in sorted(flips.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {flip_type:<15} {count:>7,}  ({pct:.1f}%)")

    if missing_qids:
        print(f"\n  WARNING: {len(missing_qids)} question IDs not found in questions.jsonl")
        for qid in sorted(missing_qids)[:5]:
            print(f"    {qid}")
        if len(missing_qids) > 5:
            print(f"    ... and {len(missing_qids)-5} more")

    print(f"\n  Written to: {results_path}")


if __name__ == "__main__":
    main()
