"""Validate and score hard human packet v2 cohort (Exp1/Exp9).

Outputs a deterministic cohort status summary and, when complete, runs the
standard human-baseline scorer with hard-v2 answer keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = ROOT / "audit" / "human_baseline_packet"
HARD_V2_DIR = PACKET_DIR / "hard_v2"
DEFAULT_OUT_ROOT = PACKET_DIR / "runs"
ALLOWED_DECISIONS = {"PROCEED", "USE_TOOL", "FLAG_FOR_REVIEW"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def participant_id_from_path(path: Path) -> str:
    stem = path.stem
    if "_H" in stem:
        return "H" + stem.rsplit("_H", 1)[1]
    return stem


def validate_exp1_file(path: Path, expected_rows: int) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    blank_answer = 0
    blank_conf = 0
    for row in rows:
        if not (row.get("participant_answer") or "").strip():
            blank_answer += 1
        if not (row.get("participant_confidence") or "").strip():
            blank_conf += 1
    return {
        "path": str(path),
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "row_count_ok": len(rows) == expected_rows,
        "blank_participant_answer": blank_answer,
        "blank_participant_confidence": blank_conf,
        "status_ok": len(rows) == expected_rows and blank_answer == 0 and blank_conf == 0,
    }


def validate_exp9_file(path: Path, expected_rows: int) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    blank_a = 0
    blank_b = 0
    invalid_a = 0
    invalid_b = 0
    for row in rows:
        da = (row.get("decision_a") or "").strip().upper()
        db = (row.get("decision_b") or "").strip().upper()
        if not da:
            blank_a += 1
        elif da not in ALLOWED_DECISIONS:
            invalid_a += 1
        if not db:
            blank_b += 1
        elif db not in ALLOWED_DECISIONS:
            invalid_b += 1
    return {
        "path": str(path),
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "row_count_ok": len(rows) == expected_rows,
        "blank_decision_a": blank_a,
        "blank_decision_b": blank_b,
        "invalid_decision_a": invalid_a,
        "invalid_decision_b": invalid_b,
        "status_ok": len(rows) == expected_rows and blank_a == 0 and blank_b == 0 and invalid_a == 0 and invalid_b == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and score hard-v2 cohort sheets.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("hard_v2_cohort_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--hard-v2-dir", type=Path, default=HARD_V2_DIR)
    parser.add_argument("--cohort-dir", type=Path, default=None)
    parser.add_argument("--min-participants", type=int, default=20)
    parser.add_argument("--summary-stem", type=str, default="human_baseline_hardv2_cohort_summary")
    parser.add_argument("--cohort-label", type=str, default="Human Baseline Hard-v2 Cohort")
    args = parser.parse_args()

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "human_baseline_hardv2_cohort_summary.json"
    out_md = run_dir / "human_baseline_hardv2_cohort_summary.md"

    hard_v2_dir = args.hard_v2_dir
    cohort_dir = args.cohort_dir if args.cohort_dir is not None else (hard_v2_dir / "cohort")
    manifest = read_json(hard_v2_dir / "packet_manifest.json")
    exp1_expected = int(manifest.get("exp1_items_total", 512))
    exp9_expected = int(manifest.get("exp9_items_total", 336))

    exp1_files = sorted(cohort_dir.glob("exp1_response_sheet_H*.csv"), key=lambda p: p.name.lower())
    exp9_files = sorted(cohort_dir.glob("exp9_response_sheet_H*.csv"), key=lambda p: p.name.lower())
    exp1_map = {participant_id_from_path(p): p for p in exp1_files}
    exp9_map = {participant_id_from_path(p): p for p in exp9_files}
    participant_ids = sorted(set(exp1_map).intersection(exp9_map))

    validation = {"exp1": [], "exp9": []}
    for pid in participant_ids:
        validation["exp1"].append({"participant_id": pid, **validate_exp1_file(exp1_map[pid], exp1_expected)})
        validation["exp9"].append({"participant_id": pid, **validate_exp9_file(exp9_map[pid], exp9_expected)})

    n_valid_pairs = len(participant_ids)
    n_validated_ok = sum(
        1
        for e1, e9 in zip(validation["exp1"], validation["exp9"])
        if bool(e1.get("status_ok")) and bool(e9.get("status_ok"))
    )
    meets_min = n_valid_pairs >= int(args.min_participants) and n_validated_ok >= int(args.min_participants)

    summary = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "hard_v2_dir": str(hard_v2_dir),
        "cohort_dir": str(cohort_dir),
        "min_participants_required": int(args.min_participants),
        "expected_rows": {"exp1": exp1_expected, "exp9": exp9_expected},
        "counts": {
            "exp1_files_found": len(exp1_files),
            "exp9_files_found": len(exp9_files),
            "participant_pairs_found": n_valid_pairs,
            "participant_pairs_validated_ok": n_validated_ok,
        },
        "participant_ids": participant_ids,
        "validation": validation,
        "status": "pending_cohort_completion",
        "scoring": {},
    }

    if meets_min:
        score_out_dir = run_dir / "scored"
        score_out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_human_baseline_packet.py"),
            "--exp1-responses",
            *[str(exp1_map[pid]) for pid in participant_ids],
            "--exp9-responses",
            *[str(exp9_map[pid]) for pid in participant_ids],
            "--participant-ids",
            *participant_ids,
            "--exp1-key",
            str(hard_v2_dir / "answer_keys" / "exp1_answer_key.csv"),
            "--exp9-key",
            str(hard_v2_dir / "answer_keys" / "exp9_answer_key.csv"),
            "--out-dir",
            str(score_out_dir),
            "--summary-stem",
            args.summary_stem,
            "--cohort-label",
            args.cohort_label,
            "--weak-domain-rule",
            "median_or_bottom_k",
            "--fallback-bottom-k",
            "2",
            "--primary-aggregation",
            "participant_mean",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        summary["scoring"] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "score_out_dir": str(score_out_dir),
            "score_summary_json": str(score_out_dir / f"{args.summary_stem}.json"),
            "score_summary_md": str(score_out_dir / f"{args.summary_stem}.md"),
        }
        if proc.returncode == 0:
            summary["status"] = "complete"
        else:
            summary["status"] = "scoring_failed"

    write_json(out_json, summary)

    lines = [
        "# Hard-v2 Cohort Status",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Cohort dir: `{cohort_dir}`",
        f"- Min participants required: `{args.min_participants}`",
        f"- Participant pairs found: `{n_valid_pairs}`",
        f"- Participant pairs fully valid: `{n_validated_ok}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Validation",
        "",
        f"- Exp1 expected rows per participant: `{exp1_expected}`",
        f"- Exp9 expected rows per participant: `{exp9_expected}`",
        "",
    ]
    if summary["status"] == "pending_cohort_completion":
        lines.extend(
            [
                "## Action Needed",
                "",
                f"- Place complete `Hxx` hard-v2 sheets in `{cohort_dir}` with both Exp1 and Exp9 files.",
                f"- Required valid participant pairs: `{args.min_participants}`.",
                "",
            ]
        )
    elif summary["status"] == "complete":
        lines.extend(
            [
                "## Scoring",
                "",
                f"- Scoring output dir: `{summary['scoring'].get('score_out_dir')}`",
                f"- Summary JSON: `{summary['scoring'].get('score_summary_json')}`",
                f"- Summary MD: `{summary['scoring'].get('score_summary_md')}`",
                "",
            ]
        )
    elif summary["status"] == "scoring_failed":
        lines.extend(
            [
                "## Scoring Failure",
                "",
                f"- Return code: `{summary['scoring'].get('returncode')}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

