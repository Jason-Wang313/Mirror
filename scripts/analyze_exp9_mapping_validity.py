"""
Exp9 domain-component mapping validity summary (publication-ready).

Runs/reads verify_exp9_tasks outputs and emits compact reporting artifacts:
  - exp9_mapping_validity_summary.json
  - exp9_mapping_validity_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = ROOT / "scripts" / "verify_exp9_tasks.py"
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "results"
DEFAULT_CHECKPOINT_DIR = ROOT / "data" / "exp9_verification_runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_verifier(
    run_id: str,
    checkpoint_dir: Path,
    verify_with_api: bool,
    fixed_only: bool,
    max_workers: int,
    resume: bool,
) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--run-id",
        run_id,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--max-workers",
        str(max_workers),
    ]
    if fixed_only:
        cmd.append("--fixed-only")
    if verify_with_api:
        cmd.append("--api")
    if resume:
        cmd.append("--resume")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def make_markdown(path: Path, summary: dict) -> None:
    st = summary["static"]
    llm = summary.get("llm_verification", {})
    lines = [
        "# Exp9 Mapping Validity Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Report source: `{summary['report_path']}`",
        "",
        "## Static Verification",
        "",
        f"- Total tasks: {st['total_tasks']}",
        f"- Passed static checks: {st['passed_static']}",
        f"- Failed static checks: {st['failed_static']}",
        f"- Circularity-free tasks: {st['circularity_free_count']}",
        f"- Domain-slot coverage: {st['domain_component_counts']}",
        "",
        "## Cross-Verifier Results",
        "",
        f"- Cross-verifier executed: {summary['cross_verifier_executed']}",
        f"- Tasks verified: {llm.get('tasks_verified')}",
        f"- Flagged total: {llm.get('flagged_count')}",
        f"- Flagged part A: {llm.get('flagged_part_a')}",
        f"- Flagged part B: {llm.get('flagged_part_b')}",
        "",
        "## Publication Statement",
        "",
        summary["publication_statement"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp9 mapping-validity publication summary.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("exp9_mapping_validity_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--verify-with-api", action="store_true", default=False)
    parser.add_argument("--fixed-only", action="store_true", default=False)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--skip-verify-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress_log.jsonl"
    summary_json = run_dir / "exp9_mapping_validity_summary.json"
    summary_md = run_dir / "exp9_mapping_validity_summary.md"

    append_jsonl(progress_log, {"ts_utc": utc_now_iso(), "event": "start", "run_id": args.run_id})

    verifier_run_id = f"{args.run_id}_verify"
    if not args.skip_verify_run:
        code, out, err = run_verifier(
            run_id=verifier_run_id,
            checkpoint_dir=args.checkpoint_dir,
            verify_with_api=bool(args.verify_with_api),
            fixed_only=bool(args.fixed_only),
            max_workers=max(1, args.max_workers),
            resume=bool(args.resume),
        )
        append_jsonl(
            progress_log,
            {
                "ts_utc": utc_now_iso(),
                "event": "verify_run_complete",
                "returncode": code,
                "stdout_tail": out[-2000:],
                "stderr_tail": err[-2000:],
            },
        )
        if code != 0:
            raise RuntimeError(f"verify_exp9_tasks.py failed with code {code}\n{err}\n{out}")

    report_path = args.checkpoint_dir / verifier_run_id / "exp9_verification_report.json"
    if not report_path.exists():
        # fallback to stable latest alias
        fallback = ROOT / "data" / "exp9_verification_report.json"
        if fallback.exists():
            report_path = fallback
        else:
            raise FileNotFoundError(f"Missing verification report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    static = report.get("static", {})
    llm = report.get("llm_verification")

    passed = int(static.get("passed_static", 0))
    total = int(static.get("total_tasks", 0))
    flagged = int((llm or {}).get("flagged_count", 0))
    verified = int((llm or {}).get("tasks_verified", 0))
    static_pass_rate = (passed / total) if total else None
    flag_rate = (flagged / verified) if verified else None

    statement = (
        "Static schema/domain-component checks pass on the full fixed-task bank with "
        f"{passed}/{total} tasks passing static validation"
    )
    if llm:
        statement += (
            f"; cross-verifier auditing covers {verified} tasks with {flagged} flagged "
            f"({(100.0 * flag_rate):.2f}% of verified tasks)."
        )
    else:
        statement += "; cross-verifier stage was not executed in this run."

    summary = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "report_path": str(report_path),
        "cross_verifier_executed": llm is not None,
        "static": static,
        "llm_verification": llm or {},
        "metrics": {
            "static_pass_rate": static_pass_rate,
            "cross_verifier_flag_rate": flag_rate,
        },
        "publication_statement": statement,
    }
    write_json(summary_json, summary)
    make_markdown(summary_md, summary)
    append_jsonl(progress_log, {"ts_utc": utc_now_iso(), "event": "complete"})
    print(f"Done: {summary_json}")


if __name__ == "__main__":
    main()
