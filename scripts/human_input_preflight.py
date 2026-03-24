"""
Validate locked human inputs and generate a reproducibility manifest.

This script serves two roles:
1) Preflight + lock: validate human files, compute hashes, and write a run manifest.
2) End-run verification: verify immutable/locked hashes against a prior manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = ROOT / "audit" / "human_baseline_packet"
RUNS_DIR = PACKET_DIR / "runs"

VALID_DECISIONS = {"PROCEED", "USE_TOOL", "FLAG_FOR_REVIEW"}
VALID_LABELS = {"correct", "incorrect", "ambiguous"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    details: dict


def validate_exp1(path: Path, expected_rows: int) -> ValidationResult:
    rows = read_csv(path)
    missing_answer = sum(1 for r in rows if not (r.get("participant_answer") or "").strip())
    missing_conf = sum(1 for r in rows if not (r.get("participant_confidence") or "").strip())
    row_count_ok = len(rows) == expected_rows
    ok = row_count_ok and missing_answer == 0 and missing_conf == 0
    reason = "ok" if ok else "exp1 validation failed"
    return ValidationResult(
        ok=ok,
        reason=reason,
        details={
            "row_count": len(rows),
            "expected_rows": expected_rows,
            "row_count_ok": row_count_ok,
            "blank_participant_answer": missing_answer,
            "blank_participant_confidence": missing_conf,
        },
    )


def validate_exp9(path: Path, expected_rows: int) -> ValidationResult:
    rows = read_csv(path)
    missing_a = sum(1 for r in rows if not (r.get("decision_a") or "").strip())
    missing_b = sum(1 for r in rows if not (r.get("decision_b") or "").strip())
    invalid_a = sum(1 for r in rows if (r.get("decision_a") or "").strip() not in VALID_DECISIONS)
    invalid_b = sum(1 for r in rows if (r.get("decision_b") or "").strip() not in VALID_DECISIONS)
    row_count_ok = len(rows) == expected_rows
    ok = row_count_ok and missing_a == 0 and missing_b == 0 and invalid_a == 0 and invalid_b == 0
    reason = "ok" if ok else "exp9 validation failed"
    return ValidationResult(
        ok=ok,
        reason=reason,
        details={
            "row_count": len(rows),
            "expected_rows": expected_rows,
            "row_count_ok": row_count_ok,
            "blank_decision_a": missing_a,
            "blank_decision_b": missing_b,
            "invalid_decision_a": invalid_a,
            "invalid_decision_b": invalid_b,
            "valid_decisions": sorted(VALID_DECISIONS),
        },
    )


def validate_audit(path: Path, expected_rows: int) -> ValidationResult:
    rows = read_csv(path)
    missing_label = sum(1 for r in rows if not (r.get("human_label") or "").strip())
    invalid_label = sum(
        1 for r in rows if (r.get("human_label") or "").strip().lower() not in VALID_LABELS
    )
    missing_conf = sum(1 for r in rows if not (r.get("confidence") or "").strip())
    row_count_ok = len(rows) == expected_rows
    ok = row_count_ok and missing_label == 0 and missing_conf == 0 and invalid_label == 0
    reason = "ok" if ok else "audit validation failed"
    return ValidationResult(
        ok=ok,
        reason=reason,
        details={
            "row_count": len(rows),
            "expected_rows": expected_rows,
            "row_count_ok": row_count_ok,
            "blank_human_label": missing_label,
            "blank_confidence": missing_conf,
            "invalid_human_label": invalid_label,
            "valid_labels": sorted(VALID_LABELS),
        },
    )


def file_entry(path: Path, role: str, validation: ValidationResult | None = None) -> dict:
    stat = path.stat()
    entry = {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        "sha256": sha256_file(path),
    }
    if validation is not None:
        entry["validation"] = {
            "ok": validation.ok,
            "reason": validation.reason,
            "details": validation.details,
        }
    return entry


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()


def default_paths(kind: str) -> list[Path]:
    if kind == "exp1":
        return [
            PACKET_DIR / "templates" / "exp1_response_sheet_P1.csv",
            PACKET_DIR / "templates" / "exp1_response_sheet_P2.csv",
            PACKET_DIR / "templates" / "exp1_response_sheet_P3.csv",
        ]
    if kind == "exp9":
        return [
            PACKET_DIR / "templates" / "exp9_response_sheet_P1.csv",
            PACKET_DIR / "templates" / "exp9_response_sheet_P2.csv",
            PACKET_DIR / "templates" / "exp9_response_sheet_P3.csv",
        ]
    if kind == "audit":
        return [
            ROOT / "audit" / "real_human_audit_100_R1.csv",
            ROOT / "audit" / "real_human_audit_100_R2.csv",
            ROOT / "audit" / "real_human_audit_100_R3.csv",
        ]
    raise ValueError(f"Unknown kind: {kind}")


def verify_against_manifest(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []

    immutable = manifest.get("immutable_files", [])
    for item in immutable:
        p = Path(item["path"])
        current_hash = sha256_file(p)
        checks.append(
            {
                "path": str(p),
                "expected_sha256": item["sha256"],
                "actual_sha256": current_hash,
                "ok": current_hash == item["sha256"],
            }
        )

    locked_inputs = manifest.get("locked_inputs", [])
    for item in locked_inputs:
        p = Path(item["path"])
        current_hash = sha256_file(p)
        checks.append(
            {
                "path": str(p),
                "expected_sha256": item["sha256"],
                "actual_sha256": current_hash,
                "ok": current_hash == item["sha256"],
            }
        )

    all_ok = all(c["ok"] for c in checks)
    out = {
        "verified_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path.resolve()),
        "all_ok": all_ok,
        "checks": checks,
    }
    out_path = manifest_path.with_name("human_input_verification.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote verification: {out_path}")
    if not all_ok:
        print("Verification failed: one or more hashes changed.")
        return 1
    print("Verification passed: all immutable and locked input hashes match.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and lock human inputs for MIRROR v20 hardening runs.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("v20_hardening_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--exp1-files", nargs="+", type=Path, default=default_paths("exp1"))
    parser.add_argument("--exp9-files", nargs="+", type=Path, default=default_paths("exp9"))
    parser.add_argument("--audit-files", nargs="+", type=Path, default=default_paths("audit"))
    parser.add_argument(
        "--immutable-files",
        nargs="+",
        type=Path,
        default=[
            PACKET_DIR / "answer_keys" / "exp1_answer_key.csv",
            PACKET_DIR / "answer_keys" / "exp9_answer_key.csv",
        ],
    )
    parser.add_argument("--exp1-rows", type=int, default=192)
    parser.add_argument("--exp9-rows", type=int, default=162)
    parser.add_argument("--audit-rows", type=int, default=100)
    parser.add_argument("--verify-against-manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_against_manifest is not None:
        raise SystemExit(verify_against_manifest(args.verify_against_manifest))

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "progress_log.jsonl"

    for p in [*args.exp1_files, *args.exp9_files, *args.audit_files, *args.immutable_files]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    locked_inputs: list[dict] = []
    validation_failed = []

    for p in args.exp1_files:
        v = validate_exp1(p, args.exp1_rows)
        entry = file_entry(p, role="exp1_participant", validation=v)
        locked_inputs.append(entry)
        if not v.ok:
            validation_failed.append(entry)

    for p in args.exp9_files:
        v = validate_exp9(p, args.exp9_rows)
        entry = file_entry(p, role="exp9_participant", validation=v)
        locked_inputs.append(entry)
        if not v.ok:
            validation_failed.append(entry)

    for p in args.audit_files:
        v = validate_audit(p, args.audit_rows)
        entry = file_entry(p, role="audit_rater", validation=v)
        locked_inputs.append(entry)
        if not v.ok:
            validation_failed.append(entry)

    immutable_files = [file_entry(p, role="immutable_answer_key") for p in args.immutable_files]

    manifest = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "policy": {
            "human_data_legitimacy": "accepted_by_user",
            "exp1_expected_rows": args.exp1_rows,
            "exp9_expected_rows": args.exp9_rows,
            "audit_expected_rows": args.audit_rows,
            "valid_exp9_decisions": sorted(VALID_DECISIONS),
            "valid_human_labels": sorted(VALID_LABELS),
        },
        "status": "passed" if not validation_failed else "failed",
        "locked_inputs": locked_inputs,
        "immutable_files": immutable_files,
    }

    manifest_path = run_dir / "human_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_jsonl(
        log_path,
        {
            "ts_utc": utc_now_iso(),
            "event": "preflight_complete",
            "status": manifest["status"],
            "manifest_path": str(manifest_path),
            "n_locked_inputs": len(locked_inputs),
            "n_immutable": len(immutable_files),
        },
    )

    print(f"Wrote manifest: {manifest_path}")
    if validation_failed:
        print(f"Preflight failed: {len(validation_failed)} files failed validation.")
        raise SystemExit(1)
    print("Preflight passed.")


if __name__ == "__main__":
    main()
