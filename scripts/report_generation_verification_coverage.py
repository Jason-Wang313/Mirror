"""Generate experiment-level data-generation / verification coverage summary.

The goal is to make labeling provenance explicit (human vs programmatic) with
deterministic references to available audit artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "runs"
DEFAULT_RESULTS_DIR = ROOT / "audit" / "human_baseline_packet" / "results"


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


def as_int(x: object, default: int = 0) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except Exception:
        return default


def as_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generation/verification coverage summary table.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("verification_coverage_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "generation_verification_coverage_summary.json"
    out_md = run_dir / "generation_verification_coverage_summary.md"

    multirater = read_json(args.results_dir / "human_audit_multirater_summary.json")
    mapping = read_json(args.results_dir / "exp9_mapping_validity_summary.json")
    human_ann = read_json(ROOT / "audit" / "human_annotations_report.json")

    n_raters = as_int(multirater.get("n_raters"), 0)
    per_exp = multirater.get("per_experiment", {}) if isinstance(multirater.get("per_experiment"), dict) else {}
    exp1_human = as_int((per_exp.get("exp1") or {}).get("n"), 0)  # type: ignore[index]
    exp9_human_multirater = as_int((per_exp.get("exp9") or {}).get("n"), 0)  # type: ignore[index]
    exp1_unanimous = as_float((per_exp.get("exp1") or {}).get("unanimous_rate"))  # type: ignore[index]
    exp9_unanimous = as_float((per_exp.get("exp9") or {}).get("unanimous_rate"))  # type: ignore[index]

    by_exp_ann = (human_ann.get("counts") or {}).get("by_experiment", {}) if isinstance(human_ann.get("counts"), dict) else {}
    exp4_human_ann = as_int(by_exp_ann.get("exp4"), 0)
    exp9_human_ann = as_int(by_exp_ann.get("exp9"), 0)

    exp9_static_tasks = as_int(((mapping.get("static") or {}).get("total_tasks")), 0)
    exp9_static_pass = as_int(((mapping.get("static") or {}).get("passed_static")), 0)

    rows = [
        {
            "experiment": "Exp1",
            "primary_label_source": "Programmatic scoring against fixed answer keys",
            "human_verified_n": exp1_human,
            "human_verification_depth": f"{n_raters}-rater audit, unanimous={exp1_unanimous:.3f}" if not math.isnan(exp1_unanimous) else f"{n_raters}-rater audit",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Human verification sourced from multi-rater audit summary.",
        },
        {
            "experiment": "Exp2",
            "primary_label_source": "Programmatic / deterministic pipeline outputs",
            "human_verified_n": 0,
            "human_verification_depth": "No dedicated human-audit artifact in this stage",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Bench-level QA remains in reproducibility scripts.",
        },
        {
            "experiment": "Exp3",
            "primary_label_source": "Programmatic / deterministic pipeline outputs",
            "human_verified_n": 0,
            "human_verification_depth": "No dedicated human-audit artifact in this stage",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Pair-balanced diagnostics and bootstrap checks are automated.",
        },
        {
            "experiment": "Exp4",
            "primary_label_source": "Hybrid (programmatic + targeted human annotation packet)",
            "human_verified_n": exp4_human_ann,
            "human_verification_depth": "Targeted annotation packet",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Counts sourced from human_annotations_report.json.",
        },
        {
            "experiment": "Exp5",
            "primary_label_source": "Programmatic / deterministic pipeline outputs",
            "human_verified_n": 0,
            "human_verification_depth": "No dedicated human-audit artifact in this stage",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Adversarial-condition checks are scripted.",
        },
        {
            "experiment": "Exp6",
            "primary_label_source": "Programmatic + curated adversarial metadata",
            "human_verified_n": 0,
            "human_verification_depth": "See manuscript appendix note for full-verification run details",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "This table reports machine-readable artifacts only.",
        },
        {
            "experiment": "Exp8",
            "primary_label_source": "Programmatic / deterministic pipeline outputs",
            "human_verified_n": 0,
            "human_verification_depth": "No dedicated human-audit artifact in this stage",
            "llm_labeled_n": 0,
            "static_schema_verified_n": 0,
            "notes": "Scaling analysis is computed from benchmark outputs.",
        },
        {
            "experiment": "Exp9",
            "primary_label_source": "Programmatic routing outcomes + static task verification",
            "human_verified_n": exp9_human_multirater + exp9_human_ann,
            "human_verification_depth": (
                f"{n_raters}-rater audit core={exp9_human_multirater}, targeted packet={exp9_human_ann}, "
                f"unanimous={exp9_unanimous:.3f}"
                if not math.isnan(exp9_unanimous)
                else f"{n_raters}-rater audit core={exp9_human_multirater}, targeted packet={exp9_human_ann}"
            ),
            "llm_labeled_n": 0,
            "static_schema_verified_n": exp9_static_pass,
            "notes": f"Static domain-component checks: {exp9_static_pass}/{exp9_static_tasks} passed.",
        },
    ]

    totals = {
        "human_verified_n_total": sum(int(r["human_verified_n"]) for r in rows),
        "static_schema_verified_n_total": sum(int(r["static_schema_verified_n"]) for r in rows),
        "n_experiments_reported": len(rows),
    }

    payload = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "artifact_inputs": {
            "multirater_summary": str(args.results_dir / "human_audit_multirater_summary.json"),
            "exp9_mapping_validity_summary": str(args.results_dir / "exp9_mapping_validity_summary.json"),
            "human_annotations_report": str(ROOT / "audit" / "human_annotations_report.json"),
        },
        "rows": rows,
        "totals": totals,
        "status": "complete",
    }
    write_json(out_json, payload)

    lines = [
        "# Data Generation and Verification Coverage",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Experiments reported: `{totals['n_experiments_reported']}`",
        f"- Human-verified rows/items (artifact-counted): `{totals['human_verified_n_total']}`",
        f"- Static schema-verified tasks (artifact-counted): `{totals['static_schema_verified_n_total']}`",
        "",
        "## Coverage Table",
        "",
        "| Experiment | Primary label source | Human verified (n) | Verification depth | Static schema verified (n) | Notes |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['primary_label_source']} | {r['human_verified_n']} | "
            f"{r['human_verification_depth']} | {r['static_schema_verified_n']} | {r['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This table is artifact-driven: it summarizes machine-readable evidence currently present in-repo.",
            "- Human vs programmatic coverage can be expanded by adding experiment-specific verification artifacts.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

