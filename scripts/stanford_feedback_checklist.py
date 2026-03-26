"""
Stanford-feedback closure checklist for MIRROR v20.

This is a deterministic gate used in the hardening loop to flag unresolved
non-production-log weaknesses before claiming closure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_TEX = ROOT / "paper" / "mirror_draft_v20.tex"
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "results"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def contains_all(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return all(n.lower() in t for n in needles)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stanford-feedback checklist gate.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("stanford_gate_%Y%m%dT%H%M%S"))
    parser.add_argument("--paper-tex", type=Path, default=PAPER_TEX)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stanford_feedback_checklist.json"
    out_md = out_dir / "stanford_feedback_checklist.md"

    tex = args.paper_tex.read_text(encoding="utf-8") if args.paper_tex.exists() else ""

    checks = [
        {
            "id": "parse_missingness",
            "description": "Parse/API missingness analysis is reported in manuscript.",
            "pass": contains_all(tex, ["missingness", "mcar", "mnar"]),
            "evidence_hint": "Look for explicit MCAR/MNAR and sensitivity text.",
        },
        {
            "id": "mnar_bounds_reporting",
            "description": "Explicit MNAR bound scenarios are reported for headline metrics.",
            "pass": contains_all(tex, ["mnar", "bound", "c1", "c4"]),
            "evidence_hint": "Look for worst-case MNAR bounds tied to C1→C4 effects.",
        },
        {
            "id": "exp3_difficulty_control",
            "description": "Exp3 difficulty-controlled ablation is reported.",
            "pass": contains_all(tex, ["strong--strong", "strong--weak", "weak--weak", "ablation"]),
            "evidence_hint": "Look for controlled mixture reporting in Exp3 discussion/appendix.",
        },
        {
            "id": "cce_main_text_formula_example",
            "description": "CCE is formalized in main text with a worked example.",
            "pass": contains_all(tex, ["worked example", "cce", "equation"]),
            "evidence_hint": "Look for CCE equation and worked numeric interpretation in main Findings text.",
        },
        {
            "id": "mci_channel_robustness",
            "description": "MCI channel-disagreement diagnostics are reported.",
            "pass": contains_all(tex, ["leave-one", "no-wagering", "mci"]),
            "evidence_hint": "Look for no-wagering / leave-one-channel sensitivity text.",
        },
        {
            "id": "non_oracle_utility_main_text",
            "description": "Non-oracle/fallible-resolver utility appears in main claim path.",
            "pass": contains_all(tex, ["fallible", "resolver", "system success"]),
            "evidence_hint": "Look for explicit resolver-quality sensitivity in main text.",
        },
        {
            "id": "utility_frontier_prominence",
            "description": "Cost/latency-aware Pareto or frontier reporting appears in main text path.",
            "pass": contains_all(tex, ["pareto", "cost-aware", "latency"]),
            "evidence_hint": "Look for a cost/latency-aware frontier summary tied to C1/C4 utility.",
        },
        {
            "id": "instance_baseline_expanded",
            "description": "Instance-level baselines are discussed beyond narrow legacy frame.",
            "pass": contains_all(tex, ["instance-level", "confidence-threshold", "self-consistency", "conformal"]),
            "evidence_hint": "Look for expanded frame details near Exp9 comparisons.",
        },
        {
            "id": "instance_baseline_robust_set",
            "description": "Robust baseline set includes calibrated confidence and target-error grid framing.",
            "pass": contains_all(tex, ["transformed-platt", "target-error", "risk-coverage"]),
            "evidence_hint": "Look for calibrated-confidence and conformal target-error grid reporting.",
        },
        {
            "id": "mapping_validity",
            "description": "Exp9 domain-component mapping validation is reported.",
            "pass": contains_all(tex, ["domain-component", "verification", "exp9"]),
            "evidence_hint": "Look for static/cross-verifier mapping-validation summary.",
        },
        {
            "id": "related_work_expanded",
            "description": "Recent metacognitive-control/abstention related work is integrated in main text.",
            "pass": contains_all(tex, ["monitor-generate-verify"]) or contains_all(tex, ["medcog"]) or contains_all(tex, ["metaclass"]),
            "evidence_hint": "Look for added related-work anchors in main section.",
        },
    ]

    failed = [c for c in checks if not c["pass"]]
    summary = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "paper_tex": str(args.paper_tex),
        "n_checks": len(checks),
        "n_pass": len(checks) - len(failed),
        "n_fail": len(failed),
        "checks": checks,
        "actionable_items": [
            {
                "id": c["id"],
                "description": c["description"],
                "hint": c["evidence_hint"],
            }
            for c in failed
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stanford Feedback Checklist",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Pass: {summary['n_pass']}/{summary['n_checks']}",
        f"- Fail: {summary['n_fail']}",
        "",
        "## Checks",
        "",
        "| ID | Pass | Description |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| `{c['id']}` | {'yes' if c['pass'] else 'no'} | {c['description']} |")
    if failed:
        lines.extend(["", "## Actionable Items", ""])
        for c in failed:
            lines.append(f"- `{c['id']}`: {c['description']} ({c['evidence_hint']})")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Done: {out_json}")


if __name__ == "__main__":
    main()
