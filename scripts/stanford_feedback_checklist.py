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
DEFAULT_RESULTS_DIR = ROOT / "audit" / "human_baseline_packet" / "results"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def contains_all(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return all(n.lower() in t for n in needles)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stanford-feedback checklist gate.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("stanford_gate_%Y%m%dT%H%M%S"))
    parser.add_argument("--paper-tex", type=Path, default=PAPER_TEX)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "stanford_feedback_checklist.json"
    out_md = out_dir / "stanford_feedback_checklist.md"

    tex = args.paper_tex.read_text(encoding="utf-8") if args.paper_tex.exists() else ""
    frontier_md = (args.results_dir / "exp9_policy_frontier_summary.md").read_text(encoding="utf-8", errors="ignore") if (args.results_dir / "exp9_policy_frontier_summary.md").exists() else ""
    hard_v2_md = (args.results_dir / "human_baseline_hardv2_summary.md").read_text(encoding="utf-8", errors="ignore") if (args.results_dir / "human_baseline_hardv2_summary.md").exists() else ""
    ood_json = read_json(args.results_dir / "exp9_ood_holdout_summary.json")
    mech_json = read_json(args.results_dir / "mechanistic_probe_summary.json")
    coverage_json = read_json(args.results_dir / "generation_verification_coverage_summary.json")
    hard_v2_cohort_json = read_json(args.results_dir / "human_baseline_hardv2_cohort_summary.json")

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
        {
            "id": "baseline_main_text_prominence",
            "description": "Main text explicitly foregrounds budget-matched and frontier baseline comparisons.",
            "pass": contains_all(tex, ["budget-matched", "risk-coverage", "instance-level"]),
            "evidence_hint": "Look for compact main-text baseline summary (not appendix-only framing).",
        },
        {
            "id": "weak_domain_frontier_reporting",
            "description": "Weak-domain policy-family frontier (median/bottom-k/absolute/quantile) is reported.",
            "pass": contains_all(frontier_md, ["median_or_bottom_k", "matched_escalation", "matched_autonomy"]) and contains_all(frontier_md, ["absolute_threshold", "quantile_threshold"]),
            "evidence_hint": "Look for exp9_policy_frontier_summary.md with matched-escalation/autonomy slices.",
        },
        {
            "id": "exp3_expanded_sample_size_disclosure",
            "description": "Manuscript discloses expanded Exp3 pair-level scale beyond the 112-task v2 bank.",
            "pass": contains_all(tex, ["16 tasks per pair"]) or contains_all(tex, ["448-task"]) or contains_all(tex, ["448 task"]),
            "evidence_hint": "Look for explicit expanded-pair sample-size statement in Exp3 results/discussion.",
        },
        {
            "id": "hard_packet_v2_integration",
            "description": "Hard human packet v2 evidence is integrated (summary artifact + manuscript mention).",
            "pass": (contains_all(hard_v2_md, ["512", "336"]) or contains_all(hard_v2_md, ["hard", "participant-mean"])) and contains_all(tex, ["hard packet", "human baseline"]),
            "evidence_hint": "Expect human_baseline_hardv2_summary.md and main-text reference to harder packet execution.",
        },
        {
            "id": "proper_score_primary_claim_path",
            "description": "Main claim path explicitly prioritizes strictly proper scoring (Brier/log/ECE) over MIRROR-gap-only framing.",
            "pass": contains_all(tex, ["strictly proper scoring", "brier", "log score"]) and contains_all(tex, ["primary"]),
            "evidence_hint": "Look for explicit claim-path language making proper scoring primary for L0.",
        },
        {
            "id": "cfr_plus_utility_coreport",
            "description": "CFR is co-reported with system success/autonomy/cost/latency in claim-critical sections.",
            "pass": contains_all(tex, ["cfr", "system success", "autonomy", "cost", "latency"]),
            "evidence_hint": "Look for explicit anti-gaming framing that pairs CFR with end-to-end utility metrics.",
        },
        {
            "id": "baseline_operating_point_disclosure",
            "description": "Baseline operating points disclose matched budget/autonomy/escalation details.",
            "pass": contains_all(tex, ["budget-matched", "autonomy", "escalation"]) and contains_all(tex, ["confidence-threshold", "self-consistency", "conformal"]),
            "evidence_hint": "Look for main-text (or core table) operating-point disclosure rather than appendix-only mention.",
        },
        {
            "id": "sar_ai_interpretation_block",
            "description": "SAR/AI interpretation is explicitly provided with scale intuition in manuscript text.",
            "pass": contains_all(tex, ["the \\ai{} measures"]) and contains_all(tex, ["\\sar{}", "range"]),
            "evidence_hint": "Look for concise SAR/AI interpretation text near Exp4 or metrics discussion.",
        },
        {
            "id": "ood_holdout_evidence",
            "description": "OOD holdout-domain routing stress artifact is present and passes stability status.",
            "pass": bool(ood_json) and (ood_json.get("macro_summary", {}).get("ood_generalization_status") == "pass"),
            "evidence_hint": "Expect exp9_ood_holdout_summary.json with macro_summary.ood_generalization_status='pass'.",
        },
        {
            "id": "mechanistic_probe_evidence",
            "description": "Targeted open-weight mechanistic probe artifact is present with sufficient model coverage.",
            "pass": bool(mech_json) and int(mech_json.get("macro_summary", {}).get("n_models_scored", 0)) >= 8,
            "evidence_hint": "Expect mechanistic_probe_summary.json with macro_summary.n_models_scored >= 8.",
        },
        {
            "id": "verification_coverage_table_evidence",
            "description": "Generation/verification coverage table artifact exists and is marked complete.",
            "pass": bool(coverage_json) and str(coverage_json.get("status", "")).lower() == "complete",
            "evidence_hint": "Expect generation_verification_coverage_summary.json with status='complete'.",
        },
        {
            "id": "hard_v2_cohort_completion",
            "description": "Hard-v2 cohort execution is complete with at least 20 validated participant pairs.",
            "pass": bool(hard_v2_cohort_json)
            and str(hard_v2_cohort_json.get("status", "")).lower() == "complete"
            and int(hard_v2_cohort_json.get("counts", {}).get("participant_pairs_validated_ok", 0)) >= 20,
            "evidence_hint": "Expect human_baseline_hardv2_cohort_summary.json status='complete' and participant_pairs_validated_ok >= 20.",
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
