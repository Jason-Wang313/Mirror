"""
Analyze deployment and red-team context files from the MIRROR human packet.

Outputs publication-ready JSON/Markdown summaries:
  - ecological_validity_summary.{json,md}
  - oracle_realism_sensitivity.{json,md}
  - goodhart_redteam_summary.{json,md}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKET_DIR = ROOT / "audit" / "human_baseline_packet"
DEPLOY_DIR = PACKET_DIR / "deployment"
REDTEAM_DIR = PACKET_DIR / "redteam"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def analyze_ecological(tasks: list[dict], gold: list[dict]) -> dict:
    by_task = {r.get("task_id", ""): r for r in tasks}
    matched = [r for r in gold if r.get("task_id", "") in by_task]

    workflow_counts = Counter(r.get("workflow", "") for r in tasks)
    domain_counts = Counter(r.get("domain", "") for r in tasks)
    risk_counts = Counter(r.get("risk_tier", "") for r in tasks)
    resolver_counts = Counter(r.get("resolver_type", "") for r in matched)
    outcome_counts = Counter(r.get("ground_truth_outcome", "") for r in matched)
    source_counts = Counter(r.get("ground_truth_source", "") for r in matched)

    return {
        "n_tasks": len(tasks),
        "n_gold_rows": len(gold),
        "n_matched": len(matched),
        "coverage_rate": pct(len(matched), len(tasks)),
        "workflow_counts": dict(sorted(workflow_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "risk_tier_counts": dict(sorted(risk_counts.items())),
        "resolver_type_counts": dict(sorted(resolver_counts.items())),
        "ground_truth_outcome_counts": dict(sorted(outcome_counts.items())),
        "ground_truth_source_counts": dict(sorted(source_counts.items())),
    }


def analyze_oracle_realism(rows: list[dict]) -> dict:
    total = len(rows)
    correct_n = sum(1 for r in rows if (r.get("resolution_correctness", "") or "").strip().lower() == "correct")

    by_resolver: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_risk: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        resolver = (r.get("final_resolver_type", "") or "").strip().lower()
        risk = (r.get("risk_tier", "") or "").strip().lower()
        is_correct = (r.get("resolution_correctness", "") or "").strip().lower() == "correct"

        by_resolver[resolver]["n"] += 1
        by_risk[risk]["n"] += 1
        if is_correct:
            by_resolver[resolver]["correct"] += 1
            by_risk[risk]["correct"] += 1

    by_resolver_rates = {
        k: {
            "n": v["n"],
            "correct_n": v["correct"],
            "accuracy": pct(v["correct"], v["n"]),
        }
        for k, v in sorted(by_resolver.items())
    }
    by_risk_rates = {
        k: {
            "n": v["n"],
            "correct_n": v["correct"],
            "accuracy": pct(v["correct"], v["n"]),
        }
        for k, v in sorted(by_risk.items())
    }

    # Fallible-reviewer sensitivity: projected final correctness if escalated components
    # are resolved with reviewer accuracy q rather than oracle=1.0.
    q_grid = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    sensitivity = []
    for q in q_grid:
        sensitivity.append(
            {
                "reviewer_accuracy": q,
                "projected_correct_rate": q,
                "delta_vs_oracle": q - 1.0,
            }
        )

    return {
        "n_components": total,
        "observed_correct_n": correct_n,
        "observed_correct_rate": pct(correct_n, total),
        "by_resolver": by_resolver_rates,
        "by_risk_tier": by_risk_rates,
        "fallible_reviewer_sensitivity": sensitivity,
    }


def analyze_redteam(rows: list[dict]) -> dict:
    attack_type_counts = Counter((r.get("attack_type", "") or "").strip().lower() for r in rows)
    target_mode_counts = Counter((r.get("target_failure_mode", "") or "").strip().lower() for r in rows)
    domain_counts = Counter((r.get("target_domain", "") or "").strip().lower() for r in rows)
    exploit_signal_counts = Counter((r.get("expected_exploit_signal", "") or "").strip().lower() for r in rows)
    difficulty_counts = Counter((r.get("difficulty", "") or "").strip().lower() for r in rows)
    designer_counts = Counter((r.get("designer", "") or "").strip().lower() for r in rows)

    return {
        "n_attacks": len(rows),
        "attack_type_counts": dict(sorted(attack_type_counts.items())),
        "target_failure_mode_counts": dict(sorted(target_mode_counts.items())),
        "target_domain_counts": dict(sorted(domain_counts.items())),
        "expected_exploit_signal_counts": dict(sorted(exploit_signal_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "designer_counts": dict(sorted(designer_counts.items())),
        "mitigation_bundle": [
            "hidden canary attacks in each release",
            "periodic attack-set rotation with held-out templates",
            "routing-trigger drift monitors on escalation patterns",
            "cross-signal anomaly checks (confidence vs action mismatch)",
        ],
    }


def write_markdown(out_path: Path, eco: dict, oracle: dict, redteam: dict) -> None:
    lines = [
        "# Human Packet Context Analysis",
        "",
        "## Ecological Validity Packet",
        "",
        f"- Tasks: {eco['n_tasks']}",
        f"- Gold rows: {eco['n_gold_rows']}",
        f"- Matched coverage: {eco['coverage_rate']:.1%}",
        f"- Workflows: {eco['workflow_counts']}",
        f"- Domains: {eco['domain_counts']}",
        f"- Risk tiers: {eco['risk_tier_counts']}",
        f"- Resolver types: {eco['resolver_type_counts']}",
        "",
        "## Oracle Realism Sensitivity",
        "",
        f"- Escalation components: {oracle['n_components']}",
        f"- Observed correctness in oracle-eval table: {oracle['observed_correct_rate']:.1%}",
        "",
        "| Reviewer Accuracy q | Projected Correct Rate | Delta vs Oracle |",
        "|---:|---:|---:|",
    ]
    for row in oracle["fallible_reviewer_sensitivity"]:
        lines.append(
            f"| {row['reviewer_accuracy']:.2f} | {row['projected_correct_rate']:.2f} | {row['delta_vs_oracle']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Goodhart Red-Team Packet",
            "",
            f"- Attack count: {redteam['n_attacks']}",
            f"- Attack types: {redteam['attack_type_counts']}",
            f"- Target failure modes: {redteam['target_failure_mode_counts']}",
            f"- Expected exploit signals: {redteam['expected_exploit_signal_counts']}",
            "",
            "### Mitigation Bundle",
            "",
        ]
    )
    for item in redteam["mitigation_bundle"]:
        lines.append(f"- {item}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MIRROR human packet context files.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEPLOY_DIR / "ecological_validity_tasks.csv",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEPLOY_DIR / "ecological_validity_gold.csv",
    )
    parser.add_argument(
        "--oracle-eval",
        type=Path,
        default=DEPLOY_DIR / "escalation_oracle_eval.csv",
    )
    parser.add_argument(
        "--redteam",
        type=Path,
        default=REDTEAM_DIR / "goodhart_attack_set.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACKET_DIR / "results",
    )
    args = parser.parse_args()

    tasks = read_csv(args.tasks)
    gold = read_csv(args.gold)
    oracle_rows = read_csv(args.oracle_eval)
    redteam_rows = read_csv(args.redteam)

    eco = analyze_ecological(tasks, gold)
    oracle = analyze_oracle_realism(oracle_rows)
    redteam = analyze_redteam(redteam_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "ecological_validity_summary.json", eco)
    write_json(args.out_dir / "oracle_realism_sensitivity.json", oracle)
    write_json(args.out_dir / "goodhart_redteam_summary.json", redteam)
    write_markdown(args.out_dir / "context_hardening_summary.md", eco, oracle, redteam)

    print(f"Wrote: {args.out_dir / 'ecological_validity_summary.json'}")
    print(f"Wrote: {args.out_dir / 'oracle_realism_sensitivity.json'}")
    print(f"Wrote: {args.out_dir / 'goodhart_redteam_summary.json'}")
    print(f"Wrote: {args.out_dir / 'context_hardening_summary.md'}")


if __name__ == "__main__":
    main()
