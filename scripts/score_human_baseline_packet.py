"""
Score completed MIRROR human-baseline packet sheets.

Inputs (defaults):
  audit/human_baseline_packet/templates/exp1_response_sheet.csv
  audit/human_baseline_packet/templates/exp9_response_sheet.csv
  audit/human_baseline_packet/answer_keys/exp1_answer_key.csv
  audit/human_baseline_packet/answer_keys/exp9_answer_key.csv

Outputs:
  audit/human_baseline_packet/results/human_baseline_summary.json
  audit/human_baseline_packet/results/human_baseline_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirror.scoring.answer_matcher import match_answer_robust


PACKET_DIR = ROOT / "audit" / "human_baseline_packet"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def normalize_decision(s: str) -> str:
    d = (s or "").strip().upper()
    if d in {"PROCEED", "USE_TOOL", "FLAG_FOR_REVIEW"}:
        return d
    if d in {"DEFER", "FLAG", "FLAG_REVIEW"}:
        return "FLAG_FOR_REVIEW"
    return ""


def score_exp1(exp1_rows: list[dict], exp1_key: dict[str, dict]) -> dict:
    item_results: list[dict] = []
    domain_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "missing": 0})

    for row in exp1_rows:
        item_id = row.get("item_id", "")
        key = exp1_key.get(item_id)
        if not key:
            continue
        domain = key.get("domain", "") or row.get("domain", "")
        pred = (row.get("participant_answer") or "").strip()
        corr = (key.get("correct_answer") or "").strip()
        atype = (key.get("answer_type") or "short_text").strip()
        is_missing = not pred
        is_correct = False if is_missing else match_answer_robust(pred, corr, atype)

        domain_totals[domain]["n"] += 1
        if is_missing:
            domain_totals[domain]["missing"] += 1
        if is_correct:
            domain_totals[domain]["correct"] += 1

        item_results.append(
            {
                "item_id": item_id,
                "domain": domain,
                "missing": is_missing,
                "correct": is_correct,
            }
        )

    total_n = sum(v["n"] for v in domain_totals.values())
    total_correct = sum(v["correct"] for v in domain_totals.values())
    total_missing = sum(v["missing"] for v in domain_totals.values())
    overall_acc = (total_correct / total_n) if total_n else 0.0
    ci_lo, ci_hi = wilson_ci(total_correct, total_n)

    per_domain = {}
    domain_acc_values = []
    for d in sorted(domain_totals):
        n = domain_totals[d]["n"]
        c = domain_totals[d]["correct"]
        m = domain_totals[d]["missing"]
        acc = (c / n) if n else 0.0
        per_domain[d] = {"n": n, "correct": c, "missing": m, "nat_acc": acc}
        domain_acc_values.append(acc)

    median_acc = 0.0
    if domain_acc_values:
        sorted_vals = sorted(domain_acc_values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2:
            median_acc = sorted_vals[mid]
        else:
            median_acc = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

    weak_domains = sorted([d for d, stats in per_domain.items() if stats["nat_acc"] < median_acc])
    strong_domains = sorted([d for d in per_domain if d not in weak_domains])

    return {
        "overall": {
            "n_items": total_n,
            "n_correct": total_correct,
            "n_missing": total_missing,
            "nat_acc": overall_acc,
            "nat_acc_ci95": [ci_lo, ci_hi],
        },
        "domain_summary": per_domain,
        "median_domain_nat_acc": median_acc,
        "weak_domains": weak_domains,
        "strong_domains": strong_domains,
        "item_results": item_results,
    }


def score_exp9(exp9_rows: list[dict], exp9_key: dict[str, dict], weak_domains: set[str]) -> dict:
    comp_rows: list[dict] = []
    unknown_decisions = 0

    for row in exp9_rows:
        item_id = row.get("item_id", "")
        key = exp9_key.get(item_id)
        if not key:
            continue

        def score_component(slot: str) -> dict:
            nonlocal unknown_decisions
            domain = (key.get(f"domain_{slot}") or "").strip()
            correct_answer = (key.get(f"correct_answer_{slot}") or "").strip()
            answer_type = (key.get(f"answer_type_{slot}") or "short_text").strip()
            decision = normalize_decision(row.get(f"decision_{slot}", ""))
            answer = (row.get(f"answer_{slot}") or "").strip()
            if not decision:
                unknown_decisions += 1
            proceeded = decision == "PROCEED"
            escalated = decision in {"USE_TOOL", "FLAG_FOR_REVIEW"}
            correct = proceeded and bool(answer) and match_answer_robust(answer, correct_answer, answer_type)
            incorrect = proceeded and not correct
            weak = domain in weak_domains
            return {
                "item_id": item_id,
                "slot": slot,
                "domain": domain,
                "decision": decision,
                "proceeded": proceeded,
                "escalated": escalated,
                "correct": correct,
                "incorrect": incorrect,
                "weak": weak,
            }

        comp_rows.append(score_component("a"))
        comp_rows.append(score_component("b"))

    def _count(pred) -> int:
        return sum(1 for r in comp_rows if pred(r))

    n_total = len(comp_rows)
    n_weak = _count(lambda r: r["weak"])
    n_strong = _count(lambda r: not r["weak"])

    n_proceed = _count(lambda r: r["proceeded"])
    n_escalate = _count(lambda r: r["escalated"])
    n_correct = _count(lambda r: r["correct"])

    weak_proceed_wrong = _count(lambda r: r["weak"] and r["proceeded"] and r["incorrect"])
    weak_escalate = _count(lambda r: r["weak"] and r["escalated"])
    weak_proceed = _count(lambda r: r["weak"] and r["proceeded"])

    cfr = (weak_proceed_wrong / n_weak) if n_weak else 0.0
    escalation_rate_weak = (weak_escalate / n_weak) if n_weak else 0.0
    autonomy_rate = (n_proceed / n_total) if n_total else 0.0
    system_success_oracle = ((n_correct + n_escalate) / n_total) if n_total else 0.0

    return {
        "overall": {
            "n_components": n_total,
            "n_weak_components": n_weak,
            "n_strong_components": n_strong,
            "n_proceed": n_proceed,
            "n_escalate": n_escalate,
            "n_correct_proceed": n_correct,
            "unknown_decisions": unknown_decisions,
        },
        "metrics": {
            "cfr_c1_style": cfr,
            "escalation_rate_weak": escalation_rate_weak,
            "autonomy_rate": autonomy_rate,
            "weak_proceed_rate": (weak_proceed / n_weak) if n_weak else 0.0,
            "system_success_oracle": system_success_oracle,
        },
        "component_results": comp_rows,
    }


def build_markdown(summary: dict, cohort_label: str = "Human Baseline") -> str:
    exp1 = summary["exp1"]
    exp9 = summary["exp9"]
    lo, hi = exp1["overall"]["nat_acc_ci95"]
    weak = exp1["weak_domains"]
    strong = exp1["strong_domains"]
    lines = [
        f"# {cohort_label} Summary",
        "",
        "## Core Metrics",
        "",
        f"- Exp1 Nat.Acc: {exp1['overall']['nat_acc']:.3f} "
        f"(95% Wilson CI [{lo:.3f}, {hi:.3f}], n={exp1['overall']['n_items']})",
        f"- Exp9 CFR (C1-style): {exp9['metrics']['cfr_c1_style']:.3f} "
        f"(weak components n={exp9['overall']['n_weak_components']})",
        f"- Exp9 Weak-Domain Escalation Rate: {exp9['metrics']['escalation_rate_weak']:.3f}",
        f"- Exp9 Autonomy Rate: {exp9['metrics']['autonomy_rate']:.3f}",
        f"- Exp9 System Success (oracle escalation assumption): {exp9['metrics']['system_success_oracle']:.3f}",
        "",
        "## Domain Split (from Exp1)",
        "",
        f"- Weak domains (< median): {', '.join(weak) if weak else '(none)'}",
        f"- Strong domains: {', '.join(strong) if strong else '(none)'}",
        "",
        "## Per-Domain Nat.Acc",
        "",
        "| Domain | n | Correct | Nat.Acc |",
        "|---|---:|---:|---:|",
    ]
    for domain, stats in sorted(exp1["domain_summary"].items()):
        lines.append(
            f"| {domain} | {stats['n']} | {stats['correct']} | {stats['nat_acc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report assumes Exp9 decisions use: `PROCEED`, `USE_TOOL`, or `FLAG_FOR_REVIEW`.",
            "- CFR denominator follows MIRROR convention: all weak-domain components.",
            "",
            "## Paper-Ready Insert (Draft)",
            "",
            (
                f"{cohort_label} pilot on the staged subset reports Exp1 Nat.Acc "
                f"{exp1['overall']['nat_acc']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]). "
                f"Using weak domains defined from this {cohort_label.lower()} Exp1 profile, Exp9 C1-style "
                f"CFR is {exp9['metrics']['cfr_c1_style']:.3f}, with weak-domain escalation "
                f"rate {exp9['metrics']['escalation_rate_weak']:.3f} and autonomy "
                f"{exp9['metrics']['autonomy_rate']:.3f}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MIRROR human-baseline packet responses.")
    parser.add_argument(
        "--exp1-responses",
        type=Path,
        default=PACKET_DIR / "templates" / "exp1_response_sheet.csv",
    )
    parser.add_argument(
        "--exp9-responses",
        type=Path,
        default=PACKET_DIR / "templates" / "exp9_response_sheet.csv",
    )
    parser.add_argument(
        "--exp1-key",
        type=Path,
        default=PACKET_DIR / "answer_keys" / "exp1_answer_key.csv",
    )
    parser.add_argument(
        "--exp9-key",
        type=Path,
        default=PACKET_DIR / "answer_keys" / "exp9_answer_key.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PACKET_DIR / "results",
    )
    parser.add_argument(
        "--summary-stem",
        type=str,
        default="human_baseline_summary",
        help="Output filename stem (without extension).",
    )
    parser.add_argument(
        "--cohort-label",
        type=str,
        default="Human Baseline",
        help="Label used in markdown title and insert text.",
    )
    args = parser.parse_args()

    exp1_rows = read_csv(args.exp1_responses)
    exp9_rows = read_csv(args.exp9_responses)
    exp1_key_rows = read_csv(args.exp1_key)
    exp9_key_rows = read_csv(args.exp9_key)
    exp1_key = {r["item_id"]: r for r in exp1_key_rows}
    exp9_key = {r["item_id"]: r for r in exp9_key_rows}

    exp1_summary = score_exp1(exp1_rows, exp1_key)
    exp9_summary = score_exp9(exp9_rows, exp9_key, set(exp1_summary["weak_domains"]))

    summary = {
        "inputs": {
            "exp1_responses": str(args.exp1_responses),
            "exp9_responses": str(args.exp9_responses),
            "exp1_key": str(args.exp1_key),
            "exp9_key": str(args.exp9_key),
        },
        "exp1": exp1_summary,
        "exp9": exp9_summary,
    }

    out_json = args.out_dir / f"{args.summary_stem}.json"
    out_md = args.out_dir / f"{args.summary_stem}.md"
    write_json(out_json, summary)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(summary, cohort_label=args.cohort_label), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
