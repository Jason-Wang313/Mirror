"""
Score completed MIRROR human-baseline packet sheets.

Supports:
- single-participant scoring (backward compatible)
- multi-participant scoring with participant-mean primary reporting
- pooled-response sensitivity metrics
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
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


def _select_weak_domains(
    per_domain: dict[str, dict],
    median_acc: float,
    weak_domain_rule: str,
    fallback_bottom_k: int,
) -> tuple[list[str], list[str], str]:
    median_weak = sorted(
        [d for d, stats in per_domain.items() if stats["nat_acc"] < median_acc]
    )
    ranked = sorted(
        per_domain.keys(),
        key=lambda d: (per_domain[d]["nat_acc"], d),
    )
    if ranked:
        k = max(1, min(fallback_bottom_k, len(ranked)))
        bottom_k = sorted(ranked[:k])
    else:
        bottom_k = []

    if weak_domain_rule == "median":
        return median_weak, bottom_k, "median"
    if weak_domain_rule == "bottom_k":
        return bottom_k, bottom_k, "bottom_k"
    if median_weak:
        return median_weak, bottom_k, "median"
    return bottom_k, bottom_k, "bottom_k_fallback"


def score_exp1(
    exp1_rows: list[dict],
    exp1_key: dict[str, dict],
    weak_domain_rule: str = "median_or_bottom_k",
    fallback_bottom_k: int = 2,
) -> dict:
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

    weak_domains, weak_domains_bottom_k, rule_applied = _select_weak_domains(
        per_domain=per_domain,
        median_acc=median_acc,
        weak_domain_rule=weak_domain_rule,
        fallback_bottom_k=fallback_bottom_k,
    )
    weak_domains_median = sorted(
        [d for d, stats in per_domain.items() if stats["nat_acc"] < median_acc]
    )
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
        "weak_domain_rule_requested": weak_domain_rule,
        "weak_domain_rule_applied": rule_applied,
        "weak_domain_fallback_bottom_k": fallback_bottom_k,
        "weak_domains_median": weak_domains_median,
        "weak_domains_bottom_k": weak_domains_bottom_k,
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


def summarize_distribution(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    mean_v = statistics.fmean(values)
    sd_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean_v, "sd": sd_v, "min": min(values), "max": max(values)}


def aggregate_participants(participants: list[dict]) -> dict:
    exp1_nat = [p["exp1"]["overall"]["nat_acc"] for p in participants]
    exp9_cfr = [p["exp9"]["metrics"]["cfr_c1_style"] for p in participants]
    exp9_esc = [p["exp9"]["metrics"]["escalation_rate_weak"] for p in participants]
    exp9_auto = [p["exp9"]["metrics"]["autonomy_rate"] for p in participants]
    exp9_sys = [p["exp9"]["metrics"]["system_success_oracle"] for p in participants]

    weak_freq: dict[str, int] = defaultdict(int)
    weak_sets = []
    rule_counts: dict[str, int] = defaultdict(int)
    for p in participants:
        weak = set(p["exp1"].get("weak_domains", []))
        weak_sets.append(weak)
        for d in weak:
            weak_freq[d] += 1
        rule_counts[p["exp1"].get("weak_domain_rule_applied", "unknown")] += 1

    weak_union = sorted(set().union(*weak_sets)) if weak_sets else []
    weak_intersection = sorted(set.intersection(*weak_sets)) if weak_sets else []

    return {
        "n_participants": len(participants),
        "metrics": {
            "exp1_nat_acc": summarize_distribution(exp1_nat),
            "exp9_cfr_c1_style": summarize_distribution(exp9_cfr),
            "exp9_escalation_rate_weak": summarize_distribution(exp9_esc),
            "exp9_autonomy_rate": summarize_distribution(exp9_auto),
            "exp9_system_success_oracle": summarize_distribution(exp9_sys),
        },
        "weak_domains": {
            "frequency": dict(sorted(weak_freq.items())),
            "union": weak_union,
            "intersection": weak_intersection,
        },
        "weak_domain_rule_applied_counts": dict(sorted(rule_counts.items())),
    }


def select_primary(participants: list[dict], aggregate: dict, pooled: dict, mode: str) -> dict:
    if mode == "pooled":
        exp1 = pooled["exp1"]
        exp9 = pooled["exp9"]
        lo, hi = exp1["overall"]["nat_acc_ci95"]
        return {
            "mode": "pooled",
            "exp1_nat_acc": exp1["overall"]["nat_acc"],
            "exp1_nat_acc_ci95": [lo, hi],
            "exp9_cfr_c1_style": exp9["metrics"]["cfr_c1_style"],
            "exp9_escalation_rate_weak": exp9["metrics"]["escalation_rate_weak"],
            "exp9_autonomy_rate": exp9["metrics"]["autonomy_rate"],
            "exp9_system_success_oracle": exp9["metrics"]["system_success_oracle"],
            "weak_domains": exp1.get("weak_domains", []),
            "weak_domain_rule_applied": exp1.get("weak_domain_rule_applied", "unknown"),
            "n_exp1_items": exp1["overall"]["n_items"],
            "n_exp9_weak_components": exp9["overall"]["n_weak_components"],
        }

    if mode == "worst_case":
        chosen = max(participants, key=lambda p: p["exp9"]["metrics"]["cfr_c1_style"])
        exp1 = chosen["exp1"]
        exp9 = chosen["exp9"]
        lo, hi = exp1["overall"]["nat_acc_ci95"]
        return {
            "mode": "worst_case_participant",
            "participant_id": chosen["participant_id"],
            "exp1_nat_acc": exp1["overall"]["nat_acc"],
            "exp1_nat_acc_ci95": [lo, hi],
            "exp9_cfr_c1_style": exp9["metrics"]["cfr_c1_style"],
            "exp9_escalation_rate_weak": exp9["metrics"]["escalation_rate_weak"],
            "exp9_autonomy_rate": exp9["metrics"]["autonomy_rate"],
            "exp9_system_success_oracle": exp9["metrics"]["system_success_oracle"],
            "weak_domains": exp1.get("weak_domains", []),
            "weak_domain_rule_applied": exp1.get("weak_domain_rule_applied", "unknown"),
            "n_exp1_items": exp1["overall"]["n_items"],
            "n_exp9_weak_components": exp9["overall"]["n_weak_components"],
        }

    nat = aggregate["metrics"]["exp1_nat_acc"]
    cfr = aggregate["metrics"]["exp9_cfr_c1_style"]
    esc = aggregate["metrics"]["exp9_escalation_rate_weak"]
    auto = aggregate["metrics"]["exp9_autonomy_rate"]
    sys_ok = aggregate["metrics"]["exp9_system_success_oracle"]
    pooled_lo, pooled_hi = pooled["exp1"]["overall"]["nat_acc_ci95"]
    return {
        "mode": "participant_mean",
        "exp1_nat_acc": nat["mean"],
        "exp1_nat_acc_sd": nat["sd"],
        "exp1_nat_acc_range": [nat["min"], nat["max"]],
        "exp1_nat_acc_pooled_ci95": [pooled_lo, pooled_hi],
        "exp9_cfr_c1_style": cfr["mean"],
        "exp9_cfr_c1_style_sd": cfr["sd"],
        "exp9_cfr_c1_style_range": [cfr["min"], cfr["max"]],
        "exp9_escalation_rate_weak": esc["mean"],
        "exp9_escalation_rate_weak_sd": esc["sd"],
        "exp9_escalation_rate_weak_range": [esc["min"], esc["max"]],
        "exp9_autonomy_rate": auto["mean"],
        "exp9_autonomy_rate_sd": auto["sd"],
        "exp9_autonomy_rate_range": [auto["min"], auto["max"]],
        "exp9_system_success_oracle": sys_ok["mean"],
        "exp9_system_success_oracle_sd": sys_ok["sd"],
        "exp9_system_success_oracle_range": [sys_ok["min"], sys_ok["max"]],
        "weak_domains_union": aggregate["weak_domains"]["union"],
        "weak_domains_intersection": aggregate["weak_domains"]["intersection"],
        "weak_domain_rule_applied_counts": aggregate["weak_domain_rule_applied_counts"],
        "n_participants": aggregate["n_participants"],
        "n_exp1_items_per_participant": participants[0]["exp1"]["overall"]["n_items"] if participants else 0,
        "n_exp9_weak_components_pooled": pooled["exp9"]["overall"]["n_weak_components"],
    }


def build_markdown_single(summary: dict, cohort_label: str = "Human Baseline") -> str:
    exp1 = summary["exp1"]
    exp9 = summary["exp9"]
    lo, hi = exp1["overall"]["nat_acc_ci95"]
    weak = exp1["weak_domains"]
    strong = exp1["strong_domains"]
    weak_rule_req = exp1.get("weak_domain_rule_requested", "median_or_bottom_k")
    weak_rule_applied = exp1.get("weak_domain_rule_applied", "median")
    bottom_k = exp1.get("weak_domain_fallback_bottom_k", 2)
    weak_median = exp1.get("weak_domains_median", [])
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
        f"- Weak-domain rule requested: `{weak_rule_req}`",
        f"- Weak-domain rule applied: `{weak_rule_applied}`",
        f"- Weak domains (active for Exp9): {', '.join(weak) if weak else '(none)'}",
        f"- Weak domains (< median): {', '.join(weak_median) if weak_median else '(none)'}",
        f"- Bottom-{bottom_k} fallback set: {', '.join(exp1.get('weak_domains_bottom_k', [])) if exp1.get('weak_domains_bottom_k') else '(none)'}",
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
            "- If median-split yields no weak domains, `median_or_bottom_k` applies a deterministic bottom-k fallback to keep the denominator estimable.",
            "",
            "## Paper-Ready Insert (Draft)",
            "",
            (
                f"{cohort_label} pilot on the staged subset reports Exp1 Nat.Acc "
                f"{exp1['overall']['nat_acc']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]). "
                f"Using weak domains defined from this {cohort_label.lower()} Exp1 profile "
                f"(rule `{weak_rule_applied}`), Exp9 C1-style "
                f"CFR is {exp9['metrics']['cfr_c1_style']:.3f}, with weak-domain escalation "
                f"rate {exp9['metrics']['escalation_rate_weak']:.3f} and autonomy "
                f"{exp9['metrics']['autonomy_rate']:.3f}."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_markdown_multi(summary: dict, cohort_label: str) -> str:
    participants = summary["participants"]
    aggregate = summary["aggregate"]
    primary = summary["primary"]
    pooled = aggregate["pooled_sensitivity"]

    nat = aggregate["participant_mean"]["metrics"]["exp1_nat_acc"]
    cfr = aggregate["participant_mean"]["metrics"]["exp9_cfr_c1_style"]
    esc = aggregate["participant_mean"]["metrics"]["exp9_escalation_rate_weak"]
    auto = aggregate["participant_mean"]["metrics"]["exp9_autonomy_rate"]
    sys_ok = aggregate["participant_mean"]["metrics"]["exp9_system_success_oracle"]
    plo, phi = pooled["exp1"]["overall"]["nat_acc_ci95"]

    lines = [
        f"# {cohort_label} Multi-Participant Summary",
        "",
        f"Primary aggregation mode: `{summary['aggregation']['primary_mode']}`",
        "",
        "## Primary Metrics (Participant Mean)",
        "",
        f"- Exp1 Nat.Acc: {nat['mean']:.3f} ± {nat['sd']:.3f} (range {nat['min']:.3f}--{nat['max']:.3f}); pooled 95% Wilson CI [{plo:.3f}, {phi:.3f}]",
        f"- Exp9 CFR (C1-style): {cfr['mean']:.3f} ± {cfr['sd']:.3f} (range {cfr['min']:.3f}--{cfr['max']:.3f})",
        f"- Exp9 Weak-Domain Escalation Rate: {esc['mean']:.3f} ± {esc['sd']:.3f} (range {esc['min']:.3f}--{esc['max']:.3f})",
        f"- Exp9 Autonomy Rate: {auto['mean']:.3f} ± {auto['sd']:.3f} (range {auto['min']:.3f}--{auto['max']:.3f})",
        f"- Exp9 System Success (oracle escalation assumption): {sys_ok['mean']:.3f} ± {sys_ok['sd']:.3f} (range {sys_ok['min']:.3f}--{sys_ok['max']:.3f})",
        "",
        "## Per-Participant Metrics",
        "",
        "| Participant | Exp1 Nat.Acc | Exp9 CFR | Exp9 Weak Esc. | Exp9 Autonomy | Exp9 Oracle Success | Weak Domains | Rule |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    for p in participants:
        e1 = p["exp1"]
        e9 = p["exp9"]
        lines.append(
            f"| {p['participant_id']} | {e1['overall']['nat_acc']:.3f} | {e9['metrics']['cfr_c1_style']:.3f} | "
            f"{e9['metrics']['escalation_rate_weak']:.3f} | {e9['metrics']['autonomy_rate']:.3f} | "
            f"{e9['metrics']['system_success_oracle']:.3f} | {', '.join(e1.get('weak_domains', [])) or '(none)'} | "
            f"{e1.get('weak_domain_rule_applied', 'unknown')} |"
        )

    weak = aggregate["participant_mean"]["weak_domains"]
    lines.extend(
        [
            "",
            "## Weak-Domain Stability",
            "",
            f"- Weak-domain union: {', '.join(weak['union']) if weak['union'] else '(none)'}",
            f"- Weak-domain intersection: {', '.join(weak['intersection']) if weak['intersection'] else '(none)'}",
            f"- Rule-applied counts: {aggregate['participant_mean']['weak_domain_rule_applied_counts']}",
            f"- Weak-domain frequency: {weak['frequency']}",
            "",
            "## Pooled Sensitivity",
            "",
            f"- Pooled Exp1 Nat.Acc: {pooled['exp1']['overall']['nat_acc']:.3f} "
            f"(95% Wilson CI [{plo:.3f}, {phi:.3f}], n={pooled['exp1']['overall']['n_items']})",
            f"- Pooled Exp9 CFR (C1-style): {pooled['exp9']['metrics']['cfr_c1_style']:.3f} "
            f"(weak components n={pooled['exp9']['overall']['n_weak_components']})",
            f"- Pooled Exp9 Weak-Domain Escalation Rate: {pooled['exp9']['metrics']['escalation_rate_weak']:.3f}",
            f"- Pooled Exp9 Autonomy Rate: {pooled['exp9']['metrics']['autonomy_rate']:.3f}",
            f"- Pooled Exp9 System Success (oracle escalation assumption): {pooled['exp9']['metrics']['system_success_oracle']:.3f}",
            "",
            "## Paper-Ready Insert (Draft)",
            "",
            (
                f"{cohort_label} on the staged subset reports participant-mean Exp1 Nat.Acc "
                f"{nat['mean']:.3f}±{nat['sd']:.3f} (range {nat['min']:.3f}--{nat['max']:.3f}; pooled 95% CI [{plo:.3f}, {phi:.3f}]). "
                f"Using pre-specified weak-domain derivation (`median_or_bottom_k`, fallback k=2), participant-mean Exp9 C1-style CFR is "
                f"{cfr['mean']:.3f}±{cfr['sd']:.3f} (range {cfr['min']:.3f}--{cfr['max']:.3f}), with weak-domain escalation "
                f"{esc['mean']:.3f}±{esc['sd']:.3f} and autonomy {auto['mean']:.3f}±{auto['sd']:.3f}."
            ),
            "",
            "## Primary Snapshot",
            "",
            f"- Selected primary mode: `{primary['mode']}`",
        ]
    )

    return "\n".join(lines) + "\n"


def strip_detailed_fields(summary: dict) -> dict:
    slim = copy.deepcopy(summary)

    if "exp1" in slim:
        slim["exp1"].pop("item_results", None)
    if "exp9" in slim:
        slim["exp9"].pop("component_results", None)

    for p in slim.get("participants", []):
        if "exp1" in p:
            p["exp1"].pop("item_results", None)
        if "exp9" in p:
            p["exp9"].pop("component_results", None)

    pooled = slim.get("aggregate", {}).get("pooled_sensitivity", {})
    if "exp1" in pooled:
        pooled["exp1"].pop("item_results", None)
    if "exp9" in pooled:
        pooled["exp9"].pop("component_results", None)

    return slim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MIRROR human-baseline packet responses.")
    parser.add_argument(
        "--exp1-responses",
        nargs="+",
        type=Path,
        default=[PACKET_DIR / "templates" / "exp1_response_sheet.csv"],
    )
    parser.add_argument(
        "--exp9-responses",
        nargs="+",
        type=Path,
        default=[PACKET_DIR / "templates" / "exp9_response_sheet.csv"],
    )
    parser.add_argument(
        "--participant-ids",
        nargs="+",
        type=str,
        default=None,
        help="Optional participant IDs aligned with response files.",
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
    parser.add_argument(
        "--weak-domain-rule",
        type=str,
        choices=["median", "bottom_k", "median_or_bottom_k"],
        default="median_or_bottom_k",
        help="Rule for deriving weak domains from Exp1 per-domain Nat.Acc.",
    )
    parser.add_argument(
        "--fallback-bottom-k",
        type=int,
        default=2,
        help="Bottom-k used when weak-domain rule requires fallback.",
    )
    parser.add_argument(
        "--primary-aggregation",
        type=str,
        choices=["participant_mean", "pooled", "worst_case"],
        default="participant_mean",
        help="Primary aggregation mode for multi-participant summaries.",
    )
    parser.add_argument(
        "--include-detailed-results",
        action="store_true",
        help="Keep item-level/component-level result arrays in output JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.exp1_responses) != len(args.exp9_responses):
        raise ValueError("--exp1-responses and --exp9-responses must have equal length.")

    n_participants = len(args.exp1_responses)
    if args.participant_ids is not None and len(args.participant_ids) != n_participants:
        raise ValueError("--participant-ids length must match number of response files.")

    participant_ids = (
        args.participant_ids
        if args.participant_ids is not None
        else [f"P{i+1}" for i in range(n_participants)]
    )

    exp1_key_rows = read_csv(args.exp1_key)
    exp9_key_rows = read_csv(args.exp9_key)
    exp1_key = {r["item_id"]: r for r in exp1_key_rows}
    exp9_key = {r["item_id"]: r for r in exp9_key_rows}

    participants = []
    pooled_exp1_rows: list[dict] = []
    pooled_exp9_rows: list[dict] = []
    for pid, exp1_path, exp9_path in zip(participant_ids, args.exp1_responses, args.exp9_responses):
        exp1_rows = read_csv(exp1_path)
        exp9_rows = read_csv(exp9_path)
        pooled_exp1_rows.extend(exp1_rows)
        pooled_exp9_rows.extend(exp9_rows)

        exp1_summary = score_exp1(
            exp1_rows,
            exp1_key,
            weak_domain_rule=args.weak_domain_rule,
            fallback_bottom_k=args.fallback_bottom_k,
        )
        exp9_summary = score_exp9(exp9_rows, exp9_key, set(exp1_summary["weak_domains"]))
        participants.append(
            {
                "participant_id": pid,
                "inputs": {
                    "exp1_responses": str(exp1_path),
                    "exp9_responses": str(exp9_path),
                },
                "exp1": exp1_summary,
                "exp9": exp9_summary,
            }
        )

    if n_participants == 1:
        summary = {
            "inputs": {
                "exp1_responses": str(args.exp1_responses[0]),
                "exp9_responses": str(args.exp9_responses[0]),
                "exp1_key": str(args.exp1_key),
                "exp9_key": str(args.exp9_key),
            },
            "exp1": participants[0]["exp1"],
            "exp9": participants[0]["exp9"],
            "participants": participants,
        }
        markdown = build_markdown_single(summary, cohort_label=args.cohort_label)
    else:
        pooled_exp1 = score_exp1(
            pooled_exp1_rows,
            exp1_key,
            weak_domain_rule=args.weak_domain_rule,
            fallback_bottom_k=args.fallback_bottom_k,
        )
        pooled_exp9 = score_exp9(pooled_exp9_rows, exp9_key, set(pooled_exp1["weak_domains"]))
        pooled = {"exp1": pooled_exp1, "exp9": pooled_exp9}
        aggregate_pm = aggregate_participants(participants)
        primary = select_primary(participants, aggregate_pm, pooled, mode=args.primary_aggregation)

        summary = {
            "inputs": {
                "exp1_responses": [str(p) for p in args.exp1_responses],
                "exp9_responses": [str(p) for p in args.exp9_responses],
                "participant_ids": participant_ids,
                "exp1_key": str(args.exp1_key),
                "exp9_key": str(args.exp9_key),
            },
            "aggregation": {
                "primary_mode": args.primary_aggregation,
                "weak_domain_rule": args.weak_domain_rule,
                "fallback_bottom_k": args.fallback_bottom_k,
            },
            "participants": participants,
            "aggregate": {
                "participant_mean": aggregate_pm,
                "pooled_sensitivity": pooled,
            },
            "primary": primary,
        }
        markdown = build_markdown_multi(summary, cohort_label=args.cohort_label)

    out_json = args.out_dir / f"{args.summary_stem}.json"
    out_md = args.out_dir / f"{args.summary_stem}.md"
    json_payload = summary if args.include_detailed_results else strip_detailed_fields(summary)
    write_json(out_json, json_payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
