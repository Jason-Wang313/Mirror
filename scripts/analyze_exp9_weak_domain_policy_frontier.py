"""Weak-domain policy frontier analysis for Exp9.

Sweeps policy families (median/bottom-k, absolute thresholds, quantiles),
computes CFR/autonomy/escalation/system-success, and emits matched-escalation
and matched-autonomy comparisons versus the canonical median_or_bottom_k policy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "runs"
DEFAULT_DEPLOY_PACKET = ROOT / "deployment_packet"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def sorted_result_files(result_glob: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(result_glob), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def load_exp1_accuracy() -> dict[str, dict[str, float]]:
    # Prefer richest per-model maps (highest domain coverage).
    quality: dict[str, int] = {}
    out: dict[str, dict[str, float]] = {}
    for fp in sorted_result_files("exp1_*_accuracy.json"):
        if "meta" in fp.name or "counterfactual" in fp.name:
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for model, dd in payload.items():
            if not isinstance(dd, dict):
                continue
            m: dict[str, float] = {}
            valid = 0
            for domain, rec in dd.items():
                if not isinstance(rec, dict):
                    continue
                nat = rec.get("natural_acc")
                if nat is None:
                    continue
                try:
                    m[str(domain)] = float(nat)
                    valid += 1
                except Exception:
                    continue
            if valid >= quality.get(model, -1):
                quality[model] = valid
                out[model] = m
    return out


def load_exp9_components(
    result_glob: str,
    conditions: set[int],
    paradigms: set[int],
) -> dict[str, list[dict]]:
    # latest-file-wins dedupe by (model, task_id, condition, paradigm, is_false_score_control)
    dedup: dict[tuple, dict] = {}
    for fp in sorted_result_files(result_glob):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    rec.get("model"),
                    rec.get("task_id"),
                    rec.get("condition"),
                    rec.get("paradigm"),
                    rec.get("is_false_score_control", False),
                )
                dedup[key] = rec

    by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in dedup.values():
        if int(rec.get("condition", -1)) not in conditions:
            continue
        if int(rec.get("paradigm", -1)) not in paradigms:
            continue
        if bool(rec.get("is_false_score_control", False)):
            continue
        if not bool(rec.get("api_success", True)):
            continue
        model = str(rec.get("model") or "").strip()
        if not model:
            continue
        task_id = str(rec.get("task_id") or "").strip()
        for slot in ("a", "b"):
            domain = str(rec.get(f"domain_{slot}") or "").strip()
            if not domain:
                continue
            by_model[model].append(
                {
                    "task_id": task_id,
                    "slot": slot,
                    "domain": domain,
                    "correct": bool(rec.get(f"component_{slot}_correct", False)),
                }
            )
    return by_model


def median_or_bottom_k(acc_map: dict[str, float], k: int) -> set[str]:
    if not acc_map:
        return set()
    vals = sorted(acc_map.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    weak = {d for d, a in acc_map.items() if a < med}
    if weak:
        return weak
    ordered = sorted(acc_map.items(), key=lambda kv: (kv[1], kv[0]))
    return {d for d, _ in ordered[: max(1, min(k, len(ordered)))]}


def bottom_k(acc_map: dict[str, float], k: int) -> set[str]:
    if not acc_map:
        return set()
    ordered = sorted(acc_map.items(), key=lambda kv: (kv[1], kv[0]))
    return {d for d, _ in ordered[: max(1, min(k, len(ordered)))]}


def absolute_threshold(acc_map: dict[str, float], threshold: float, fallback_k: int) -> set[str]:
    weak = {d for d, a in acc_map.items() if a < threshold}
    if weak:
        return weak
    return bottom_k(acc_map, fallback_k)


def quantile_threshold(acc_map: dict[str, float], q: float, fallback_k: int) -> set[str]:
    if not acc_map:
        return set()
    q = clamp01(q)
    vals = sorted(acc_map.values())
    idx = int((len(vals) - 1) * q)
    thresh = vals[idx]
    weak = {d for d, a in acc_map.items() if a <= thresh}
    if weak:
        return weak
    return bottom_k(acc_map, fallback_k)


@dataclass
class Metrics:
    weak_cfr: float
    escalation_rate: float
    autonomy_rate: float
    overall_failure: float
    system_success_oracle: float
    n_components: int
    n_weak: int


def evaluate_policy(rows: list[dict], weak_domains: set[str]) -> Metrics:
    n = len(rows)
    if n == 0:
        return Metrics(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)
    weak_idx = [i for i, r in enumerate(rows) if r["domain"] in weak_domains]
    n_weak = len(weak_idx)
    escalated = [r["domain"] in weak_domains for r in rows]
    proceed = [not x for x in escalated]
    fail = [proceed[i] and (not rows[i]["correct"]) for i in range(n)]
    weak_fail = sum(1 for i in weak_idx if fail[i])
    weak_cfr = weak_fail / n_weak if n_weak > 0 else float("nan")
    esc_rate = sum(1 for x in escalated if x) / n
    auto_rate = 1.0 - esc_rate
    overall_failure = sum(1 for x in fail if x) / n
    return Metrics(
        weak_cfr=weak_cfr,
        escalation_rate=esc_rate,
        autonomy_rate=auto_rate,
        overall_failure=overall_failure,
        system_success_oracle=1.0 - overall_failure,
        n_components=n,
        n_weak=n_weak,
    )


def load_deployment_cost_latency(packet_dir: Path) -> dict[str, float]:
    # Global means used as lightweight expected-cost/latency proxy.
    out = {"mean_cost_usd": float("nan"), "mean_total_latency_ms": float("nan"), "mean_resolver_latency_ms": float("nan")}
    cost_path = packet_dir / "cost_latency.csv"
    if not cost_path.exists():
        return out
    import csv

    costs: list[float] = []
    lats: list[float] = []
    resolver_lats: list[float] = []
    with cost_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                costs.append(float(row.get("cost_usd", "")))
            except Exception:
                pass
            try:
                lats.append(float(row.get("total_latency_ms", "")))
            except Exception:
                pass
            try:
                resolver_lats.append(float(row.get("resolver_latency_ms", "")))
            except Exception:
                pass
    if costs:
        out["mean_cost_usd"] = mean(costs)
    if lats:
        out["mean_total_latency_ms"] = mean(lats)
    if resolver_lats:
        out["mean_resolver_latency_ms"] = mean(resolver_lats)
    return out


def nearest_by_key(candidates: list[dict], target: float, key: str) -> dict | None:
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(float(x[key]) - target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weak-domain policy frontiers for Exp9.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp9_policy_frontier_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--result-glob", type=str, default="exp9*_results*.jsonl")
    parser.add_argument("--conditions", nargs="+", type=int, default=[1])
    parser.add_argument("--paradigms", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--fallback-bottom-k", type=int, default=2)
    parser.add_argument("--bottom-k-grid", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--absolute-grid", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--quantile-grid", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--deployment-packet-dir", type=Path, default=DEFAULT_DEPLOY_PACKET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "exp9_policy_frontier_summary.json"
    out_md = run_dir / "exp9_policy_frontier_summary.md"

    exp1 = load_exp1_accuracy()
    exp9 = load_exp9_components(args.result_glob, conditions=set(args.conditions), paradigms=set(args.paradigms))
    cost_profile = load_deployment_cost_latency(args.deployment_packet_dir)

    per_model_rows: list[dict] = []
    macro_rows: list[dict] = []
    family_groups: dict[str, list[dict]] = defaultdict(list)

    for model, rows in sorted(exp9.items()):
        acc_map = exp1.get(model, {})
        if not rows or not acc_map:
            continue
        reference_weak = median_or_bottom_k(acc_map, k=args.fallback_bottom_k)
        ref = evaluate_policy(rows, reference_weak)
        ref_row = {
            "model": model,
            "family": "median_or_bottom_k",
            "policy_label": f"median_or_bottom_k(k={args.fallback_bottom_k})",
            "param": float(args.fallback_bottom_k),
            "weak_cfr": ref.weak_cfr,
            "escalation_rate": ref.escalation_rate,
            "autonomy_rate": ref.autonomy_rate,
            "overall_failure": ref.overall_failure,
            "system_success_oracle": ref.system_success_oracle,
            "n_components": ref.n_components,
            "n_weak": ref.n_weak,
            "matched_escalation_to_ref": True,
            "matched_autonomy_to_ref": True,
        }
        per_model_rows.append(ref_row)
        family_groups["median_or_bottom_k"].append(ref_row)

        candidates_bottom: list[dict] = []
        for k in sorted(set(int(x) for x in args.bottom_k_grid)):
            weak = bottom_k(acc_map, k=k)
            m = evaluate_policy(rows, weak)
            candidates_bottom.append(
                {
                    "model": model,
                    "family": "bottom_k",
                    "policy_label": f"bottom_k(k={k})",
                    "param": float(k),
                    "weak_cfr": m.weak_cfr,
                    "escalation_rate": m.escalation_rate,
                    "autonomy_rate": m.autonomy_rate,
                    "overall_failure": m.overall_failure,
                    "system_success_oracle": m.system_success_oracle,
                    "n_components": m.n_components,
                    "n_weak": m.n_weak,
                }
            )

        candidates_abs: list[dict] = []
        for t in sorted(set(float(x) for x in args.absolute_grid)):
            weak = absolute_threshold(acc_map, threshold=t, fallback_k=args.fallback_bottom_k)
            m = evaluate_policy(rows, weak)
            candidates_abs.append(
                {
                    "model": model,
                    "family": "absolute_threshold",
                    "policy_label": f"absolute_threshold(t={t:.2f})",
                    "param": t,
                    "weak_cfr": m.weak_cfr,
                    "escalation_rate": m.escalation_rate,
                    "autonomy_rate": m.autonomy_rate,
                    "overall_failure": m.overall_failure,
                    "system_success_oracle": m.system_success_oracle,
                    "n_components": m.n_components,
                    "n_weak": m.n_weak,
                }
            )

        candidates_q: list[dict] = []
        for q in sorted(set(float(x) for x in args.quantile_grid)):
            weak = quantile_threshold(acc_map, q=q, fallback_k=args.fallback_bottom_k)
            m = evaluate_policy(rows, weak)
            candidates_q.append(
                {
                    "model": model,
                    "family": "quantile_threshold",
                    "policy_label": f"quantile_threshold(q={q:.2f})",
                    "param": q,
                    "weak_cfr": m.weak_cfr,
                    "escalation_rate": m.escalation_rate,
                    "autonomy_rate": m.autonomy_rate,
                    "overall_failure": m.overall_failure,
                    "system_success_oracle": m.system_success_oracle,
                    "n_components": m.n_components,
                    "n_weak": m.n_weak,
                }
            )

        all_cands = candidates_bottom + candidates_abs + candidates_q
        for c in all_cands:
            c["matched_escalation_to_ref"] = False
            c["matched_autonomy_to_ref"] = False

        by_family = {
            "bottom_k": candidates_bottom,
            "absolute_threshold": candidates_abs,
            "quantile_threshold": candidates_q,
        }
        for fam, fam_cands in by_family.items():
            m_esc = nearest_by_key(fam_cands, target=ref.escalation_rate, key="escalation_rate")
            m_auto = nearest_by_key(fam_cands, target=ref.autonomy_rate, key="autonomy_rate")
            if m_esc is not None:
                m_esc["matched_escalation_to_ref"] = True
            if m_auto is not None:
                m_auto["matched_autonomy_to_ref"] = True
            for c in fam_cands:
                per_model_rows.append(c)
                family_groups[fam].append(c)

    # macro aggregation
    for fam, rows in sorted(family_groups.items()):
        # full family average
        macro_rows.append(
            {
                "family": fam,
                "slice": "all",
                "weak_cfr_mean": mean([float(r["weak_cfr"]) for r in rows if not math.isnan(float(r["weak_cfr"]))]),
                "escalation_rate_mean": mean([float(r["escalation_rate"]) for r in rows if not math.isnan(float(r["escalation_rate"]))]),
                "autonomy_rate_mean": mean([float(r["autonomy_rate"]) for r in rows if not math.isnan(float(r["autonomy_rate"]))]),
                "system_success_mean": mean([float(r["system_success_oracle"]) for r in rows if not math.isnan(float(r["system_success_oracle"]))]),
                "n_rows": len(rows),
            }
        )
        for slice_name, flag in [("matched_escalation", "matched_escalation_to_ref"), ("matched_autonomy", "matched_autonomy_to_ref")]:
            srows = [r for r in rows if bool(r.get(flag))]
            if not srows:
                continue
            macro_rows.append(
                {
                    "family": fam,
                    "slice": slice_name,
                    "weak_cfr_mean": mean([float(r["weak_cfr"]) for r in srows if not math.isnan(float(r["weak_cfr"]))]),
                    "escalation_rate_mean": mean([float(r["escalation_rate"]) for r in srows if not math.isnan(float(r["escalation_rate"]))]),
                    "autonomy_rate_mean": mean([float(r["autonomy_rate"]) for r in srows if not math.isnan(float(r["autonomy_rate"]))]),
                    "system_success_mean": mean([float(r["system_success_oracle"]) for r in srows if not math.isnan(float(r["system_success_oracle"]))]),
                    "n_rows": len(srows),
                }
            )

    # attach lightweight deployment cost/latency proxy
    for row in macro_rows:
        esc = float(row.get("escalation_rate_mean", float("nan")))
        base_cost = cost_profile.get("mean_cost_usd", float("nan"))
        base_lat = cost_profile.get("mean_total_latency_ms", float("nan"))
        resolver_lat = cost_profile.get("mean_resolver_latency_ms", float("nan"))
        if math.isnan(base_cost):
            row["expected_cost_usd_proxy"] = float("nan")
        else:
            row["expected_cost_usd_proxy"] = base_cost * (1.0 + 0.5 * esc)
        if math.isnan(base_lat) or math.isnan(resolver_lat):
            row["expected_latency_ms_proxy"] = float("nan")
        else:
            row["expected_latency_ms_proxy"] = base_lat + (esc * resolver_lat)

    payload = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "conditions": list(args.conditions),
        "paradigms": list(args.paradigms),
        "fallback_bottom_k": args.fallback_bottom_k,
        "bottom_k_grid": list(args.bottom_k_grid),
        "absolute_grid": list(args.absolute_grid),
        "quantile_grid": list(args.quantile_grid),
        "cost_latency_profile": cost_profile,
        "n_models": len({r["model"] for r in per_model_rows}),
        "macro_summary": macro_rows,
        "per_model_rows": per_model_rows,
    }
    write_json(out_json, payload)

    lines = [
        "# Exp9 Weak-Domain Policy Frontier Summary",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Conditions: `{args.conditions}` | Paradigms: `{args.paradigms}`",
        f"- Models covered: `{payload['n_models']}`",
        "",
        "## Macro Frontier",
        "",
        "| Family | Slice | Weak CFR | Escalation | Autonomy | System Success | Cost Proxy | Latency Proxy (ms) | N |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in macro_rows:
        lines.append(
            "| {family} | {slice} | {weak_cfr_mean:.3f} | {escalation_rate_mean:.3f} | {autonomy_rate_mean:.3f} | "
            "{system_success_mean:.3f} | {expected_cost_usd_proxy:.4f} | {expected_latency_ms_proxy:.1f} | {n_rows} |".format(
                **r
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `median_or_bottom_k` is treated as the canonical reference policy.",
            "- `matched_escalation` and `matched_autonomy` slices choose nearest policy settings per model.",
            "- Cost/latency columns are packet-derived operational proxies, not claim-critical primary evidence.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

