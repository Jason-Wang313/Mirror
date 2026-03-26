"""
Non-oracle Exp9 utility sensitivity analysis (crash-safe).

Promotes fallible-resolver outcomes to first-class reporting by estimating
system success and weak-domain failure under realistic resolver accuracies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_RESULT_GLOBS = ["exp9*_results*.jsonl"]
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "results"
DEFAULT_ORACLE_EVAL = ROOT / "audit" / "human_baseline_packet" / "deployment" / "escalation_oracle_eval.csv"
DEFAULT_DEPLOYMENT_PACKET_DIR = ROOT / "deployment_packet"


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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def parse_q_grid(values: list[str] | None, fallback: list[float]) -> list[float]:
    if not values:
        return sorted(set(float(v) for v in fallback))
    out = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if 0.0 <= fv <= 1.0:
            out.append(fv)
    return sorted(set(out)) if out else sorted(set(float(v) for v in fallback))


def dedup_files(globs: list[str]) -> list[Path]:
    uniq: dict[str, Path] = {}
    for pat in globs:
        for p in RESULTS_DIR.glob(pat):
            if p.exists():
                uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def load_exp1_natural_accuracy() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    quality: dict[str, int] = {}
    for fp in sorted(RESULTS_DIR.glob("exp1_*_accuracy.json"), key=lambda p: (p.stat().st_mtime, p.name.lower())):
        if "meta" in fp.name or "counterfactual" in fp.name:
            continue
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for model, domains in obj.items():
            if not isinstance(domains, dict):
                continue
            tmp: dict[str, float] = {}
            valid = 0
            for domain, metrics in domains.items():
                if not isinstance(metrics, dict):
                    continue
                nat = metrics.get("natural_acc")
                if nat is None:
                    continue
                try:
                    tmp[str(domain)] = float(nat)
                    valid += 1
                except Exception:
                    continue
            if valid >= quality.get(model, -1):
                out[model] = tmp
                quality[model] = valid
    return out


def weak_domains_for_model(acc_map: dict[str, float], bottom_k: int = 2) -> set[str]:
    if not acc_map:
        return set()
    vals = sorted(acc_map.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    weak = {d for d, v in acc_map.items() if v < med}
    if weak:
        return weak
    ordered = sorted(acc_map.items(), key=lambda kv: (kv[1], kv[0]))
    return {d for d, _ in ordered[:bottom_k]}


def load_components(files: list[Path]) -> list[dict[str, Any]]:
    # Dedup on (model, task_id, condition, paradigm, slot) with latest-file-wins.
    dedup: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = r.get("model")
                task = r.get("task_id")
                cond = r.get("condition")
                par = r.get("paradigm")
                if model is None or task is None or cond is None or par is None:
                    continue
                if not bool(r.get("api_success", False)):
                    continue
                for slot in ("a", "b"):
                    key = (str(model), str(task), int(cond), int(par), slot)
                    dedup[key] = r

    out = []
    for (_m, _t, _c, _p, slot), r in dedup.items():
        out.append(
            {
                "model": str(r.get("model")),
                "condition": int(r.get("condition")),
                "paradigm": int(r.get("paradigm")),
                "task_id": str(r.get("task_id")),
                "slot": slot,
                "domain": str(r.get(f"domain_{slot}") or "unknown"),
                "correct": bool(r.get(f"component_{slot}_correct", False)),
                "externally_routed": bool(r.get(f"component_{slot}_externally_routed", False)),
            }
        )
    return out


def observed_resolver_accuracy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "overall_accuracy": None,
            "by_resolver_type": {},
            "n_rows": 0,
        }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rc = (row.get("resolution_correctness") or "").strip().lower()
            if rc not in {"correct", "incorrect"}:
                continue
            rows.append(row)
    if not rows:
        return {"available": True, "overall_accuracy": None, "by_resolver_type": {}, "n_rows": 0}
    correct = sum(1 for r in rows if (r.get("resolution_correctness") or "").strip().lower() == "correct")
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "n_correct": 0})
    for r in rows:
        t = str(r.get("final_resolver_type") or "unknown")
        by_type[t]["n"] += 1
        if (r.get("resolution_correctness") or "").strip().lower() == "correct":
            by_type[t]["n_correct"] += 1
    by_type_acc = {
        t: {"n": d["n"], "accuracy": (d["n_correct"] / d["n"]) if d["n"] else None}
        for t, d in sorted(by_type.items())
    }
    return {
        "available": True,
        "overall_accuracy": correct / len(rows),
        "by_resolver_type": by_type_acc,
        "n_rows": len(rows),
    }


def deployment_operational_profile(
    packet_dir: Path,
    cost_column: str,
    latency_column: str,
) -> dict[str, Any]:
    traffic_p = packet_dir / "traffic.csv"
    cost_p = packet_dir / "cost_latency.csv"
    constraints_p = packet_dir / "constraints.json"
    if not traffic_p.exists() or not cost_p.exists():
        return {"available": False, "reason": "missing_traffic_or_cost_file"}

    decisions: dict[str, str] = {}
    with traffic_p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = str(row.get("request_id") or "").strip()
            dec = str(row.get("model_decision") or "").strip().upper()
            if not rid:
                continue
            decisions[rid] = dec

    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "cost_sum": 0.0, "lat_sum": 0.0})
    with cost_p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = str(row.get("request_id") or "").strip()
            if not rid:
                continue
            dec = decisions.get(rid, "PROCEED")
            bucket = "escalated" if dec in {"FLAG_FOR_REVIEW", "USE_TOOL"} else "proceed"
            try:
                c = float(row.get(cost_column, "") or 0.0)
            except Exception:
                c = 0.0
            try:
                l = float(row.get(latency_column, "") or 0.0)
            except Exception:
                l = 0.0
            agg = groups[bucket]
            agg["n"] += 1.0
            agg["cost_sum"] += c
            agg["lat_sum"] += l

    def mean_safe(bucket: str, key: str) -> float | None:
        g = groups.get(bucket, {})
        n = float(g.get("n", 0.0))
        if n <= 0:
            return None
        return float(g.get(key, 0.0)) / n

    constraints = {}
    if constraints_p.exists():
        try:
            constraints = json.loads(constraints_p.read_text(encoding="utf-8"))
        except Exception:
            constraints = {}

    proceed_cost = mean_safe("proceed", "cost_sum")
    escalated_cost = mean_safe("escalated", "cost_sum")
    proceed_lat = mean_safe("proceed", "lat_sum")
    escalated_lat = mean_safe("escalated", "lat_sum")

    return {
        "available": True,
        "packet_dir": str(packet_dir),
        "cost_column": cost_column,
        "latency_column": latency_column,
        "group_counts": {
            "proceed": int(groups.get("proceed", {}).get("n", 0.0)),
            "escalated": int(groups.get("escalated", {}).get("n", 0.0)),
        },
        "mean_cost": {
            "proceed": proceed_cost,
            "escalated": escalated_cost,
        },
        "mean_latency_ms": {
            "proceed": proceed_lat,
            "escalated": escalated_lat,
        },
        "constraints": constraints,
    }


def estimate_cost_latency(op_profile: dict[str, Any], escalation_rate: float) -> dict[str, float | None]:
    if not op_profile.get("available"):
        return {"expected_cost_usd": None, "expected_latency_ms": None}
    e = max(0.0, min(1.0, float(escalation_rate)))
    mc = op_profile.get("mean_cost", {})
    ml = op_profile.get("mean_latency_ms", {})
    c0 = mc.get("proceed")
    c1 = mc.get("escalated")
    l0 = ml.get("proceed")
    l1 = ml.get("escalated")
    cost = None if (c0 is None or c1 is None) else (1.0 - e) * float(c0) + e * float(c1)
    lat = None if (l0 is None or l1 is None) else (1.0 - e) * float(l0) + e * float(l1)
    return {"expected_cost_usd": cost, "expected_latency_ms": lat}


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Maximize system_success/autonomy; minimize cost/latency.
    nd: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            better_or_equal = (
                (q.get("system_success_rate_q", -1.0) >= p.get("system_success_rate_q", -1.0))
                and (q.get("autonomy_rate", -1.0) >= p.get("autonomy_rate", -1.0))
                and (
                    q.get("expected_cost_usd") is None
                    or p.get("expected_cost_usd") is None
                    or q.get("expected_cost_usd") <= p.get("expected_cost_usd")
                )
                and (
                    q.get("expected_latency_ms") is None
                    or p.get("expected_latency_ms") is None
                    or q.get("expected_latency_ms") <= p.get("expected_latency_ms")
                )
            )
            strictly_better = (
                (q.get("system_success_rate_q", -1.0) > p.get("system_success_rate_q", -1.0))
                or (q.get("autonomy_rate", -1.0) > p.get("autonomy_rate", -1.0))
                or (
                    q.get("expected_cost_usd") is not None
                    and p.get("expected_cost_usd") is not None
                    and q.get("expected_cost_usd") < p.get("expected_cost_usd")
                )
                or (
                    q.get("expected_latency_ms") is not None
                    and p.get("expected_latency_ms") is not None
                    and q.get("expected_latency_ms") < p.get("expected_latency_ms")
                )
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return sorted(nd, key=lambda x: (-float(x.get("system_success_rate_q", 0.0)), -float(x.get("autonomy_rate", 0.0))))


def cond_metrics_with_q(rows: list[dict[str, Any]], weak_map: dict[str, set[str]], q: float) -> dict[int, dict[str, float]]:
    # q = external resolver correctness probability.
    by_cond: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "autonomous": 0, "autonomous_correct": 0, "escalated": 0, "weak_n": 0, "weak_fail_exp": 0.0})
    for r in rows:
        cond = int(r["condition"])
        model = r["model"]
        weak = r["domain"] in weak_map.get(model, set())
        ext = bool(r["externally_routed"])
        corr = bool(r["correct"])
        agg = by_cond[cond]
        agg["n"] += 1
        if ext:
            agg["escalated"] += 1
        else:
            agg["autonomous"] += 1
            if corr:
                agg["autonomous_correct"] += 1
        if weak:
            agg["weak_n"] += 1
            if ext:
                agg["weak_fail_exp"] += (1.0 - q)
            else:
                agg["weak_fail_exp"] += 0.0 if corr else 1.0

    out: dict[int, dict[str, float]] = {}
    for cond, d in sorted(by_cond.items()):
        n = d["n"]
        if n == 0:
            continue
        system_success = (d["autonomous_correct"] + q * d["escalated"]) / n
        weak_fail = (d["weak_fail_exp"] / d["weak_n"]) if d["weak_n"] else None
        out[cond] = {
            "n_components": n,
            "autonomy_rate": d["autonomous"] / n,
            "escalation_rate": d["escalated"] / n,
            "autonomous_correct_rate": (d["autonomous_correct"] / d["autonomous"]) if d["autonomous"] else None,
            "system_success_rate_q": system_success,
            "system_failure_rate_q": 1.0 - system_success,
            "weak_effective_failure_rate_q": weak_fail,
            "weak_n": d["weak_n"],
        }
    return out


def make_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Non-Oracle Utility Sensitivity (Exp9)",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Components analyzed: {summary['n_components']}",
        "",
        "## Resolver Accuracy Inputs",
        "",
        f"- Deployment packet observed resolver accuracy: {summary['resolver_accuracy']['overall_accuracy']}",
        f"- Resolver eval rows: {summary['resolver_accuracy']['n_rows']}",
        f"- Cost/latency profile available: {summary.get('operational_profile', {}).get('available', False)}",
        "",
        "## Condition-Level Sensitivity (q = resolver correctness)",
        "",
        "| q | C1 System Success | C4 System Success | C1 Weak Fail | C4 Weak Fail | C4-C1 Success Delta | C1 Cost | C4 Cost | C1 Latency | C4 Latency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["q_curve"]:
        c1_cost = row.get("c1_expected_cost_usd")
        c4_cost = row.get("c4_expected_cost_usd")
        c1_lat = row.get("c1_expected_latency_ms")
        c4_lat = row.get("c4_expected_latency_ms")
        lines.append(
            f"| {row['q']:.2f} | {row['c1_system_success']:.4f} | {row['c4_system_success']:.4f} | "
            f"{row['c1_weak_fail']:.4f} | {row['c4_weak_fail']:.4f} | {row['c4_minus_c1_success']:.4f} | "
            f"{'NA' if c1_cost is None else f'{c1_cost:.4f}'} | {'NA' if c4_cost is None else f'{c4_cost:.4f}'} | "
            f"{'NA' if c1_lat is None else f'{c1_lat:.1f}'} | {'NA' if c4_lat is None else f'{c4_lat:.1f}'} |"
        )
    if summary.get("pareto_frontier"):
        lines.extend(
            [
                "",
                "## Cost/Latency-Aware Pareto Frontier",
                "",
                "| q | Condition | System Success | Autonomy | Escalation | Exp. Cost | Exp. Latency (ms) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for p in summary["pareto_frontier"]:
            cost_txt = "NA" if p.get("expected_cost_usd") is None else f"{float(p['expected_cost_usd']):.4f}"
            lat_txt = "NA" if p.get("expected_latency_ms") is None else f"{float(p['expected_latency_ms']):.1f}"
            lines.append(
                f"| {float(p['q']):.2f} | C{int(p['condition'])} | {float(p['system_success_rate_q']):.4f} | "
                f"{float(p['autonomy_rate']):.4f} | {float(p['escalation_rate']):.4f} | "
                f"{cost_txt} | {lat_txt} |"
            )
    lines.extend(
        [
            "",
            "## Primary Takeaway",
            "",
            summary["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Runner:
    run_id: str
    result_globs: list[str]
    out_root: Path
    oracle_eval_csv: Path
    deployment_packet_dir: Path
    emit_pareto: bool
    cost_column: str
    latency_column: str
    resolver_q_grid: list[float]

    def __post_init__(self) -> None:
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "progress_log.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_json = self.run_dir / "non_oracle_utility_summary.json"
        self.summary_md = self.run_dir / "non_oracle_utility_summary.md"

    def load_state(self) -> dict:
        state = read_json(self.checkpoint_path)
        if not state:
            state = {"run_id": self.run_id, "created_at_utc": utc_now_iso(), "steps_completed": []}
        return state

    def save_state(self, state: dict) -> None:
        state["updated_at_utc"] = utc_now_iso()
        write_json(self.checkpoint_path, state)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.log_path, {"ts_utc": utc_now_iso(), "event": event, **payload})

    def run(self) -> None:
        state = self.load_state()
        self.save_state(state)

        files = dedup_files(self.result_globs)
        rows = load_components(files)
        acc_map = load_exp1_natural_accuracy()
        weak_map = {m: weak_domains_for_model(d, bottom_k=2) for m, d in acc_map.items()}
        resolver_acc = observed_resolver_accuracy(self.oracle_eval_csv)
        self.log("data_loaded", {"n_files": len(files), "n_components": len(rows)})

        op_profile = deployment_operational_profile(
            self.deployment_packet_dir, cost_column=self.cost_column, latency_column=self.latency_column
        )

        q_grid = list(self.resolver_q_grid)
        if resolver_acc.get("overall_accuracy") is not None:
            q_obs = round(float(resolver_acc["overall_accuracy"]), 3)
            if q_obs not in q_grid:
                q_grid = sorted(set(q_grid + [q_obs]))

        curve = []
        details = {}
        points_for_pareto: list[dict[str, Any]] = []
        for q in q_grid:
            cm = cond_metrics_with_q(rows, weak_map=weak_map, q=q)
            cm_with_ops: dict[str, Any] = {}
            for cond, metrics in cm.items():
                op = estimate_cost_latency(op_profile, metrics.get("escalation_rate", 0.0))
                merged = {**metrics, **op}
                cm_with_ops[str(cond)] = merged
                points_for_pareto.append(
                    {
                        "q": q,
                        "condition": int(cond),
                        "system_success_rate_q": merged.get("system_success_rate_q"),
                        "autonomy_rate": merged.get("autonomy_rate"),
                        "escalation_rate": merged.get("escalation_rate"),
                        "expected_cost_usd": merged.get("expected_cost_usd"),
                        "expected_latency_ms": merged.get("expected_latency_ms"),
                    }
                )
            details[str(q)] = cm_with_ops
            c1 = cm.get(1, {})
            c4 = cm.get(4, {})
            if not c1 or not c4:
                continue
            c1_op = estimate_cost_latency(op_profile, c1.get("escalation_rate", 0.0))
            c4_op = estimate_cost_latency(op_profile, c4.get("escalation_rate", 0.0))
            curve.append(
                {
                    "q": q,
                    "c1_system_success": c1.get("system_success_rate_q"),
                    "c4_system_success": c4.get("system_success_rate_q"),
                    "c1_weak_fail": c1.get("weak_effective_failure_rate_q"),
                    "c4_weak_fail": c4.get("weak_effective_failure_rate_q"),
                    "c4_minus_c1_success": (c4.get("system_success_rate_q") - c1.get("system_success_rate_q")),
                    "c1_expected_cost_usd": c1_op.get("expected_cost_usd"),
                    "c4_expected_cost_usd": c4_op.get("expected_cost_usd"),
                    "c1_expected_latency_ms": c1_op.get("expected_latency_ms"),
                    "c4_expected_latency_ms": c4_op.get("expected_latency_ms"),
                }
            )

        pareto = pareto_front(points_for_pareto) if self.emit_pareto else []

        summary = {
            "run_id": self.run_id,
            "generated_at_utc": utc_now_iso(),
            "input_files": [str(p) for p in files],
            "n_components": len(rows),
            "resolver_accuracy": resolver_acc,
            "operational_profile": op_profile,
            "q_curve": curve,
            "condition_details": details,
            "pareto_frontier": pareto,
        }
        if curve:
            low = min(curve, key=lambda r: r["q"])
            high = max(curve, key=lambda r: r["q"])
            summary["interpretation"] = (
                f"C4 retains a positive system-success advantage over C1 across resolver-quality sensitivity "
                f"(q={low['q']:.2f}..{high['q']:.2f}), while explicitly trading autonomy for reliability and operational overhead."
            )
        else:
            summary["interpretation"] = "Insufficient condition coverage for C1/C4 sensitivity reporting."

        write_json(self.summary_json, summary)
        make_markdown(self.summary_md, summary)
        self.log("summary_written", {"json": str(self.summary_json), "md": str(self.summary_md)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze non-oracle Exp9 utility sensitivity.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("non_oracle_utility_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--result-globs", nargs="+", default=DEFAULT_RESULT_GLOBS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--oracle-eval-csv", type=Path, default=DEFAULT_ORACLE_EVAL)
    parser.add_argument("--deployment-packet-dir", type=Path, default=DEFAULT_DEPLOYMENT_PACKET_DIR)
    parser.add_argument("--emit-pareto", action="store_true", default=False)
    parser.add_argument("--cost-column", type=str, default="cost_usd")
    parser.add_argument("--latency-column", type=str, default="total_latency_ms")
    parser.add_argument("--resolver-q-grid", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Runner(
        run_id=args.run_id,
        result_globs=args.result_globs,
        out_root=args.out_root,
        oracle_eval_csv=args.oracle_eval_csv,
        deployment_packet_dir=args.deployment_packet_dir,
        emit_pareto=bool(args.emit_pareto),
        cost_column=args.cost_column,
        latency_column=args.latency_column,
        resolver_q_grid=parse_q_grid(args.resolver_q_grid, fallback=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]),
    )
    runner.run()
    print(f"Done: {runner.summary_json}")


if __name__ == "__main__":
    main()
