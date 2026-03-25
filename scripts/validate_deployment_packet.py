"""
Validate and score production deployment packet.

Expected packet files:
  - traffic.csv
  - outcomes.csv
  - cost_latency.csv
  - constraints.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKET_DIR = ROOT / "deployment_packet"
DEFAULT_OUT_DIR = ROOT / "audit" / "human_baseline_packet" / "results"

REQ_TRAFFIC = [
    "request_id",
    "timestamp_utc",
    "workflow",
    "risk_tier",
    "input_text",
    "model_decision",
    "model_answer",
]
REQ_OUTCOMES = [
    "request_id",
    "final_correct",
    "resolver_type",
    "resolver_decision",
    "resolved_at_utc",
]
REQ_COST = [
    "request_id",
    "model_latency_ms",
    "resolver_latency_ms",
    "total_latency_ms",
    "cost_usd",
    "tokens_in",
    "tokens_out",
]
REQ_CONSTRAINT_KEYS = [
    "max_escalation_rate",
    "max_p95_latency_ms",
    "max_cost_per_request_usd",
    "critical_failure_definitions",
    "policy_notes",
]

ESCALATE_DECISIONS = {"USE_TOOL", "FLAG_FOR_REVIEW", "ESCALATE", "DEFER"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_columns(rows: list[dict[str, str]], required: list[str], file_name: str) -> list[str]:
    if not rows:
        return [f"{file_name}: no rows"]
    cols = set(rows[0].keys())
    missing = [c for c in required if c not in cols]
    return [f"{file_name}: missing column `{c}`" for c in missing]


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def to_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def to_int(v: str) -> int | None:
    try:
        return int(float(v))
    except Exception:
        return None


def validate_packet(packet_dir: Path) -> tuple[bool, dict[str, Any]]:
    traffic_p = packet_dir / "traffic.csv"
    outcomes_p = packet_dir / "outcomes.csv"
    cost_p = packet_dir / "cost_latency.csv"
    constraints_p = packet_dir / "constraints.json"

    exists = {
        "traffic.csv": traffic_p.exists(),
        "outcomes.csv": outcomes_p.exists(),
        "cost_latency.csv": cost_p.exists(),
        "constraints.json": constraints_p.exists(),
    }
    missing_files = [k for k, v in exists.items() if not v]
    if missing_files:
        return False, {
            "status": "missing_files",
            "missing_files": missing_files,
            "exists": exists,
        }

    traffic = read_csv(traffic_p)
    outcomes = read_csv(outcomes_p)
    cost = read_csv(cost_p)
    constraints = json.loads(constraints_p.read_text(encoding="utf-8"))

    errors: list[str] = []
    errors.extend(require_columns(traffic, REQ_TRAFFIC, "traffic.csv"))
    errors.extend(require_columns(outcomes, REQ_OUTCOMES, "outcomes.csv"))
    errors.extend(require_columns(cost, REQ_COST, "cost_latency.csv"))
    for k in REQ_CONSTRAINT_KEYS:
        if k not in constraints:
            errors.append(f"constraints.json: missing key `{k}`")

    # Core row-level validation
    final_correct_bad = sum(1 for r in outcomes if str(r.get("final_correct", "")).strip() not in {"0", "1"})
    resolver_bad = sum(1 for r in outcomes if str(r.get("resolver_type", "")).strip() not in {"human", "tool"})
    if final_correct_bad > 0:
        errors.append(f"outcomes.csv: invalid final_correct rows={final_correct_bad}")
    if resolver_bad > 0:
        errors.append(f"outcomes.csv: invalid resolver_type rows={resolver_bad}")

    ids_t = {r["request_id"] for r in traffic if r.get("request_id")}
    ids_o = {r["request_id"] for r in outcomes if r.get("request_id")}
    ids_c = {r["request_id"] for r in cost if r.get("request_id")}
    shared_ids = sorted(ids_t & ids_o & ids_c)
    if not shared_ids:
        errors.append("No shared request_id across traffic/outcomes/cost_latency.")

    status = "ok" if not errors else "failed"
    payload = {
        "status": status,
        "errors": errors,
        "counts": {
            "traffic_rows": len(traffic),
            "outcomes_rows": len(outcomes),
            "cost_rows": len(cost),
            "shared_request_ids": len(shared_ids),
        },
        "constraints": constraints,
        "files": {
            "traffic": str(traffic_p),
            "outcomes": str(outcomes_p),
            "cost_latency": str(cost_p),
            "constraints": str(constraints_p),
        },
        "manifests": {
            "traffic.csv": {"sha256": sha256_file(traffic_p), "size_bytes": traffic_p.stat().st_size},
            "outcomes.csv": {"sha256": sha256_file(outcomes_p), "size_bytes": outcomes_p.stat().st_size},
            "cost_latency.csv": {"sha256": sha256_file(cost_p), "size_bytes": cost_p.stat().st_size},
            "constraints.json": {"sha256": sha256_file(constraints_p), "size_bytes": constraints_p.stat().st_size},
        },
    }

    if errors:
        return False, payload

    # Compute deployment metrics on shared ids.
    traffic_by = {r["request_id"]: r for r in traffic if r.get("request_id") in shared_ids}
    outcomes_by = {r["request_id"]: r for r in outcomes if r.get("request_id") in shared_ids}
    cost_by = {r["request_id"]: r for r in cost if r.get("request_id") in shared_ids}

    rows = []
    for rid in shared_ids:
        t = traffic_by[rid]
        o = outcomes_by[rid]
        c = cost_by[rid]
        dec = str(t.get("model_decision", "")).strip().upper()
        escalated = dec in ESCALATE_DECISIONS
        final_correct = int(o["final_correct"])
        total_latency = to_float(c.get("total_latency_ms", "")) or 0.0
        total_cost = to_float(c.get("cost_usd", "")) or 0.0
        rows.append(
            {
                "request_id": rid,
                "workflow": str(t.get("workflow", "")).strip().lower(),
                "risk_tier": str(t.get("risk_tier", "")).strip().lower(),
                "decision": dec,
                "escalated": escalated,
                "final_correct": final_correct,
                "resolver_type": str(o.get("resolver_type", "")).strip().lower(),
                "total_latency_ms": total_latency,
                "cost_usd": total_cost,
            }
        )

    n = len(rows)
    escalated_n = sum(1 for r in rows if r["escalated"])
    autonomy_n = n - escalated_n
    success_n = sum(1 for r in rows if r["final_correct"] == 1)
    failure_n = n - success_n
    cfr = failure_n / autonomy_n if autonomy_n else 0.0

    latencies = [r["total_latency_ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows]

    by_risk: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "success": 0, "escalated": 0})
    for r in rows:
        k = r["risk_tier"] or "unknown"
        by_risk[k]["n"] += 1
        by_risk[k]["success"] += int(r["final_correct"] == 1)
        by_risk[k]["escalated"] += int(r["escalated"])

    by_risk_out = {}
    for k, v in sorted(by_risk.items()):
        by_risk_out[k] = {
            "n": v["n"],
            "success_rate": pct(v["success"], v["n"]),
            "escalation_rate": pct(v["escalated"], v["n"]),
        }

    cons = constraints
    max_escalation_rate = float(cons["max_escalation_rate"])
    max_p95_latency_ms = float(cons["max_p95_latency_ms"])
    max_cost = float(cons["max_cost_per_request_usd"])

    p95_latency = quantile(latencies, 0.95)
    mean_cost = sum(costs) / n if n else float("nan")
    escalation_rate = pct(escalated_n, n)

    sla = {
        "escalation_rate_pass": escalation_rate <= max_escalation_rate,
        "p95_latency_pass": p95_latency <= max_p95_latency_ms,
        "mean_cost_pass": mean_cost <= max_cost,
    }
    sla["all_pass"] = all(sla.values())

    payload["deployment_metrics"] = {
        "n_requests": n,
        "success_rate": pct(success_n, n),
        "failure_rate": pct(failure_n, n),
        "cfr_autonomy_only": cfr,
        "escalation_rate": escalation_rate,
        "autonomy_rate": pct(autonomy_n, n),
        "latency_ms": {
            "p50": quantile(latencies, 0.50),
            "p95": p95_latency,
            "mean": sum(latencies) / n if n else float("nan"),
        },
        "cost_usd": {
            "p50": quantile(costs, 0.50),
            "p95": quantile(costs, 0.95),
            "mean": mean_cost,
        },
        "decision_counts": dict(sorted(Counter(r["decision"] for r in rows).items())),
        "resolver_counts": dict(sorted(Counter(r["resolver_type"] for r in rows).items())),
        "risk_tier_breakdown": by_risk_out,
        "sla": sla,
    }
    payload["data_profile"] = {
        "workflows": dict(sorted(Counter(r["workflow"] for r in rows).items())),
        "risk_tiers": dict(sorted(Counter(r["risk_tier"] for r in rows).items())),
    }
    return True, payload


def write_validation_report(path: Path, payload: dict[str, Any]) -> None:
    status = payload.get("status", "unknown")
    lines = [
        "# Deployment Packet Validation Report",
        "",
        f"- Generated: {utc_now_iso()}",
        f"- Status: **{status.upper()}**",
        "",
    ]
    if status != "ok":
        lines.append("## Errors")
        lines.append("")
        for e in payload.get("errors", []) or payload.get("missing_files", []):
            lines.append(f"- {e}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    m = payload["deployment_metrics"]
    lines.extend(
        [
            "## Core Metrics",
            "",
            f"- Requests joined: {m['n_requests']}",
            f"- Success rate: {m['success_rate']:.3f}",
            f"- Failure rate: {m['failure_rate']:.3f}",
            f"- CFR (autonomy-only): {m['cfr_autonomy_only']:.3f}",
            f"- Escalation rate: {m['escalation_rate']:.3f}",
            f"- Autonomy rate: {m['autonomy_rate']:.3f}",
            "",
            "## Latency / Cost",
            "",
            f"- Latency p50/p95 (ms): {m['latency_ms']['p50']:.1f} / {m['latency_ms']['p95']:.1f}",
            f"- Cost mean/p95 (USD): {m['cost_usd']['mean']:.4f} / {m['cost_usd']['p95']:.4f}",
            "",
            "## SLA Checks",
            "",
            f"- Escalation budget pass: {m['sla']['escalation_rate_pass']}",
            f"- P95 latency pass: {m['sla']['p95_latency_pass']}",
            f"- Mean cost pass: {m['sla']['mean_cost_pass']}",
            f"- All pass: {m['sla']['all_pass']}",
            "",
            "## Risk-Tier Breakdown",
            "",
            "| Risk Tier | N | Success Rate | Escalation Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for tier, row in m["risk_tier_breakdown"].items():
        lines.append(
            f"| `{tier}` | {row['n']} | {row['success_rate']:.3f} | {row['escalation_rate']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and analyze deployment packet.")
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok, payload = validate_packet(args.packet_dir)
    payload["generated_at_utc"] = utc_now_iso()
    payload["packet_dir"] = str(args.packet_dir.resolve())

    report_md = args.out_dir / "validation_report.md"
    profile_json = args.out_dir / "data_profile.json"
    source_json = args.out_dir / "source_manifest.json"
    summary_json = args.out_dir / "deployment_packet_summary.json"

    write_json(summary_json, payload)
    write_json(source_json, {"packet_dir": payload["packet_dir"], "manifests": payload.get("manifests", {})})
    write_json(profile_json, payload.get("data_profile", {"status": payload.get("status")}))
    write_validation_report(report_md, payload)

    print(f"Wrote: {summary_json}")
    print(f"Wrote: {source_json}")
    print(f"Wrote: {profile_json}")
    print(f"Wrote: {report_md}")

    if not ok and payload.get("status") == "missing_files" and args.allow_missing:
        print("Deployment packet missing; marked pending (allow-missing enabled).")
        return
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
