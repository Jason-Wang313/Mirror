"""OOD holdout-domain stress test for Exp9 routing.

This analysis tests whether domain-level routing decisions remain stable when a
single domain is treated as held out during weak-domain construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sorted_result_files(result_glob: str) -> list[Path]:
    return sorted(RESULTS_DIR.glob(result_glob), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def median_value(xs: list[float]) -> float:
    vals = sorted(xs)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return float(vals[n // 2])
    return 0.5 * float(vals[n // 2 - 1] + vals[n // 2])


def load_exp1_accuracy() -> dict[str, dict[str, float]]:
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
        for model, domain_map in payload.items():
            if not isinstance(domain_map, dict):
                continue
            clean: dict[str, float] = {}
            valid = 0
            for domain, rec in domain_map.items():
                if not isinstance(rec, dict):
                    continue
                nat = rec.get("natural_acc")
                try:
                    if nat is not None:
                        clean[str(domain)] = float(nat)
                        valid += 1
                except Exception:
                    continue
            if valid >= quality.get(model, -1):
                quality[model] = valid
                out[model] = clean
    return out


def load_exp9_components(result_glob: str, conditions: set[int], paradigms: set[int]) -> dict[str, list[dict]]:
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
                    "domain": domain,
                    "correct": bool(rec.get(f"component_{slot}_correct", False)),
                }
            )
    return by_model


def bottom_k(acc_map: dict[str, float], k: int) -> set[str]:
    if not acc_map:
        return set()
    ordered = sorted(acc_map.items(), key=lambda kv: (kv[1], kv[0]))
    return {d for d, _ in ordered[: max(1, min(k, len(ordered)))]}


def median_or_bottom_k(acc_map: dict[str, float], fallback_k: int) -> set[str]:
    if not acc_map:
        return set()
    med = median_value(list(acc_map.values()))
    weak = {d for d, a in acc_map.items() if a < med}
    return weak if weak else bottom_k(acc_map, fallback_k)


def holdout_policy(acc_map: dict[str, float], holdout_domain: str, fallback_k: int) -> tuple[set[str], float, bool]:
    train = {d: a for d, a in acc_map.items() if d != holdout_domain}
    if holdout_domain not in acc_map or len(train) < 3:
        return set(), float("nan"), False
    med = median_value(list(train.values()))
    weak_train = {d for d, a in train.items() if a < med}
    if not weak_train:
        weak_train = bottom_k(train, fallback_k)
    holdout_is_weak = bool(acc_map[holdout_domain] < med)
    weak = set(weak_train)
    if holdout_is_weak:
        weak.add(holdout_domain)
    return weak, med, holdout_is_weak


def evaluate_rows(rows: list[dict], weak_domains: set[str]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "n_components": 0,
            "n_weak_components": 0,
            "escalation_rate": float("nan"),
            "autonomy_rate": float("nan"),
            "system_success_oracle": float("nan"),
            "overall_failure": float("nan"),
            "weak_cfr": float("nan"),
        }
    escalated = [r["domain"] in weak_domains for r in rows]
    proceed = [not x for x in escalated]
    fail = [proceed[i] and (not rows[i]["correct"]) for i in range(n)]
    weak_idx = [i for i, r in enumerate(rows) if r["domain"] in weak_domains]
    n_weak = len(weak_idx)
    weak_fail = sum(1 for i in weak_idx if fail[i])
    weak_cfr = weak_fail / n_weak if n_weak > 0 else float("nan")
    overall_failure = sum(1 for x in fail if x) / n
    escalation_rate = sum(1 for x in escalated if x) / n
    return {
        "n_components": float(n),
        "n_weak_components": float(n_weak),
        "escalation_rate": escalation_rate,
        "autonomy_rate": 1.0 - escalation_rate,
        "system_success_oracle": 1.0 - overall_failure,
        "overall_failure": overall_failure,
        "weak_cfr": weak_cfr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp9 OOD holdout-domain routing stress.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp9_ood_holdout_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--result-glob", type=str, default="exp9*_results*.jsonl")
    parser.add_argument("--conditions", nargs="+", type=int, default=[1])
    parser.add_argument("--paradigms", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--fallback-bottom-k", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "exp9_ood_holdout_summary.json"
    out_md = run_dir / "exp9_ood_holdout_summary.md"
    checkpoint = run_dir / "checkpoint.json"
    progress = run_dir / "progress_log.jsonl"

    if args.resume and checkpoint.exists() and out_json.exists() and out_md.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            print(f"Resume hit completed run: {run_dir}")
            return

    append_jsonl(progress, {"ts_utc": utc_now_iso(), "event": "start", "run_id": args.run_id})
    exp1 = load_exp1_accuracy()
    exp9 = load_exp9_components(args.result_glob, set(args.conditions), set(args.paradigms))
    append_jsonl(
        progress,
        {
            "ts_utc": utc_now_iso(),
            "event": "loaded_inputs",
            "n_models_exp1": len(exp1),
            "n_models_exp9": len(exp9),
        },
    )

    rows_out: list[dict] = []
    skipped_models: dict[str, str] = {}

    for model, rows in sorted(exp9.items()):
        acc_map = exp1.get(model, {})
        if len(acc_map) < 4 or not rows:
            skipped_models[model] = "insufficient_exp1_or_exp9_coverage"
            continue
        full_weak = median_or_bottom_k(acc_map, fallback_k=args.fallback_bottom_k)
        row_domains = sorted({str(r["domain"]) for r in rows if r.get("domain")})
        if not row_domains:
            skipped_models[model] = "no_domains_in_exp9_rows"
            continue
        for holdout in row_domains:
            rows_holdout = [r for r in rows if r["domain"] == holdout]
            if not rows_holdout:
                continue
            holdout_weak_set, train_median, holdout_is_weak_ood = holdout_policy(
                acc_map,
                holdout_domain=holdout,
                fallback_k=args.fallback_bottom_k,
            )
            if not holdout_weak_set and math.isnan(train_median):
                continue

            full_metrics = evaluate_rows(rows_holdout, full_weak)
            ood_metrics = evaluate_rows(rows_holdout, holdout_weak_set)
            holdout_is_weak_full = holdout in full_weak
            rows_out.append(
                {
                    "model": model,
                    "holdout_domain": holdout,
                    "n_components": int(ood_metrics["n_components"]),
                    "holdout_acc_exp1": float(acc_map.get(holdout, float("nan"))),
                    "train_median_exp1": train_median,
                    "holdout_is_weak_full_policy": bool(holdout_is_weak_full),
                    "holdout_is_weak_ood_policy": bool(holdout_is_weak_ood),
                    "policy_match_full_vs_ood": bool(holdout_is_weak_full == holdout_is_weak_ood),
                    "full_escalation_rate": float(full_metrics["escalation_rate"]),
                    "ood_escalation_rate": float(ood_metrics["escalation_rate"]),
                    "full_autonomy_rate": float(full_metrics["autonomy_rate"]),
                    "ood_autonomy_rate": float(ood_metrics["autonomy_rate"]),
                    "full_system_success_oracle": float(full_metrics["system_success_oracle"]),
                    "ood_system_success_oracle": float(ood_metrics["system_success_oracle"]),
                    "delta_escalation_rate_ood_minus_full": float(
                        ood_metrics["escalation_rate"] - full_metrics["escalation_rate"]
                    ),
                    "delta_autonomy_rate_ood_minus_full": float(
                        ood_metrics["autonomy_rate"] - full_metrics["autonomy_rate"]
                    ),
                    "delta_system_success_ood_minus_full": float(
                        ood_metrics["system_success_oracle"] - full_metrics["system_success_oracle"]
                    ),
                }
            )

    deltas_success = [float(r["delta_system_success_ood_minus_full"]) for r in rows_out]
    deltas_auto = [float(r["delta_autonomy_rate_ood_minus_full"]) for r in rows_out]
    deltas_esc = [float(r["delta_escalation_rate_ood_minus_full"]) for r in rows_out]
    matches = [1.0 if r["policy_match_full_vs_ood"] else 0.0 for r in rows_out]

    macro = {
        "n_models_covered": len({r["model"] for r in rows_out}),
        "n_holdout_slices": len(rows_out),
        "policy_match_rate": mean(matches),
        "delta_system_success_mean": mean(deltas_success),
        "delta_autonomy_mean": mean(deltas_auto),
        "delta_escalation_mean": mean(deltas_esc),
    }
    macro["ood_generalization_status"] = (
        "pass"
        if (
            not math.isnan(macro["policy_match_rate"])
            and macro["policy_match_rate"] >= 0.75
            and not math.isnan(macro["delta_system_success_mean"])
            and abs(macro["delta_system_success_mean"]) <= 0.03
            and not math.isnan(macro["delta_autonomy_mean"])
            and abs(macro["delta_autonomy_mean"]) <= 0.03
        )
        else "needs_attention"
    )

    payload = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "conditions": list(args.conditions),
        "paradigms": list(args.paradigms),
        "fallback_bottom_k": args.fallback_bottom_k,
        "macro_summary": macro,
        "per_model_holdout_rows": rows_out,
        "skipped_models": skipped_models,
        "interpretation": (
            "OOD holdout-domain routing remains directionally stable when policy match is high "
            "and mean system-success/autonomy deltas remain close to zero."
        ),
    }
    write_json(out_json, payload)

    lines = [
        "# Exp9 OOD Holdout-Domain Stress Summary",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Conditions: `{args.conditions}` | Paradigms: `{args.paradigms}`",
        f"- Models covered: `{macro['n_models_covered']}`",
        f"- Holdout slices: `{macro['n_holdout_slices']}`",
        f"- Policy match rate (full vs OOD): `{macro['policy_match_rate']:.3f}`",
        f"- Mean delta system success (OOD-full): `{macro['delta_system_success_mean']:.4f}`",
        f"- Mean delta autonomy (OOD-full): `{macro['delta_autonomy_mean']:.4f}`",
        f"- Mean delta escalation (OOD-full): `{macro['delta_escalation_mean']:.4f}`",
        f"- OOD generalization status: `{macro['ood_generalization_status']}`",
        "",
        "## Notes",
        "",
        "- Full policy = canonical `median_or_bottom_k` built using all 8 domains.",
        "- OOD policy = hold one domain out when constructing weak-domain threshold.",
        "- Near-zero mean deltas imply domain-holdout robustness for routing behavior.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    write_json(
        checkpoint,
        {
            "run_id": args.run_id,
            "updated_at_utc": utc_now_iso(),
            "status": "complete",
            "out_json": str(out_json),
            "out_md": str(out_md),
        },
    )
    append_jsonl(progress, {"ts_utc": utc_now_iso(), "event": "complete", "out_json": str(out_json)})
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()

