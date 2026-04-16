"""Targeted open-weight probe: monitoring-vs-control signal mismatch.

This is a lightweight, behavior-level probe (not hidden-state mechanistic
analysis) to quantify whether stronger monitoring signals translate to control
decisions in Exp9.
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


def quantile(xs: list[float], q: float) -> float:
    vals = sorted(xs)
    if not vals:
        return float("nan")
    idx = int(max(0, min(len(vals) - 1, round((len(vals) - 1) * q))))
    return float(vals[idx])


def average_ranks(values: list[float]) -> list[float]:
    idx_vals = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx_vals):
        j = i
        while j + 1 < len(idx_vals) and idx_vals[j + 1][1] == idx_vals[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[idx_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    den = den_x * den_y
    if den == 0:
        return float("nan")
    return num / den


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return pearson(average_ranks(x), average_ranks(y))


def auc_binary(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or len(labels) < 2:
        return float("nan")
    pos = sum(1 for v in labels if v == 1)
    neg = sum(1 for v in labels if v == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    u = rank_sum_pos - (pos * (pos + 1) / 2.0)
    return u / (pos * neg)


def load_open_weight_models() -> set[str]:
    try:
        from mirror.api.models import MODEL_REGISTRY  # type: ignore

        return {name for name, cfg in MODEL_REGISTRY.items() if bool(cfg.get("open_weight", False))}
    except Exception:
        return {
            "llama-3.1-8b",
            "llama-3.1-70b",
            "llama-3.1-405b",
            "deepseek-r1",
            "mistral-large",
            "qwen-3-235b",
            "gpt-oss-120b",
            "gemma-3-12b",
            "gemma-3-27b",
            "kimi-k2",
            "llama-3.2-3b",
            "llama-3.3-70b",
            "mixtral-8x22b",
            "phi-4",
            "qwen3-next-80b",
        }


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


def normalize_decision(x: str) -> str:
    t = (x or "").strip().lower()
    if t in {"proceed", "answer", "continue"}:
        return "proceed"
    if t in {"use_tool", "tool", "delegate", "defer_tool"}:
        return "use_tool"
    if t in {"flag_for_review", "review", "escalate", "defer", "human_review"}:
        return "flag_for_review"
    return t


def load_probe_rows(
    result_glob: str,
    conditions: set[int],
    paradigms: set[int],
    exp1_acc: dict[str, dict[str, float]],
    exclude_externally_routed: bool,
) -> dict[str, list[dict]]:
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
        if not model or model not in exp1_acc:
            continue
        for slot in ("a", "b"):
            if exclude_externally_routed and bool(rec.get(f"component_{slot}_externally_routed", False)):
                continue
            domain = str(rec.get(f"domain_{slot}") or "").strip()
            if not domain:
                continue
            monitor_acc = exp1_acc[model].get(domain)
            if monitor_acc is None:
                continue
            decision = normalize_decision(str(rec.get(f"component_{slot}_decision") or ""))
            proceed = 1 if decision == "proceed" else 0
            by_model[model].append(
                {
                    "monitor_acc": float(monitor_acc),
                    "proceed": proceed,
                    "correct": 1 if bool(rec.get(f"component_{slot}_correct", False)) else 0,
                    "condition": int(rec.get("condition", -1)),
                    "paradigm": int(rec.get("paradigm", -1)),
                    "domain": domain,
                }
            )
    return by_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted monitoring-vs-control probe on open-weight models.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp9_mech_probe_%Y%m%dT%H%M%S"))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--result-glob", type=str, default="exp9*_results*.jsonl")
    parser.add_argument("--conditions", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--paradigms", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--exclude-externally-routed", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "mechanistic_probe_summary.json"
    out_md = run_dir / "mechanistic_probe_summary.md"
    checkpoint = run_dir / "checkpoint.json"
    progress = run_dir / "progress_log.jsonl"

    if args.resume and checkpoint.exists() and out_json.exists() and out_md.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            print(f"Resume hit completed run: {run_dir}")
            return

    append_jsonl(progress, {"ts_utc": utc_now_iso(), "event": "start", "run_id": args.run_id})
    exp1_acc = load_exp1_accuracy()
    open_models = load_open_weight_models()
    rows_by_model = load_probe_rows(
        result_glob=args.result_glob,
        conditions=set(args.conditions),
        paradigms=set(args.paradigms),
        exp1_acc=exp1_acc,
        exclude_externally_routed=bool(args.exclude_externally_routed),
    )
    append_jsonl(
        progress,
        {
            "ts_utc": utc_now_iso(),
            "event": "loaded_inputs",
            "n_models_open_weight_registry": len(open_models),
            "n_models_with_probe_rows": len(rows_by_model),
        },
    )

    per_model: list[dict] = []
    skipped: dict[str, str] = {}
    for model in sorted(open_models):
        rows = rows_by_model.get(model, [])
        if len(rows) < 50:
            skipped[model] = "insufficient_rows"
            continue
        monitor = [float(r["monitor_acc"]) for r in rows]
        y_correct = [int(r["correct"]) for r in rows]
        y_proceed = [int(r["proceed"]) for r in rows]

        q25 = quantile(monitor, 0.25)
        q75 = quantile(monitor, 0.75)
        low_idx = [i for i, m in enumerate(monitor) if m <= q25]
        high_idx = [i for i, m in enumerate(monitor) if m >= q75]

        low_proceed = mean([y_proceed[i] for i in low_idx]) if low_idx else float("nan")
        high_proceed = mean([y_proceed[i] for i in high_idx]) if high_idx else float("nan")
        control_alignment_gap = (
            high_proceed - low_proceed if not math.isnan(high_proceed) and not math.isnan(low_proceed) else float("nan")
        )

        low_fail_if_proceed_vals = [1 - y_correct[i] for i in low_idx if y_proceed[i] == 1]
        low_fail_if_proceed = mean(low_fail_if_proceed_vals) if low_fail_if_proceed_vals else float("nan")

        auc_correct = auc_binary(y_correct, monitor)
        auc_proceed = auc_binary(y_proceed, monitor)
        monitor_strength = auc_correct - 0.5 if not math.isnan(auc_correct) else float("nan")
        mismatch_score = (
            monitor_strength - control_alignment_gap
            if not math.isnan(monitor_strength) and not math.isnan(control_alignment_gap)
            else float("nan")
        )
        per_model.append(
            {
                "model": model,
                "n_components": len(rows),
                "auc_monitor_for_correctness": auc_correct,
                "auc_monitor_for_proceed": auc_proceed,
                "spearman_monitor_vs_proceed": spearman(monitor, [float(v) for v in y_proceed]),
                "q25_monitor_acc": q25,
                "q75_monitor_acc": q75,
                "low_monitor_proceed_rate": low_proceed,
                "high_monitor_proceed_rate": high_proceed,
                "control_alignment_gap_high_minus_low": control_alignment_gap,
                "low_monitor_failure_rate_when_proceed": low_fail_if_proceed,
                "monitor_signal_strength_auc_minus_0_5": monitor_strength,
                "mismatch_score_monitor_minus_control_gap": mismatch_score,
            }
        )

    mismatch_vals = [float(r["mismatch_score_monitor_minus_control_gap"]) for r in per_model if not math.isnan(float(r["mismatch_score_monitor_minus_control_gap"]))]
    macro = {
        "n_models_scored": len(per_model),
        "mean_auc_monitor_for_correctness": mean(
            [float(r["auc_monitor_for_correctness"]) for r in per_model if not math.isnan(float(r["auc_monitor_for_correctness"]))]
        ),
        "mean_auc_monitor_for_proceed": mean(
            [float(r["auc_monitor_for_proceed"]) for r in per_model if not math.isnan(float(r["auc_monitor_for_proceed"]))]
        ),
        "mean_control_alignment_gap_high_minus_low": mean(
            [float(r["control_alignment_gap_high_minus_low"]) for r in per_model if not math.isnan(float(r["control_alignment_gap_high_minus_low"]))]
        ),
        "mean_mismatch_score": mean(mismatch_vals),
        "share_models_with_mismatch_score_gt_0_10": (
            sum(1 for v in mismatch_vals if v > 0.10) / len(mismatch_vals) if mismatch_vals else float("nan")
        ),
    }
    macro["status"] = "pass" if (not math.isnan(macro["mean_mismatch_score"]) and macro["mean_mismatch_score"] >= 0.0) else "needs_attention"

    payload = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "conditions": list(args.conditions),
        "paradigms": list(args.paradigms),
        "exclude_externally_routed": bool(args.exclude_externally_routed),
        "macro_summary": macro,
        "per_model": per_model,
        "skipped_models": skipped,
        "interpretation": (
            "Positive mismatch score indicates monitoring signal quality outpaces control translation, "
            "consistent with a monitoring-to-control gap."
        ),
    }
    write_json(out_json, payload)

    lines = [
        "# Targeted Mechanistic Probe (Open-Weight Models)",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Models scored: `{macro['n_models_scored']}`",
        f"- Mean AUC(monitor->correctness): `{macro['mean_auc_monitor_for_correctness']:.3f}`",
        f"- Mean AUC(monitor->proceed): `{macro['mean_auc_monitor_for_proceed']:.3f}`",
        f"- Mean control alignment gap (high-low proceed): `{macro['mean_control_alignment_gap_high_minus_low']:.3f}`",
        f"- Mean mismatch score: `{macro['mean_mismatch_score']:.3f}`",
        f"- Status: `{macro['status']}`",
        "",
        "## Notes",
        "",
        "- This is a targeted behavioral probe, not hidden-state mechanistic tracing.",
        "- `mismatch_score = (AUC_monitor->correctness - 0.5) - (high-low proceed gap)`.",
        "- Higher mismatch implies monitoring signals are stronger than control translation.",
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

