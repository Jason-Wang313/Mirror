"""
Crash-safe Exp1 proper-scoring primary analysis.

Primary evidence path:
  - Brier score (strictly proper)
  - Log score / negative log-likelihood (strictly proper)

Inputs:
  - Exp1 JSONL result shards and full files

Outputs:
  - proper_scoring_primary_summary.json
  - proper_scoring_primary_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_OUT_ROOT = RESULTS_DIR / "exp1_proper_scoring"
DEFAULT_GLOBS = ["exp1_*_results.jsonl", "exp1_*_shard.jsonl"]
DEFAULT_MODEL_LIST_FILE = RESULTS_DIR / "exp9_combined_16model_escalation.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def resolve_result_files(globs: list[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for pat in globs:
        for p in RESULTS_DIR.glob(pat):
            files[str(p.resolve())] = p
    return sorted(files.values(), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return float("nan")
    n = len(probs)
    return sum((probs[i] - labels[i]) ** 2 for i in range(n)) / n


def neg_log_score(probs: list[float], labels: list[int], eps: float = 1e-6) -> float:
    if not probs:
        return float("nan")
    n = len(probs)
    acc = 0.0
    for p, y in zip(probs, labels):
        p_clip = min(1.0 - eps, max(eps, p))
        if y == 1:
            acc += -math.log(p_clip)
        else:
            acc += -math.log(1.0 - p_clip)
    return acc / n


def bootstrap_ci(
    probs: list[float],
    labels: list[int],
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not probs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(probs)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        p = [probs[i] for i in idx]
        y = [labels[i] for i in idx]
        vals.append(metric_fn(p, y))
    vals.sort()
    lo_idx = max(0, int((alpha / 2.0) * n_boot) - 1)
    hi_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    return vals[lo_idx], vals[hi_idx]


def model_table_from_records(files: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # Deduplicate on (model, question_id, channel) with latest-file-wins semantics.
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_stats = {"files": [str(p) for p in files], "file_rows": {}, "kept_channel1_rows": 0}
    for fp in files:
        rows = 0
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("channel") != 1:
                    continue
                model = r.get("model")
                qid = r.get("question_id")
                if not model or not qid:
                    continue
                parsed = r.get("parsed") or {}
                bet = parsed.get("bet")
                correct = r.get("answer_correct")
                if bet is None or correct is None:
                    continue
                key = (str(model), str(qid), 1)
                latest[key] = r
        source_stats["file_rows"][str(fp)] = rows

    by_model: dict[str, dict[str, Any]] = {}
    for (_model, _qid, _channel), r in latest.items():
        model = r["model"]
        parsed = r.get("parsed") or {}
        bet = float(parsed.get("bet"))
        p = max(0.0, min(1.0, bet / 10.0))
        y = 1 if bool(r.get("answer_correct")) else 0
        entry = by_model.setdefault(model, {"probs": [], "labels": []})
        entry["probs"].append(p)
        entry["labels"].append(y)

    source_stats["kept_channel1_rows"] = sum(len(v["probs"]) for v in by_model.values())
    return by_model, source_stats


def load_target_models(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    per_model = obj.get("per_model", {})
    if isinstance(per_model, dict):
        return {str(k) for k in per_model.keys()}
    return set()


def safe_mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def make_markdown(path: Path, summary: dict) -> None:
    models = summary["per_model"]
    macro = summary["macro_summary"]
    lines = [
        "# Exp1 Proper-Scoring Primary Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Created: {summary['created_at_utc']}",
        f"- Models analyzed: {summary['n_models']}",
        f"- Primary scoring rule: `{summary['primary_scoring_rule']}`",
        "",
        "## Macro Summary",
        "",
        f"- Mean Brier: {macro['mean_brier']:.4f}",
        f"- Mean Log Score (NLL): {macro['mean_log_score']:.4f}",
        "",
        "## Per-Model (Proper Scoring)",
        "",
        "| Model | N | Brier | 95% CI (Brier) | Log Score | 95% CI (Log) | Mean Conf | Mean Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted(models):
        row = models[model]
        b_lo, b_hi = row["brier_ci95"]
        l_lo, l_hi = row["log_score_ci95"]
        lines.append(
            f"| `{model}` | {row['n']} | {row['brier']:.4f} | [{b_lo:.4f}, {b_hi:.4f}] | "
            f"{row['log_score']:.4f} | [{l_lo:.4f}, {l_hi:.4f}] | {row['mean_confidence']:.4f} | {row['mean_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Inputs are deduplicated on `(model, question_id, channel=1)` with latest-file-wins semantics.",
            "- Brier and log score are both strictly proper scoring rules.",
            "- This report is intended as the primary scoring evidence where wagering-rule concerns are raised.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp1 proper-scoring primary analysis.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("exp1_proper_scoring_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--result-globs", nargs="+", default=DEFAULT_GLOBS)
    parser.add_argument("--model-list-file", type=Path, default=DEFAULT_MODEL_LIST_FILE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "progress_log.jsonl"
    checkpoint_path = run_dir / "checkpoint.json"
    summary_json = run_dir / "proper_scoring_primary_summary.json"
    summary_md = run_dir / "proper_scoring_primary_summary.md"

    state = read_json(checkpoint_path) or {
        "run_id": args.run_id,
        "created_at_utc": utc_now_iso(),
        "steps_completed": [],
    }
    write_json(checkpoint_path, state)

    files = resolve_result_files(args.result_globs)
    if not files:
        raise FileNotFoundError(f"No files matched globs: {args.result_globs}")

    append_jsonl(log_path, {"ts_utc": utc_now_iso(), "event": "files_resolved", "n_files": len(files)})
    by_model, source_stats = model_table_from_records(files)
    target_models = load_target_models(args.model_list_file)
    if target_models:
        by_model = {m: v for m, v in by_model.items() if m in target_models}
    source_stats["target_model_list_file"] = str(args.model_list_file) if args.model_list_file else None
    source_stats["target_models_used"] = sorted(target_models) if target_models else []

    per_model: dict[str, dict[str, Any]] = {}
    for model in sorted(by_model):
        probs = by_model[model]["probs"]
        labels = by_model[model]["labels"]
        brier = brier_score(probs, labels)
        logv = neg_log_score(probs, labels)
        brier_ci = bootstrap_ci(
            probs,
            labels,
            brier_score,
            n_boot=max(100, args.bootstrap_samples),
            seed=args.seed,
        )
        log_ci = bootstrap_ci(
            probs,
            labels,
            neg_log_score,
            n_boot=max(100, args.bootstrap_samples),
            seed=args.seed + 17,
        )
        per_model[model] = {
            "n": len(probs),
            "brier": brier,
            "brier_ci95": [brier_ci[0], brier_ci[1]],
            "log_score": logv,
            "log_score_ci95": [log_ci[0], log_ci[1]],
            "mean_confidence": safe_mean(probs),
            "mean_accuracy": safe_mean(labels),
        }

    brier_vals = [v["brier"] for v in per_model.values()]
    log_vals = [v["log_score"] for v in per_model.values()]
    summary = {
        "run_id": args.run_id,
        "created_at_utc": state["created_at_utc"],
        "generated_at_utc": utc_now_iso(),
        "primary_scoring_rule": "brier",
        "secondary_scoring_rule": "log_score",
        "n_models": len(per_model),
        "source": source_stats,
        "macro_summary": {
            "mean_brier": safe_mean(brier_vals),
            "mean_log_score": safe_mean(log_vals),
        },
        "per_model": per_model,
    }
    write_json(summary_json, summary)
    make_markdown(summary_md, summary)

    append_jsonl(
        log_path,
        {
            "ts_utc": utc_now_iso(),
            "event": "run_complete",
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "n_models": len(per_model),
        },
    )
    state["steps_completed"] = ["complete"]
    state["updated_at_utc"] = utc_now_iso()
    write_json(checkpoint_path, state)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
