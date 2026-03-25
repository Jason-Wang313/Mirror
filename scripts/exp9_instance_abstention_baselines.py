"""
Crash-safe Exp9 instance-level abstention baseline comparison.

Compares MIRROR domain routing against three instance-level baselines on the
same Exp9 frame (Condition 1, Paradigm 3, successful API parses):
  1) confidence-threshold routing (uncertainty proxy from hedge/decomp/tokens)
  2) self-consistency proxy routing (decomposition count)
  3) conformal-style risk-controlled thresholding

Outputs are written to a run-scoped directory with checkpoint + append-only logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_RESULT_FILES = []
DEFAULT_RESULT_GLOB = "exp9*_results*.jsonl"
DEFAULT_MODEL_LIST_FILE = RESULTS_DIR / "exp9_combined_16model_escalation.json"
DEFAULT_OUT_ROOT = RESULTS_DIR / "exp9_instance_baselines"


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


def nan_to_none(v: float) -> float | None:
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def resolve_result_files(explicit_files: list[Path], result_glob: str | None) -> list[Path]:
    files: list[Path] = []
    if explicit_files:
        files.extend(explicit_files)
    if result_glob:
        files.extend(RESULTS_DIR.glob(result_glob))
    # Keep only extant files; sort by mtime for deterministic "latest wins" behavior.
    uniq: dict[str, Path] = {}
    for p in files:
        if p.exists():
            uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def normalize_0_1(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def top_k_mask(scores: list[float], rate: float) -> list[bool]:
    n = len(scores)
    if n == 0 or rate <= 0:
        return [False] * n
    if rate >= 1:
        return [True] * n
    k = int(round(rate * n))
    k = max(0, min(n, k))
    order = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
    mask = [False] * n
    for idx in order[:k]:
        mask[idx] = True
    return mask


def hash_split(task_id: str, slot: str) -> int:
    digest = hashlib.sha1(f"{task_id}|{slot}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def evaluate_strategy(mask_escalate: list[bool], weak: list[bool], correct: list[bool]) -> dict[str, float]:
    n = len(mask_escalate)
    if n == 0:
        return {
            "escalation_rate": float("nan"),
            "autonomy_rate": float("nan"),
            "overall_failure_rate": float("nan"),
            "overall_oracle_success_rate": float("nan"),
            "weak_cfr": float("nan"),
            "weak_escalation_rate": float("nan"),
        }

    proceed = [not x for x in mask_escalate]
    fail = [proceed[i] and (not correct[i]) for i in range(n)]
    weak_idx = [i for i, is_weak in enumerate(weak) if is_weak]
    weak_total = len(weak_idx)
    weak_fail = sum(1 for i in weak_idx if fail[i])
    weak_esc = sum(1 for i in weak_idx if mask_escalate[i])

    escalation_rate = sum(1 for x in mask_escalate if x) / n
    autonomy_rate = 1.0 - escalation_rate
    overall_failure_rate = sum(1 for x in fail if x) / n
    # Oracle success assumption: escalated items are resolved correctly.
    overall_oracle_success_rate = 1.0 - overall_failure_rate
    weak_cfr = weak_fail / weak_total if weak_total > 0 else float("nan")
    weak_escalation_rate = weak_esc / weak_total if weak_total > 0 else float("nan")

    return {
        "escalation_rate": escalation_rate,
        "autonomy_rate": autonomy_rate,
        "overall_failure_rate": overall_failure_rate,
        "overall_oracle_success_rate": overall_oracle_success_rate,
        "weak_cfr": weak_cfr,
        "weak_escalation_rate": weak_escalation_rate,
    }


def conformal_style_mask(
    scores: list[float],
    correct: list[bool],
    task_ids: list[str],
    slots: list[str],
    target_error: float,
) -> tuple[list[bool], dict[str, Any]]:
    n = len(scores)
    if n == 0:
        return [], {"status": "empty"}

    calib_idx = [i for i in range(n) if hash_split(task_ids[i], slots[i]) == 0]
    test_idx = [i for i in range(n) if hash_split(task_ids[i], slots[i]) == 1]
    if not calib_idx or not test_idx:
        # Fallback to a simple median threshold if split is degenerate.
        median_score = sorted(scores)[n // 2]
        mask = [s > median_score for s in scores]
        return mask, {
            "status": "fallback_degenerate_split",
            "threshold": median_score,
            "target_error": target_error,
            "calibration_size": len(calib_idx),
            "test_size": len(test_idx),
        }

    calib_scores = [scores[i] for i in calib_idx]
    uniq_thresholds = sorted(set(calib_scores))
    best_t = None
    best_cov = -1.0
    best_err = None

    for t in uniq_thresholds:
        proceed_calib = [i for i in calib_idx if scores[i] <= t]
        if len(proceed_calib) < 20:
            continue
        err = sum(1 for i in proceed_calib if not correct[i]) / len(proceed_calib)
        cov = len(proceed_calib) / len(calib_idx)
        if err <= target_error and cov > best_cov:
            best_cov = cov
            best_t = t
            best_err = err

    if best_t is None:
        # No threshold satisfies the target error; escalate everything.
        return [True] * n, {
            "status": "no_feasible_threshold",
            "target_error": target_error,
            "calibration_size": len(calib_idx),
            "test_size": len(test_idx),
        }

    mask = [s > best_t for s in scores]
    proceed_test = [i for i in test_idx if not mask[i]]
    test_err = (
        sum(1 for i in proceed_test if not correct[i]) / len(proceed_test)
        if proceed_test
        else float("nan")
    )

    return mask, {
        "status": "ok",
        "threshold": best_t,
        "target_error": target_error,
        "calibration_size": len(calib_idx),
        "test_size": len(test_idx),
        "calibration_coverage": best_cov,
        "calibration_error": best_err,
        "test_error": test_err,
    }


def derive_weak_domains(acc_by_domain: dict[str, float], bottom_k: int = 2) -> tuple[set[str], dict[str, Any]]:
    if not acc_by_domain:
        return set(), {"rule_applied": "no_accuracy_data", "median": None, "bottom_k": bottom_k}

    vals = list(acc_by_domain.values())
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    if n % 2 == 1:
        median = vals_sorted[n // 2]
    else:
        median = 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])

    weak = {d for d, v in acc_by_domain.items() if v < median}
    rule = "median_split"
    if not weak:
        sorted_domains = sorted(acc_by_domain.items(), key=lambda kv: (kv[1], kv[0]))
        weak = {d for d, _ in sorted_domains[:bottom_k]}
        rule = "bottom_k_fallback"

    return weak, {
        "rule_applied": rule,
        "median": median,
        "bottom_k": bottom_k,
        "weak_domains": sorted(weak),
    }


def load_exp1_natural_accuracy() -> dict[str, dict[str, float]]:
    files = sorted(RESULTS_DIR.glob("exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
    out: dict[str, dict[str, float]] = {}
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue
        # Only ingest model/domain payloads that actually contain domain-level natural_acc.
        # This avoids accidental overwrite from ranking/meta blobs that also match *_accuracy.json.
        for model, domains in obj.items():
            if not isinstance(domains, dict):
                continue
            model_acc = out.setdefault(model, {})
            for domain, channel_metrics in domains.items():
                if not isinstance(channel_metrics, dict):
                    continue
                nat = channel_metrics.get("natural_acc")
                if nat is None:
                    continue
                try:
                    model_acc[domain] = float(nat)
                except Exception:
                    continue

    return {m: d for m, d in out.items() if d}


def load_target_models(model_list_file: Path) -> list[str]:
    if not model_list_file.exists():
        return []
    obj = read_json(model_list_file)
    per_model = obj.get("per_model", {})
    if isinstance(per_model, dict):
        return sorted(per_model.keys())
    return []


def iter_model_components(model: str, result_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str, str]] = set()
    for file_path in result_files:
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trial.get("model") != model:
                    continue
                if trial.get("condition") != 1 or trial.get("paradigm") != 3:
                    continue
                if not trial.get("api_success", False):
                    continue
                task_id = str(trial.get("task_id", ""))
                for slot in ("a", "b"):
                    domain = trial.get(f"domain_{slot}")
                    if not domain:
                        continue
                    key = (
                        model,
                        int(trial.get("condition", -1)),
                        int(trial.get("paradigm", -1)),
                        task_id,
                        slot,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "task_id": task_id,
                            "slot": slot,
                            "domain": domain,
                            "correct": bool(trial.get(f"component_{slot}_correct", False)),
                            "hedge": float(trial.get(f"hedge_count_{slot}", 0.0) or 0.0),
                            "decomp": float(trial.get(f"decomp_count_{slot}", 0.0) or 0.0),
                            "tokens": float(trial.get(f"token_count_{slot}", 0.0) or 0.0),
                        }
                    )
    return rows


@dataclass
class Runner:
    run_id: str
    result_files: list[Path]
    model_list_file: Path
    out_root: Path
    bottom_k: int
    conformal_target_error: float
    max_workers: int
    result_glob: str | None

    def __post_init__(self) -> None:
        self.run_dir = self.out_root / self.run_id
        self.shards_dir = self.run_dir / "shards"
        self.log_path = self.run_dir / "progress_log.jsonl"
        self.retry_queue_path = self.run_dir / "retry_queue.json"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_json = self.run_dir / "instance_baseline_summary.json"
        self.summary_md = self.run_dir / "instance_baseline_summary.md"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict:
        state = read_json(self.checkpoint_path)
        if not state:
            state = {
                "run_id": self.run_id,
                "created_at_utc": utc_now_iso(),
                "completed_models": [],
                "failed_models": [],
                "steps_completed": [],
            }
        return state

    def save_state(self, state: dict) -> None:
        state["updated_at_utc"] = utc_now_iso()
        write_json(self.checkpoint_path, state)

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.log_path,
            {
                "ts_utc": utc_now_iso(),
                "event": event,
                **payload,
            },
        )

    def mark_step(self, state: dict, step: str) -> None:
        if step not in state["steps_completed"]:
            state["steps_completed"].append(step)
            self.save_state(state)

    def prepare_manifest(self, state: dict) -> None:
        step = "prepare_manifest"
        if step in state["steps_completed"]:
            return

        exp1_acc = load_exp1_natural_accuracy()
        target_models = load_target_models(self.model_list_file)

        manifest = {
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "result_files": [str(p) for p in self.result_files],
            "existing_result_files": [str(p) for p in self.result_files if p.exists()],
            "result_glob": self.result_glob,
            "model_list_file": str(self.model_list_file),
            "target_models_from_list": target_models,
            "exp1_models_available": sorted(exp1_acc.keys()),
            "bottom_k": self.bottom_k,
            "conformal_target_error": self.conformal_target_error,
            "max_workers": self.max_workers,
        }
        write_json(self.run_dir / "manifest.json", manifest)
        self.log_event("manifest_written", {"path": str(self.run_dir / "manifest.json")})
        self.mark_step(state, step)

    def evaluate_one_model(self, model: str, exp1_acc: dict[str, dict[str, float]]) -> tuple[str, int, str]:
        try:
            rows = iter_model_components(model, self.result_files)
            if not rows:
                write_json(
                    self.shards_dir / f"{model}.json",
                    {
                        "model": model,
                        "status": "skipped",
                        "reason": "no_condition1_paradigm3_api_success_rows",
                    },
                )
                return model, 0, ""

            model_acc = exp1_acc.get(model, {})
            weak_domains, weak_meta = derive_weak_domains(model_acc, bottom_k=self.bottom_k)
            if not weak_domains:
                write_json(
                    self.shards_dir / f"{model}.json",
                    {
                        "model": model,
                        "status": "skipped",
                        "reason": "no_exp1_accuracy_available_for_weak_domain_policy",
                    },
                )
                return model, 0, ""

            task_ids = [r["task_id"] for r in rows]
            slots = [r["slot"] for r in rows]
            weak = [r["domain"] in weak_domains for r in rows]
            correct = [bool(r["correct"]) for r in rows]

            hedge = [float(r["hedge"]) for r in rows]
            decomp = [float(r["decomp"]) for r in rows]
            tokens = [float(r["tokens"]) for r in rows]

            hedge_n = normalize_0_1(hedge)
            decomp_n = normalize_0_1(decomp)
            tokens_n = normalize_0_1(tokens)

            # Higher score => more uncertainty => more likely to escalate.
            confidence_uncertainty = [
                0.50 * hedge_n[i] + 0.30 * decomp_n[i] + 0.20 * tokens_n[i]
                for i in range(len(rows))
            ]
            self_consistency_uncertainty = decomp_n

            domain_mask = [is_weak for is_weak in weak]
            domain_budget = sum(1 for x in domain_mask if x) / len(domain_mask) if domain_mask else 0.0

            no_mask = [False] * len(rows)
            conf_mask = top_k_mask(confidence_uncertainty, domain_budget)
            sc_mask = top_k_mask(self_consistency_uncertainty, domain_budget)
            conformal_mask, conformal_meta = conformal_style_mask(
                confidence_uncertainty,
                correct,
                task_ids,
                slots,
                self.conformal_target_error,
            )

            strategies = {
                "no_routing": evaluate_strategy(no_mask, weak, correct),
                "mirror_domain_routing": evaluate_strategy(domain_mask, weak, correct),
                "confidence_threshold_budget_matched": evaluate_strategy(conf_mask, weak, correct),
                "self_consistency_budget_matched": evaluate_strategy(sc_mask, weak, correct),
                "conformal_style": evaluate_strategy(conformal_mask, weak, correct),
            }

            weak_cfr_no = strategies["no_routing"]["weak_cfr"]
            for name, metrics in strategies.items():
                cfr = metrics["weak_cfr"]
                if weak_cfr_no is None or isinstance(weak_cfr_no, float) and math.isnan(weak_cfr_no):
                    reduction = float("nan")
                elif weak_cfr_no <= 0:
                    reduction = float("nan")
                elif cfr is None or isinstance(cfr, float) and math.isnan(cfr):
                    reduction = float("nan")
                else:
                    reduction = 1.0 - (cfr / weak_cfr_no)
                metrics["weak_cfr_reduction_vs_no_routing"] = reduction

            out = {
                "model": model,
                "status": "complete",
                "n_components": len(rows),
                "n_weak_components": sum(1 for x in weak if x),
                "weak_policy": weak_meta,
                "strategy_metrics": {
                    k: {kk: nan_to_none(vv) for kk, vv in m.items()} for k, m in strategies.items()
                },
                "conformal_meta": {k: nan_to_none(v) if isinstance(v, float) else v for k, v in conformal_meta.items()},
            }
            write_json(self.shards_dir / f"{model}.json", out)
            return model, 0, ""
        except Exception as e:
            return model, 1, str(e)

    def run_model_shards(self, state: dict) -> None:
        step = "run_model_shards"
        if step in state["steps_completed"]:
            return

        exp1_acc = load_exp1_natural_accuracy()
        target_models = load_target_models(self.model_list_file)
        if not target_models:
            target_models = sorted(exp1_acc.keys())

        pending = [m for m in target_models if m not in state["completed_models"]]
        failed_jobs: list[dict[str, Any]] = []
        self.log_event("model_shards_start", {"pending_models": pending})

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as ex:
            futures = {ex.submit(self.evaluate_one_model, model, exp1_acc): model for model in pending}
            for fut in as_completed(futures):
                model, code, err = fut.result()
                if code == 0:
                    if model not in state["completed_models"]:
                        state["completed_models"].append(model)
                    self.save_state(state)
                    self.log_event("model_complete", {"model": model})
                else:
                    failed_jobs.append({"model": model, "error": err})
                    self.log_event("model_failed", {"model": model, "error": err})

        state["failed_models"] = [j["model"] for j in failed_jobs]
        self.save_state(state)
        write_json(self.retry_queue_path, {"ts_utc": utc_now_iso(), "failed_jobs": failed_jobs})
        if failed_jobs:
            raise RuntimeError(f"{len(failed_jobs)} model shard(s) failed; see retry queue.")
        self.mark_step(state, step)

    def aggregate(self, state: dict) -> None:
        step = "aggregate"
        if step in state["steps_completed"]:
            return

        shard_files = sorted(self.shards_dir.glob("*.json"))
        rows = [read_json(p) for p in shard_files]

        completed = [r for r in rows if r.get("status") == "complete"]
        skipped = [r for r in rows if r.get("status") != "complete"]

        strategies = [
            "no_routing",
            "mirror_domain_routing",
            "confidence_threshold_budget_matched",
            "self_consistency_budget_matched",
            "conformal_style",
        ]

        macro: dict[str, dict[str, float | None]] = {}
        for strategy in strategies:
            weak_cfr_vals = []
            overall_fail_vals = []
            autonomy_vals = []
            escalation_vals = []
            reduction_vals = []
            for r in completed:
                m = r.get("strategy_metrics", {}).get(strategy, {})
                weak_cfr = m.get("weak_cfr")
                overall_fail = m.get("overall_failure_rate")
                autonomy = m.get("autonomy_rate")
                escalation = m.get("escalation_rate")
                reduction = m.get("weak_cfr_reduction_vs_no_routing")

                if weak_cfr is not None:
                    weak_cfr_vals.append(weak_cfr)
                if overall_fail is not None:
                    overall_fail_vals.append(overall_fail)
                if autonomy is not None:
                    autonomy_vals.append(autonomy)
                if escalation is not None:
                    escalation_vals.append(escalation)
                if reduction is not None:
                    reduction_vals.append(reduction)

            def mean_or_none(vals: list[float]) -> float | None:
                return sum(vals) / len(vals) if vals else None

            macro[strategy] = {
                "mean_weak_cfr": mean_or_none(weak_cfr_vals),
                "mean_overall_failure_rate": mean_or_none(overall_fail_vals),
                "mean_autonomy_rate": mean_or_none(autonomy_vals),
                "mean_escalation_rate": mean_or_none(escalation_vals),
                "mean_weak_cfr_reduction_vs_no_routing": mean_or_none(reduction_vals),
                "n_models": len(weak_cfr_vals),
            }

        summary = {
            "run_id": self.run_id,
            "created_at_utc": read_json(self.run_dir / "manifest.json").get("created_at_utc"),
            "completed_models": sorted([r["model"] for r in completed]),
            "skipped_models": sorted([r.get("model", "unknown") for r in skipped]),
            "n_models_complete": len(completed),
            "n_models_skipped": len(skipped),
            "macro_summary": {k: {kk: nan_to_none(vv) for kk, vv in v.items()} for k, v in macro.items()},
            "per_model": completed,
            "skipped_details": skipped,
        }
        write_json(self.summary_json, summary)

        def pct(v: float | None) -> str:
            return "NA" if v is None else f"{100.0 * v:.1f}%"

        lines = [
            "# Exp9 Instance-Level Abstention Baseline Comparison",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Completed models: {len(completed)}",
            f"- Skipped models: {len(skipped)}",
            "",
            "## Macro Summary (Across Completed Models)",
            "",
            "| Strategy | Mean Weak CFR | Mean Weak CFR Reduction vs No Routing | Mean Autonomy | Mean Escalation | Mean Overall Failure | N Models |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in strategies:
            m = summary["macro_summary"][s]
            lines.append(
                f"| `{s}` | {pct(m['mean_weak_cfr'])} | {pct(m['mean_weak_cfr_reduction_vs_no_routing'])} | "
                f"{pct(m['mean_autonomy_rate'])} | {pct(m['mean_escalation_rate'])} | "
                f"{pct(m['mean_overall_failure_rate'])} | {m['n_models']} |"
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Frame: Condition 1, Paradigm 3, `api_success=true` rows only.",
                "- Weak-domain policy: `median_or_bottom_k` (fallback `k=2`) from merged Exp1 natural accuracy.",
                "- `mirror_domain_routing` escalates all weak-domain components.",
                "- `confidence_threshold_budget_matched` and `self_consistency_budget_matched` match MIRROR domain-routing escalation budget per model.",
                "- `conformal_style` is a split-calibrated thresholding baseline over the confidence-uncertainty proxy.",
                "",
            ]
        )
        self.summary_md.write_text("\n".join(lines), encoding="utf-8")
        self.log_event("aggregate_written", {"summary_json": str(self.summary_json), "summary_md": str(self.summary_md)})
        self.mark_step(state, step)

    def run(self) -> None:
        state = self.load_state()
        self.save_state(state)
        self.prepare_manifest(state)
        self.run_model_shards(state)
        self.aggregate(state)
        self.log_event("run_complete", {"run_id": self.run_id})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-safe Exp9 instance-level abstention baseline comparison.")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp9_instance_baselines_%Y%m%dT%H%M%S"))
    parser.add_argument("--result-files", nargs="+", type=Path, default=DEFAULT_RESULT_FILES)
    parser.add_argument("--result-glob", type=str, default=DEFAULT_RESULT_GLOB)
    parser.add_argument("--model-list-file", type=Path, default=DEFAULT_MODEL_LIST_FILE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--bottom-k", type=int, default=2)
    parser.add_argument("--conformal-target-error", type=float, default=0.25)
    parser.add_argument("--max-workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) // 2)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = resolve_result_files(args.result_files, args.result_glob)
    if not result_files:
        raise FileNotFoundError(
            f"No Exp9 result files found from --result-files and --result-glob={args.result_glob!r}"
        )
    runner = Runner(
        run_id=args.run_id,
        result_files=result_files,
        model_list_file=args.model_list_file,
        out_root=args.out_root,
        bottom_k=max(1, args.bottom_k),
        conformal_target_error=args.conformal_target_error,
        max_workers=max(1, args.max_workers),
        result_glob=args.result_glob,
    )
    runner.run()
    print(f"Done: {runner.run_dir}")


if __name__ == "__main__":
    main()
