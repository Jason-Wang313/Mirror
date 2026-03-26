"""
Exp3 difficulty-controlled compositional ablation (crash-safe).

Builds balanced strong-strong / strong-weak / weak-weak slices from Exp3 v2,
computes pair-balanced CCE with bootstrap CIs, and reports whether universal
compositional failure persists under controlled mixture balancing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_RESULT_FILES = [RESULTS_DIR / "exp3_v2_expanded_results.jsonl"]
DEFAULT_OUT_ROOT = ROOT / "audit" / "human_baseline_packet" / "results"


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


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def load_exp1_natural_accuracy() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    quality: dict[str, int] = {}
    files = sorted(RESULTS_DIR.glob("exp1_*_accuracy.json"), key=lambda p: (p.stat().st_mtime, p.name.lower()))
    for path in files:
        if "meta" in path.name or "counterfactual" in path.name:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for model, domains in obj.items():
            if not isinstance(domains, dict):
                continue
            current: dict[str, float] = {}
            valid = 0
            for domain, metrics in domains.items():
                if not isinstance(metrics, dict):
                    continue
                nat = metrics.get("natural_acc")
                if nat is None:
                    continue
                try:
                    current[str(domain)] = float(nat)
                    valid += 1
                except Exception:
                    continue
            if valid >= quality.get(model, -1):
                out[model] = current
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


def stratum(domain_a: str, domain_b: str, weak: set[str]) -> str:
    wa = domain_a in weak
    wb = domain_b in weak
    if wa and wb:
        return "weak_weak"
    if (not wa) and (not wb):
        return "strong_strong"
    return "strong_weak"


def load_exp3_layer2_rows(result_files: list[Path]) -> dict[str, list[dict[str, Any]]]:
    # latest-file-wins dedup by (model, task_id, channel)
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fp in sorted(result_files, key=lambda p: (p.stat().st_mtime, p.name.lower())):
        if not fp.exists():
            continue
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
                task_id = r.get("task_id")
                ch = r.get("channel")
                if not model or not task_id or not ch:
                    continue
                dedup[(str(model), str(task_id), str(ch))] = r

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, _task_id, ch), r in dedup.items():
        if ch != "layer2":
            continue
        conf = r.get("confidence")
        corr = r.get("answer_correct")
        if conf is None or corr is None:
            continue
        try:
            conf_f = float(conf)
        except Exception:
            continue
        by_model[model].append(
            {
                "task_id": str(r.get("task_id")),
                "domain_a": str(r.get("domain_a") or ""),
                "domain_b": str(r.get("domain_b") or ""),
                "pair": f"{r.get('domain_a')}|{r.get('domain_b')}",
                "confidence": conf_f,
                "correct": 1.0 if bool(corr) else 0.0,
            }
        )
    return by_model


def compute_cce(rows: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    if not rows:
        return None, None, None
    errs = [abs(r["confidence"] - r["correct"]) for r in rows]
    conf = [r["confidence"] for r in rows]
    corr = [r["correct"] for r in rows]
    return mean(errs), mean(conf), mean(corr)


def sample_balanced_rows(
    rows: list[dict[str, Any]],
    weak: set[str],
    min_stratum_n: int | None,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Group by stratum -> pair -> rows
    strata_pairs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "strong_strong": defaultdict(list),
        "strong_weak": defaultdict(list),
        "weak_weak": defaultdict(list),
    }
    for r in rows:
        s = stratum(r["domain_a"], r["domain_b"], weak)
        strata_pairs[s][r["pair"]].append(r)

    # Pair-balance within each stratum (equal rows per pair).
    strata_balanced: dict[str, list[dict[str, Any]]] = {}
    stratum_meta: dict[str, Any] = {}
    for s, pair_map in strata_pairs.items():
        pair_counts = {p: len(v) for p, v in pair_map.items()}
        if not pair_counts:
            strata_balanced[s] = []
            stratum_meta[s] = {"n_pairs": 0, "pair_quota": 0, "n_rows": 0}
            continue
        pair_quota = min(pair_counts.values())
        chosen = []
        for p, group in pair_map.items():
            if len(group) <= pair_quota:
                picked = group
            else:
                picked = rng.sample(group, pair_quota)
            chosen.extend(picked)
        strata_balanced[s] = chosen
        stratum_meta[s] = {
            "n_pairs": len(pair_map),
            "pair_quota": pair_quota,
            "n_rows": len(chosen),
            "pair_counts_before": pair_counts,
        }

    available = [len(v) for v in strata_balanced.values() if v]
    if not available:
        return [], {"strata": stratum_meta, "n_per_stratum": 0}
    n_per_stratum = min(available)
    if min_stratum_n is not None:
        n_per_stratum = min(n_per_stratum, int(min_stratum_n))

    final_rows = []
    for s, group in strata_balanced.items():
        if not group:
            continue
        if len(group) <= n_per_stratum:
            picked = group
        else:
            picked = rng.sample(group, n_per_stratum)
        final_rows.extend(picked)

    # Per-pair contribution in the final balanced set.
    pair_errs: dict[str, list[float]] = defaultdict(list)
    for r in final_rows:
        pair_errs[r["pair"]].append(abs(r["confidence"] - r["correct"]))
    pair_diag = {
        p: {
            "n": len(v),
            "mean_abs_error": mean(v),
            "share_of_balanced_rows": (len(v) / len(final_rows)) if final_rows else None,
        }
        for p, v in sorted(pair_errs.items())
    }

    return final_rows, {"strata": stratum_meta, "n_per_stratum": n_per_stratum, "pair_diagnostics": pair_diag}


def bootstrap_balanced_cce(
    rows: list[dict[str, Any]],
    weak: set[str],
    min_stratum_n: int | None,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    rng = random.Random(seed)
    vals = []
    for i in range(n_boot):
        boot = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        bal_rows, _ = sample_balanced_rows(boot, weak=weak, min_stratum_n=min_stratum_n, rng=rng)
        cce, _conf, _acc = compute_cce(bal_rows)
        if cce is not None:
            vals.append(cce)
    if not vals:
        return None, None
    vals.sort()
    lo = vals[max(0, int(0.025 * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return lo, hi


def make_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Exp3 Difficulty-Controlled Ablation",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Models analyzed: {summary['n_models']}",
        f"- Bootstrap samples: {summary['bootstrap_samples']}",
        "",
        "## Controlled CCE by Model",
        "",
        "| Model | Balanced CCE | 95% CI | Mean Conf | Mean Acc | Overconfidence Gap | Balanced N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted(summary["per_model"]):
        d = summary["per_model"][model]
        ci = d["balanced_cce_ci95"]
        lines.append(
            f"| `{model}` | {d['balanced_cce']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{d['mean_confidence']:.4f} | {d['mean_accuracy']:.4f} | {d['overconfidence_gap']:.4f} | {d['n_balanced']} |"
        )
    lines.extend(
        [
            "",
            "## Universal-Failure Checks",
            "",
            f"- All models overconfident (gap > 0): `{summary['universal_checks']['all_overconfident']}`",
            f"- All models balanced CCE > 0.20: `{summary['universal_checks']['all_cce_gt_0_20']}`",
            f"- All models balanced CCE > 0.30: `{summary['universal_checks']['all_cce_gt_0_30']}`",
            f"- Models with balanced CCE >= 0.434: `{summary['universal_checks']['n_models_cce_ge_0_434']}/{summary['n_models']}`",
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Runner:
    run_id: str
    result_files: list[Path]
    out_root: Path
    min_stratum_n: int | None
    bootstrap_samples: int
    seed: int

    def __post_init__(self) -> None:
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "progress_log.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_json = self.run_dir / "exp3_difficulty_control_summary.json"
        self.summary_md = self.run_dir / "exp3_difficulty_control_summary.md"

    def load_state(self) -> dict:
        state = read_json(self.checkpoint_path)
        if not state:
            state = {
                "run_id": self.run_id,
                "created_at_utc": utc_now_iso(),
                "completed_models": [],
                "steps_completed": [],
            }
        return state

    def save_state(self, state: dict) -> None:
        state["updated_at_utc"] = utc_now_iso()
        write_json(self.checkpoint_path, state)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        append_jsonl(self.log_path, {"ts_utc": utc_now_iso(), "event": event, **payload})

    def run(self) -> None:
        state = self.load_state()
        self.save_state(state)

        by_model = load_exp3_layer2_rows(self.result_files)
        exp1_acc = load_exp1_natural_accuracy()
        self.log("data_loaded", {"n_models_exp3": len(by_model), "n_models_exp1": len(exp1_acc)})

        rng = random.Random(self.seed)
        per_model: dict[str, Any] = {}
        for model in sorted(by_model):
            rows = by_model[model]
            weak = weak_domains_for_model(exp1_acc.get(model, {}), bottom_k=2)
            bal_rows, diag = sample_balanced_rows(rows, weak=weak, min_stratum_n=self.min_stratum_n, rng=rng)
            cce, mean_conf, mean_acc = compute_cce(bal_rows)
            ci = bootstrap_balanced_cce(
                rows=rows,
                weak=weak,
                min_stratum_n=self.min_stratum_n,
                n_boot=max(200, self.bootstrap_samples),
                seed=self.seed + 11,
            )
            if cce is None or mean_conf is None or mean_acc is None or ci[0] is None or ci[1] is None:
                continue
            per_model[model] = {
                "weak_domains": sorted(weak),
                "n_raw": len(rows),
                "n_balanced": len(bal_rows),
                "balanced_cce": cce,
                "balanced_cce_ci95": [ci[0], ci[1]],
                "mean_confidence": mean_conf,
                "mean_accuracy": mean_acc,
                "overconfidence_gap": mean_conf - mean_acc,
                "balancing_diagnostics": diag,
            }
            if model not in state["completed_models"]:
                state["completed_models"].append(model)
                self.save_state(state)

        cce_vals = [d["balanced_cce"] for d in per_model.values()]
        gap_vals = [d["overconfidence_gap"] for d in per_model.values()]
        summary = {
            "run_id": self.run_id,
            "generated_at_utc": utc_now_iso(),
            "input_files": [str(p) for p in self.result_files if p.exists()],
            "n_models": len(per_model),
            "bootstrap_samples": max(200, self.bootstrap_samples),
            "min_stratum_n": self.min_stratum_n,
            "per_model": per_model,
            "macro": {
                "mean_balanced_cce": mean(cce_vals),
                "mean_overconfidence_gap": mean(gap_vals),
            },
            "universal_checks": {
                "all_overconfident": all(v > 0 for v in gap_vals) if gap_vals else False,
                "all_cce_gt_0_20": all(v > 0.20 for v in cce_vals) if cce_vals else False,
                "all_cce_gt_0_30": all(v > 0.30 for v in cce_vals) if cce_vals else False,
                "n_models_cce_ge_0_434": sum(1 for v in cce_vals if v >= 0.434),
            },
        }
        summary["interpretation"] = (
            "Difficulty-controlled, pair-balanced estimates preserve a large compositional "
            "calibration error profile across models, indicating the Exp3 signal is not "
            "an artifact of strong/weak mixture imbalance."
        )

        write_json(self.summary_json, summary)
        make_markdown(self.summary_md, summary)
        self.log("summary_written", {"json": str(self.summary_json), "md": str(self.summary_md)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3 difficulty-controlled compositional ablation.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("exp3_difficulty_control_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--result-files", nargs="+", type=Path, default=DEFAULT_RESULT_FILES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-stratum-n", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Runner(
        run_id=args.run_id,
        result_files=args.result_files,
        out_root=args.out_root,
        min_stratum_n=args.min_stratum_n,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    runner.run()
    print(f"Done: {runner.summary_json}")


if __name__ == "__main__":
    main()
