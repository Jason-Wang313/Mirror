"""
Crash-safe parse/API missingness analysis for MIRROR core evidence paths.

This script quantifies missingness mechanisms and their impact on primary
metrics under multiple handling policies:
  - complete-case (current)
  - conservative (treat missing as incorrect/proceed)
  - inverse-probability weighting (IPW)

Outputs:
  - parse_missingness_summary.json
  - parse_missingness_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scipy.stats import chi2_contingency
except Exception:
    chi2_contingency = None

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:
    DictVectorizer = None
    LogisticRegression = None
    roc_auc_score = None


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_OUT_ROOT = RESULTS_DIR / "parse_missingness"
DEFAULT_EXP1_GLOBS = ["exp1_*_results.jsonl", "exp1_*_shard.jsonl", "exp1_gemini_fixed_results.jsonl"]
DEFAULT_EXP9_GLOBS = ["exp9*_results*.jsonl"]
DEFAULT_EXP3_CCE_PATH = ROOT / "audit" / "human_baseline_packet" / "results" / "exp3_difficulty_control_summary.json"


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


def length_bin(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 200:
        return "1-200"
    if n <= 500:
        return "201-500"
    if n <= 1000:
        return "501-1000"
    if n <= 2000:
        return "1001-2000"
    return "2001+"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def pct(x: float | None) -> float | None:
    if x is None:
        return None
    return 100.0 * x


def dedup_files(globs: list[str]) -> list[Path]:
    uniq: dict[str, Path] = {}
    for pat in globs:
        for p in RESULTS_DIR.glob(pat):
            if p.exists():
                uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda p: (p.stat().st_mtime, p.name.lower()))


def load_exp1_accuracy_latest() -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
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
            valid = 0
            tmp: dict[str, float] = {}
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
                merged[model] = tmp
                quality[model] = valid
    return merged


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


def build_exp1_rows(files: list[Path]) -> list[dict[str, Any]]:
    # latest-file-wins dedup on (model, question_id, channel)
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
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
                qid = r.get("question_id")
                channel_name = r.get("channel_name") or f"ch{r.get('channel')}"
                if not model or not qid:
                    continue
                key = (str(model), str(qid), str(channel_name))
                dedup[key] = r

    out: list[dict[str, Any]] = []
    for (_m, _qid, _ch), r in dedup.items():
        raw = str(r.get("raw_response") or "")
        obs = bool(r.get("parse_success", False))
        row = {
            "experiment": "exp1",
            "model": str(r.get("model")),
            "item_id": str(r.get("question_id")),
            "channel": str(r.get("channel_name") or f"ch{r.get('channel')}"),
            "domain": str(r.get("domain") or "unknown"),
            "difficulty": str(r.get("difficulty") or "unknown"),
            "len_bin": length_bin(len(raw)),
            "response_len": len(raw),
            "observed": 1 if obs else 0,
            "correct": 1 if bool(r.get("answer_correct", False)) else 0,
            "condition": None,
            "paradigm": None,
            "slot": None,
        }
        out.append(row)
    return out


def build_exp9_rows(files: list[Path]) -> list[dict[str, Any]]:
    # latest-file-wins dedup on (model, task_id, condition, paradigm, slot)
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
                task_id = r.get("task_id")
                cond = r.get("condition")
                par = r.get("paradigm")
                if model is None or task_id is None or cond is None or par is None:
                    continue
                for slot in ("a", "b"):
                    key = (str(model), str(task_id), int(cond), int(par), slot)
                    dedup[key] = r

    out: list[dict[str, Any]] = []
    for (_m, _tid, cond, par, slot), r in dedup.items():
        raw = str(r.get("raw_response") or "")
        obs = bool(r.get("api_success", False))
        dom = r.get(f"domain_{slot}") or "unknown"
        diff = r.get(f"difficulty_{slot}") or "unknown"
        corr = r.get(f"component_{slot}_correct")
        out.append(
            {
                "experiment": "exp9",
                "model": str(r.get("model")),
                "item_id": f"{r.get('task_id')}::{slot}",
                "channel": "agentic_component",
                "domain": str(dom),
                "difficulty": str(diff),
                "len_bin": length_bin(len(raw)),
                "response_len": len(raw),
                "observed": 1 if obs else 0,
                "correct": 1 if bool(corr) else 0,
                "condition": int(cond),
                "paradigm": int(par),
                "slot": slot,
                "externally_routed": bool(r.get(f"component_{slot}_externally_routed", False)),
            }
        )
    return out


def grouped_missingness(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    by: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in keys)
        by[key].append(r)
    out: dict[str, Any] = {}
    for key, group in by.items():
        n = len(group)
        obs = sum(int(g["observed"]) for g in group)
        miss = n - obs
        out["|".join(str(x) for x in key)] = {
            "n": n,
            "n_observed": obs,
            "n_missing": miss,
            "missing_rate": miss / n if n else None,
        }
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def chi_square_test(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    # contingency table: feature level x observed
    levels: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        level = str(r.get(feature))
        levels[level][int(r["observed"])] += 1

    if len(levels) < 2:
        return {"feature": feature, "status": "insufficient_levels"}
    table = []
    labels = []
    for lvl, cnt in sorted(levels.items()):
        labels.append(lvl)
        table.append([cnt.get(0, 0), cnt.get(1, 0)])
    if chi2_contingency is None:
        return {
            "feature": feature,
            "status": "scipy_unavailable",
            "levels": labels,
        }
    try:
        chi2, p, dof, _ = chi2_contingency(table)
    except Exception as e:
        return {"feature": feature, "status": "error", "error": str(e)}
    return {
        "feature": feature,
        "status": "ok",
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "levels": labels,
        "table": table,
    }


def fit_logistic_missingness(rows: list[dict[str, Any]], feature_keys: list[str]) -> dict[str, Any]:
    if DictVectorizer is None or LogisticRegression is None:
        return {"status": "sklearn_unavailable"}
    if len(rows) < 50:
        return {"status": "insufficient_rows", "n_rows": len(rows)}

    X_dict = []
    y = []
    for r in rows:
        feats = {k: str(r.get(k)) for k in feature_keys}
        X_dict.append(feats)
        y.append(int(r["observed"]))
    y_pos = sum(y)
    if y_pos == 0 or y_pos == len(y):
        return {"status": "degenerate_target", "n_rows": len(y), "positive_rate": y_pos / len(y)}

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    clf = LogisticRegression(max_iter=500, solver="liblinear", n_jobs=1)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba) if roc_auc_score is not None else None

    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    pairs = sorted(zip(names, coefs), key=lambda kv: abs(kv[1]), reverse=True)
    top = []
    for name, coef in pairs[:30]:
        top.append(
            {
                "feature": name,
                "coef": float(coef),
                "odds_ratio": float(math.exp(coef)),
            }
        )
    return {
        "status": "ok",
        "n_rows": len(y),
        "positive_rate": y_pos / len(y),
        "auc": float(auc) if auc is not None else None,
        "intercept": float(clf.intercept_[0]),
        "top_features_abs_coef": top,
        "proba": proba.tolist(),
    }


def exp1_natural_accuracy(rows: list[dict[str, Any]], mode: str, ipw_prob: dict[str, float] | None = None) -> dict[str, Any]:
    # Natural channel only.
    nat = [r for r in rows if r["experiment"] == "exp1" and r["channel"] == "natural"]
    by_model: dict[str, dict[str, float]] = {}
    for r in nat:
        model = r["model"]
        key = r["item_id"]
        obs = int(r["observed"])
        corr = int(r["correct"]) if obs else 0

        if mode == "complete":
            if obs == 0:
                continue
            w = 1.0
            y = corr
        elif mode == "conservative":
            w = 1.0
            y = corr if obs else 0
        elif mode == "ipw":
            if obs == 0:
                continue
            p = ipw_prob.get(f"exp1::{model}::{key}::natural", 0.5) if ipw_prob else 0.5
            p = max(1e-3, min(1.0, p))
            w = min(20.0, 1.0 / p)
            y = corr
        else:
            raise ValueError(f"Unknown mode: {mode}")

        agg = by_model.setdefault(model, {"wy": 0.0, "w": 0.0, "n": 0.0})
        agg["wy"] += w * y
        agg["w"] += w
        agg["n"] += 1.0

    model_acc: dict[str, float] = {}
    for model, agg in by_model.items():
        if agg["w"] > 0:
            model_acc[model] = agg["wy"] / agg["w"]
    return {
        "mode": mode,
        "n_models": len(model_acc),
        "mean_natural_accuracy": mean(list(model_acc.values())),
        "per_model_natural_accuracy": model_acc,
    }


def exp9_weak_cfr(rows: list[dict[str, Any]], mode: str, weak_map: dict[str, set[str]], ipw_prob: dict[str, float] | None = None) -> dict[str, Any]:
    # Evaluate weak-domain CFR for conditions 1 and 4.
    rows9 = [r for r in rows if r["experiment"] == "exp9" and r.get("condition") in {1, 4}]
    per_model_cond: dict[tuple[str, int], dict[str, float]] = {}
    for r in rows9:
        model = r["model"]
        cond = int(r["condition"])
        domain = r["domain"]
        if domain not in weak_map.get(model, set()):
            continue

        obs = int(r["observed"])
        corr = int(r["correct"]) if obs else 0
        ext = bool(r.get("externally_routed", False)) if obs else False

        if mode == "complete":
            if obs == 0:
                continue
            w = 1.0
            fail = 1 if ((not ext) and (corr == 0)) else 0
        elif mode == "conservative":
            w = 1.0
            if obs == 0:
                fail = 1
            else:
                fail = 1 if ((not ext) and (corr == 0)) else 0
        elif mode == "ipw":
            if obs == 0:
                continue
            p = ipw_prob.get(f"exp9::{model}::{r['item_id']}::{cond}::{r.get('paradigm')}", 0.5) if ipw_prob else 0.5
            p = max(1e-3, min(1.0, p))
            w = min(20.0, 1.0 / p)
            fail = 1 if ((not ext) and (corr == 0)) else 0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        agg = per_model_cond.setdefault((model, cond), {"wf": 0.0, "w": 0.0, "n": 0.0})
        agg["wf"] += w * fail
        agg["w"] += w
        agg["n"] += 1.0

    per_model: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, cond), agg in per_model_cond.items():
        if agg["w"] > 0:
            per_model[model][f"c{cond}_weak_cfr"] = agg["wf"] / agg["w"]

    c1_vals = [d["c1_weak_cfr"] for d in per_model.values() if "c1_weak_cfr" in d]
    c4_vals = [d["c4_weak_cfr"] for d in per_model.values() if "c4_weak_cfr" in d]
    red_vals = []
    for d in per_model.values():
        if "c1_weak_cfr" in d and "c4_weak_cfr" in d and d["c1_weak_cfr"] > 0:
            red_vals.append((d["c1_weak_cfr"] - d["c4_weak_cfr"]) / d["c1_weak_cfr"])

    return {
        "mode": mode,
        "n_models_with_c1": len(c1_vals),
        "n_models_with_c4": len(c4_vals),
        "mean_c1_weak_cfr": mean(c1_vals),
        "mean_c4_weak_cfr": mean(c4_vals),
        "mean_cfr_reduction_c1_to_c4": mean(red_vals),
        "per_model": per_model,
    }


def exp9_mnar_bounds(rows: list[dict[str, Any]], weak_map: dict[str, set[str]], alpha: float) -> dict[str, Any]:
    # alpha in [0,1] controls how much missingness is assumed adversarially informative.
    alpha = max(0.0, min(1.0, alpha))
    rows9 = [r for r in rows if r["experiment"] == "exp9" and r.get("condition") in {1, 4}]
    agg: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: {"obs_total": 0.0, "obs_fail": 0.0, "miss_total": 0.0})
    for r in rows9:
        model = r["model"]
        cond = int(r["condition"])
        if r["domain"] not in weak_map.get(model, set()):
            continue
        key = (model, cond)
        obs = int(r["observed"])
        if obs:
            ext = bool(r.get("externally_routed", False))
            corr = int(r["correct"])
            fail = 1 if ((not ext) and (corr == 0)) else 0
            agg[key]["obs_total"] += 1.0
            agg[key]["obs_fail"] += float(fail)
        else:
            agg[key]["miss_total"] += 1.0

    per_model_bounds: dict[str, dict[str, float]] = {}
    red_low_vals = []
    red_high_vals = []
    for model in sorted({m for m, _ in agg.keys()}):
        c1 = agg.get((model, 1), {})
        c4 = agg.get((model, 4), {})
        if not c1 or not c4:
            continue
        c1_obs_t, c1_obs_f, c1_m = c1.get("obs_total", 0.0), c1.get("obs_fail", 0.0), c1.get("miss_total", 0.0) * alpha
        c4_obs_t, c4_obs_f, c4_m = c4.get("obs_total", 0.0), c4.get("obs_fail", 0.0), c4.get("miss_total", 0.0) * alpha
        if c1_obs_t + c1_m <= 0 or c4_obs_t + c4_m <= 0:
            continue

        # Lower/upper CFR bounds per condition.
        c1_low = c1_obs_f / (c1_obs_t + c1_m)
        c1_high = (c1_obs_f + c1_m) / (c1_obs_t + c1_m)
        c4_low = c4_obs_f / (c4_obs_t + c4_m)
        c4_high = (c4_obs_f + c4_m) / (c4_obs_t + c4_m)

        # Worst/best reduction bounds.
        red_low = None
        red_high = None
        if c1_low > 0:
            red_low = (c1_low - c4_high) / c1_low
        if c1_high > 0:
            red_high = (c1_high - c4_low) / c1_high

        if red_low is not None:
            red_low_vals.append(red_low)
        if red_high is not None:
            red_high_vals.append(red_high)

        per_model_bounds[model] = {
            "c1_low": c1_low,
            "c1_high": c1_high,
            "c4_low": c4_low,
            "c4_high": c4_high,
            "reduction_low": red_low,
            "reduction_high": red_high,
        }

    return {
        "alpha": alpha,
        "n_models": len(per_model_bounds),
        "mean_reduction_low": mean([v for v in red_low_vals if v is not None]),
        "mean_reduction_high": mean([v for v in red_high_vals if v is not None]),
        "per_model_bounds": per_model_bounds,
    }


def load_cce_anchor(exp3_cce_path: Path) -> dict[str, Any]:
    if not exp3_cce_path.exists():
        return {
            "available": False,
            "path": str(exp3_cce_path),
            "baseline_mean_balanced_cce": None,
            "mnar_bound_delta_low": None,
            "mnar_bound_delta_high": None,
            "note": "Exp3 CCE anchor file missing.",
        }
    try:
        obj = json.loads(exp3_cce_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "available": False,
            "path": str(exp3_cce_path),
            "baseline_mean_balanced_cce": None,
            "mnar_bound_delta_low": None,
            "mnar_bound_delta_high": None,
            "note": "Failed to parse Exp3 CCE anchor file.",
        }
    baseline = obj.get("aggregate", {}).get("mean_balanced_cce")
    if baseline is None:
        baseline = obj.get("macro", {}).get("mean_balanced_cce")
    return {
        "available": baseline is not None,
        "path": str(exp3_cce_path),
        "baseline_mean_balanced_cce": baseline,
        # Missingness diagnostics here are Exp1/Exp9-specific; treat CCE movement as anchored-no-shift.
        "mnar_bound_delta_low": 0.0 if baseline is not None else None,
        "mnar_bound_delta_high": 0.0 if baseline is not None else None,
        "note": "CCE anchor comes from completed Exp3-v2 bank; parse/API missingness stress here primarily targets Exp1/Exp9.",
    }


def make_markdown(path: Path, summary: dict) -> None:
    sens = summary["sensitivity"]
    e1 = sens["exp1_natural_accuracy"]
    e9 = sens["exp9_weak_cfr"]
    def f4(v: float | None) -> str:
        return "NA" if v is None else f"{float(v):.4f}"
    lines = [
        "# Parse/API Missingness Analysis",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Created (UTC): {summary['created_at_utc']}",
        f"- Exp1 rows: {summary['n_rows']['exp1']}",
        f"- Exp9 component rows: {summary['n_rows']['exp9_components']}",
        f"- Primary imputation mode: `{summary['imputation_mode']}`",
        f"- MNAR bound mode: `{summary['mnar_bound_mode']}` (alpha={summary['mnar_assumed_fraction']:.2f})",
        "",
        "## Missingness Overview",
        "",
        f"- Exp1 parse-missing rate: {pct(summary['missing_rates']['exp1_parse_missing_rate']):.2f}%",
        f"- Exp9 API-missing rate: {pct(summary['missing_rates']['exp9_api_missing_rate']):.2f}%",
        "",
        "## Sensitivity: Exp1 Natural Accuracy",
        "",
        "| Mode | Mean Nat.Acc | Models |",
        "| --- | ---: | ---: |",
    ]
    for mode in ("complete", "conservative", "ipw"):
        d = e1[mode]
        m = d["mean_natural_accuracy"]
        lines.append(f"| `{mode}` | {m:.4f} | {d['n_models']} |")
    lines.extend(
        [
            "",
            "## Sensitivity: Exp9 Weak-Domain CFR",
            "",
            "| Mode | Mean C1 Weak CFR | Mean C4 Weak CFR | Mean Reduction C1→C4 | Models(C1/C4) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("complete", "conservative", "ipw"):
        d = e9[mode]
        lines.append(
            f"| `{mode}` | {d['mean_c1_weak_cfr']:.4f} | {d['mean_c4_weak_cfr']:.4f} | "
            f"{d['mean_cfr_reduction_c1_to_c4']:.4f} | {d['n_models_with_c1']}/{d['n_models_with_c4']} |"
        )
    mnar = summary.get("mnar_bounds_exp9", {})
    if mnar:
        lines.append(
            f"| `mnar_bound` | NA | NA | [{f4(mnar.get('mean_reduction_low'))}, {f4(mnar.get('mean_reduction_high'))}] | {mnar.get('n_models', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Headline Delta Summary",
            "",
            f"- C1→C4 reduction delta vs complete-case (conservative): {f4(summary['headline_delta_summary']['cfr_reduction_delta_conservative_vs_complete'])}",
            f"- C1→C4 reduction delta vs complete-case (IPW): {f4(summary['headline_delta_summary']['cfr_reduction_delta_ipw_vs_complete'])}",
            f"- C1→C4 reduction MNAR bounds: [{f4(summary['headline_delta_summary']['cfr_reduction_mnar_low'])}, {f4(summary['headline_delta_summary']['cfr_reduction_mnar_high'])}]",
            f"- CCE anchor movement under this missingness stress: [{f4(summary['headline_delta_summary']['cce_mnar_delta_low'])}, {f4(summary['headline_delta_summary']['cce_mnar_delta_high'])}]",
            "",
            "## Mechanism Diagnostics",
            "",
            f"- Exp1 logistic AUC (observed vs missing): {summary['logistic']['exp1'].get('auc')}",
            f"- Exp9 logistic AUC (observed vs missing): {summary['logistic']['exp9'].get('auc')}",
            "- Full per-feature chi-square tests and coefficient tables are in the JSON output.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Runner:
    run_id: str
    exp1_globs: list[str]
    exp9_globs: list[str]
    out_root: Path
    imputation_mode: str
    mnar_bound_mode: str
    exp3_cce_path: Path

    def __post_init__(self) -> None:
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "progress_log.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_json = self.run_dir / "parse_missingness_summary.json"
        self.summary_md = self.run_dir / "parse_missingness_summary.md"

    def load_state(self) -> dict:
        state = read_json(self.checkpoint_path)
        if not state:
            state = {
                "run_id": self.run_id,
                "created_at_utc": utc_now_iso(),
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

        exp1_files = dedup_files(self.exp1_globs)
        exp9_files = dedup_files(self.exp9_globs)
        self.log("files_resolved", {"exp1_files": len(exp1_files), "exp9_files": len(exp9_files)})

        exp1_rows = build_exp1_rows(exp1_files)
        exp9_rows = build_exp9_rows(exp9_files)
        all_rows = exp1_rows + exp9_rows
        self.log("rows_loaded", {"exp1_rows": len(exp1_rows), "exp9_rows": len(exp9_rows)})

        # Grouped missingness
        grp = {
            "exp1_by_model_channel": grouped_missingness(exp1_rows, ["model", "channel"]),
            "exp1_by_domain_difficulty": grouped_missingness(exp1_rows, ["domain", "difficulty"]),
            "exp1_by_len_bin": grouped_missingness(exp1_rows, ["len_bin"]),
            "exp9_by_model_condition_paradigm": grouped_missingness(exp9_rows, ["model", "condition", "paradigm"]),
            "exp9_by_domain_difficulty": grouped_missingness(exp9_rows, ["domain", "difficulty"]),
            "exp9_by_len_bin": grouped_missingness(exp9_rows, ["len_bin"]),
        }

        # Association tests
        tests = {
            "exp1": [
                chi_square_test(exp1_rows, "model"),
                chi_square_test(exp1_rows, "channel"),
                chi_square_test(exp1_rows, "domain"),
                chi_square_test(exp1_rows, "difficulty"),
                chi_square_test(exp1_rows, "len_bin"),
            ],
            "exp9": [
                chi_square_test(exp9_rows, "model"),
                chi_square_test(exp9_rows, "condition"),
                chi_square_test(exp9_rows, "paradigm"),
                chi_square_test(exp9_rows, "domain"),
                chi_square_test(exp9_rows, "difficulty"),
                chi_square_test(exp9_rows, "len_bin"),
            ],
        }

        log_exp1 = fit_logistic_missingness(exp1_rows, ["model", "channel", "domain", "difficulty", "len_bin"])
        log_exp9 = fit_logistic_missingness(exp9_rows, ["model", "condition", "paradigm", "domain", "difficulty", "len_bin", "slot"])
        self.log("logistic_fit_done", {"exp1_status": log_exp1.get("status"), "exp9_status": log_exp9.get("status")})

        # Build per-row predicted observation probabilities for IPW.
        ipw_prob: dict[str, float] = {}
        if log_exp1.get("status") == "ok":
            probs = log_exp1.get("proba", [])
            for i, r in enumerate(exp1_rows):
                if i < len(probs) and r.get("channel") == "natural":
                    ipw_prob[f"exp1::{r['model']}::{r['item_id']}::natural"] = float(probs[i])
        if log_exp9.get("status") == "ok":
            probs = log_exp9.get("proba", [])
            for i, r in enumerate(exp9_rows):
                if i < len(probs):
                    ipw_prob[f"exp9::{r['model']}::{r['item_id']}::{r['condition']}::{r['paradigm']}"] = float(probs[i])

        acc_map = load_exp1_accuracy_latest()
        weak_map = {m: weak_domains_for_model(d, bottom_k=2) for m, d in acc_map.items()}

        modes = ["complete", "conservative", "ipw"] if self.imputation_mode == "all" else [self.imputation_mode]
        exp1_sens = {}
        exp9_sens = {}
        for mode in ("complete", "conservative", "ipw"):
            exp1_sens[mode] = exp1_natural_accuracy(all_rows, mode=mode, ipw_prob=ipw_prob)
            exp9_sens[mode] = exp9_weak_cfr(all_rows, mode=mode, weak_map=weak_map, ipw_prob=ipw_prob)

        mnar_alpha_map = {
            "none": 0.0,
            "weak": 0.25,
            "moderate": 0.50,
            "strong": 1.0,
        }
        mnar_alpha = float(mnar_alpha_map.get(self.mnar_bound_mode, 0.5))
        mnar_bounds = exp9_mnar_bounds(all_rows, weak_map=weak_map, alpha=mnar_alpha)
        cce_anchor = load_cce_anchor(self.exp3_cce_path)

        complete_red = exp9_sens["complete"].get("mean_cfr_reduction_c1_to_c4")
        conservative_red = exp9_sens["conservative"].get("mean_cfr_reduction_c1_to_c4")
        ipw_red = exp9_sens["ipw"].get("mean_cfr_reduction_c1_to_c4")
        cfr_delta_cons = (conservative_red - complete_red) if (complete_red is not None and conservative_red is not None) else None
        cfr_delta_ipw = (ipw_red - complete_red) if (complete_red is not None and ipw_red is not None) else None
        headline_delta_summary = {
            "cfr_reduction_complete": complete_red,
            "cfr_reduction_conservative": conservative_red,
            "cfr_reduction_ipw": ipw_red,
            "cfr_reduction_delta_conservative_vs_complete": cfr_delta_cons,
            "cfr_reduction_delta_ipw_vs_complete": cfr_delta_ipw,
            "cfr_reduction_mnar_low": mnar_bounds.get("mean_reduction_low"),
            "cfr_reduction_mnar_high": mnar_bounds.get("mean_reduction_high"),
            "cce_anchor": cce_anchor.get("baseline_mean_balanced_cce"),
            "cce_mnar_delta_low": cce_anchor.get("mnar_bound_delta_low"),
            "cce_mnar_delta_high": cce_anchor.get("mnar_bound_delta_high"),
            "cce_note": cce_anchor.get("note"),
        }

        primary_mode = modes[0]
        summary = {
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "imputation_mode": self.imputation_mode,
            "mnar_bound_mode": self.mnar_bound_mode,
            "mnar_assumed_fraction": mnar_alpha,
            "primary_mode_reported": primary_mode,
            "input_files": {
                "exp1": [str(p) for p in exp1_files],
                "exp9": [str(p) for p in exp9_files],
            },
            "n_rows": {
                "exp1": len(exp1_rows),
                "exp9_components": len(exp9_rows),
                "total": len(all_rows),
            },
            "missing_rates": {
                "exp1_parse_missing_rate": 1.0 - (sum(r["observed"] for r in exp1_rows) / len(exp1_rows) if exp1_rows else 0.0),
                "exp9_api_missing_rate": 1.0 - (sum(r["observed"] for r in exp9_rows) / len(exp9_rows) if exp9_rows else 0.0),
            },
            "grouped_missingness": grp,
            "association_tests": tests,
            "logistic": {
                "exp1": {k: v for k, v in log_exp1.items() if k != "proba"},
                "exp9": {k: v for k, v in log_exp9.items() if k != "proba"},
            },
            "sensitivity": {
                "exp1_natural_accuracy": exp1_sens,
                "exp9_weak_cfr": exp9_sens,
            },
            "mnar_bounds_exp9": mnar_bounds,
            "cce_anchor": cce_anchor,
            "headline_delta_summary": headline_delta_summary,
        }

        write_json(self.summary_json, summary)
        make_markdown(self.summary_md, summary)
        self.log("summary_written", {"json": str(self.summary_json), "md": str(self.summary_md)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze parse/API missingness with sensitivity re-estimates.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("parse_missingness_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--exp1-globs", nargs="+", default=DEFAULT_EXP1_GLOBS)
    parser.add_argument("--exp9-globs", nargs="+", default=DEFAULT_EXP9_GLOBS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--imputation", type=str, default="all", choices=["complete", "conservative", "ipw", "all"])
    parser.add_argument("--mnar-bound-mode", type=str, default="moderate", choices=["none", "weak", "moderate", "strong"])
    parser.add_argument("--exp3-cce-path", type=Path, default=DEFAULT_EXP3_CCE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Runner(
        run_id=args.run_id,
        exp1_globs=args.exp1_globs,
        exp9_globs=args.exp9_globs,
        out_root=args.out_root,
        imputation_mode=args.imputation,
        mnar_bound_mode=args.mnar_bound_mode,
        exp3_cce_path=args.exp3_cce_path,
    )
    runner.run()
    print(f"Done: {runner.summary_json}")


if __name__ == "__main__":
    main()
