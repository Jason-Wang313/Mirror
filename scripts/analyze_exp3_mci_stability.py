"""
Analyze Exp3 v2 MCI stability and balanced-composition coverage.

Outputs:
  - exp3_mci_stability_summary.json
  - exp3_mci_stability_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scipy.stats import kendalltau as scipy_kendalltau
    from scipy.stats import spearmanr as scipy_spearmanr
except Exception:
    scipy_kendalltau = None
    scipy_spearmanr = None


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
DEFAULT_RESULTS = [RESULTS_DIR / "exp3_v2_expanded_results.jsonl"]
DEFAULT_OUT = ROOT / "paper" / "tables"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def rank(vals: list[float]) -> list[float]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx <= 0 or deny <= 0:
        return float("nan")
    return num / (denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    if scipy_spearmanr is not None:
        rho, _ = scipy_spearmanr(xs, ys)
        return float(rho) if rho is not None else float("nan")
    return pearson(rank(xs), rank(ys))


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    if scipy_kendalltau is not None:
        tau, _ = scipy_kendalltau(xs, ys)
        return float(tau) if tau is not None else float("nan")
    num = 0
    den = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            den += 1
            num += 1 if dx * dy > 0 else -1
    if den == 0:
        return float("nan")
    return num / den


def same_sign(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    if a == 0.0 and b == 0.0:
        return True
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def load_exp1_natural_accuracy() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    files = sorted(RESULTS_DIR.glob("exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for model, domains in obj.items():
            if not isinstance(domains, dict):
                continue
            model_acc = out.setdefault(model, {})
            for domain, metrics in domains.items():
                if not isinstance(metrics, dict):
                    continue
                nat = metrics.get("natural_acc")
                if nat is None:
                    continue
                try:
                    model_acc[domain] = float(nat)
                except Exception:
                    continue
    return {m: d for m, d in out.items() if d}


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


def composition_stratum(domain_a: str, domain_b: str, weak: set[str]) -> str:
    wa = domain_a in weak
    wb = domain_b in weak
    if wa and wb:
        return "weak_weak"
    if (not wa) and (not wb):
        return "strong_strong"
    return "strong_weak"


def score_from_record(r: dict[str, Any]) -> float | None:
    ch = str(r.get("channel"))
    if ch in {"wagering", "layer2"}:
        conf = r.get("confidence")
        if conf is not None:
            try:
                return float(conf)
            except Exception:
                return None
    correct = r.get("answer_correct")
    if correct is None:
        return None
    return 1.0 if bool(correct) else 0.0


def correctness_score(r: dict[str, Any]) -> float | None:
    correct = r.get("answer_correct")
    if correct is None:
        return None
    return 1.0 if bool(correct) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp3 v2 MCI stability.")
    parser.add_argument("--result-files", nargs="+", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-stem", type=str, default="exp3_mci_stability_summary")
    parser.add_argument("--min-pair-quota", type=int, default=4)
    parser.add_argument("--min-shared-for-corr", type=int, default=50)
    parser.add_argument("--max-rho-tau-gap", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [p for p in args.result_files if p.exists()]
    if not files:
        raise FileNotFoundError("No exp3 result files found.")

    # latest-file-wins dedup
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fp in sorted(files, key=lambda p: (p.stat().st_mtime, p.name.lower())):
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
                ch = r.get("channel")
                if not model or not task or not ch:
                    continue
                dedup[(str(model), str(task), str(ch))] = r

    by_model_task: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for (model, task, ch), r in dedup.items():
        by_model_task[model][task][ch] = r

    exp1_acc = load_exp1_natural_accuracy()
    channels = ["wagering", "natural", "layer2"]
    channel_pairs = [("wagering", "natural"), ("wagering", "layer2"), ("natural", "layer2")]

    per_model: dict[str, dict[str, Any]] = {}
    for model in sorted(by_model_task):
        tasks = by_model_task[model]
        weak = weak_domains_for_model(exp1_acc.get(model, {}), bottom_k=2)

        pair_counts: dict[str, int] = defaultdict(int)
        stratum_counts: dict[str, int] = {"strong_strong": 0, "strong_weak": 0, "weak_weak": 0}
        for task_id, ch_map in tasks.items():
            # Take domains from any channel record for this task.
            any_row = next(iter(ch_map.values()))
            d_a = str(any_row.get("domain_a", ""))
            d_b = str(any_row.get("domain_b", ""))
            if d_a and d_b:
                pair_key = f"{d_a}|{d_b}"
                pair_counts[pair_key] += 1
                stratum = composition_stratum(d_a, d_b, weak)
                stratum_counts[stratum] += 1

        corr_rows = []
        rho_vals = []
        tau_vals = []
        min_shared = None
        for c1, c2 in channel_pairs:
            xs, ys = [], []
            raw_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for task_id, ch_map in tasks.items():
                r1 = ch_map.get(c1)
                r2 = ch_map.get(c2)
                if not r1 or not r2:
                    continue
                s1 = score_from_record(r1)
                s2 = score_from_record(r2)
                if s1 is None or s2 is None:
                    continue
                xs.append(s1)
                ys.append(s2)
                raw_pairs.append((r1, r2))
            # If one side is constant (common with parsed confidence fallbacks),
            # back off to correctness-based task difficulty for that channel.
            if xs and len(set(xs)) <= 1:
                x2 = []
                for r1, _r2 in raw_pairs:
                    cx = correctness_score(r1)
                    if cx is not None:
                        x2.append(cx)
                if len(x2) == len(xs):
                    xs = x2
            if ys and len(set(ys)) <= 1:
                y2 = []
                for _r1, r2 in raw_pairs:
                    cy = correctness_score(r2)
                    if cy is not None:
                        y2.append(cy)
                if len(y2) == len(ys):
                    ys = y2
            shared = len(xs)
            if min_shared is None or shared < min_shared:
                min_shared = shared
            rho = spearman(xs, ys) if shared >= 3 else float("nan")
            tau = kendall_tau(xs, ys) if shared >= 3 else float("nan")
            corr_rows.append(
                {
                    "pair": f"{c1}-{c2}",
                    "n_shared": shared,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                }
            )
            if not math.isnan(rho):
                rho_vals.append(rho)
            if not math.isnan(tau):
                tau_vals.append(tau)

        mci_s = sum(rho_vals) / len(rho_vals) if rho_vals else float("nan")
        mci_k = sum(tau_vals) / len(tau_vals) if tau_vals else float("nan")
        # Tau is scale-compressed relative to rho; use scaled comparison.
        rho_tau_gap = (
            abs(mci_s - (1.5 * mci_k))
            if (not math.isnan(mci_s) and not math.isnan(mci_k))
            else float("nan")
        )
        sign_concordance = all(
            same_sign(row["spearman_rho"], row["kendall_tau"])
            for row in corr_rows
            if row["n_shared"] >= 3
        )

        n_pairs = len(pair_counts)
        pair_quota_pass = (
            n_pairs == 28
            and min(pair_counts.values(), default=0) >= int(args.min_pair_quota)
        )
        corr_support_pass = (min_shared or 0) >= int(args.min_shared_for_corr)
        stability_pass = (
            pair_quota_pass
            and corr_support_pass
            and (not math.isnan(rho_tau_gap))
            and rho_tau_gap <= args.max_rho_tau_gap
            and sign_concordance
        )

        per_model[model] = {
            "n_tasks": len(tasks),
            "n_domain_pairs": n_pairs,
            "min_tasks_per_pair": min(pair_counts.values(), default=0),
            "max_tasks_per_pair": max(pair_counts.values(), default=0),
            "stratum_counts": stratum_counts,
            "pair_quota_pass": pair_quota_pass,
            "weak_domains_used": sorted(weak),
            "channel_pair_correlations": corr_rows,
            "mci_spearman": mci_s,
            "mci_kendall": mci_k,
            "rho_tau_gap": rho_tau_gap,
            "min_shared_for_corr": min_shared or 0,
            "corr_support_pass": corr_support_pass,
            "sign_concordance": sign_concordance,
            "stability_pass": stability_pass,
        }

    pass_n = sum(1 for v in per_model.values() if v["stability_pass"])
    summary = {
        "generated_at_utc": utc_now_iso(),
        "input_files": [str(p) for p in files],
        "n_models": len(per_model),
        "criteria": {
            "required_domain_pairs": 28,
            "min_pair_quota": int(args.min_pair_quota),
            "min_shared_for_corr": int(args.min_shared_for_corr),
            "max_rho_tau_gap_scaled": float(args.max_rho_tau_gap),
            "rho_tau_gap_formula": "|rho - 1.5*tau|",
            "sign_concordance_required": True,
        },
        "pass_counts": {"stability_pass": pass_n, "stability_fail": len(per_model) - pass_n},
        "per_model": per_model,
    }

    out_json = args.out_dir / f"{args.summary_stem}.json"
    out_md = args.out_dir / f"{args.summary_stem}.md"
    write_json(out_json, summary)

    lines = [
        "# Exp3 MCI Stability Summary",
        "",
        f"- Generated: {summary['generated_at_utc']}",
        f"- Models analyzed: {summary['n_models']}",
        f"- Stability pass: {pass_n}/{len(per_model)}",
        "",
        "## Criteria",
        "",
        f"- Required domain pairs: {summary['criteria']['required_domain_pairs']}",
        f"- Min tasks per pair: {summary['criteria']['min_pair_quota']}",
        f"- Min shared samples for correlation: {summary['criteria']['min_shared_for_corr']}",
        f"- Max |MCI_spearman - 1.5*MCI_kendall|: {summary['criteria']['max_rho_tau_gap_scaled']}",
        "",
        "## Per-Model",
        "",
        "| Model | Tasks | Pairs | Min/Pair | Strata (SS/SW/WW) | MCI-S | MCI-K | Gap | Stability |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for model in sorted(per_model):
        m = per_model[model]
        s = m["stratum_counts"]
        lines.append(
            f"| `{model}` | {m['n_tasks']} | {m['n_domain_pairs']} | {m['min_tasks_per_pair']} | "
            f"{s['strong_strong']}/{s['strong_weak']}/{s['weak_weak']} | "
            f"{m['mci_spearman']:.3f} | {m['mci_kendall']:.3f} | {m['rho_tau_gap']:.3f} | "
            f"{'PASS' if m['stability_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This table tests whether MCI estimates are supported by balanced pair coverage and stable rank-correlation estimators.",
            "- Estimator concordance uses signed agreement plus scaled-magnitude consistency (`|rho - 1.5*tau|`).",
            "- If a model fails, MCI for that model should remain diagnostic/secondary rather than primary.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
