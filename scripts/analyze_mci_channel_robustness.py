"""
MCI channel-disagreement robustness diagnostics for Exp3 v2.

Computes MCI under:
  - full channels
  - no-wagering
  - leave-one-channel-out variants

Reports rank stability and sign-shift diagnostics (Spearman + Kendall).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
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


def kendall(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    if scipy_kendalltau is not None:
        tau, _ = scipy_kendalltau(xs, ys)
        return float(tau) if tau is not None else float("nan")
    # fallback
    n = len(xs)
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


def score_from_record(r: dict[str, Any]) -> float | None:
    ch = str(r.get("channel"))
    if ch in {"wagering", "layer2"}:
        conf = r.get("confidence")
        if conf is not None:
            try:
                return float(conf)
            except Exception:
                return None
    corr = r.get("answer_correct")
    if corr is None:
        return None
    return 1.0 if bool(corr) else 0.0


def load_exp3(result_files: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    # model -> task_id -> channel -> score
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

    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (model, task_id, ch), r in dedup.items():
        score = score_from_record(r)
        if score is None:
            continue
        out[model][task_id][ch] = score
    return out


def compute_variant(
    model_tasks: dict[str, dict[str, float]],
    channels: list[str],
    min_shared: int,
) -> dict[str, Any]:
    pair_rows = []
    rho_vals = []
    tau_vals = []
    for c1, c2 in combinations(channels, 2):
        xs = []
        ys = []
        for _tid, cmap in model_tasks.items():
            if c1 in cmap and c2 in cmap:
                xs.append(cmap[c1])
                ys.append(cmap[c2])
        n = len(xs)
        if n < min_shared:
            pair_rows.append(
                {
                    "pair": f"{c1}-{c2}",
                    "n_shared": n,
                    "spearman_rho": None,
                    "kendall_tau": None,
                    "included": False,
                }
            )
            continue
        rho = spearman(xs, ys)
        tau = kendall(xs, ys)
        pair_rows.append(
            {
                "pair": f"{c1}-{c2}",
                "n_shared": n,
                "spearman_rho": None if math.isnan(rho) else rho,
                "kendall_tau": None if math.isnan(tau) else tau,
                "included": True,
            }
        )
        if not math.isnan(rho):
            rho_vals.append(rho)
        if not math.isnan(tau):
            tau_vals.append(tau)
    mci_s = mean(rho_vals)
    mci_k = mean(tau_vals)
    return {
        "channels": channels,
        "pair_correlations": pair_rows,
        "mci_spearman": mci_s,
        "mci_kendall": mci_k,
        "n_pairs_included": sum(1 for p in pair_rows if p["included"]),
    }


def rank_corr(a: dict[str, float], b: dict[str, float]) -> float | None:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 3:
        return None
    va = [a[m] for m in common]
    vb = [b[m] for m in common]
    rho = spearman(va, vb)
    return None if math.isnan(rho) else rho


def sign_of(x: float | None) -> int:
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def make_markdown(path: Path, summary: dict) -> None:
    def f4(v: float | None) -> str:
        return "NA" if v is None else f"{v:.4f}"

    lines = [
        "# MCI Channel Robustness (Exp3 v2)",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated (UTC): {summary['generated_at_utc']}",
        f"- Models analyzed: {summary['n_models']}",
        "",
        "## Per-Model Variant MCI (Spearman)",
        "",
        "| Model | Full | No-Wagering | Leave-Out Natural | Leave-Out Layer2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted(summary["per_model"]):
        pm = summary["per_model"][model]
        lines.append(
            f"| `{model}` | {f4(pm['full']['mci_spearman'])} | "
            f"{f4(pm['no_wagering']['mci_spearman'])} | "
            f"{f4(pm['leave_out_natural']['mci_spearman'])} | "
            f"{f4(pm['leave_out_layer2']['mci_spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Rank Stability",
            "",
            f"- Full vs No-Wagering rank Spearman: {summary['rank_stability']['full_vs_no_wagering_rank_spearman']}",
            f"- Full vs Leave-Out-Natural rank Spearman: {summary['rank_stability']['full_vs_leave_out_natural_rank_spearman']}",
            f"- Full vs Leave-Out-Layer2 rank Spearman: {summary['rank_stability']['full_vs_leave_out_layer2_rank_spearman']}",
            "",
            "## Sign Shifts",
            "",
            f"- Full→No-Wagering sign shifts: {summary['sign_shift_counts']['full_to_no_wagering']}",
            f"- Full→Leave-Out-Natural sign shifts: {summary['sign_shift_counts']['full_to_leave_out_natural']}",
            f"- Full→Leave-Out-Layer2 sign shifts: {summary['sign_shift_counts']['full_to_leave_out_layer2']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class Runner:
    run_id: str
    result_files: list[Path]
    out_root: Path
    min_shared: int

    def __post_init__(self) -> None:
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "progress_log.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.summary_json = self.run_dir / "mci_channel_robustness_summary.json"
        self.summary_md = self.run_dir / "mci_channel_robustness_summary.md"

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

        data = load_exp3(self.result_files)
        self.log("data_loaded", {"n_models": len(data)})

        per_model: dict[str, Any] = {}
        for model in sorted(data):
            tasks = data[model]
            full = compute_variant(tasks, ["wagering", "natural", "layer2"], min_shared=self.min_shared)
            no_w = compute_variant(tasks, ["natural", "layer2"], min_shared=self.min_shared)
            loo_w = compute_variant(tasks, ["natural", "layer2"], min_shared=self.min_shared)
            loo_n = compute_variant(tasks, ["wagering", "layer2"], min_shared=self.min_shared)
            loo_l2 = compute_variant(tasks, ["wagering", "natural"], min_shared=self.min_shared)
            if full["mci_spearman"] is None:
                continue
            per_model[model] = {
                "full": full,
                "no_wagering": no_w,
                "leave_out_wagering": loo_w,
                "leave_out_natural": loo_n,
                "leave_out_layer2": loo_l2,
            }

        full_map = {m: d["full"]["mci_spearman"] for m, d in per_model.items()}
        now_map = {m: d["no_wagering"]["mci_spearman"] for m, d in per_model.items() if d["no_wagering"]["mci_spearman"] is not None}
        lon_map = {m: d["leave_out_natural"]["mci_spearman"] for m, d in per_model.items() if d["leave_out_natural"]["mci_spearman"] is not None}
        lol_map = {m: d["leave_out_layer2"]["mci_spearman"] for m, d in per_model.items() if d["leave_out_layer2"]["mci_spearman"] is not None}

        def sign_shift_count(other: dict[str, float]) -> int:
            n = 0
            for m, v in full_map.items():
                if m not in other:
                    continue
                if sign_of(v) != sign_of(other[m]):
                    n += 1
            return n

        summary = {
            "run_id": self.run_id,
            "generated_at_utc": utc_now_iso(),
            "input_files": [str(p) for p in self.result_files if p.exists()],
            "n_models": len(per_model),
            "min_shared_for_corr": self.min_shared,
            "per_model": per_model,
            "rank_stability": {
                "full_vs_no_wagering_rank_spearman": rank_corr(full_map, now_map),
                "full_vs_leave_out_natural_rank_spearman": rank_corr(full_map, lon_map),
                "full_vs_leave_out_layer2_rank_spearman": rank_corr(full_map, lol_map),
            },
            "sign_shift_counts": {
                "full_to_no_wagering": sign_shift_count(now_map),
                "full_to_leave_out_natural": sign_shift_count(lon_map),
                "full_to_leave_out_layer2": sign_shift_count(lol_map),
            },
        }

        write_json(self.summary_json, summary)
        make_markdown(self.summary_md, summary)
        self.log("summary_written", {"json": str(self.summary_json), "md": str(self.summary_md)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCI channel-disagreement robustness diagnostics.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("mci_channel_robustness_%Y%m%dT%H%M%S"),
    )
    parser.add_argument("--result-files", nargs="+", type=Path, default=DEFAULT_RESULT_FILES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-shared-for-corr", type=int, default=30)
    parser.add_argument("--exclude-channels", nargs="*", default=[])
    parser.add_argument("--leave-one-out", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # CLI compatibility: parameters are accepted and currently diagnostics are
    # computed for canonical variants regardless of exclusions.
    runner = Runner(
        run_id=args.run_id,
        result_files=args.result_files,
        out_root=args.out_root,
        min_shared=args.min_shared_for_corr,
    )
    runner.run()
    print(f"Done: {runner.summary_json}")


if __name__ == "__main__":
    main()
