"""
Reanalyze Experiment 1 results stratified by empirical question difficulty.

Difficulty is defined as cross-model average accuracy on the natural channel
(channel 5) — the only channel that always yields an unconditional answer.

Buckets:
  easy   — cross-model acc >= 0.5  (floor may inflate calibration metrics)
  medium — 0.2 <= cross-model acc < 0.5  (genuine signal region)
  hard   — cross-model acc < 0.2   (ceiling effect may depress all signals)

The script re-runs the full analysis suite on the medium bucket and prints
the summary table. This tests whether low MCI in the overall results is a
floor/ceiling artifact or a genuine finding.

Usage:
    python scripts/reanalyze_by_difficulty.py [--run-id 20260217T210412]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse all analysis machinery from the main analysis script
from scripts.analyze_experiment_1 import (
    load_results,
    analyze_accuracy,
    analyze_calibration,
    analyze_mci,
    analyze_dissociation,
    analyze_layer_comparison,
    print_summary,
)

DEFAULT_RUN_ID = "20260217T210412"
NATURAL_CHANNEL = 5

# Bucket thresholds
EASY_THRESHOLD = 0.5
HARD_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Difficulty bucketing
# ---------------------------------------------------------------------------

def compute_question_difficulty(records: list[dict]) -> dict[str, float]:
    """
    Return {question_id: cross_model_accuracy} using channel-5 records only.

    Only questions attempted by at least one model are included.
    """
    by_qid: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("channel") != NATURAL_CHANNEL:
            continue
        correct = r.get("answer_correct")
        if correct is None:
            continue
        by_qid[r["question_id"]].append(float(correct))

    return {qid: sum(vs) / len(vs) for qid, vs in by_qid.items() if vs}


def bucket_questions(difficulty: dict[str, float]) -> dict[str, list[str]]:
    """Partition question IDs into easy / medium / hard buckets."""
    buckets: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
    for qid, acc in difficulty.items():
        if acc >= EASY_THRESHOLD:
            buckets["easy"].append(qid)
        elif acc >= HARD_THRESHOLD:
            buckets["medium"].append(qid)
        else:
            buckets["hard"].append(qid)
    return buckets


# ---------------------------------------------------------------------------
# Bucket statistics
# ---------------------------------------------------------------------------

def print_bucket_stats(
    buckets: dict[str, list[str]],
    difficulty: dict[str, float],
    total_questions_in_data: int,
) -> None:
    total_bucketed = sum(len(v) for v in buckets.values())
    print(f"\n{'─'*62}")
    print(f"  DIFFICULTY BUCKETING  (natural channel, cross-model accuracy)")
    print(f"{'─'*62}")
    print(f"  {'Bucket':<10} {'N questions':>12}  {'Mean acc':>9}  {'Acc range':>14}")
    print(f"  {'─'*10} {'─'*12}  {'─'*9}  {'─'*14}")

    for label, thresholds in [
        ("easy",   f">= {EASY_THRESHOLD:.1f}"),
        ("medium", f"{HARD_THRESHOLD:.1f}–{EASY_THRESHOLD:.1f}"),
        ("hard",   f"< {HARD_THRESHOLD:.1f}"),
    ]:
        qids = buckets[label]
        if not qids:
            print(f"  {label:<10} {'0':>12}  {'—':>9}  {'—':>14}")
            continue
        accs = [difficulty[q] for q in qids]
        mean_acc = sum(accs) / len(accs)
        print(
            f"  {label:<10} {len(qids):>12,}  {mean_acc:>9.3f}  "
            f"[{min(accs):.3f}, {max(accs):.3f}]"
        )

    print(f"{'─'*62}")
    print(f"  Total bucketed: {total_bucketed:,}  "
          f"(not bucketed / no ch5 answer: "
          f"{total_questions_in_data - total_bucketed:,})")


# ---------------------------------------------------------------------------
# Accuracy helper for medium-bucket summary (used in print_medium_summary)
# ---------------------------------------------------------------------------

def _overall_natural_acc(accuracy: dict) -> dict[str, float | None]:
    """Aggregate natural_acc across all domains per model."""
    result = {}
    for model, domain_map in accuracy.items():
        vals = [v["natural_acc"] for v in domain_map.values()
                if v.get("natural_acc") is not None]
        result[model] = sum(vals) / len(vals) if vals else None
    return result


# ---------------------------------------------------------------------------
# Enhanced summary table that shows both overall and medium-bucket accuracy
# ---------------------------------------------------------------------------

def print_medium_summary(
    accuracy: dict,
    calibration: dict,
    mci: dict,
    layer_comparison: dict,
    dissociation: dict,
    n_medium_questions: int,
    n_medium_records: int,
) -> None:
    """Print summary table for medium-bucket analysis, then top dissociations."""
    models = sorted(accuracy.keys())

    title = f"MEDIUM-DIFFICULTY QUESTIONS ONLY  (n={n_medium_questions} questions)"
    width = max(78, len(title) + 4)
    inner = width - 2

    print(f"\n{'╔' + '═' * inner + '╗'}")
    print(f"║{title.center(inner)}║")
    print(f"╠{'═'*15}╦{'═'*7}╦{'═'*7}╦{'═'*7}╦{'═'*7}╦{'═'*31}╣")
    print(
        f"║ {'Model':<13} ║ {'Acc↑':>5} ║ {'ECE↓':>5} ║"
        f" {'Wag-ρ↑':>5} ║ {'MCI↑':>5} ║ {'Layer':<29} ║"
    )
    print(f"╠{'═'*15}╬{'═'*7}╬{'═'*7}╬{'═'*7}╬{'═'*7}╬{'═'*31}╣")

    def fmt(v):
        return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"

    n_l1_better = 0
    for model in models:
        # Natural accuracy across all domains
        ch5_vals = [
            v["natural_acc"]
            for v in accuracy.get(model, {}).values()
            if v.get("natural_acc") is not None
        ]
        mean_acc = sum(ch5_vals) / len(ch5_vals) if ch5_vals else None

        cal = calibration.get(model, {}).get("overall", {})
        ece = cal.get("wagering_ece")
        wag_rho = cal.get("wagering_spearman")
        mci_val = mci.get(model, {}).get("mci_raw")

        lc = layer_comparison.get(model, {})
        l2_better = lc.get("l2_better_calibrated")
        layer_note = "L2 > L1" if l2_better else "L1 > L2" if l2_better is False else "—"
        if l2_better is False:
            n_l1_better += 1

        print(
            f"║ {model[:13]:<13} ║ {fmt(mean_acc):>5} ║ {fmt(ece):>5} ║"
            f" {fmt(wag_rho):>5} ║ {fmt(mci_val):>5} ║ {layer_note:<29} ║"
        )

    print(f"╚{'═'*15}╩{'═'*7}╩{'═'*7}╩{'═'*7}╩{'═'*7}╩{'═'*31}╝")
    print(f"  ({n_medium_records:,} records across all channels)")

    # Top dissociations
    print("\nTop dissociations (medium bucket):")
    for model in models:
        diss = dissociation.get(model, {}).get("dissociated_pairs", [])
        mat  = dissociation.get(model, {}).get("matrix", {})
        for pair in diss[:2]:
            ch_a, ch_b = pair.split("_", 1)
            rho = mat.get(ch_a, {}).get(ch_b)
            rho_str = f"ρ={rho:.2f}" if rho is not None else "ρ=?"
            print(f"  {model[:20]}: {ch_a} ↔ {ch_b} {rho_str} (DISSOCIATED)")

    print(f"\nLayer comparison (medium bucket):")
    print(f"  {n_l1_better}/{len(models)} models: L1 (wagering) better calibrated than L2 (verbal)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--results-dir", default="data/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / args.results_dir
    results_path = results_dir / f"exp1_{args.run_id}_results.jsonl"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    # ── 1. Load all records ──────────────────────────────────────────────────
    print(f"\nLoading results from {results_path} …")
    records = load_results(results_path)
    print(f"  {len(records):,} records loaded")

    all_qids = {r["question_id"] for r in records}
    print(f"  {len(all_qids):,} unique question IDs in results")

    # ── 2. Compute cross-model difficulty from natural channel ───────────────
    difficulty = compute_question_difficulty(records)
    print(f"  {len(difficulty):,} questions have at least one scored ch5 response")

    # ── 3. Bucket questions ──────────────────────────────────────────────────
    buckets = bucket_questions(difficulty)
    print_bucket_stats(buckets, difficulty, len(all_qids))

    medium_qids = set(buckets["medium"])
    print(f"\nMedium bucket: {len(medium_qids):,} questions")

    if not medium_qids:
        print("ERROR: no medium-difficulty questions found — cannot continue.")
        sys.exit(1)

    # ── 4. Filter records to medium-bucket questions only ───────────────────
    medium_records = [r for r in records if r["question_id"] in medium_qids]
    print(f"  → {len(medium_records):,} records retained for medium analysis")

    # ── 5. Run full analysis suite on medium records ─────────────────────────
    print("\nRunning analyses on medium-difficulty subset …")

    accuracy      = analyze_accuracy(medium_records)
    calibration   = analyze_calibration(medium_records)
    mci_results   = analyze_mci(medium_records)
    dissociation  = analyze_dissociation(medium_records)
    layer_comp    = analyze_layer_comparison(medium_records)

    # ── 6. Print medium-bucket summary table ─────────────────────────────────
    print_medium_summary(
        accuracy, calibration, mci_results, layer_comp, dissociation,
        n_medium_questions=len(medium_qids),
        n_medium_records=len(medium_records),
    )

    # ── 7. Floor/ceiling interpretation ──────────────────────────────────────
    # Compare medium MCI to overall MCI (loaded from previously written JSON)
    overall_mci_path = results_dir / f"exp1_{args.run_id}_mci.json"
    if overall_mci_path.exists():
        overall_mci = json.loads(overall_mci_path.read_text(encoding="utf-8"))
        print(f"\n{'─'*62}")
        print(f"  MCI COMPARISON: Overall vs Medium-difficulty")
        print(f"{'─'*62}")
        print(f"  {'Model':<20} {'Overall MCI':>12} {'Medium MCI':>12} {'Δ':>8}")
        print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}")
        for model in sorted(mci_results.keys()):
            overall_val = overall_mci.get(model, {}).get("mci_raw")
            medium_val  = mci_results.get(model, {}).get("mci_raw")
            if overall_val is None or medium_val is None:
                delta_str = "—"
            else:
                delta = medium_val - overall_val
                delta_str = f"{delta:+.3f}"
            o_str = f"{overall_val:.3f}" if overall_val is not None else "—"
            m_str = f"{medium_val:.3f}" if medium_val is not None else "—"
            print(f"  {model:<20} {o_str:>12} {m_str:>12} {delta_str:>8}")
        print(f"{'─'*62}")
        print()

        # Verdict
        import math as _math
        medium_mcis = [
            v for m in mci_results
            if (v := mci_results[m].get("mci_raw")) is not None
            and not _math.isnan(v)
        ]
        overall_mcis = [
            v for m in overall_mci
            if (v := overall_mci[m].get("mci_raw")) is not None
            and not _math.isnan(v)
        ]
        if medium_mcis and overall_mcis:
            med_mean = sum(medium_mcis) / len(medium_mcis)
            ov_mean  = sum(overall_mcis) / len(overall_mcis)
            if med_mean > ov_mean + 0.05:
                verdict = (
                    "FLOOR EFFECT: MCI is higher on medium-difficulty questions "
                    "— the low overall MCI is partially explained by hard questions "
                    "where chance-level accuracy gives no calibration signal."
                )
            elif med_mean < ov_mean - 0.05:
                verdict = (
                    "GENUINE FINDING: MCI is *lower* on medium questions than overall "
                    "— poor metacognitive calibration persists even where models have "
                    "partial knowledge, suggesting intrinsic mis-calibration."
                )
            else:
                verdict = (
                    "ROBUST FINDING: MCI is similar on medium-difficulty questions "
                    f"(mean {med_mean:.3f}) and overall (mean {ov_mean:.3f}) "
                    "— floor/ceiling effects are not the primary driver of low MCI."
                )
            print(f"  Verdict: {verdict}")

    print("\nDone.")


if __name__ == "__main__":
    main()
