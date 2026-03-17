"""
Experiment 8: Scaling Analysis
================================

Examines how MIRROR metacognitive calibration metrics scale with model size,
using the Llama family as the primary scaling ladder.

Design:
  Primary scaling series (3+ points):
    llama-3.1-8b    →   8B params
    llama-3.1-70b   →   70B params
    llama-3.1-405b  →  405B params
    llama-3.3-70b   →   70B params (newer generation — separate curve)

  Secondary diversity points (for cross-family comparison):
    phi-4           →   ~4B params (Microsoft)
    gemma-3-27b     →   27B params (Google)
    mistral-large   →  ~675B params (Mistral; sparse / not comparable)

  What we measure:
    For each model we extract from EXISTING Exp1–5 data:
      • natural_acc       — raw accuracy across 8 domains
      • wagering_acc      — accuracy on wagered (high-confidence) questions
      • MIRROR_gap        — |wagering_acc − natural_acc|
      • MCI               — Metacognitive Convergence Index (from Exp1 analysis)
      • ECE               — Expected Calibration Error (from Exp1 analysis)
      • adversarial_ars   — Adversarial Robustness Score (from Exp5, if available)

  We then fit:
    metric ~ log2(params) using OLS regression
    Report: R², slope, 95% CI, p-value

  Fallback: if any model is missing from Exp1 data, we run a lightweight
  subset of Exp1 (80 questions) to generate baseline accuracy.

Usage:
  # Step 1: Extract from existing data (no API calls needed)
  python scripts/run_experiment_8.py --extract-only

  # Step 2: Run gap-fill for missing models (uses API)
  python scripts/run_experiment_8.py --fill-gaps

  # Step 3: Full re-run of Llama family through all channels (fresh data)
  python scripts/run_experiment_8.py --mode full

  # Step 4: Analyze (must run after extraction/fill)
  python scripts/analyze_experiment_8.py
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import (
    build_channel1_prompt,
    build_channel2_prompt,
    build_channel5_prompt,
    parse_channel1,
    parse_channel2,
    parse_channel5,
)
from mirror.scoring.answer_matcher import match_answer_robust


# ── Scaling Model Registry ─────────────────────────────────────────────────────

# Model name → approximate parameter count (billions), family, generation
SCALING_MODELS = {
    # Primary Llama 3.1 series (same architecture family, same generation)
    "llama-3.1-8b":   {"params_b": 8,   "family": "llama-3.1", "gen": 1, "include_primary": True},
    "llama-3.1-70b":  {"params_b": 70,  "family": "llama-3.1", "gen": 1, "include_primary": True},
    "llama-3.1-405b": {"params_b": 405, "family": "llama-3.1", "gen": 1, "include_primary": True},
    # Llama 3.3 (updated weights, same scale as 70b)
    "llama-3.3-70b":  {"params_b": 70,  "family": "llama-3.3", "gen": 2, "include_primary": False},
    # Smaller open models for cross-family comparison
    "phi-4":          {"params_b": 4,   "family": "phi",       "gen": 1, "include_primary": False},
    "gemma-3-27b":    {"params_b": 27,  "family": "gemma-3",   "gen": 1, "include_primary": False},
    # Frontier / diversity (not on scaling curve, plotted separately)
    "mistral-large":  {"params_b": 675, "family": "mistral",   "gen": 1, "include_primary": False},
    "deepseek-r1":    {"params_b": 671, "family": "deepseek",  "gen": 1, "include_primary": False},
    "deepseek-v3":    {"params_b": 671, "family": "deepseek",  "gen": 2, "include_primary": False},
    "gpt-oss-120b":   {"params_b": 120, "family": "openai",    "gen": 1, "include_primary": False},
}

# For gap-fill runs: limited question set
GAP_FILL_N_QUESTIONS = 80  # per model
GAP_FILL_CHANNELS = ["wagering", "natural"]

CALL_TIMEOUT = 90
MAX_RETRIES = 5
CONCURRENCY = 8


# ── Data Extraction ────────────────────────────────────────────────────────────

def load_exp1_accuracy() -> dict:
    """Load all available exp1 accuracy files, merging them (newer wins)."""
    results_dir = Path("data/results")
    accuracy_files = sorted(
        results_dir.glob("exp1_*_accuracy.json"),
        key=lambda p: p.stat().st_mtime
    )
    merged = {}
    for af in accuracy_files:
        with open(af) as f:
            data = json.load(f)
        for model, metrics in data.items():
            merged[model] = metrics  # newer overrides older
    return merged


def load_exp1_calibration() -> dict:
    """Load MCI and ECE from latest Exp1 analysis."""
    results_dir = Path("data/results")
    calib_files = sorted(
        results_dir.glob("exp1_*_calibration.json"),
        key=lambda p: p.stat().st_mtime
    )
    if not calib_files:
        return {}
    with open(calib_files[-1]) as f:
        return json.load(f)


def load_exp1_mci() -> dict:
    """Load MCI per model from latest Exp1 analysis."""
    results_dir = Path("data/results")
    mci_files = sorted(
        results_dir.glob("exp1_*_mci.json"),
        key=lambda p: p.stat().st_mtime
    )
    if not mci_files:
        return {}
    with open(mci_files[-1]) as f:
        return json.load(f)


def load_exp5_ars() -> dict:
    """Load Adversarial Robustness Score from Exp5 analysis."""
    results_dir = Path("data/results")
    exp5_files = sorted(
        results_dir.glob("exp5_*_analysis.json"),
        key=lambda p: p.stat().st_mtime
    )
    if not exp5_files:
        return {}
    with open(exp5_files[-1]) as f:
        data = json.load(f)
    summary = data.get("model_summary", {})
    return {m: v.get("ars") for m, v in summary.items() if v.get("ars") is not None}


def load_exp6_robustness() -> dict:
    """Load TRI and EHS from Exp6 analysis."""
    results_dir = Path("data/results")
    exp6_files = sorted(
        results_dir.glob("exp6_*_analysis.json"),
        key=lambda p: p.stat().st_mtime
    )
    if not exp6_files:
        return {}
    with open(exp6_files[-1]) as f:
        data = json.load(f)
    result = {}
    r6a = data.get("results_6a", {}).get("per_model", {})
    r6b = data.get("results_6b", {}).get("per_model", {})
    all_models = set(list(r6a.keys()) + list(r6b.keys()))
    for m in all_models:
        result[m] = {
            "tri": r6a.get(m, {}).get("trust_robustness_index"),
            "ehs": r6b.get(m, {}).get("epistemic_hygiene_score"),
        }
    return result


def extract_scaling_data(
    exp1_acc: dict,
    exp1_calib: dict,
    exp1_mci: dict,
    exp5_ars: dict,
    exp6_rob: dict,
) -> dict:
    """
    For each scaling model, collect all available metrics into a flat record.

    Returns:
        {model_slug: {params_b, family, natural_acc, wagering_acc, mirror_gap, mci, ece, ars, tri, ehs, ...}}
    """
    records = {}

    for model_slug, meta in SCALING_MODELS.items():
        rec = {
            "model": model_slug,
            "params_b": meta["params_b"],
            "log2_params": round(float(np.log2(meta["params_b"])), 4),
            "family": meta["family"],
            "generation": meta["gen"],
            "include_primary_scaling_curve": meta["include_primary"],
        }

        # Exp1 accuracy
        # Initialize defaults first (prevents KeyError if model is in exp1_acc but has no valid data)
        rec["natural_acc"] = None
        rec["wagering_acc"] = None
        rec["mirror_gap"] = None
        rec["domain_natural_acc"] = {}

        if model_slug in exp1_acc:
            m_acc = exp1_acc[model_slug]
            # Aggregate across domains
            nat_vals = [v.get("natural_acc") for v in m_acc.values()
                        if isinstance(v, dict) and v.get("natural_acc") is not None]
            wag_vals = [v.get("wagering_acc") for v in m_acc.values()
                        if isinstance(v, dict) and v.get("wagering_acc") is not None]
            if nat_vals:
                rec["natural_acc"] = round(float(np.mean(nat_vals)), 4)
            if wag_vals:
                rec["wagering_acc"] = round(float(np.mean(wag_vals)), 4)
            if nat_vals and wag_vals:
                rec["mirror_gap"] = round(abs(rec["wagering_acc"] - rec["natural_acc"]), 4)
            # Domain-level breakdown for plotting
            rec["domain_natural_acc"] = {
                d: v.get("natural_acc")
                for d, v in m_acc.items()
                if isinstance(v, dict) and v.get("natural_acc") is not None
            }

        # Exp1 calibration (MCI, ECE)
        if model_slug in exp1_mci:
            rec["mci"] = exp1_mci[model_slug].get("mci")
        else:
            rec["mci"] = None

        if model_slug in exp1_calib:
            rec["ece"] = exp1_calib[model_slug].get("ece")
        else:
            rec["ece"] = None

        # Exp5 adversarial robustness
        rec["adversarial_ars"] = exp5_ars.get(model_slug)

        # Exp6 ecosystem robustness
        if model_slug in exp6_rob:
            rec["trust_robustness_index"] = exp6_rob[model_slug].get("tri")
            rec["epistemic_hygiene_score"] = exp6_rob[model_slug].get("ehs")
        else:
            rec["trust_robustness_index"] = None
            rec["epistemic_hygiene_score"] = None

        # Data completeness flag
        has_exp1 = rec["natural_acc"] is not None
        rec["data_complete"] = has_exp1
        rec["missing_data"] = [
            k for k in ["natural_acc", "wagering_acc", "mirror_gap", "mci"]
            if rec.get(k) is None
        ]

        records[model_slug] = rec

    return records


# ── Gap Fill: Lightweight Exp1 for Missing Models ─────────────────────────────

async def call_with_retry(client, model, messages, temperature, max_tokens, metadata):
    for attempt in range(MAX_RETRIES):
        try:
            coro = client.complete(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, metadata=metadata,
            )
            resp = await asyncio.wait_for(coro, timeout=CALL_TIMEOUT)
            if isinstance(resp, dict) and "error" in resp:
                err = str(resp["error"])
                if any(x in err for x in ["429", "rate", "quota"]):
                    await asyncio.sleep((2 ** attempt) * 2)
                    continue
            return resp
        except asyncio.TimeoutError:
            await asyncio.sleep((2 ** attempt) * 2)
        except Exception as e:
            err = str(e)
            if attempt < MAX_RETRIES - 1 and any(
                x in err for x in ["429", "rate", "timeout", "connection", "503", "502"]
            ):
                await asyncio.sleep((2 ** attempt) * 2)
            else:
                raise
    return {"error": f"Max retries ({MAX_RETRIES}) exceeded"}


def load_questions(n: int = GAP_FILL_N_QUESTIONS) -> list[dict]:
    """Load questions from the main question bank."""
    questions_path = Path("data/questions.jsonl")
    if not questions_path.exists():
        # Try generated questions
        questions_path = Path("data/generated")
        all_qs = []
        if questions_path.is_dir():
            for qf in questions_path.glob("*.jsonl"):
                with open(qf) as f:
                    for line in f:
                        try:
                            all_qs.append(json.loads(line))
                        except Exception:
                            pass
        if not all_qs:
            raise FileNotFoundError("No question bank found at data/questions.jsonl or data/generated/")
        return all_qs[:n]

    questions = []
    with open(questions_path) as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except Exception:
                pass
    # Sample evenly across domains
    from collections import defaultdict
    by_domain = defaultdict(list)
    for q in questions:
        by_domain[q.get("domain", "unknown")].append(q)
    per_domain = max(1, n // max(1, len(by_domain)))
    sampled = []
    for domain_qs in by_domain.values():
        sampled.extend(domain_qs[:per_domain])
    return sampled[:n]


async def run_gap_fill(model: str, questions: list[dict], output_file: Path) -> int:
    """Run lightweight Exp1 (wagering + natural channels) for a single model."""
    client = UnifiedClient()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Check which questions already done
    done_keys = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("model") == model:
                        done_keys.add(f"{rec.get('question_id')}_{rec.get('channel')}")
                except Exception:
                    pass

    tasks = []
    for q in questions:
        q_id = q.get("question_id") or q.get("id") or hash(q.get("question_text", ""))
        for channel in GAP_FILL_CHANNELS:
            key = f"{q_id}_{channel}"
            if key not in done_keys:
                tasks.append((q, q_id, channel))

    if not tasks:
        print(f"  {model}: already complete ({len(questions) * len(GAP_FILL_CHANNELS)} records)")
        return 0

    written = 0

    async def run_one(q, q_id, channel):
        if channel == "wagering":
            prompt = build_channel1_prompt(q)
            parser = parse_channel1
        else:
            prompt = build_channel5_prompt(q)
            parser = parse_channel5

        messages = [{"role": "user", "content": prompt}]
        async with semaphore:
            resp = await call_with_retry(
                client=client, model=model, messages=messages,
                temperature=0.0, max_tokens=512,
                metadata={"experiment": "8_gapfill", "model": model, "channel": channel},
            )
        raw = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        api_success = "error" not in (resp if isinstance(resp, dict) else {})
        parsed = parser(raw) if api_success else {}
        raw_safe = (raw or "")[:200]
        is_correct = match_answer_robust(
            parsed.get("answer", raw_safe), q.get("correct_answer", ""), "short_text"
        ) if api_success else False
        return {
            "experiment": "8_gapfill",
            "model": model,
            "question_id": q_id,
            "domain": q.get("domain", ""),
            "channel": channel,
            "correct_answer": q.get("correct_answer", ""),
            "raw_response": raw,
            "parsed": parsed,
            "api_success": api_success,
            "is_correct": is_correct,
            "timestamp": datetime.utcnow().isoformat(),
        }

    with open(output_file, "a", encoding="utf-8") as f:
        futures = [run_one(q, q_id, ch) for q, q_id, ch in tasks]
        for i, fut in enumerate(asyncio.as_completed(futures), 1):
            rec = await fut
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            written += 1
            status = "✓" if rec.get("api_success") else "✗"
            print(f"\r    {model}: {i}/{len(tasks)} {status}", end="", flush=True)

    print()
    return written


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args):
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    output_file = results_dir / f"exp8_{run_id}_gapfill.jsonl"
    scaling_data_file = results_dir / f"exp8_{run_id}_scaling_data.json"

    # ── Step 1: Extract from existing data ──
    print("Loading existing experiment data...")
    exp1_acc = load_exp1_accuracy()
    exp1_calib = load_exp1_calibration()
    exp1_mci = load_exp1_mci()
    exp5_ars = load_exp5_ars()
    exp6_rob = load_exp6_robustness()

    print(f"  Exp1 accuracy: {len(exp1_acc)} models")
    print(f"  Exp1 calibration: {len(exp1_calib)} models")
    print(f"  Exp1 MCI: {len(exp1_mci)} models")
    print(f"  Exp5 ARS: {len(exp5_ars)} models")
    print(f"  Exp6 robustness: {len(exp6_rob)} models")

    scaling_data = extract_scaling_data(exp1_acc, exp1_calib, exp1_mci, exp5_ars, exp6_rob)

    # Report coverage
    print("\n── Model Coverage ──")
    print(f"{'Model':<25} {'Params':>8} {'Nat.Acc':>10} {'Wag.Acc':>10} {'MCI':>8} {'Status':>10}")
    print("-" * 72)
    missing_models = []
    for slug, rec in scaling_data.items():
        nat = f"{rec['natural_acc']:.3f}" if rec["natural_acc"] is not None else "—"
        wag = f"{rec['wagering_acc']:.3f}" if rec["wagering_acc"] is not None else "—"
        mci = f"{rec['mci']:.3f}" if rec["mci"] is not None else "—"
        status = "COMPLETE" if rec["data_complete"] else "MISSING"
        print(f"{slug:<25} {rec['params_b']:>7}B {nat:>10} {wag:>10} {mci:>8} {status:>10}")
        if not rec["data_complete"]:
            missing_models.append(slug)

    if args.extract_only:
        with open(scaling_data_file, "w") as f:
            json.dump(scaling_data, f, indent=2)
        print(f"\nScaling data saved: {scaling_data_file}")
        print(f"\nMissing models ({len(missing_models)}): {missing_models}")
        print("Run --fill-gaps to collect data for missing models.")
        return

    # ── Step 2: Gap fill for missing models ──
    if missing_models and (args.fill_gaps or args.mode == "full"):
        print(f"\n── Gap Fill: {len(missing_models)} models need Exp1 data ──")
        questions = load_questions(GAP_FILL_N_QUESTIONS)
        print(f"Loaded {len(questions)} questions for gap fill")

        for model_slug in missing_models:
            if model_slug not in SCALING_MODELS:
                continue
            print(f"\n  Running {model_slug}...")
            written = await run_gap_fill(model_slug, questions, output_file)
            print(f"  {model_slug}: wrote {written} new records")

        # Recompute from gap fill data
        if output_file.exists():
            gf_acc = compute_accuracy_from_gapfill(output_file)
            for model_slug, model_acc in gf_acc.items():
                if model_slug not in exp1_acc or not scaling_data[model_slug]["data_complete"]:
                    exp1_acc[model_slug] = model_acc

            # Rerun extraction with gap fill data
            scaling_data = extract_scaling_data(exp1_acc, exp1_calib, exp1_mci, exp5_ars, exp6_rob)

    # ── Save final scaling data ──
    with open(scaling_data_file, "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\nScaling data saved: {scaling_data_file}")
    print(f"\nNext step: python scripts/analyze_experiment_8.py {scaling_data_file}")


def compute_accuracy_from_gapfill(gapfill_file: Path) -> dict:
    """Compute per-model per-domain accuracy from gap fill results."""
    from collections import defaultdict
    model_domain_channel = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(gapfill_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("api_success"):
                    continue
                model = rec["model"]
                domain = rec.get("domain", "unknown")
                channel = rec.get("channel")
                is_correct = rec.get("is_correct", False)
                model_domain_channel[model][domain][channel].append(1 if is_correct else 0)
            except Exception:
                pass

    result = {}
    for model, domain_data in model_domain_channel.items():
        model_acc = {}
        for domain, channel_data in domain_data.items():
            domain_entry = {}
            if "natural" in channel_data:
                domain_entry["natural_acc"] = float(np.mean(channel_data["natural"]))
            if "wagering" in channel_data:
                domain_entry["wagering_acc"] = float(np.mean(channel_data["wagering"]))
            model_acc[domain] = domain_entry
        result[model] = model_acc
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 8: Scaling Analysis")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract from existing data, no API calls")
    parser.add_argument("--fill-gaps", action="store_true",
                        help="Run lightweight gap-fill for models with missing data")
    parser.add_argument("--mode", choices=["full"], default="full",
                        help="Full extraction + gap fill")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model override for gap fill")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
