"""
Experiment 1: Counterfactual Questions Runner (B4)

Runs all 104 counterfactual questions through all 13 models.
For each model, computes:
  - Mean confidence on counterfactual questions
  - Comparison to mean confidence on standard questions (from Exp1)
  - Counterfactual Confidence Suppression (CCS): standard_conf - counterfactual_conf
    High CCS = model genuinely lowers confidence on false/counterfactual premises (good)
    Low CCS = model applies same high confidence regardless of premise truth (template matching)

Results: data/results/exp1_counterfactual_{run_id}_results.jsonl
Analysis: data/results/exp1_counterfactual_{run_id}_analysis.json

Usage:
  python scripts/run_exp1_counterfactual.py
  python scripts/run_exp1_counterfactual.py --models llama-3.1-8b,deepseek-r1
  python scripts/run_exp1_counterfactual.py --resume
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient

RESULTS_DIR = Path("data/results")
CF_DIR = Path("data/counterfactual")
DOMAINS = ["arithmetic", "factual", "linguistic", "logical",
           "procedural", "social", "spatial", "temporal"]

ALL_MODELS = [
    "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
    "deepseek-r1", "mistral-large", "gpt-oss-120b",
    "deepseek-v3", "gemini-2.5-pro",
    "gemma-3-27b", "kimi-k2", "phi-4", "llama-3.3-70b",
]

CONCURRENCY = 8
CALL_TIMEOUT = 90
MAX_RETRIES = 5


# ── API helper ─────────────────────────────────────────────────────────────

async def call_with_retry(client, model, messages, temperature=0.0, max_tokens=800):
    for attempt in range(MAX_RETRIES):
        try:
            coro = client.complete(model=model, messages=messages,
                                   temperature=temperature, max_tokens=max_tokens,
                                   metadata={"exp": "cf"})
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
            if attempt < MAX_RETRIES - 1 and any(
                x in str(e) for x in ["429", "rate", "timeout", "connection", "503", "502"]
            ):
                await asyncio.sleep((2 ** attempt) * 2)
            else:
                raise
    return {"error": "Max retries exceeded"}


# ── Prompt builder ──────────────────────────────────────────────────────────

CONFIDENCE_PROMPT = """{question}

Please answer this question and then provide your confidence level.
Format your response as:
ANSWER: [your answer]
CONFIDENCE: [0-100, where 100 = completely certain]
REASONING: [brief explanation]"""


def extract_confidence(raw: str) -> float | None:
    if not raw:
        return None
    for line in raw.splitlines():
        ls = line.strip().upper()
        if ls.startswith("CONFIDENCE:"):
            try:
                val = float(ls.split(":", 1)[1].strip().split()[0].rstrip("%"))
                return max(0.0, min(100.0, val))
            except Exception:
                pass
    return None


# ── Load counterfactual questions ──────────────────────────────────────────

def load_counterfactual_questions():
    questions = []
    for domain in DOMAINS:
        f = CF_DIR / f"{domain}.jsonl"
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    q = json.loads(line)
                    q["domain"] = domain
                    questions.append(q)
                except Exception:
                    pass
    return questions


# ── Load Exp1 standard confidence baseline ─────────────────────────────────

def load_exp1_confidence_baseline():
    """Load mean wagering confidence per model from Exp1 results."""
    baseline = {}
    for f in sorted(RESULTS_DIR.glob("exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for model, model_data in data.items():
                # Look for wagering_acc or confidence fields
                wagering_acc = model_data.get("wagering_acc")
                if wagering_acc is not None:
                    baseline[model] = float(wagering_acc) * 100  # convert to 0-100 scale
        except Exception:
            pass
    return baseline


# ── Trial runner ────────────────────────────────────────────────────────────

async def run_trial(client, model, question, semaphore):
    q_text = question.get("question_text") or question.get("text") or question.get("question", "")
    prompt = CONFIDENCE_PROMPT.format(question=q_text)

    async with semaphore:
        resp = await call_with_retry(client, model, [{"role": "user", "content": prompt}])

    raw = (resp.get("content", "") or "") if isinstance(resp, dict) else ""
    ok = isinstance(resp, dict) and "error" not in resp
    confidence = extract_confidence(raw) if ok else None

    return {
        "model": model,
        "question_id": question.get("id") or question.get("source_id", ""),
        "domain": question.get("domain", ""),
        "question_type": "counterfactual",
        "counterfactual_premise": question.get("counterfactual_premise", ""),
        "correct_answer": question.get("correct_answer", ""),
        "raw_response": raw[:500] if raw else "",
        "confidence": confidence,
        "api_success": ok,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Main ────────────────────────────────────────────────────────────────────

async def run_counterfactual(models, resume=False):
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"exp1_counterfactual_{run_id}_results.jsonl"

    questions = load_counterfactual_questions()
    print(f"Loaded {len(questions)} counterfactual questions")
    print(f"Models: {models}")
    print(f"Output: {out_file}")

    # Resume: load completed (model, question_id) pairs
    done = set()
    if resume:
        existing = sorted(RESULTS_DIR.glob("exp1_counterfactual_*_results.jsonl"),
                          key=lambda p: p.stat().st_mtime)
        if existing:
            out_file = existing[-1]
            for line in open(out_file, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["question_id"]))
                except Exception:
                    pass
            print(f"Resuming from {out_file} ({len(done)} done)")

    client = UnifiedClient()
    sem = asyncio.Semaphore(CONCURRENCY)

    with open(out_file, "a", encoding="utf-8") as out_f:
        all_jobs = [
            (model, q) for model in models for q in questions
            if (model, q.get("id") or q.get("source_id", "")) not in done
        ]
        print(f"Jobs to run: {len(all_jobs)}")

        async def safe_trial(model, q):
            try:
                return await run_trial(client, model, q, sem)
            except Exception as e:
                return {"model": model, "question_id": q.get("id",""), "domain": q.get("domain",""),
                        "api_success": False, "confidence": None, "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()}

        futs = [safe_trial(m, q) for m, q in all_jobs]
        for i, fut in enumerate(asyncio.as_completed(futs), 1):
            rec = await fut
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            print(f"\r  {i}/{len(all_jobs)} {'✓' if rec.get('api_success') else '✗'}", end="", flush=True)
    print(f"\nDone. Output: {out_file}")

    # Analysis
    print("\nAnalyzing counterfactual confidence suppression...")
    baseline = load_exp1_confidence_baseline()
    model_cf_confs = {}
    for line in open(out_file, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("api_success") and r.get("confidence") is not None:
                m = r["model"]
                model_cf_confs.setdefault(m, []).append(float(r["confidence"]))
        except Exception:
            pass

    analysis = {}
    for model, cf_confs in model_cf_confs.items():
        mean_cf = float(np.mean(cf_confs)) if cf_confs else None
        std_name = baseline.get(model)
        ccs = float(std_name - mean_cf) if (std_name is not None and mean_cf is not None) else None
        analysis[model] = {
            "mean_cf_confidence": mean_cf,
            "std_confidence_baseline": std_name,
            "counterfactual_confidence_suppression": ccs,
            "n_trials": len(cf_confs),
            "interpretation": (
                "genuine capability assessment (lowers confidence on false premises)"
                if ccs is not None and ccs > 10 else
                "template matching (maintains high confidence despite false premise)"
                if ccs is not None and ccs < 5 else
                "moderate suppression" if ccs is not None else "insufficient data"
            ),
        }

    ranked = sorted(
        [(m, v["counterfactual_confidence_suppression"]) for m, v in analysis.items()
         if v["counterfactual_confidence_suppression"] is not None],
        key=lambda x: x[1], reverse=True
    )

    analysis_out = {
        "run_id": run_id,
        "n_questions": len(questions),
        "n_models": len(model_cf_confs),
        "per_model": analysis,
        "ccs_ranking": [{"model": m, "ccs": ccs} for m, ccs in ranked],
        "note": "CCS = standard_confidence - counterfactual_confidence. Higher CCS = model correctly lowers confidence on false/counterfactual premises.",
    }

    a_path = RESULTS_DIR / f"exp1_counterfactual_{run_id}_analysis.json"
    with open(a_path, "w") as f:
        json.dump(analysis_out, f, indent=2)
    print(f"Analysis saved: {a_path}")

    print("\nCounterfactual Confidence Suppression (CCS) ranking:")
    for entry in ranked[:10]:
        interp = analysis[entry["model"]]["interpretation"]
        print(f"  {entry['model']:30s}  CCS={entry['ccs']:.1f}  [{interp}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list (default: all 12)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS
    asyncio.run(run_counterfactual(models, resume=args.resume))


if __name__ == "__main__":
    main()
