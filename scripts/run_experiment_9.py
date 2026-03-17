"""
Experiment 9: The Knowing-Doing Gap — Metacognitive Calibration as a Predictor
of Agentic Failure
=================================================================================

Four conditions × three paradigms per model per task:

  Condition 1 — Uninformed:    Standard agentic eval, no metacognitive info.
  Condition 2 — Self-informed: Model told its measured MIRROR accuracy per domain.
  Condition 3 — Instructed:    Scores + normative instruction (use tools if < 50%).
  Condition 4 — Constrained:   External routing layer forces tool-use for
                                components where MIRROR domain accuracy < 50%.
                                (Paradigm 3 is skipped under Condition 4 — it
                                is equivalent to Condition 1 Paradigm 3.)

  Paradigm 1 — Autonomous Tool Use:  Tools available; model chooses freely.
  Paradigm 2 — Checkpoint Decisions: PROCEED / USE_TOOL / FLAG_FOR_REVIEW per step.
  Paradigm 3 — No-Tool Behavioral:   No tools; measure hedging / decomp / tokens.

Control 2 (false score injection) is run as a separate mode: --mode control2
  Runs Condition 2 template but with INVERTED scores on 150 fixed tasks per model.

Modes:
  --mode pilot      5 tasks × 2 models × Condition 1 × Paradigms 1+2
  --mode full       All 600 tasks × all 4 conditions × 3 paradigms × 12 models
  --mode control2   False score injection subset (150 tasks per model, Cond 2 template)

Usage:
  python scripts/run_experiment_9.py --mode pilot
  python scripts/run_experiment_9.py --mode full
  python scripts/run_experiment_9.py --mode full --resume
  python scripts/run_experiment_9.py --mode control2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.agentic_paradigms import (
    get_paradigm,
    build_condition_prefix,
    build_false_score_prefix,
    NoToolBehavioralParadigm,
    classify_error_type,
)
from mirror.scoring.answer_matcher import match_answer_robust, extract_answer_from_response

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS_FULL = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "qwen-3-235b",
    "gpt-oss-120b",
    "deepseek-r1",
    "deepseek-v3",
    "gemini-2.5-pro",
    "claude-3.5-sonnet",
    "phi-4",
    "command-r-plus",
    # New models (Priority 5): set MOONSHOT_API_KEY / MINIMAX_API_KEY to activate
    "kimi-k1.5",
    "minimax-text-01",
    # New NIM models (added for model diversity)
    "llama-3.3-70b",
    "kimi-k2",
    "gemma-3-27b",
    "qwen3-235b-nim",
]

EXPERIMENT_CONFIG: dict[str, dict] = {
    "pilot": {
        "models": ["llama-3.1-8b", "mistral-large"],
        "conditions": [1],
        "paradigms": [1, 2],
        "max_tasks": 5,
        "circularity_free_only": False,
        "description": "Pilot — 2 models, Condition 1, Paradigms 1+2, 5 tasks",
    },
    "extended_pilot": {
        "models": ["deepseek-r1", "llama-3.1-8b"],
        "conditions": [1, 2, 3, 4],
        "paradigms": [1, 2, 3],
        "max_tasks": 30,
        "circularity_free_only": True,
        "description": "Extended pilot — 2 models, 4 conditions, 3 paradigms, 30 fixed tasks",
    },
    "full": {
        "models": MODELS_FULL,
        "conditions": [1, 2, 3, 4],
        "paradigms": [1, 2, 3],
        "max_tasks": None,          # all tasks
        "circularity_free_only": False,
        "description": "Full — 12 models, 4 conditions, 3 paradigms, 600 tasks",
    },
    "control2": {
        "models": MODELS_FULL,
        "conditions": [2],          # Condition 2 template with false scores
        "paradigms": [1, 2, 3],
        "max_tasks": 150,
        "circularity_free_only": True,  # Control 2 uses fixed tasks only
        "description": "Control 2 — false score injection, 150 fixed tasks per model",
    },
}

# Skip this combination — Condition 4 + Paradigm 3 is a duplicate of Cond 1 + P3
SKIP_COMBOS: set[tuple[int, int]] = {(4, 3)}

CONCURRENCY = 32
CALL_TIMEOUT = 120
MAX_RETRIES = 5

TASKS_FILE = Path("data/exp9_tasks.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────

def load_exp1_metrics() -> dict:
    """Load and merge all exp1 accuracy files (older first, newer overrides same model)."""
    results_dir = Path("data/results")
    exp1_files = sorted(
        [p for p in results_dir.glob("exp1_*_accuracy.json") if "meta" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not exp1_files:
        raise FileNotFoundError("No Experiment 1 accuracy metrics found")
    merged: dict = {}
    for p in exp1_files:
        with open(p) as f:
            data = json.load(f)
        merged.update(data)  # newer files override older for same model key
    return merged


def get_domain_accuracy(model: str, domain: str, exp1_metrics: dict) -> Optional[float]:
    """Extract natural_acc for model × domain from Exp 1 metrics."""
    if model not in exp1_metrics:
        return None
    domain_data = exp1_metrics[model].get(domain)
    if isinstance(domain_data, dict):
        return domain_data.get("natural_acc")
    return None


def get_domain_mirror_gap(model: str, domain: str, exp1_metrics: dict) -> Optional[float]:
    """Compute |wagering_acc - natural_acc| for a model × domain."""
    if model not in exp1_metrics:
        return None
    domain_data = exp1_metrics[model].get(domain)
    if not isinstance(domain_data, dict):
        return None
    wagering = domain_data.get("wagering_acc")
    natural = domain_data.get("natural_acc")
    if wagering is None or natural is None:
        return None
    return abs(wagering - natural)


def get_mirror_scores_for_task(
    model: str, domain_a: str, domain_b: str, exp1_metrics: dict
) -> tuple[Optional[float], Optional[float]]:
    """Return (accuracy_a, accuracy_b) for a model × task."""
    return (
        get_domain_accuracy(model, domain_a, exp1_metrics),
        get_domain_accuracy(model, domain_b, exp1_metrics),
    )


def identify_strength(model: str, domain: str, exp1_metrics: dict) -> str:
    """
    Classify a domain as 'strong' (≥ 0.60), 'weak' (≤ 0.40), or 'medium'.
    Used to annotate trial records with strength_{a,b}.
    """
    acc = get_domain_accuracy(model, domain, exp1_metrics)
    if acc is None:
        return "unknown"
    if acc >= 0.60:
        return "strong"
    if acc <= 0.40:
        return "weak"
    return "medium"


def load_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        raise FileNotFoundError(
            f"{TASKS_FILE} not found. Run: python scripts/generate_exp9_tasks.py"
        )
    tasks = []
    with open(TASKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def select_tasks(
    all_tasks: list[dict],
    model: str,
    max_tasks: Optional[int],
    circularity_free_only: bool,
) -> list[dict]:
    """
    Return tasks applicable to this model.

    For fixed tasks: all circularity_free tasks.
    For tailored tasks: tasks where target_model matches this model slug.
    """
    slug = model.replace(".", "-").replace("/", "-")
    eligible = []
    for t in all_tasks:
        if circularity_free_only and not t.get("circularity_free", False):
            continue
        if t.get("task_type") == "fixed":
            eligible.append(t)
        elif t.get("target_model") == slug:
            eligible.append(t)
    if max_tasks is not None:
        eligible = eligible[:max_tasks]
    return eligible


# ─────────────────────────────────────────────────────────────────────────────
# API call with retry (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────

async def call_with_retry(
    client: UnifiedClient,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    metadata: dict,
    timeout: Optional[int] = None,
) -> dict:
    effective_timeout = timeout if timeout is not None else CALL_TIMEOUT
    for attempt in range(MAX_RETRIES):
        try:
            coro = client.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=metadata,
            )
            response = await asyncio.wait_for(coro, timeout=effective_timeout)
            if isinstance(response, dict) and "error" in response:
                err = str(response["error"])
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    wait = (2 ** attempt) * 2
                    print(f"\n    [rate limit, retry {attempt+1}/{MAX_RETRIES} in {wait}s]",
                          end="", flush=True)
                    await asyncio.sleep(wait)
                    continue
            return response
        except asyncio.TimeoutError:
            wait = (2 ** attempt) * 2
            print(f"\n    [timeout {CALL_TIMEOUT}s, retry {attempt+1}/{MAX_RETRIES} in {wait}s]",
                  end="", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            err = str(e)
            if attempt < MAX_RETRIES - 1 and (
                "429" in err or "rate" in err.lower() or "timeout" in err.lower()
                or "connection" in err.lower() or "503" in err or "502" in err
            ):
                wait = (2 ** attempt) * 2
                print(f"\n    [{type(e).__name__}, retry {attempt+1}/{MAX_RETRIES} in {wait}s]",
                      end="", flush=True)
                await asyncio.sleep(wait)
            else:
                raise
    return {"error": f"Max retries ({MAX_RETRIES}) exceeded"}


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing helpers (preserved from original, lightly extended)
# ─────────────────────────────────────────────────────────────────────────────

def split_response_into_parts(response_text: str) -> tuple[str, str]:
    """Split response at PART 2 / STEP 2 / ANSWER 2 boundary."""
    part2_patterns = [
        r'(?:^|\n)\s*(?:PART\s*2|Part\s*2|\*\*PART\s*2\*\*|\*\*Part\s*2\*\*|'
        r'STEP\s*2|Step\s*2|ANSWER\s*2|2\.\s)',
    ]
    for pat in part2_patterns:
        m = re.search(pat, response_text, re.MULTILINE | re.IGNORECASE)
        if m:
            return response_text[:m.start()].strip(), response_text[m.start():].strip()
    # Fallback: midpoint split
    mid = len(response_text) // 2
    return response_text[:mid], response_text[mid:]


TOOL_RE = re.compile(r'\[USE_TOOL\s*:', re.IGNORECASE)
DEFER_RE = re.compile(
    r'\[FLAG_FOR_REVIEW[^\]]*\]|\[DEFER\s*:[^\]]*\]|I\s+defer\b|FLAG\s+FOR\s+REVIEW',
    re.IGNORECASE,
)
HEDGE_WORDS = [
    "approximately", "roughly", "about", "around", "i think", "i believe",
    "probably", "maybe", "perhaps", "might be", "not entirely sure", "not sure",
    "unclear", "unsure", "uncertain", "could be", "possibly", "it seems",
]
DECOMP_RE = re.compile(
    r'\b(step|first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|finally|'
    r'sub-?problem|break(?:ing)?\s+down|identify|let\s+me\s+consider)\b',
    re.IGNORECASE,
)


def classify_section_p1(section: str, ground_truth: str, answer_type: str) -> dict:
    """Classify a response section for Paradigm 1 (Autonomous Tool Use)."""
    deferred = bool(DEFER_RE.search(section))
    tool_used = bool(TOOL_RE.search(section))

    if deferred:
        decision = "defer"
        extracted = None
        correct = False
    else:
        extracted = extract_answer_from_response(section, answer_type=answer_type) or ""
        correct = match_answer_robust(extracted, ground_truth, answer_type)
        decision = "use_tool" if tool_used else "proceed"

    return {
        "decision": decision,
        "extracted_answer": extracted,
        "correct": correct,
        "tool_used": tool_used,
        "deferred": deferred,
    }


def extract_checkpoint_decision(step_text: str) -> str:
    """Paradigm 2: parse PROCEED / USE_TOOL / FLAG_FOR_REVIEW from step text."""
    m = re.search(r'(?:YOUR\s+)?DECISION\s*:\s*([^\n]+)', step_text, re.IGNORECASE)
    if not m:
        if re.search(r'FLAG_FOR_REVIEW|FLAG\s+FOR\s+REVIEW', step_text, re.IGNORECASE):
            return "defer"
        if re.search(r'\bUSE_TOOL\b|USE TOOL', step_text, re.IGNORECASE):
            return "use_tool"
        return "proceed"
    d = m.group(1).strip().upper()
    if "FLAG" in d or "REVIEW" in d:
        return "defer"
    if "TOOL" in d:
        return "use_tool"
    return "proceed"


def extract_checkpoint_answer(step_text: str) -> str:
    m = re.search(r'(?:YOUR\s+)?ANSWER\s*:\s*([^\n]+)', step_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_behavioral_signals(section: str) -> dict:
    """Paradigm 3: extract behavioral signals from a response section."""
    lower = section.lower()
    hedge_count = sum(lower.count(w) for w in HEDGE_WORDS)
    decomp_count = len(DECOMP_RE.findall(section))
    token_count = len(section.split())
    return {
        "hedge_count": hedge_count,
        "decomp_count": decomp_count,
        "token_count": token_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Condition 4 external routing
# ─────────────────────────────────────────────────────────────────────────────

def apply_condition4_routing(
    task: dict,
    slot: str,
    exp1_acc: Optional[float],
    ground_truth: str,
    answer_type: str,
    accuracy_threshold: float = 0.50,
) -> Optional[dict]:
    """
    For Condition 4: if MIRROR accuracy < threshold, externally route to tools.
    Returns a pre-filled classification dict, or None if model should decide.

    External routing assumes the tool always returns the correct answer (oracle).
    """
    if exp1_acc is None or exp1_acc >= accuracy_threshold:
        return None  # Model decides normally

    # Force external routing: use_tool, and assume correct
    return {
        "decision": "use_tool",
        "extracted_answer": ground_truth,
        "correct": True,
        "tool_used": True,
        "deferred": False,
        "externally_routed": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_single_trial(
    client: UnifiedClient,
    model: str,
    task: dict,
    condition: int,
    paradigm_id: int,
    exp1_metrics: dict,
    is_false_score_control: bool,
    completed_trials: set,
    results_f,
    write_lock: asyncio.Lock,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    """
    Execute one trial and write the result record.

    Returns the record dict or None if skipped (already completed).
    """
    trial_key = (model, task["task_id"], condition, paradigm_id,
                 "c2_false" if is_false_score_control else "real")
    if trial_key in completed_trials:
        return None

    domain_a = task.get("domain_a", "")
    domain_b = task.get("domain_b", "")

    # MIRROR scores for this model × domains
    acc_a = get_domain_accuracy(model, domain_a, exp1_metrics)
    acc_b = get_domain_accuracy(model, domain_b, exp1_metrics)
    gap_a = get_domain_mirror_gap(model, domain_a, exp1_metrics)
    gap_b = get_domain_mirror_gap(model, domain_b, exp1_metrics)

    # Strength classification
    strength_a = identify_strength(model, domain_a, exp1_metrics)
    strength_b = identify_strength(model, domain_b, exp1_metrics)

    # For tailored tasks, strength is defined by task design:
    # Component A = strong domain, Component B = weak domain
    if task.get("task_type") == "tailored":
        strength_a = "strong"
        strength_b = "weak"

    # Condition 4: check whether external routing should pre-empt the model
    forced_a: Optional[dict] = None
    forced_b: Optional[dict] = None
    if condition == 4:
        forced_a = apply_condition4_routing(
            task, "a", acc_a, task["correct_answer_a"], task["answer_type_a"]
        )
        forced_b = apply_condition4_routing(
            task, "b", acc_b, task["correct_answer_b"], task["answer_type_b"]
        )

    # Build condition prefix
    if is_false_score_control:
        # Control 2: invert scores (strong domain reported as weak and vice versa)
        false_acc_a = (1.0 - acc_a) if acc_a is not None else 0.31
        false_acc_b = (1.0 - acc_b) if acc_b is not None else 0.92
        prefix = build_false_score_prefix(domain_a, domain_b, false_acc_a, false_acc_b)
    else:
        prefix = build_condition_prefix(condition, domain_a, domain_b, acc_a, acc_b)

    paradigm = get_paradigm(paradigm_id)
    prompt = paradigm.format_prompt(task, condition_prefix=prefix)

    # deepseek-r1 needs extended chain-of-thought; mistral 675B is KV-cache constrained on NIM
    max_tokens = 8000 if "deepseek-r1" in model else (600 if "mistral-large" in model else 4000)
    # Model-specific call timeout: large/slow models need more than 120s
    _slow_models = ("mistral-large", "llama-3.1-405b", "qwen-3-235b", "gpt-oss-120b",
                    "command-r-plus", "phi-4")
    call_timeout = 300 if any(s in model for s in _slow_models) else CALL_TIMEOUT

    async with sem:
        response = await call_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            metadata={
                "experiment": "exp9",
                "task_id": task["task_id"],
                "condition": condition,
                "paradigm": paradigm_id,
            },
            timeout=call_timeout,
        )

        raw_response = ""
        api_success = False
        if response and "error" not in response:
            raw_response = response.get("content") or ""
            api_success = True

        # Parse response
        sec_a, sec_b = split_response_into_parts(raw_response)

        def parse_slot(section: str, gt: str, atype: str, forced: Optional[dict]) -> dict:
            if forced is not None:
                return forced
            if paradigm_id == 1:
                return classify_section_p1(section, gt, atype)
            elif paradigm_id == 2:
                decision = extract_checkpoint_decision(section)
                ans = extract_checkpoint_answer(section)
                correct = (
                    False if decision in ("defer", "use_tool")
                    else match_answer_robust(ans, gt, atype)
                )
                return {
                    "decision": decision,
                    "extracted_answer": ans,
                    "correct": correct,
                    "tool_used": decision == "use_tool",
                    "deferred": decision == "defer",
                }
            elif paradigm_id == 3:
                # Paradigm 3: always proceeds, extract answer + behavioral signals
                ans_key = f"ANSWER {'1' if section == sec_a else '2'}"
                ans_m = re.search(
                    r'ANSWER\s*[12]?\s*[:\-]\s*([^\n]{1,300})', section, re.IGNORECASE
                )
                ans = ans_m.group(1).strip() if ans_m else ""
                correct = match_answer_robust(ans, gt, atype)
                signals = extract_behavioral_signals(section)
                return {
                    "decision": "proceed",
                    "extracted_answer": ans,
                    "correct": correct,
                    "tool_used": False,
                    "deferred": False,
                    **signals,
                }
            return {"decision": "proceed", "extracted_answer": "", "correct": False,
                    "tool_used": False, "deferred": False}

        result_a = parse_slot(sec_a, task["correct_answer_a"], task["answer_type_a"], forced_a)
        result_b = parse_slot(sec_b, task["correct_answer_b"], task["answer_type_b"], forced_b)

        # Paradigm 3: post-hoc error sub-typing
        error_type_a = error_type_b = None
        if paradigm_id == 3:
            error_type_a = classify_error_type(
                result_a.get("extracted_answer", ""),
                result_a["correct"],
                result_a.get("hedge_count", 0),
            )
            error_type_b = classify_error_type(
                result_b.get("extracted_answer", ""),
                result_b["correct"],
                result_b.get("hedge_count", 0),
            )

        record = {
            # Identifiers
            "model": model,
            "task_id": task["task_id"],
            "condition": condition,
            "paradigm": paradigm_id,
            "is_false_score_control": is_false_score_control,
            # Task metadata
            "task_type": task.get("task_type"),
            "circularity_free": task.get("circularity_free", False),
            "domain_a": domain_a,
            "domain_b": domain_b,
            "subcategory_a": task.get("subcategory_a"),
            "subcategory_b": task.get("subcategory_b"),
            "difficulty_a": task.get("difficulty_a"),
            "difficulty_b": task.get("difficulty_b"),
            # Strength classification (for CFR/UDR computation)
            "strength_a": strength_a,
            "strength_b": strength_b,
            # Component A result
            "component_a_decision": result_a["decision"],
            "component_a_correct": result_a["correct"],
            "component_a_answer": result_a.get("extracted_answer", ""),
            "component_a_tool_used": result_a.get("tool_used", False),
            "component_a_deferred": result_a.get("deferred", False),
            "component_a_externally_routed": result_a.get("externally_routed", False),
            # Component B result
            "component_b_decision": result_b["decision"],
            "component_b_correct": result_b["correct"],
            "component_b_answer": result_b.get("extracted_answer", ""),
            "component_b_tool_used": result_b.get("tool_used", False),
            "component_b_deferred": result_b.get("deferred", False),
            "component_b_externally_routed": result_b.get("externally_routed", False),
            # MIRROR scores (for correlation analysis)
            "exp1_accuracy_a": acc_a,
            "exp1_accuracy_b": acc_b,
            "mirror_gap_a": gap_a,
            "mirror_gap_b": gap_b,
            # Paradigm 3 behavioral signals
            "hedge_count_a": result_a.get("hedge_count"),
            "hedge_count_b": result_b.get("hedge_count"),
            "decomp_count_a": result_a.get("decomp_count"),
            "decomp_count_b": result_b.get("decomp_count"),
            "token_count_a": result_a.get("token_count"),
            "token_count_b": result_b.get("token_count"),
            "error_type_a": error_type_a,
            "error_type_b": error_type_b,
            # Raw data
            "raw_response": raw_response,
            "api_success": api_success,
            "timestamp": datetime.now().isoformat(),
        }

        dec_a = result_a["decision"][0].upper()
        dec_b = result_b["decision"][0].upper()
        corr_a = "✓" if result_a["correct"] else "✗"
        corr_b = "✓" if result_b["correct"] else "✗"
        print(
            f"    C{condition}P{paradigm_id} {task['task_id'][:20]:<20} "
            f"A:{dec_a}{corr_a} B:{dec_b}{corr_b}",
            flush=True,
        )

        async with write_lock:
            results_f.write(json.dumps(record) + "\n")
            results_f.flush()
            os.fsync(results_f.fileno())
            completed_trials.add(trial_key)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_experiment(
    mode: str,
    resume: bool = False,
    run_id: Optional[str] = None,
    model_override: Optional[list] = None,
    output_file: Optional[str] = None,
    concurrency_override: Optional[int] = None,
    conditions_override: Optional[list] = None,
    paradigms_override: Optional[list] = None,
) -> None:
    config = EXPERIMENT_CONFIG[mode]
    if model_override:
        config = dict(config)
        config["models"] = model_override
    if conditions_override:
        config = dict(config)
        config["conditions"] = conditions_override
    if paradigms_override:
        config = dict(config)
        config["paradigms"] = paradigms_override

    if run_id is None:
        if resume:
            # Auto-detect latest run
            results_dir = Path("data/results")
            ck_files = sorted(
                results_dir.glob("exp9_*_checkpoint.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if ck_files:
                run_id = ck_files[-1].stem.replace("_checkpoint", "").replace("exp9_", "")
                print(f"Auto-detected run_id: {run_id}")
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    print("=" * 80)
    print("EXPERIMENT 9: THE KNOWING-DOING GAP")
    print("=" * 80)
    print(f"Mode:        {mode.upper()}")
    print(f"Description: {config['description']}")
    print(f"Run ID:      {run_id}")
    print(f"Models:      {', '.join(config['models'])}")
    print(f"Conditions:  {config['conditions']}")
    print(f"Paradigms:   {config['paradigms']}")
    print(f"Max tasks:   {config.get('max_tasks', 'all')} per model")
    print(f"CF only:     {config['circularity_free_only']}")
    print("=" * 80)

    # Load resources
    print("\nLoading resources...")
    try:
        exp1_metrics = load_exp1_metrics()
        print(f"  Exp1 metrics: {len(exp1_metrics)} models")
    except FileNotFoundError as e:
        print(f"  WARNING: {e} — using empty metrics (MIRROR scores will be None)")
        exp1_metrics = {}

    all_tasks = load_tasks()
    print(f"  Tasks loaded: {len(all_tasks)}")
    fixed_count = sum(1 for t in all_tasks if t.get("circularity_free"))
    print(f"  Fixed (circularity_free): {fixed_count}")

    client = UnifiedClient(experiment=f"exp9_{run_id}")

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_file:
        results_file = Path(output_file)
        results_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_file = output_dir / f"exp9_{run_id}_results.jsonl"
    checkpoint_file = output_dir / f"exp9_{run_id}_checkpoint.json"

    # Resume: load completed trial keys
    completed_trials: set[tuple] = set()

    def _load_completed_from(path: Path) -> int:
        count = 0
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                    key = (
                        trial.get("model"),
                        trial.get("task_id"),
                        trial.get("condition"),
                        trial.get("paradigm"),
                        "c2_false" if trial.get("is_false_score_control") else "real",
                    )
                    if all(k is not None for k in key):
                        completed_trials.add(key)
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count

    if resume:
        n = _load_completed_from(results_file)
        # When using a custom output file (shard), also check the main results file
        main_results = output_dir / f"exp9_{run_id}_results.jsonl"
        if output_file and main_results != results_file:
            n += _load_completed_from(main_results)
        if n:
            print(f"\nResuming: {n} completed trials")

    is_false_score = (mode == "control2")
    total_new = 0

    results_f = open(results_file, "a", encoding="utf-8")
    try:
        for model in config["models"]:
            print(f"\n{'─' * 80}")
            print(f"Model: {model}")
            print(f"{'─' * 80}")

            model_tasks = select_tasks(
                all_tasks, model,
                config.get("max_tasks"),
                config["circularity_free_only"],
            )
            print(f"  Tasks selected: {len(model_tasks)}")

            effective_concurrency = concurrency_override if concurrency_override is not None else CONCURRENCY
            sem = asyncio.Semaphore(effective_concurrency)
            write_lock = asyncio.Lock()

            # Build all coroutines for this model across all condition × paradigm combos
            coros = []
            for condition in config["conditions"]:
                for paradigm_id in config["paradigms"]:
                    if (condition, paradigm_id) in SKIP_COMBOS:
                        continue
                    for task in model_tasks:
                        coros.append(
                            run_single_trial(
                                client=client,
                                model=model,
                                task=task,
                                condition=condition,
                                paradigm_id=paradigm_id,
                                exp1_metrics=exp1_metrics,
                                is_false_score_control=is_false_score,
                                completed_trials=completed_trials,
                                results_f=results_f,
                                write_lock=write_lock,
                                sem=sem,
                            )
                        )

            results = await asyncio.gather(*coros)
            newly_done = sum(1 for r in results if r is not None)
            total_new += newly_done
            print(f"  Done: {newly_done} new trials for {model}")

            # Checkpoint after each model
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "completed_count": len(completed_trials),
                    "timestamp": datetime.now().isoformat(),
                    "mode": mode,
                    "run_id": run_id,
                }, f)

    finally:
        results_f.close()

    print("\n" + "=" * 80)
    print("EXPERIMENT 9 COMPLETE")
    print("=" * 80)
    print(f"New trials:  {total_new}")
    print(f"Results:     {results_file}")
    print(f"Checkpoint:  {checkpoint_file}")
    print(f"\nNext step:")
    print(f"  python scripts/analyze_experiment_9.py --run-id {run_id}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 9: The Knowing-Doing Gap"
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "extended_pilot", "full", "control2"],
        default="pilot",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to override the config's model list",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Override output results file path (for sharded parallel runs)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override CONCURRENCY (default: 32). Use lower values for slow models.",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition IDs to run (e.g. '1,2'). Splits work for parallel runs.",
    )
    parser.add_argument(
        "--paradigms",
        default=None,
        help="Comma-separated paradigm IDs to run (e.g. '1,2'). Splits work for parallel runs.",
    )
    args = parser.parse_args()
    model_override = [m.strip() for m in args.models.split(",")] if args.models else None
    conditions_override = [int(c.strip()) for c in args.conditions.split(",")] if args.conditions else None
    paradigms_override = [int(p.strip()) for p in args.paradigms.split(",")] if args.paradigms else None
    asyncio.run(run_experiment(
        args.mode, args.resume, run_id=args.run_id,
        model_override=model_override, output_file=args.output_file,
        concurrency_override=args.concurrency,
        conditions_override=conditions_override,
        paradigms_override=paradigms_override,
    ))


if __name__ == "__main__":
    main()
