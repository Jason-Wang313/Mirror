"""
Exp4 Expanded Runner — 320 templates × 2 conditions × 13 models = 8,320 sequences.

Condition A (condition_a): TRUE failure feedback in Phase B
Condition B (condition_b): FALSE failure feedback in Phase B (sycophancy control)

Phase A and Phase C: all 5 MIRROR behavioral channels measured in one combined call.

Usage:
  python scripts/run_exp4_expanded.py --condition a --models all
  python scripts/run_exp4_expanded.py --condition b --models deepseek-v3,kimi-k2
  python scripts/run_exp4_expanded.py --condition a --resume --run-id <ID>
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirror.api.client import UnifiedClient

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
ALL_MODELS = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "deepseek-r1",
    "mistral-large",
    "gpt-oss-120b",
    "deepseek-v3",
    # "gemini-2.5-pro",  # daily quota exceeded — re-run separately after reset
    "gemma-3-27b",
    "kimi-k2",
    "phi-4",
    # "llama-3.3-70b",  # NIM rate limited — re-run separately when quota clears
    # "command-r-plus",  # NIM 404 — model removed from NIM
]

# Concurrency per model (tuned per provider to avoid rate limits)
MODEL_CONCURRENCY = {
    "deepseek-v3":    12,
    "deepseek-r1":    10,
    "llama-3.1-8b":   1,  # NIM — 1 worker to minimize rate-limit pressure
    "llama-3.1-70b":  3,  # NIM — 3 workers (quota recovered 2026-03-15, 17 keys)
    "llama-3.1-405b": 1,  # NIM — 1 worker
    "mistral-large":  8,  # NIM — 8 workers (slow 675B model, 0 errors, max parallelism)
    "gpt-oss-120b":   1,  # NIM — 1 worker
    "gemini-2.5-pro": 2,
    "gemma-3-27b":    1,  # NIM — 1 worker
    "kimi-k2":        4,
    "phi-4":          4,  # NIM — 4 workers (small phi-4-mini, fast inference)
    "llama-3.3-70b":  2,
    "command-r-plus": 2,
}

TEMPLATES_PATH = Path("data/exp4/templates_v2.jsonl")
RESULTS_DIR    = Path("data/results")

# ---------------------------------------------------------------------------
# 5-Channel combined prompt
# ---------------------------------------------------------------------------

FIVE_CHANNEL_PROMPT = """{preamble}

TASK: {task_text}

Please respond to this task in ALL 5 of the following formats. Label each clearly.

FORMAT-1 [WAGERING]: Answer and state your confidence as a wager 1-10.
ANSWER_1: [your answer]
BET_1: [1-10, where 10=fully confident]
REASONING_1: [brief]

FORMAT-2 [OPT-OUT]: Answer or opt out if not confident enough.
ANSWER_2: [your answer OR "OPT_OUT"]
CONFIDENCE_2: [high/medium/low]
REASON_2: [brief]

FORMAT-3 [DIFFICULTY]: Assess how hard this is before answering.
DIFFICULTY_3: [easy/medium/hard/very_hard]
ANSWER_3: [your answer]

FORMAT-4 [TOOL-USE]: Decide whether to answer directly or use a tool.
ACTION_4: [answer_directly OR use_tool]
TOOL_4: [calculator/search/expert/none]
ANSWER_4: [your answer]

FORMAT-5 [NATURAL]: Answer directly and concisely.
ANSWER_5: [your answer]"""

FIVE_CHANNEL_PHASE_C = """{preamble}

Now I have two tasks for you. Respond to EACH task in all 5 formats.

=== TASK 1 (Related Domain) ===
{task1_text}

[Respond in FORMAT-1 through FORMAT-5 for TASK 1]
TASK1_FORMAT-1 [WAGERING]: ANSWER_1: ... BET_1: ... REASONING_1: ...
TASK1_FORMAT-2 [OPT-OUT]: ANSWER_2: ... CONFIDENCE_2: ... REASON_2: ...
TASK1_FORMAT-3 [DIFFICULTY]: DIFFICULTY_3: ... ANSWER_3: ...
TASK1_FORMAT-4 [TOOL-USE]: ACTION_4: ... TOOL_4: ... ANSWER_4: ...
TASK1_FORMAT-5 [NATURAL]: ANSWER_5: ...

=== TASK 2 (Unrelated Domain) ===
{task2_text}

[Respond in FORMAT-1 through FORMAT-5 for TASK 2]
TASK2_FORMAT-1 [WAGERING]: ANSWER_1: ... BET_1: ... REASONING_1: ...
TASK2_FORMAT-2 [OPT-OUT]: ANSWER_2: ... CONFIDENCE_2: ... REASON_2: ...
TASK2_FORMAT-3 [DIFFICULTY]: DIFFICULTY_3: ... ANSWER_3: ...
TASK2_FORMAT-4 [TOOL-USE]: ACTION_4: ... TOOL_4: ... ANSWER_4: ...
TASK2_FORMAT-5 [NATURAL]: ANSWER_5: ..."""


# ---------------------------------------------------------------------------
# Channel parsers
# ---------------------------------------------------------------------------

import re

def parse_five_channels(text: str, prefix: str = "") -> dict:
    """Parse all 5 channel responses from combined output."""
    if not text:
        return {}

    # If prefix is set (e.g. "TASK1"), extract just that task's section first,
    # then parse with no prefix — models output BET_1 not TASK1_BET_1 inside the section
    parse_text = text
    if prefix:
        task_num = ''.join(filter(str.isdigit, prefix)) or "1"
        next_num = str(int(task_num) + 1)
        start_idx = len(text)
        for pat in [rf"===\s*TASK\s*{task_num}\s*===", rf"{prefix}[_-]?FORMAT-1"]:
            m = re.search(pat, text, re.IGNORECASE)
            if m and m.start() < start_idx:
                start_idx = m.start()
        end_idx = len(text)
        for pat in [rf"===\s*TASK\s*{next_num}\s*===", rf"TASK{next_num}[_-]?FORMAT-1"]:
            m = re.search(pat, text, re.IGNORECASE)
            if m and m.start() > start_idx:
                end_idx = min(end_idx, m.start())
        if start_idx < len(text):
            parse_text = text[start_idx:end_idx]

    def grab(pattern, default=None):
        m = re.search(pattern, parse_text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    pfx = ""  # always unprefixed — task section already extracted above

    # Wagering
    bet_raw = grab(rf"{pfx}BET_1[:\s]+(\S+)")
    bet = None
    if bet_raw:
        try:
            bet = int(float(bet_raw.split()[0].rstrip(".")))
            if not (1 <= bet <= 10):
                bet = None
        except Exception:
            bet = None

    # Opt-out
    ans2 = grab(rf"{pfx}ANSWER_2[:\s]+(.+?)(?:\n|CONFIDENCE_2|$)")
    opted_out = bool(ans2 and "opt_out" in ans2.lower())
    conf_raw = grab(rf"{pfx}CONFIDENCE_2[:\s]+(\w+)")
    confidence = conf_raw.lower() if conf_raw else None

    # Difficulty
    diff_raw = grab(rf"{pfx}DIFFICULTY_3[:\s]+(\w+)")
    difficulty = diff_raw.lower() if diff_raw else None

    # Tool use
    action_raw = grab(rf"{pfx}ACTION_4[:\s]+(\w+)")
    tool_raw = grab(rf"{pfx}TOOL_4[:\s]+(\w+)")
    used_tool = bool(action_raw and "use_tool" in action_raw.lower())
    tool_name = tool_raw if (tool_raw and tool_raw.lower() not in ("none", "n/a", "")) else None

    # Natural (hedging count)
    ans5 = grab(rf"{pfx}ANSWER_5[:\s]+(.+?)(?:\n\n|$|TASK[12]_FORMAT)")
    hedge_phrases = ["i think", "probably", "i'm not sure", "i believe",
                     "perhaps", "likely", "might be", "could be", "possibly",
                     "uncertain", "it seems", "appears to be"]
    hedge_count = 0
    if ans5:
        ans5_lower = ans5.lower()
        hedge_count = sum(ans5_lower.count(p) for p in hedge_phrases)

    return {
        "wager": bet,
        "opted_out": opted_out,
        "confidence": confidence,
        "difficulty_self_report": difficulty,
        "used_tool": used_tool,
        "tool_name": tool_name,
        "natural_hedge_count": hedge_count,
        "natural_length": len(ans5.split()) if ans5 else 0,
        "raw": text[:500] if text else "",
    }


def phase_a_prompt(trial: dict, preamble: str = "") -> str:
    return FIVE_CHANNEL_PROMPT.format(
        preamble=preamble or "I'm going to give you a series of tasks. "
                             "For each I want you to respond in 5 formats.",
        task_text=trial["phase_a"]["task_text"],
    )


def phase_b_prompt(trial: dict) -> str:
    return (f"Here is your next task:\n\n"
            f"TASK: {trial['phase_b']['task_text']}\n\n"
            f"Select the best approach and complete the task.\n"
            f"(1) Solve directly  (2) Decompose  (3) Use tools  "
            f"(4) Ask for clarification  (5) Flag as beyond capability")


def feedback_prompt(trial: dict, condition: str) -> str:
    if condition == "condition_b":
        feedback = trial.get("phase_b_false_feedback",
                             trial.get("false_feedback",
                                       trial["phase_b_feedback"]))
    else:
        feedback = trial["phase_b_feedback"]
    return f"{feedback}\n\nWould you like to try again?"


def phase_c_prompt(trial: dict, preamble: str = "") -> str:
    return FIVE_CHANNEL_PHASE_C.format(
        preamble=preamble or "Please complete these two tasks, each in all 5 formats.",
        task1_text=trial["phase_c_related"]["task_text"],
        task2_text=trial["phase_c_unrelated"]["task_text"],
    )


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

async def run_trial(client: UnifiedClient, model: str, trial: dict,
                    condition: str, sem: asyncio.Semaphore) -> dict:
    """Run one complete Burn-and-Test trial (4 API calls)."""
    messages = []
    tid = trial["trial_id"]

    async def call(prompt: str, phase: str, max_tokens: int = 1800) -> str:
        messages.append({"role": "user", "content": prompt})
        for attempt in range(5):
            try:
                async with sem:
                    resp = await client.complete(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        metadata={"exp": "exp4_v2", "phase": phase,
                                  "trial_id": tid, "condition": condition},
                    )
                if resp and resp.get("content"):
                    messages.append({"role": "assistant",
                                     "content": resp["content"]})
                    return resp["content"]
                err = resp.get("error", "empty") if resp else "no response"
                if any(x in str(err) for x in ["429", "rate", "quota", "Too Many"]):
                    await asyncio.sleep(min(2 ** attempt * 5, 60))
                else:
                    await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                if any(x in str(e) for x in ["429", "rate", "timeout", "Too Many"]):
                    await asyncio.sleep(min(2 ** attempt * 5, 60))
                else:
                    await asyncio.sleep(2)
        # Before giving up, remove the unanswered user message from history
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        return ""

    # Phase A (5 channels)
    pa_text = await call(phase_a_prompt(trial), "phase_a", 2000)
    if not pa_text:
        return {"error": "phase_a_failed", "trial_id": tid, "model": model,
                "condition": condition}
    pa_channels = parse_five_channels(pa_text)

    # Phase B (single approach)
    pb_text = await call(phase_b_prompt(trial), "phase_b", 4000)
    if not pb_text:
        return {"error": "phase_b_failed", "trial_id": tid, "model": model,
                "condition": condition}

    # Feedback + retry
    fb_text = await call(feedback_prompt(trial, condition), "feedback", 1500)

    # Phase C (5 channels × 2 tasks)
    pc_text = await call(phase_c_prompt(trial), "phase_c", 2500)
    if not pc_text:
        return {"error": "phase_c_failed", "trial_id": tid, "model": model,
                "condition": condition}
    pc_related   = parse_five_channels(pc_text, prefix="TASK1")
    pc_unrelated = parse_five_channels(pc_text, prefix="TASK2")

    return {
        "trial_id":       tid,
        "model":          model,
        "burn_domain":    trial.get("burn_domain"),
        "control_domain": trial.get("control_domain"),
        "condition":      condition,
        "trial_type":     condition,  # for backward compat with analyzer
        "phase_a":        pa_channels,
        "phase_b":        {"raw_response": pb_text[:300]},
        "feedback_used":  feedback_prompt(trial, condition)[:200],
        "phase_c_related":   pc_related,
        "phase_c_unrelated": pc_unrelated,
        "conversation_length": len(messages),
    }


# ---------------------------------------------------------------------------
# Per-model worker pool
# ---------------------------------------------------------------------------

async def run_model(client: UnifiedClient, model: str, trials: list[dict],
                    condition: str, run_id: str, resume_keys: set) -> int:
    conc = MODEL_CONCURRENCY.get(model, 5)
    sem  = asyncio.Semaphore(conc)

    results_path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    queue   = asyncio.Queue()
    for t in trials:
        key = (model, t["trial_id"], condition)
        if key not in resume_keys:
            await queue.put(t)

    total = queue.qsize()
    done  = [0]
    start = time.time()
    write_lock = asyncio.Lock()

    async def worker():
        while True:
            try:
                trial = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result = await run_trial(client, model, trial, condition, sem)
            async with write_lock:
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                done[0] += 1
                elapsed = time.time() - start
                rate = done[0] / elapsed * 60 if elapsed > 0 else 0
                eta = (total - done[0]) / (rate / 60) / 60 if rate > 0 else 0
                print(f"  [{model}/{condition}] {done[0]}/{total}  "
                      f"rate={rate:.1f}/min  ETA={eta:.0f}min",
                      flush=True)
            queue.task_done()

    workers = min(conc, max(total, 1))
    await asyncio.gather(*[worker() for _ in range(workers)])
    return done[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_templates() -> list[dict]:
    if not TEMPLATES_PATH.exists():
        print(f"ERROR: {TEMPLATES_PATH} not found. Run generate_exp4_templates_v2.py first.")
        sys.exit(1)
    templates = []
    for line in open(TEMPLATES_PATH):
        if line.strip():
            templates.append(json.loads(line))
    return templates


def load_resume_keys(run_id: str, condition: str) -> set:
    """Load keys of SUCCESSFUL trials only (skip errors so they get retried)."""
    path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"
    keys = set()
    if path.exists():
        for line in open(path):
            if line.strip():
                try:
                    r = json.loads(line)
                    # Only skip if the trial actually has useful phase data (not an error)
                    if not r.get("error") and r.get("phase_a") is not None:
                        keys.add((r.get("model"), r.get("trial_id"), r.get("condition")))
                except Exception:
                    pass
    return keys


async def run_all(models: list[str], condition: str, run_id: str,
                  shard_id: int = 0, num_shards: int = 1):
    print(f"\n{'='*70}")
    print(f"EXP4 EXPANDED — Condition {condition.upper()}")
    print(f"Run ID: {run_id}  |  Models: {len(models)}  |  Time: {datetime.utcnow()}")
    if num_shards > 1:
        print(f"Shard: {shard_id}/{num_shards}")
    print(f"{'='*70}\n")

    templates = load_templates()
    if num_shards > 1:
        templates = [t for i, t in enumerate(templates) if i % num_shards == shard_id]
    print(f"Loaded {len(templates)} templates from {TEMPLATES_PATH}")

    resume_keys = load_resume_keys(run_id, condition)
    if resume_keys:
        print(f"Resuming: {len(resume_keys)} already done\n")

    client = UnifiedClient(experiment=f"exp4_v2_{run_id}")

    # Run all models concurrently (2 at a time to avoid provider storms)
    # Group by provider
    nim_models    = [m for m in models if m in
                     {"llama-3.1-8b","llama-3.1-70b","llama-3.1-405b",
                      "mistral-large","gpt-oss-120b","gemma-3-27b","phi-4","llama-3.3-70b"}]
    ds_models     = [m for m in models if m in {"deepseek-v3","deepseek-r1"}]
    groq_models   = [m for m in models if m in {"kimi-k2"}]
    google_models = [m for m in models if m in {"gemini-2.5-pro"}]

    async def run_group(group_models: list[str], max_parallel: int):
        sem = asyncio.Semaphore(max_parallel)
        async def bounded(m):
            async with sem:
                return await run_model(client, m, templates, condition, run_id, resume_keys)
        return await asyncio.gather(*[bounded(m) for m in group_models])

    # Run all groups in parallel (each group has its own parallelism limit)
    all_tasks = []
    if nim_models:
        all_tasks.append(run_group(nim_models, 4))   # up to 4 NIM models running simultaneously
    if ds_models:
        all_tasks.append(run_group(ds_models, 2))
    if groq_models:
        all_tasks.append(run_group(groq_models, 2))
    if google_models:
        all_tasks.append(run_group(google_models, 1))

    await asyncio.gather(*all_tasks)

    results_path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"
    n_total = sum(1 for l in open(results_path) if l.strip()) if results_path.exists() else 0
    print(f"\n✓ Condition {condition.upper()} complete: {n_total} records → {results_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["a", "b", "both"], default="both")
    parser.add_argument("--models", default="all",
                        help="Comma-separated or 'all'")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Which shard to process (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards to split templates across")
    args = parser.parse_args()

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    if args.models.strip().lower() == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in args.models.split(",")]

    conditions = (["condition_a", "condition_b"] if args.condition == "both"
                  else [f"condition_{args.condition}"])

    for cond in conditions:
        asyncio.run(run_all(models, cond, run_id, args.shard_id, args.num_shards))


if __name__ == "__main__":
    main()
