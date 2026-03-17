"""
Force-retry stuck llama-3.1-70b trials for Exp4 expanded via NIM (direct, key rotation).

Normal runners use load_resume_keys() which skips trials with ANY record (error or success).
This script only skips trials with SUCCESSFUL records, forcing retries of 34+94=128 stuck trials.

Uses all 4 NIM keys rotating to spread load. CONCURRENCY=1 to stay within per-key limits.
MODEL_SLUG stays "llama-3.1-70b" for result merging consistency.
"""
import asyncio, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG  = "llama-3.1-70b"
NIM_MODEL   = "meta/llama-3.1-70b-instruct"
NIM_BASE    = "https://integrate.api.nvidia.com/v1"
RUN_ID      = "20260314T135731"
CONCURRENCY = 1
MAX_TOKENS  = {"phase_a": 2000, "phase_b": 4000, "feedback": 1500, "phase_c": 2500}


def load_nim_keys() -> list:
    keys = []
    for var in ["NVIDIA_NIM_API_KEY"] + [f"NVIDIA_NIM_API_KEY_{i}" for i in range(2, 19)]:
        k = os.getenv(var)
        if k:
            keys.append(k)
    if not keys:
        raise RuntimeError("No NVIDIA_NIM_API_KEY* found in environment")
    print(f"NIM keys loaded: {len(keys)}", flush=True)
    return keys


def load_success_keys(run_id: str, condition: str) -> set:
    """Return set of trial_ids with SUCCESSFUL records only (ignores error records)."""
    cond_ltr = condition.split("_")[-1]
    path = RESULTS_DIR / f"exp4_v2_{run_id}_condition_{cond_ltr}_results.jsonl"
    done = set()
    if not path.exists():
        return done
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if (r.get("model") == MODEL_SLUG
                    and not r.get("error")
                    and r.get("phase_a") is not None):
                done.add(r["trial_id"])
        except Exception:
            pass
    return done


class NIMCaller:
    def __init__(self, keys: list):
        self._keys = keys
        self._idx = 0

    def _next_client(self) -> AsyncOpenAI:
        key = self._keys[self._idx % len(self._keys)]
        self._idx += 1
        return AsyncOpenAI(api_key=key, base_url=NIM_BASE, timeout=120)

    async def call(self, messages: list, max_tokens: int) -> str:
        for attempt in range(6):
            client = self._next_client()
            try:
                resp = await client.chat.completions.create(
                    model=NIM_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "Too Many" in err:
                    backoff = min(10 * (2 ** attempt), 120)
                    print(f"  [nim] 429 → sleep {backoff}s key_{(self._idx-1) % len(self._keys)} (attempt {attempt+1})", flush=True)
                    await asyncio.sleep(backoff)
                elif "timeout" in err.lower():
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    print(f"  [nim] error: {err[:100]}", flush=True)
                    await asyncio.sleep(min(5 * (2 ** attempt), 60))
        return ""


async def run_trial(caller: NIMCaller, trial: dict, condition: str) -> dict:
    messages = []
    tid = trial["trial_id"]

    async def call(prompt: str, phase: str) -> str:
        messages.append({"role": "user", "content": prompt})
        content = await caller.call(list(messages), MAX_TOKENS[phase])
        if content:
            messages.append({"role": "assistant", "content": content})
        elif messages and messages[-1]["role"] == "user":
            messages.pop()
        return content

    pa = await call(phase_a_prompt(trial), "phase_a")
    if not pa:
        return {"error": "phase_a_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    pb = await call(phase_b_prompt(trial), "phase_b")
    if not pb:
        return {"error": "phase_b_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    await call(feedback_prompt(trial, condition), "feedback")

    pc = await call(phase_c_prompt(trial), "phase_c")
    if not pc:
        return {"error": "phase_c_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    return {
        "trial_id":           tid,
        "model":              MODEL_SLUG,
        "burn_domain":        trial.get("burn_domain"),
        "control_domain":     trial.get("control_domain"),
        "condition":          condition,
        "trial_type":         condition,
        "phase_a":            parse_five_channels(pa),
        "phase_b":            {"raw_response": pb[:300]},
        "feedback_used":      feedback_prompt(trial, condition)[:200],
        "phase_c_related":    parse_five_channels(pc, prefix="TASK1"),
        "phase_c_unrelated":  parse_five_channels(pc, prefix="TASK2"),
        "conversation_length": len(messages),
        "_provider":          "nim_force_retry",
        "_nim_model":         NIM_MODEL,
    }


async def main():
    keys = load_nim_keys()
    caller = NIMCaller(keys)

    print(f"Force-retry via NIM: {NIM_MODEL}  Keys: {len(keys)}", flush=True)
    # Brief pause to let any lingering 429 windows clear
    print("Waiting 90s for NIM rate-limit window to clear...", flush=True)
    await asyncio.sleep(90)

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]

    results_paths = {
        "condition_a": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_a_results.jsonl",
        "condition_b": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_b_results.jsonl",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queue = asyncio.Queue()
    total = 0
    for condition in ["condition_a", "condition_b"]:
        success_keys = load_success_keys(RUN_ID, condition)
        pending = [t for t in templates if t["trial_id"] not in success_keys]
        print(f"  {condition}: {len(pending)} to force-retry (have {len(success_keys)} successful)", flush=True)
        for t in pending:
            await queue.put((t, condition))
        total += len(pending)

    if total == 0:
        print("All trials already have successful records — nothing to do.", flush=True)
        return

    print(f"Total: {total} trials  Concurrency: {CONCURRENCY}", flush=True)

    write_lock = asyncio.Lock()
    counters = {"done": 0, "ok": 0}
    start = time.time()

    async def worker(wid: int):
        while True:
            try:
                trial, condition = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result = await run_trial(caller, trial, condition)
            ok = not result.get("error") and result.get("phase_a") is not None
            cond_ltr = condition.split("_")[-1]
            async with write_lock:
                with open(results_paths[condition], "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush(); os.fsync(f.fileno())
                counters["done"] += 1
                if ok:
                    counters["ok"] += 1
                elapsed = time.time() - start
                rate = counters["done"] / elapsed * 60 if elapsed > 0 else 0
                eta = (total - counters["done"]) / rate if rate > 0 else 0
                print(
                    f"  [w{wid}][{cond_ltr}] {counters['done']}/{total} "
                    f"{'OK' if ok else 'ERR:'+str(result.get('error',''))}"
                    f"  rate={rate:.1f}/min  ETA={eta:.0f}min"
                    f"  {trial['trial_id'][:35]}",
                    flush=True
                )
            queue.task_done()

    await asyncio.gather(*[worker(i) for i in range(CONCURRENCY)])
    print(f"\n=== DONE: {counters['ok']}/{counters['done']} successful ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
