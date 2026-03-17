"""
API Airforce runner for exp4 deepseek-r1 (21 missing trials).

DeepSeek direct API: 402 Insufficient Balance.
NIM deepseek/deepseek-r1: 410 Gone (retired).
API Airforce: deepseek-r1 available, 1 RPS global rate limit.

21 trials × 4 calls = 84 calls at 1.2s spacing ≈ 2 minutes total.
MODEL_SLUG = "deepseek-r1" — matches existing records for seamless merge.
"""
import asyncio, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from openai import AsyncOpenAI

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG   = "deepseek-r1"
AIR_MODEL    = "deepseek-r1"
AIR_BASE     = "https://api.airforce/v1"
AIR_API_KEY  = os.environ.get("AIRFORCE_API_KEY", "")
RUN_ID       = "20260314T135731"
MIN_INTERVAL = 1.2   # seconds between API calls (1 RPS global limit + headroom)


class AirForceCaller:
    """Single-worker caller with strict 1.2s inter-call rate limiting."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=AIR_API_KEY, base_url=AIR_BASE, timeout=60)
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def call(self, messages: list, max_tokens: int) -> str:
        for attempt in range(6):
            async with self._lock:
                elapsed = time.monotonic() - self._last_call
                if elapsed < MIN_INTERVAL:
                    await asyncio.sleep(MIN_INTERVAL - elapsed)
                self._last_call = time.monotonic()
            try:
                resp = await self.client.chat.completions.create(
                    model=AIR_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=60,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    backoff = min(2 * (2 ** attempt), 30)
                    print(f"  [airforce] 429 → sleep {backoff}s (attempt {attempt+1})", flush=True)
                    await asyncio.sleep(backoff)
                elif "timeout" in err.lower():
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    print(f"  [airforce] error: {err[:80]}", flush=True)
                    await asyncio.sleep(min(3 * (2 ** attempt), 30))
        return ""


async def run_trial(caller: AirForceCaller, trial: dict, condition: str) -> dict:
    messages = []
    tid = trial["trial_id"]

    async def call(prompt: str, max_tokens: int) -> str:
        messages.append({"role": "user", "content": prompt})
        content = await caller.call(list(messages), max_tokens)
        if content:
            messages.append({"role": "assistant", "content": content})
        elif messages and messages[-1]["role"] == "user":
            messages.pop()
        return content

    pa = await call(phase_a_prompt(trial), 2000)
    if not pa:
        return {"error": "phase_a_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    pb = await call(phase_b_prompt(trial), 4000)
    if not pb:
        return {"error": "phase_b_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    await call(feedback_prompt(trial, condition), 1500)   # allowed to be empty

    pc = await call(phase_c_prompt(trial), 2500)
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
        "_provider":          "api_airforce",
        "_air_model":         AIR_MODEL,
    }


async def main():
    caller = AirForceCaller()
    print(f"API Airforce runner: {MODEL_SLUG} via {AIR_MODEL}", flush=True)
    print(f"Run ID: {RUN_ID}  Rate: 1 RPS  ETA: ~2 min for 21 trials", flush=True)

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    results_paths = {
        "condition_a": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_a_results.jsonl",
        "condition_b": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_b_results.jsonl",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    queue = asyncio.Queue()
    for condition in ["condition_a", "condition_b"]:
        done_keys = load_resume_keys(RUN_ID, condition)
        pending = [t for t in templates
                   if (MODEL_SLUG, t["trial_id"], condition) not in done_keys]
        print(f"  {condition}: {len(pending)} pending", flush=True)
        for t in pending:
            await queue.put((t, condition))
        total += len(pending)

    if total == 0:
        print(f"All done — deepseek-r1 has 320/320 both conditions.", flush=True)
        return

    print(f"Total: {total} trials  ETA ~{total * 4 * MIN_INTERVAL / 60:.1f} min", flush=True)

    write_lock = asyncio.Lock()
    counters = {"done": 0, "ok": 0}
    start = time.time()

    while not queue.empty():
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
                f"  [{cond_ltr}] {counters['done']}/{total} "
                f"{'OK' if ok else 'ERR:' + str(result.get('error',''))}"
                f"  rate={rate:.1f}/min  ETA={eta:.0f}min"
                f"  {trial['trial_id'][:35]}",
                flush=True
            )
        queue.task_done()

    print(f"\n=== DONE: {counters['ok']}/{total} successful ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
