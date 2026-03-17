"""
SambaNova runner for exp4 llama-3.1-70b.

NIM llama-3.1-70b: ~50-90s/call (inference backlog), 0.8/min.
SambaNova Meta-Llama-3.3-70B-Instruct: ~2-4s/call (109 tok/s), 15-20x faster.

Model mapping (same as Groq runner):
  MODEL_SLUG = "llama-3.1-70b"  — matches NIM runner, results merge via dedup
  SAMBA_MODEL = "Meta-Llama-3.3-70B-Instruct"  — same 70B Llama family

load_resume_keys() deduplication ensures NIM + SambaNova runners never
re-run a trial completed by the other.

SambaNova free tier observed limits:
  - ~60 RPM sustained
  - 5 workers x 4 calls/trial = 20 RPM peak — well within limits
  - Expected: ~15-20 trials/min → 217 pending → ~12 min total
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

MODEL_SLUG   = "llama-3.1-70b"
SAMBA_MODEL  = "Meta-Llama-3.3-70B-Instruct"
SAMBA_BASE   = "https://api.sambanova.ai/v1"
SAMBA_KEY    = "f6682363-559e-475d-81cf-80b828a1203c"
RUN_ID       = "20260314T135731"
CONCURRENCY  = 1     # 1 worker + rate limiter = no burst possible
MAX_RPM      = 8     # conservative (observed limit ~10 RPM; leave headroom)


class RateLimiter:
    """Token bucket: enforces MAX_RPM across all calls."""
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


_rate = RateLimiter(MAX_RPM)


class SambaNovaCaller:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=SAMBA_KEY, base_url=SAMBA_BASE, timeout=60)

    async def call(self, messages: list, max_tokens: int) -> str:
        for attempt in range(6):
            await _rate.acquire()
            try:
                resp = await self.client.chat.completions.create(
                    model=SAMBA_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=60,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    backoff = min(10 * (2 ** attempt), 60)
                    print(f"  [samba] 429 → sleep {backoff}s (attempt {attempt+1})", flush=True)
                    await asyncio.sleep(backoff)
                elif "timeout" in err.lower():
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    print(f"  [samba] error: {err[:80]}", flush=True)
                    await asyncio.sleep(min(3 * (2 ** attempt), 30))
        return ""


async def run_trial(caller: SambaNovaCaller, trial: dict, condition: str) -> dict:
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

    await call(feedback_prompt(trial, condition), 1500)

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
        "_provider":          "sambanova",
        "_samba_model":       SAMBA_MODEL,
    }


async def main():
    caller = SambaNovaCaller()
    print(f"SambaNova runner: {MODEL_SLUG} via {SAMBA_MODEL}", flush=True)
    print(f"Run ID: {RUN_ID}  Concurrency: {CONCURRENCY}  ~109 tok/s", flush=True)

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    results_paths = {
        "condition_a": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_a_results.jsonl",
        "condition_b": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_b_results.jsonl",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queue = asyncio.Queue()
    total = 0
    for condition in ["condition_a", "condition_b"]:
        done_keys = load_resume_keys(RUN_ID, condition)
        pending = [t for t in templates
                   if (MODEL_SLUG, t["trial_id"], condition) not in done_keys]
        print(f"  {condition}: {len(pending)} pending", flush=True)
        for t in pending:
            await queue.put((t, condition))
        total += len(pending)

    if total == 0:
        print("All done — llama-3.1-70b 320/320 both conditions.", flush=True)
        return

    eta = total / 15
    print(f"Total: {total} trials  ETA ~{eta:.0f} min at ~15/min", flush=True)

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
                if ok: counters["ok"] += 1
                elapsed = time.time() - start
                rate = counters["done"] / elapsed * 60 if elapsed > 0 else 0
                eta_min = (total - counters["done"]) / rate if rate > 0 else 0
                print(
                    f"  [w{wid}][{cond_ltr}] {counters['done']}/{total} "
                    f"{'OK' if ok else 'ERR:'+str(result.get('error',''))}"
                    f"  rate={rate:.1f}/min  ETA={eta_min:.0f}min"
                    f"  {trial['trial_id'][:35]}",
                    flush=True
                )
            queue.task_done()

    await asyncio.gather(*[worker(i) for i in range(CONCURRENCY)])
    print(f"\n=== DONE: {counters['ok']}/{total} successful ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
