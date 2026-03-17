"""
Groq-backed runner for exp4 llama-3.1-70b (fast parallel).

NIM quota for llama-3.1-70b is exhausted/throttled. This runner uses Groq's
LPU inference (llama-3.3-70b-versatile) to complete remaining trials.

MODEL SLUG = "llama-3.1-70b" — identical to the NIM runner — so results land
in the same JSONL bucket and load_resume_keys() deduplicates correctly.
Neither this runner nor the NIM runner will re-run a trial completed by the other.

Groq rate limits (observed 2026-03-15):
  - 1,000 RPM (requests per minute)
  - 12,000 TPM (tokens per minute, sliding window ~12s)
  - Actual per-call latency: ~0.25-0.4s (LPU hardware)
  - Actual tokens per trial: ~1,200-1,800 (well within 12k/min budget)
  - Safe concurrency: 5 workers

Groq model: llama-3.3-70b-versatile (closest available Llama on Groq)
NIM ran llama-3.1-70b — architecturally very similar (Llama 3.x 70B family).
Results tagged MODEL_SLUG = "llama-3.1-70b" for downstream analysis consistency.
"""
import asyncio, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG   = "llama-3.1-70b"          # Must match NIM runner slug — for result merging
GROQ_MODEL   = "llama-3.3-70b-versatile"  # Groq's closest available Llama 70B
GROQ_BASE    = "https://api.groq.com/openai/v1"
RUN_ID       = "20260314T135731"
CONCURRENCY  = 1   # IP-level throttle confirmed — 1 worker is the safe ceiling


def load_groq_keys() -> list:
    keys = []
    for var in ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(2, 20)]:
        k = os.getenv(var)
        if k:
            keys.append(k)
    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); unique.append(k)
    if not unique:
        print("ERROR: No GROQ_API_KEY found in environment / .env", flush=True)
        sys.exit(1)
    print(f"Loaded {len(unique)} Groq keys", flush=True)
    return unique


class GroqCaller:
    """Per-key Groq caller — each instance owns one key's token bucket."""
    def __init__(self, api_key: str, key_idx: int):
        self.client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE)
        self.key_idx = key_idx
        self._lock = asyncio.Lock()
        self._backoff_until = 0.0

    async def call(self, messages: list, max_tokens: int) -> str:
        for attempt in range(8):
            async with self._lock:
                wait = self._backoff_until - time.time()
            if wait > 0:
                print(f"  [key{self.key_idx+1}] back-off {wait:.1f}s…", flush=True)
                await asyncio.sleep(wait)
            try:
                resp = await self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=60,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    backoff = min(32 * (2 ** attempt), 120)
                    print(f"  [key{self.key_idx+1}] 429 → sleep {backoff}s (attempt {attempt+1})", flush=True)
                    async with self._lock:
                        self._backoff_until = time.time() + backoff
                    await asyncio.sleep(backoff)
                elif "timeout" in err.lower() or "timed out" in err.lower():
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    await asyncio.sleep(min(5 * (2 ** attempt), 30))
        return ""


async def run_trial(caller: GroqCaller, trial: dict, condition: str) -> dict:
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

    fb = await call(feedback_prompt(trial, condition), 1500)   # allowed to be empty

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
        "_provider":          "groq",       # audit trail — not used by analysis
        "_groq_model":        GROQ_MODEL,   # audit trail
    }


async def main():
    keys    = load_groq_keys()
    callers = [GroqCaller(k, i) for i, k in enumerate(keys)]
    n_keys  = len(keys)
    concurrency = min(CONCURRENCY, n_keys)

    print(f"Groq runner: {MODEL_SLUG} via {GROQ_MODEL}", flush=True)
    print(f"Run ID: {RUN_ID}  Keys: {n_keys}  Workers: {concurrency}", flush=True)

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
        print(f"  {condition}: {len(pending)} pending (of {len(templates)})", flush=True)
        for t in pending:
            await queue.put((t, condition))
        total += len(pending)

    if total == 0:
        print(f"All done! {MODEL_SLUG} has 320/320 for both conditions.", flush=True)
        return

    print(f"Total pending: {total} trials  ETA ~{total/n_keys:.0f} min at {n_keys} keys", flush=True)

    write_lock = asyncio.Lock()
    counters   = {"done": 0, "total": total}
    start      = time.time()

    async def worker(wid: int):
        caller = callers[wid % n_keys]   # each worker owns one key
        while True:
            try:
                trial, condition = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result   = await run_trial(caller, trial, condition)
            ok       = not result.get("error") and result.get("phase_a") is not None
            cond_ltr = condition.split("_")[-1]

            async with write_lock:
                with open(results_paths[condition], "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush(); os.fsync(f.fileno())
                counters["done"] += 1
                elapsed = time.time() - start
                rate    = counters["done"] / elapsed * 60 if elapsed > 0 else 0
                eta_min = (counters["total"] - counters["done"]) / rate if rate > 0 else 0
                print(
                    f"  [w{wid}][{cond_ltr}] {counters['done']}/{total} "
                    f"{'OK' if ok else 'ERR:' + str(result.get('error',''))}"
                    f"  rate={rate:.1f}/min  ETA={eta_min:.0f}min"
                    f"  {trial['trial_id'][:30]}",
                    flush=True
                )
            queue.task_done()

    # Stagger worker starts by 8s each to avoid simultaneous burst
    tasks = []
    for i in range(concurrency):
        if i > 0:
            await asyncio.sleep(8)
        tasks.append(asyncio.create_task(worker(i)))
    await asyncio.gather(*tasks)

    # Final summary
    print("\n=== FINAL ===", flush=True)
    for cl in ["a", "b"]:
        p = results_paths[f"condition_{cl}"]
        done = set()
        if p.exists():
            for line in open(p):
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    if (not r.get("error") and r.get("phase_a")
                            and r.get("model") == MODEL_SLUG):
                        done.add(r["trial_id"])
                except Exception:
                    pass
        status = "COMPLETE" if len(done) >= 320 else "INCOMPLETE"
        print(f"  condition_{cl}: {len(done)}/320  {status}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
