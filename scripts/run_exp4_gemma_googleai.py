"""
Google AI Studio runner for exp4 gemma-3-12b.

NIM queue for google/gemma-3-12b-it is 50-90s per call (inference backlog).
Google AI Studio serves the EXACT same model (gemma-3-12b-it) in ~2-5s latency.

MODEL_SLUG = "gemma-3-12b" — identical to the NIM runner — so results land in
the same JSONL bucket and load_resume_keys() deduplicates correctly.

Google AI Studio observed limits for gemma-3-12b-it (2026-03-15):
  - 15 RPM (requests per minute) — free tier
  - 1M TPM  (tokens per minute)
  - 1500 RPD (requests per day)
  - Latency: ~2-5s per call (vs. 50-90s on NIM)

Rate-limit strategy (updated):
  - Token-bucket rate limiter at 13 RPM (headroom under 15) prevents all 429s
  - CONCURRENCY=3 workers keeps all 13 slots/min utilized even with latency jitter
  - Max throughput = 13 RPM / 4 calls per trial = 3.25 trials/min
  - vs. 2.4/min before (with 429 backoffs burning 60s each)
  - 255 pending → ~78 min (vs ~105 min before)
"""
import asyncio, json, os, sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG       = "gemma-3-12b"          # Must match NIM runner slug — for result merging
GOOGLE_MODEL     = "gemma-3-12b-it"       # Exact same model as NIM's google/gemma-3-12b-it
RUN_ID           = "20260314T135731"
CONCURRENCY      = 1    # 1 worker + rate limiter = smooth 13 RPM, zero 429 bursts
MAX_RPM          = 13   # Token bucket limit (headroom under 15 RPM free tier)


class TokenBucket:
    """Shared rate limiter: max MAX_RPM calls per 60-second sliding window."""
    def __init__(self, rpm: int):
        self._rpm = rpm
        self._min_interval = 60.0 / rpm   # minimum seconds between calls
        self._timestamps: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            # Drop timestamps older than 60s
            while self._timestamps and self._timestamps[0] < now - 60.0:
                self._timestamps.popleft()
            # If at capacity, wait until the oldest call falls outside the window
            if len(self._timestamps) >= self._rpm:
                wait = self._timestamps[0] + 60.0 - now
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                    while self._timestamps and self._timestamps[0] < now - 60.0:
                        self._timestamps.popleft()
            # Also enforce minimum inter-call spacing to smooth bursts
            if self._timestamps:
                gap = now - self._timestamps[-1]
                if gap < self._min_interval:
                    await asyncio.sleep(self._min_interval - gap)
            self._timestamps.append(time.monotonic())


_rate_limiter = TokenBucket(MAX_RPM)


def load_google_key() -> str:
    key = os.getenv("GOOGLE_AI_API_KEY")
    if not key:
        print("ERROR: GOOGLE_AI_API_KEY not set in environment / .env", flush=True)
        sys.exit(1)
    return key


def _messages_to_contents(messages: list) -> list:
    """Convert OpenAI-style messages to Google AI contents format."""
    contents = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role in ("assistant", "model"):
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
        # system messages: prepend to first user message or skip (gemma-3 handles system via user turn)
    return contents


class GoogleAICaller:
    """Async caller for Google AI Studio (gemma-3-12b-it) with retry and rate-limit handling."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self._config_cache = {}

    def _make_config(self, max_tokens: int) -> types.GenerateContentConfig:
        if max_tokens not in self._config_cache:
            self._config_cache[max_tokens] = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
            )
        return self._config_cache[max_tokens]

    async def call(self, messages: list, max_tokens: int) -> str:
        contents = _messages_to_contents(messages)
        config   = self._make_config(max_tokens)

        for attempt in range(6):
            await _rate_limiter.acquire()   # enforce 13 RPM globally across all workers
            try:
                resp = await self.client.aio.models.generate_content(
                    model=GOOGLE_MODEL,
                    contents=contents,
                    config=config,
                )
                return resp.text or ""

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                    backoff = min(60 * (2 ** attempt), 300)
                    print(f"  [googleai] 429/quota → sleeping {backoff}s (attempt {attempt+1})", flush=True)
                    await asyncio.sleep(backoff)
                elif "timeout" in err.lower() or "deadline" in err.lower():
                    wait_s = 10 * (attempt + 1)
                    print(f"  [googleai] timeout → retry in {wait_s}s", flush=True)
                    await asyncio.sleep(wait_s)
                else:
                    wait_s = min(5 * (2 ** attempt), 60)
                    print(f"  [googleai] error: {err[:120]} → retry in {wait_s}s", flush=True)
                    await asyncio.sleep(wait_s)
        return ""


async def run_trial(caller: GoogleAICaller, trial: dict, condition: str) -> dict:
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
        "_provider":          "google_ai",      # audit trail
        "_google_model":      GOOGLE_MODEL,     # audit trail
    }


async def main():
    google_key = load_google_key()
    caller     = GoogleAICaller(google_key)

    print(f"Google AI runner: {MODEL_SLUG} via {GOOGLE_MODEL}", flush=True)
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Concurrency: {CONCURRENCY} workers  Rate-limit: {MAX_RPM} RPM token bucket", flush=True)

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

    print(f"Total pending: {total} trials", flush=True)
    # Estimate: 13 RPM / 4 calls per trial = 3.25 trials/min
    eta_min = total / 3.25
    print(f"Estimated completion: ~{eta_min:.0f} min (~{eta_min/60:.1f} hours) at ~3.25 trials/min (13 RPM)", flush=True)

    write_lock = asyncio.Lock()
    counters   = {"done": 0, "total": total}
    start      = time.time()

    async def worker(wid: int):
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

    await asyncio.gather(*[worker(i) for i in range(CONCURRENCY)])

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
