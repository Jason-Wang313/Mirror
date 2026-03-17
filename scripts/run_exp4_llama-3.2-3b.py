"""
Key-rotating runner for exp4 llama-3.2-3b (NIM).

NIM applies PER-MODEL rate limits — safe to run in parallel with all other model runners.
meta/llama-3.2-3b-instruct: small Meta model, fast inference, Exp8 scaling point.

Concurrency: 3 workers
Model slug written to JSONL: "llama-3.2-3b"
"""
import asyncio, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG  = "llama-3.2-3b"
NIM_MODEL   = "meta/llama-3.2-3b-instruct"
BASE_URL    = "https://integrate.api.nvidia.com/v1"
RUN_ID      = "20260314T135731"
CONCURRENCY = 3


def load_keys():
    keys = []
    k = os.getenv("NVIDIA_NIM_API_KEY")
    if k:
        keys.append(k)
    for i in range(2, 30):
        k = os.getenv(f"NVIDIA_NIM_API_KEY_{i}")
        if k:
            keys.append(k)
    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); unique.append(k)
    return unique


class KeyRotator:
    """Rotates through keys on 429. Tracks per-key throttle state."""
    def __init__(self, keys):
        self.keys = keys
        self.clients = [AsyncOpenAI(api_key=k, base_url=BASE_URL) for k in keys]
        self.throttled_until = [0.0] * len(keys)
        self._lock = asyncio.Lock()
        self.current = 0

    async def call(self, messages, max_tokens):
        n = len(self.keys)
        for full_cycle in range(3):
            for offset in range(n):
                async with self._lock:
                    idx = (self.current + offset) % n
                now = time.time()
                wait = self.throttled_until[idx] - now
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    resp = await self.clients[idx].chat.completions.create(
                        model=NIM_MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        timeout=120,
                    )
                    async with self._lock:
                        self.current = (idx + 1) % n
                    return resp.choices[0].message.content
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                        self.throttled_until[idx] = time.time() + 20
                        print(f"  [key{idx+1}] 429 → rotating to next key", flush=True)
                        continue
                    elif "timeout" in err.lower() or "timed out" in err.lower():
                        print(f"  [key{idx+1}] timeout → backing off 10s", flush=True)
                        await asyncio.sleep(10)
                        continue
                    else:
                        wait_s = min(5 * (2 ** full_cycle), 30)
                        await asyncio.sleep(wait_s)
                        continue
            soonest = min(self.throttled_until)
            wait_s = max(soonest - time.time(), 1)
            print(f"  All {n} keys throttled, waiting {wait_s:.0f}s...", flush=True)
            await asyncio.sleep(wait_s)
        return ""


async def run_trial(rotator, trial, condition):
    messages = []
    tid = trial["trial_id"]

    async def call(prompt, max_tokens):
        messages.append({"role": "user", "content": prompt})
        content = await rotator.call(list(messages), max_tokens)
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
    fb = await call(feedback_prompt(trial, condition), 1500)
    pc = await call(phase_c_prompt(trial), 3000)
    if not pc:
        return {"error": "phase_c_failed", "trial_id": tid, "model": MODEL_SLUG, "condition": condition}

    return {
        "trial_id": tid, "model": MODEL_SLUG,
        "burn_domain": trial.get("burn_domain"),
        "control_domain": trial.get("control_domain"),
        "condition": condition, "trial_type": condition,
        "phase_a": parse_five_channels(pa),
        "phase_b": {"raw_response": pb[:300]},
        "feedback_used": feedback_prompt(trial, condition)[:200],
        "phase_c_related": parse_five_channels(pc, prefix="TASK1"),
        "phase_c_unrelated": parse_five_channels(pc, prefix="TASK2"),
        "conversation_length": len(messages),
    }


async def main():
    keys = load_keys()
    print(f"Key-rotating runner: {len(keys)} keys, concurrency={CONCURRENCY}", flush=True)
    print(f"Model: {MODEL_SLUG} ({NIM_MODEL})", flush=True)
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"NIM rate limits: PER-MODEL (safe to run alongside all other runners)", flush=True)

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    results_paths = {
        "condition_a": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_a_results.jsonl",
        "condition_b": RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_b_results.jsonl",
    }

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
        print(f"All done! {MODEL_SLUG} has 320/320 for both conditions.", flush=True)
        return

    print(f"Total: {total} trials across both conditions.", flush=True)

    rotator = KeyRotator(keys)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "total": total}
    start = time.time()

    async def worker(wid):
        while True:
            try:
                trial, condition = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result = await run_trial(rotator, trial, condition)
            ok = not result.get("error") and result.get("phase_a") is not None
            cond_letter = condition.split("_")[-1]
            async with write_lock:
                with open(results_paths[condition], "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush(); os.fsync(f.fileno())
                counters["done"] += 1
                elapsed = time.time() - start
                rate = counters["done"] / elapsed * 60 if elapsed > 0 else 0
                print(
                    f"  [w{wid}][{cond_letter}] {counters['done']}/{total} "
                    f"{'OK' if ok else 'ERR:'+str(result.get('error',''))} "
                    f"rate={rate:.1f}/min  {trial['trial_id'][:30]}",
                    flush=True
                )
            queue.task_done()

    await asyncio.gather(*[worker(i) for i in range(CONCURRENCY)])

    print("\n=== FINAL ===", flush=True)
    for cl in ["a", "b"]:
        p = results_paths[f"condition_{cl}"]
        done = set()
        if p.exists():
            for l in open(p):
                l = l.strip()
                if not l: continue
                try:
                    r = json.loads(l)
                    if not r.get("error") and r.get("phase_a") and r.get("model") == MODEL_SLUG:
                        done.add(r["trial_id"])
                except Exception:
                    pass
        print(f"  condition_{cl}: {len(done)}/320 {'OK' if len(done) >= 320 else 'INCOMPLETE'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
