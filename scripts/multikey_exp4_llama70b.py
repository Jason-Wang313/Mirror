"""
Multi-key parallel runner for exp4 llama-3.1-70b.

Uses all NVIDIA_NIM_API_KEY, NVIDIA_NIM_API_KEY_2 ... NVIDIA_NIM_API_KEY_N keys
simultaneously. Each key gets its own worker pool with independent RPM limiting.
No cross-key quota contention = N× throughput.

Usage:
  python scripts/multikey_exp4_llama70b.py --run-id 20260314T135731 --rpm 5
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL_SLUG = "llama-3.1-70b"
NIM_MODEL  = "meta/llama-3.1-70b-instruct"
BASE_URL   = "https://integrate.api.nvidia.com/v1"
RUN_ID_DEFAULT = "20260314T135731"


def load_keys():
    keys = []
    k = os.getenv("NVIDIA_NIM_API_KEY")
    if k:
        keys.append(k)
    for i in range(2, 30):
        k = os.getenv(f"NVIDIA_NIM_API_KEY_{i}")
        if k:
            keys.append(k)
    # deduplicate while preserving order
    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


class RpmLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            wait = self.interval - (now - self.last)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.time()


async def call_nim(client, limiter, messages, max_tokens, key_idx):
    for attempt in range(8):
        await limiter.acquire()
        try:
            resp = await client.chat.completions.create(
                model=NIM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                wait = min(30 * (2 ** min(attempt, 4)), 300)
                print(f"  [key{key_idx+1}] 429 → wait {wait}s", flush=True)
                await asyncio.sleep(wait)
            else:
                wait = min(5 * (2 ** attempt), 60)
                await asyncio.sleep(wait)
    return ""


async def run_trial(client, limiter, trial, condition, key_idx):
    messages = []
    tid = trial["trial_id"]

    async def call(prompt, max_tokens):
        messages.append({"role": "user", "content": prompt})
        content = await call_nim(client, limiter, list(messages), max_tokens, key_idx)
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


async def worker(worker_id, key_idx, client, limiter, queue, results_paths, write_lock, counters, start_time):
    while True:
        try:
            trial, condition = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        result = await run_trial(client, limiter, trial, condition, key_idx)
        ok = not result.get("error") and result.get("phase_a") is not None
        cond_letter = condition.split("_")[-1]  # "a" or "b"
        async with write_lock:
            with open(results_paths[condition], "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            counters["done"] += 1
            elapsed = time.time() - start_time
            rate = counters["done"] / elapsed * 60 if elapsed > 0 else 0
            print(
                f"  [w{worker_id}/key{key_idx+1}][{cond_letter}] "
                f"{counters['done']}/{counters['total']} "
                f"{'OK' if ok else 'ERR:'+str(result.get('error',''))} "
                f"rate={rate:.1f}/min  trial={trial['trial_id'][:28]}",
                flush=True
            )
        queue.task_done()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--rpm", type=float, default=5.0,
                        help="Requests per minute per key (default 5)")
    args = parser.parse_args()

    keys = load_keys()
    if not keys:
        print("ERROR: No NIM keys found in environment.", flush=True)
        sys.exit(1)
    print(f"Loaded {len(keys)} NIM keys → {len(keys)} parallel key pools", flush=True)

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    results_paths = {
        "condition_a": RESULTS_DIR / f"exp4_v2_{args.run_id}_condition_a_results.jsonl",
        "condition_b": RESULTS_DIR / f"exp4_v2_{args.run_id}_condition_b_results.jsonl",
    }

    # Build pending queue across both conditions
    queue = asyncio.Queue()
    total = 0
    for condition in ["condition_a", "condition_b"]:
        done_keys = load_resume_keys(args.run_id, condition)
        pending = [t for t in templates
                   if (MODEL_SLUG, t["trial_id"], condition) not in done_keys]
        print(f"  {condition}: {len(pending)} pending", flush=True)
        for t in pending:
            await queue.put((t, condition))
        total += len(pending)

    if total == 0:
        print("Nothing to do — all trials already complete!", flush=True)
        return

    print(f"Total pending: {total} trials across both conditions", flush=True)
    print(f"Keys: {len(keys)}  RPM/key: {args.rpm}  "
          f"Total RPM: {len(keys)*args.rpm:.0f}  "
          f"ETA: ~{total*4/(len(keys)*args.rpm):.0f} min", flush=True)

    # Create clients and limiters
    clients  = [AsyncOpenAI(api_key=k, base_url=BASE_URL) for k in keys]
    limiters = [RpmLimiter(args.rpm) for _ in keys]

    write_lock = asyncio.Lock()
    counters = {"done": 0, "total": total}
    start_time = time.time()

    # One worker per key (or more if you want — simple 1:1 for clean quota isolation)
    workers = [
        worker(i, i % len(keys), clients[i % len(keys)], limiters[i % len(keys)],
               queue, results_paths, write_lock, counters, start_time)
        for i in range(len(keys))
    ]
    await asyncio.gather(*workers)

    # Final counts
    print("\n=== FINAL COUNTS ===", flush=True)
    for cond_letter in ["a", "b"]:
        p = results_paths[f"condition_{cond_letter}"]
        done = {}
        for l in open(p):
            l = l.strip()
            if not l: continue
            try:
                r = json.loads(l)
                if not r.get("error") and r.get("phase_a") and r.get("model") == MODEL_SLUG:
                    done.setdefault(MODEL_SLUG, set()).add(r["trial_id"])
            except: pass
        n = len(done.get(MODEL_SLUG, set()))
        print(f"  condition_{cond_letter}: {n}/320 {'✅' if n >= 320 else '⚠'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
