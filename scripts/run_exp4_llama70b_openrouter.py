"""
OpenRouter runner for exp4 llama-3.1-70b.

Routes to meta-llama/llama-3.1-70b-instruct via OpenRouter's multi-provider
pool. Observed: ~21 tok/s, ~1-14s per call (vs NIM's 50-90s queue wait).

MODEL_SLUG = "llama-3.1-70b" — identical to NIM runner for result merging.
load_resume_keys() deduplication ensures no trial is double-counted.

OpenRouter free tier: 20 RPM. With 3 workers × 4 calls = 12 RPM peak — safe.
Cost: ~$0.001/trial × 206 trials = ~$0.21 (well within signup free credit).
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

MODEL_SLUG    = "llama-3.1-70b"
OR_MODEL      = "meta-llama/llama-3.1-70b-instruct"
OR_BASE       = "https://openrouter.ai/api/v1"
OR_KEY        = os.environ.get("OPENROUTER_API_KEY", "")
RUN_ID        = "20260314T135731"
CONCURRENCY   = 3
EXTRA_HEADERS = {"HTTP-Referer": "https://mirror-benchmark.ai", "X-Title": "MIRROR"}


class OpenRouterCaller:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OR_KEY, base_url=OR_BASE,
                                   default_headers=EXTRA_HEADERS, timeout=60)

    async def call(self, messages: list, max_tokens: int) -> str:
        for attempt in range(6):
            try:
                resp = await self.client.chat.completions.create(
                    model=OR_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    backoff = min(15 * (2 ** attempt), 120)
                    print(f"  [or] 429 → sleep {backoff}s (attempt {attempt+1})", flush=True)
                    await asyncio.sleep(backoff)
                elif "402" in err or "credit" in err.lower() or "balance" in err.lower():
                    print(f"  [or] ❌ Out of credits — stopping", flush=True)
                    return "__OUT_OF_CREDITS__"
                elif "timeout" in err.lower():
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    print(f"  [or] error: {err[:80]}", flush=True)
                    await asyncio.sleep(min(3 * (2 ** attempt), 30))
        return ""


async def run_trial(caller: OpenRouterCaller, trial: dict, condition: str) -> dict:
    messages = []
    tid = trial["trial_id"]

    async def call(prompt: str, max_tokens: int) -> str:
        messages.append({"role": "user", "content": prompt})
        content = await caller.call(list(messages), max_tokens)
        if content and content != "__OUT_OF_CREDITS__":
            messages.append({"role": "assistant", "content": content})
        elif messages and messages[-1]["role"] == "user":
            messages.pop()
        return content

    pa = await call(phase_a_prompt(trial), 2000)
    if not pa or pa == "__OUT_OF_CREDITS__":
        return {"error": "phase_a_failed", "trial_id": tid, "model": MODEL_SLUG,
                "condition": condition, "_oor": pa == "__OUT_OF_CREDITS__"}

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
        "_provider":          "openrouter",
        "_or_model":          OR_MODEL,
    }


async def main():
    caller = OpenRouterCaller()
    print(f"OpenRouter runner: {MODEL_SLUG} via {OR_MODEL}", flush=True)
    print(f"Run ID: {RUN_ID}  Concurrency: {CONCURRENCY}", flush=True)

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

    print(f"Total: {total} trials  Workers: {CONCURRENCY}", flush=True)

    write_lock = asyncio.Lock()
    counters = {"done": 0, "ok": 0, "abort": False}
    start = time.time()

    async def worker(wid: int):
        while not counters["abort"]:
            try:
                trial, condition = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result = await run_trial(caller, trial, condition)
            if result.get("_oor"):
                print(f"  [w{wid}] Out of credits — aborting all workers", flush=True)
                counters["abort"] = True
                queue.task_done()
                break
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
                eta = (total - counters["done"]) / rate if rate > 0 else 0
                print(
                    f"  [w{wid}][{cond_ltr}] {counters['done']}/{total} "
                    f"{'OK' if ok else 'ERR:'+str(result.get('error',''))}"
                    f"  rate={rate:.1f}/min  ETA={eta:.0f}min"
                    f"  {trial['trial_id'][:35]}",
                    flush=True
                )
            queue.task_done()

    # Stagger workers 5s apart to avoid simultaneous burst
    tasks = []
    for i in range(CONCURRENCY):
        if i > 0:
            await asyncio.sleep(5)
        tasks.append(asyncio.create_task(worker(i)))
    await asyncio.gather(*tasks)
    print(f"\n=== DONE: {counters['ok']}/{counters['done']} successful ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
