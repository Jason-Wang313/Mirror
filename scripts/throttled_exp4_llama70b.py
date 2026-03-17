"""
Throttled retry runner for exp4 llama-3.1-70b.
Root cause: NIM returns 429 for llama-3.1-70b → 45% failure rate.
Fix: strict 1 call / 30s rate limit + 30 retries + 120s 429 backoff.

Writes to same results file. Resume-safe (skips valid records).
Processes BOTH conditions in parallel (each its own sequential call queue).

Usage:
  python scripts/throttled_exp4_llama70b.py --run-id 20260314T135731
  python scripts/throttled_exp4_llama70b.py --run-id 20260314T135731 --rpm 1.5
"""
import argparse, asyncio, json, os, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from mirror.api.client import UnifiedClient
from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL = "llama-3.1-70b"
MAX_RETRIES = 30


def load_templates():
    return [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]


async def throttled_call(client, model, messages, max_tokens, rpm_limiter):
    """Single API call with proper 429 backoff and NIM rate throttle."""
    await rpm_limiter.acquire()  # enforce RPM
    for attempt in range(MAX_RETRIES):
        try:
            resp = await asyncio.wait_for(
                client.complete(model=model, messages=list(messages),
                                temperature=0.0, max_tokens=max_tokens,
                                metadata={"exp": "exp4_throttled"}),
                timeout=120,
            )
            if resp and resp.get("content"):
                return resp["content"]
            err = str(resp.get("error", "")) if resp else ""
            if any(x in err for x in ["429", "rate", "quota", "Too Many"]):
                wait = min(120 * (2 ** min(attempt, 4)), 600)  # 120s→1200s cap at 600s
                print(f"    429 on attempt {attempt+1}. Waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
                await rpm_limiter.acquire()
            else:
                wait = min(10 * (attempt + 1), 60)
                await asyncio.sleep(wait)
        except asyncio.TimeoutError:
            print(f"    Timeout attempt {attempt+1}. Waiting 30s...", flush=True)
            await asyncio.sleep(30)
        except Exception as e:
            wait = min(15 * (attempt + 1), 90)
            print(f"    Error attempt {attempt+1}: {str(e)[:80]}. Waiting {wait}s...", flush=True)
            await asyncio.sleep(wait)
    return ""


async def run_trial_throttled(client, model, trial, condition, rpm_limiter):
    """Run 4-phase trial with throttled API calls."""
    messages = []
    tid = trial["trial_id"]

    async def call(prompt, phase, max_tokens):
        messages.append({"role": "user", "content": prompt})
        text = await throttled_call(client, model, messages, max_tokens, rpm_limiter)
        if text:
            messages.append({"role": "assistant", "content": text})
        return text

    pa = await call(phase_a_prompt(trial), "phase_a", 2000)
    if not pa:
        return {"error": "phase_a_failed", "trial_id": tid, "model": model, "condition": condition}

    pb = await call(phase_b_prompt(trial), "phase_b", 4000)
    if not pb:
        return {"error": "phase_b_failed", "trial_id": tid, "model": model, "condition": condition}

    fb = await call(feedback_prompt(trial, condition), "feedback", 1500)

    pc = await call(phase_c_prompt(trial), "phase_c", 2500)
    if not pc:
        return {"error": "phase_c_failed", "trial_id": tid, "model": model, "condition": condition}

    return {
        "trial_id": tid, "model": model,
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


class RpmLimiter:
    """Token-bucket rate limiter: allows at most rpm calls per minute."""
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            wait = self._last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.time()


async def run_condition(client, condition_short, run_id, templates, rpm, write_lock, report):
    condition = f"condition_{condition_short}"
    results_path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"
    limiter = RpmLimiter(rpm)

    done = load_resume_keys(run_id, condition)
    pending = [t for t in templates if (MODEL, t["trial_id"], condition) not in done]
    print(f"[{condition}] {len(pending)} trials to run at {rpm} RPM "
          f"→ ETA ~{len(pending)*4/rpm:.0f} min", flush=True)

    for i, trial in enumerate(pending):
        result = await run_trial_throttled(client, MODEL, trial, condition, limiter)
        success = not result.get("error")
        async with write_lock:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            report[condition_short]["done"] += 1
            report[condition_short]["success"] += int(success)
            d = report[condition_short]["done"]
            s = report[condition_short]["success"]
            ts = datetime.utcnow().strftime("%H:%M:%S")
            print(f"  [{ts}][{condition_short.upper()}] {d}/{len(pending)} "
                  f"success={s} {'✓' if success else 'ERR'} "
                  f"trial={trial['trial_id']}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260314T135731")
    parser.add_argument("--rpm", type=float, default=2.0,
                        help="Max NIM calls per minute (default 2, ~1 per 30s)")
    parser.add_argument("--condition", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    templates = load_templates()
    client = UnifiedClient(experiment=f"exp4_throttled_{args.run_id}")
    write_lock = asyncio.Lock()
    report = {"a": {"done": 0, "success": 0}, "b": {"done": 0, "success": 0}}

    conditions = (["a", "b"] if args.condition == "both"
                  else [args.condition])

    # Each condition uses its OWN rpm limiter so they run in parallel
    # without sharing the throttle (each gets args.rpm calls/min)
    tasks = [
        asyncio.create_task(run_condition(client, c, args.run_id, templates,
                                          args.rpm, write_lock, report))
        for c in conditions
    ]
    await asyncio.gather(*tasks)

    print(f"\n=== DONE ===")
    for c in conditions:
        rp = report[c]
        print(f"  Cond {c.upper()}: {rp['success']}/{rp['done']} succeeded")

    # Final file counts
    for c in conditions:
        p = RESULTS_DIR / f"exp4_v2_{args.run_id}_condition_{c}_results.jsonl"
        mc = Counter(json.loads(l).get("model") for l in open(p) if l.strip())
        total_ok = sum(1 for l in open(p) if l.strip() and
                       not json.loads(l).get("error") and
                       json.loads(l).get("phase_a") is not None and
                       json.loads(l).get("model") == MODEL)
        print(f"  Cond {c.upper()} llama-3.1-70b valid={total_ok}/320")


if __name__ == "__main__":
    asyncio.run(main())
