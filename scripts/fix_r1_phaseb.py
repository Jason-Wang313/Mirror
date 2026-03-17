"""
Targeted fix for deepseek-r1 phase_b failures in Exp4.

Root cause: deepseek-r1 thinking tokens count toward max_tokens.
With max_tokens=4000, complex reasoning exhausts tokens before writing answer.

Fix: retry with max_tokens=8000 for phase_b, run full trial from scratch.
Concurrency=2 to stay within DeepSeek rate limits.

Usage:
  python scripts/fix_r1_phaseb.py --run-id 20260314T135731
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from mirror.api.client import UnifiedClient
from scripts.run_exp4_expanded import (
    phase_a_prompt, phase_b_prompt, feedback_prompt, phase_c_prompt,
    parse_five_channels, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)

MODEL = "deepseek-r1"
CONCURRENCY = 2
# Higher max_tokens for phase_b to give thinking room
PHASE_B_MAX_TOKENS = 8000


async def run_trial_hightoken(client, model, trial, condition, sem):
    """Run 4-phase trial with higher max_tokens for phase_b."""
    messages = []
    tid = trial["trial_id"]

    async def call(prompt, max_tokens):
        messages.append({"role": "user", "content": prompt})
        for attempt in range(5):
            try:
                async with sem:
                    resp = await client.complete(
                        model=model,
                        messages=list(messages),
                        temperature=0.0,
                        max_tokens=max_tokens,
                        metadata={"exp": "exp4_r1_phaseb_fix"},
                    )
                if resp and resp.get("content"):
                    messages.append({"role": "assistant", "content": resp["content"]})
                    return resp["content"]
                err = str(resp.get("error", "empty")) if resp else "no_response"
                if any(x in err for x in ["429", "rate", "quota", "Too Many"]):
                    await asyncio.sleep(min(2 ** attempt * 10, 120))
                else:
                    await asyncio.sleep(min(2 ** attempt, 15))
            except Exception as e:
                await asyncio.sleep(min(2 ** attempt * 5, 60))
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        return ""

    pa = await call(phase_a_prompt(trial), 2000)
    if not pa:
        return {"error": "phase_a_failed", "trial_id": tid, "model": model, "condition": condition}

    pb = await call(phase_b_prompt(trial), PHASE_B_MAX_TOKENS)
    if not pb:
        return {"error": "phase_b_failed", "trial_id": tid, "model": model, "condition": condition}

    fb = await call(feedback_prompt(trial, condition), 1500)

    pc = await call(phase_c_prompt(trial), 3000)
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


async def run_condition(client, condition, templates, run_id, write_lock):
    results_path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"
    sem = asyncio.Semaphore(CONCURRENCY)

    done_keys = load_resume_keys(run_id, condition)
    pending = [t for t in templates if (MODEL, t["trial_id"], condition) not in done_keys]
    print(f"[{condition}] {len(pending)} pending with max_tokens={PHASE_B_MAX_TOKENS} for phase_b", flush=True)

    total = len(pending)
    done_count = [0]
    start = time.time()

    async def process(trial):
        result = await run_trial_hightoken(client, MODEL, trial, condition, sem)
        ok = not result.get("error") and result.get("phase_a") is not None
        async with write_lock:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done_count[0] += 1
            elapsed = time.time() - start
            rate = done_count[0] / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{condition}] {done_count[0]}/{total} "
                  f"{'OK' if ok else 'ERR:'+str(result.get('error',''))} "
                  f"rate={rate:.1f}/min  trial={trial['trial_id'][:24]}", flush=True)

    q = asyncio.Queue()
    for t in pending:
        await q.put(t)

    async def worker():
        while not q.empty():
            try:
                trial = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            await process(trial)

    await asyncio.gather(*[worker() for _ in range(CONCURRENCY)])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260314T135731")
    args = parser.parse_args()

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    client = UnifiedClient(experiment=f"exp4_r1_phaseb_fix_{args.run_id}")
    write_lock = asyncio.Lock()

    for condition in ["condition_a", "condition_b"]:
        await run_condition(client, condition, templates, args.run_id, write_lock)

    # Final count
    for cond in ["a", "b"]:
        p = RESULTS_DIR / f"exp4_v2_{args.run_id}_condition_{cond}_results.jsonl"
        valid = sum(1 for l in open(p) if l.strip()
                    and not json.loads(l).get("error")
                    and json.loads(l).get("phase_a") is not None
                    and json.loads(l).get("model") == MODEL)
        print(f"Final: {MODEL} condition_{cond} valid={valid}/320")


if __name__ == "__main__":
    asyncio.run(main())
