"""
Targeted low-concurrency retry for deepseek-r1 exp4 remaining trials.
Concurrency=2 to avoid DeepSeek rate limits.
"""
import asyncio, json, os, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from scripts.run_exp4_expanded import (
    run_trial, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)
from mirror.api.client import UnifiedClient

MODEL = "deepseek-r1"
CONCURRENCY = 2
RUN_ID = "20260314T135731"


async def run_condition(client, condition, templates, write_lock, done_global):
    results_path = RESULTS_DIR / f"exp4_v2_{RUN_ID}_{condition}_results.jsonl"
    sem = asyncio.Semaphore(CONCURRENCY)

    done_keys = load_resume_keys(RUN_ID, condition)
    pending = [t for t in templates if (MODEL, t["trial_id"], condition) not in done_keys]
    print(f"[{condition}] {len(pending)} pending  concurrency={CONCURRENCY}", flush=True)

    total = len(pending)
    done_cond = [0]
    start = time.time()

    async def process(trial):
        result = await run_trial(client, MODEL, trial, condition, sem)
        ok = not result.get("error") and result.get("phase_a") is not None
        async with write_lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done_cond[0] += 1
            elapsed = time.time() - start
            rate = done_cond[0] / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{condition}] {done_cond[0]}/{total} "
                  f"{'OK' if ok else 'ERR:'+str(result.get('error',''))} "
                  f"rate={rate:.1f}/min  trial={trial['trial_id'][:20]}", flush=True)

    # Worker pool
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
    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]
    client = UnifiedClient(experiment=f"exp4_r1_retry_{RUN_ID}")
    write_lock = asyncio.Lock()
    done_global = {}

    for condition in ["condition_a", "condition_b"]:
        await run_condition(client, condition, templates, write_lock, done_global)

    # Final count
    for cond in ["a", "b"]:
        p = RESULTS_DIR / f"exp4_v2_{RUN_ID}_condition_{cond}_results.jsonl"
        valid = sum(1 for l in open(p) if l.strip()
                    and not json.loads(l).get("error")
                    and json.loads(l).get("phase_a") is not None
                    and json.loads(l).get("model") == MODEL)
        print(f"Final: {MODEL} condition_{cond} valid={valid}/320")


if __name__ == "__main__":
    asyncio.run(main())
