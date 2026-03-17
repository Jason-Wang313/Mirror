"""
Parallel accelerator for exp4 llama-3.1-70b remaining trials.
Runs at concurrency=3 alongside the existing runner (total = 4 trials in-flight).
Duplicates are harmless — analysis deduplicates by (model, trial_id, condition).

Usage:
  python scripts/accelerate_exp4_llama70b.py --condition a
  python scripts/accelerate_exp4_llama70b.py --condition b
"""
import argparse, asyncio, json, os, sys, time
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv()

from scripts.run_exp4_expanded import (
    run_trial, load_resume_keys, RESULTS_DIR, TEMPLATES_PATH
)
from mirror.api.client import UnifiedClient

CONCURRENCY = 3
MODEL = "llama-3.1-70b"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["a", "b"])
    parser.add_argument("--run-id", default="20260314T135731")
    args = parser.parse_args()

    run_id = args.run_id
    condition = f"condition_{args.condition}"   # -> "condition_a" / "condition_b"
    results_path = RESULTS_DIR / f"exp4_v2_{run_id}_{condition}_results.jsonl"

    templates = [json.loads(l) for l in open(TEMPLATES_PATH) if l.strip()]

    done = load_resume_keys(run_id, condition)
    pending = [t for t in templates if (MODEL, t["trial_id"], condition) not in done]
    print(f"[ACCEL {condition}] pending={len(pending)}  concurrency={CONCURRENCY}")
    if not pending:
        print("[ACCEL] Nothing to do — already complete.")
        return

    client = UnifiedClient(experiment=f"exp4_v2_{run_id}_accel")
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    done_count = [0]
    start = time.time()
    total = len(pending)

    async def process(trial):
        result = await run_trial(client, MODEL, trial, condition, sem)
        async with write_lock:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done_count[0] += 1
            elapsed = time.time() - start
            rate = done_count[0] / elapsed * 60 if elapsed > 0 else 0
            eta = (total - done_count[0]) / (rate / 60) / 60 if rate > 0 else 99
            print(f"  [{condition}] {done_count[0]}/{total}  "
                  f"rate={rate:.1f}/min  ETA={eta:.0f}min", flush=True)

    await asyncio.gather(*[process(t) for t in pending])

    mc = Counter(json.loads(l).get("model") for l in open(results_path) if l.strip())
    print(f"\n[ACCEL] Done. {MODEL}/{condition}: {mc.get(MODEL,0)} records in file")


if __name__ == "__main__":
    asyncio.run(main())
