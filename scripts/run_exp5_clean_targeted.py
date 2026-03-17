"""
Run exp5 clean control for specific question IDs extracted from an adversarial run.

Used to fix llama-3.2-3b and mixtral-8x22b whose clean baseline used wrong domains
because exp1 accuracy was updated between adversarial and clean runs.

Appends to the existing clean baseline file so analyze_experiment_5.py can use it.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.channels import (
    build_channel1_prompt,
    build_channel2_prompt,
    build_channel4_prompt,
    build_channel5_prompt,
    parse_channel1,
    parse_channel2,
    parse_channel4,
    parse_channel5,
)

ATTACK_TYPE = "clean"
CONCURRENCY = 4
DEFAULT_DELAY = 0.1

CHANNEL_BUILDERS = {
    "wagering": build_channel1_prompt,
    "opt_out": build_channel2_prompt,
    "tool_use": build_channel4_prompt,
    "natural": build_channel5_prompt,
}
CHANNEL_PARSERS = {
    "wagering": parse_channel1,
    "opt_out": parse_channel2,
    "tool_use": parse_channel4,
    "natural": parse_channel5,
}

ADV_FILE = Path("data/results/exp5_20260315T154655_results.jsonl")
CLEAN_FILE = Path("data/results/exp5_clean_20260316T150242_results.jsonl")
TARGET_MODELS = ["llama-3.2-3b", "mixtral-8x22b"]


def load_questions_by_id() -> dict:
    qs = {}
    with open("data/questions.jsonl") as f:
        for line in f:
            q = json.loads(line)
            qs[q["source_id"]] = q
    return qs


def extract_adv_qids(model: str) -> dict:
    """Return {qid: domain_type} from adversarial run for this model."""
    result = {}
    for line in open(ADV_FILE):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("model") == model:
                qid = r.get("question_id", "")
                dt = r.get("domain_type", "")
                if qid and dt:
                    result[qid] = dt
        except Exception:
            pass
    return result


def load_completed() -> set:
    done = set()
    if not CLEAN_FILE.exists():
        return done
    for line in open(CLEAN_FILE):
        if not line.strip():
            continue
        try:
            t = json.loads(line)
            tid = (t.get("model"), t.get("question_id"), t.get("attack_type"), t.get("domain_type"))
            if all(tid):
                done.add(tid)
        except Exception:
            pass
    return done


async def run_channel_clean(client, model, question, channel) -> dict:
    builder = CHANNEL_BUILDERS[channel]
    prompt = builder(question)
    try:
        response = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = (response or {}).get("content") or ""
        parser = CHANNEL_PARSERS[channel]
        try:
            parsed = parser(raw) if raw else {}
        except Exception:
            parsed = {}
        return {"channel": channel, "attack_type": ATTACK_TYPE,
                "raw_response": raw, "parsed": parsed, "api_call_success": bool(raw)}
    except Exception as e:
        return {"channel": channel, "attack_type": ATTACK_TYPE,
                "raw_response": None, "parsed": {}, "api_call_success": False, "error": str(e)}


async def run_trial(client, model, question, domain_type, completed, results_f, write_lock, sem):
    trial_id = (model, question["source_id"], ATTACK_TYPE, domain_type)
    if trial_id in completed:
        return
    async with sem:
        channel_results = {}
        for channel in CHANNEL_BUILDERS:
            channel_results[channel] = await run_channel_clean(client, model, question, channel)
            await asyncio.sleep(DEFAULT_DELAY)

        record = {
            "model": model,
            "question_id": question["source_id"],
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            "attack_type": ATTACK_TYPE,
            "domain_type": domain_type,
            "channels": channel_results,
            "timestamp": datetime.now().isoformat(),
        }
        ok = any(v.get("api_call_success") for v in channel_results.values())
        async with write_lock:
            results_f.write(json.dumps(record) + "\n")
            results_f.flush()
        print(f"  {'✓' if ok else '✗'} {model} {question['source_id'][:50]} ({domain_type})", flush=True)


async def main():
    all_qs = load_questions_by_id()
    completed = load_completed()
    print(f"Existing completed trials in clean file: {len(completed)}")

    client = UnifiedClient(experiment="exp5_clean_targeted")
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    with open(CLEAN_FILE, "a", encoding="utf-8") as results_f:
        for model in TARGET_MODELS:
            qid_map = extract_adv_qids(model)
            pending = [(qid, dt) for qid, dt in qid_map.items()
                       if (model, qid, ATTACK_TYPE, dt) not in completed]
            print(f"\n{model}: {len(pending)}/{len(qid_map)} to run")
            if not pending:
                continue

            tasks = []
            for qid, domain_type in pending:
                q = all_qs.get(qid)
                if not q:
                    print(f"  WARNING: QID not found: {qid}")
                    continue
                tasks.append(run_trial(client, model, q, domain_type, completed, results_f, write_lock, sem))
            await asyncio.gather(*tasks)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
