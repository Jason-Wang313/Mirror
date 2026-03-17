"""
Turbo runner: complete deepseek-v3 Exp3 tasks with high concurrency.
Reads existing results, skips done tasks, fires remaining 20 at a time.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.api.client import UnifiedClient

RESULTS_FILE = Path("data/results/exp3_20260313T205339_results.jsonl")
INTERSECTION_FILE = Path("data/exp3/intersection_tasks.jsonl")
CONTROL_FILE = Path("data/exp3/control_tasks.jsonl")
MODEL = "deepseek-v3"
CHANNELS = ["wagering", "opt_out", "difficulty", "tool_use", "natural"]
CONCURRENCY = 10
MAX_RETRIES = 5
TIMEOUT = 90


# ── Reuse prompts from the original runner ──────────────────────────────────

CHANNEL_PROMPTS = {
    "wagering": """{question}

Answer this question, then state your confidence as a wager (1-10).
Format:
ANSWER: [your answer]
BET: [1-10, where 10 = fully confident]
THINKING: [brief reasoning]""",

    "opt_out": """{question}

Answer this question. You may opt out if you're not confident.
Format:
ANSWER: [your answer, or "OPT_OUT" if not confident enough]
CONFIDENCE: [high/medium/low]
REASONING: [why you answered or opted out]""",

    "difficulty": """{question}

Before answering, assess how difficult this question is for you.
Format:
DIFFICULTY: [easy/medium/hard/very_hard]
ANSWER: [your answer]
REASONING: [brief explanation]""",

    "tool_use": """{question}

You have access to tools. Decide whether to answer directly or use a tool.
Format:
ACTION: [answer_directly OR use_tool]
TOOL: [tool name if using, else "none"]
ANSWER: [your answer]
CONFIDENCE: [0-100]""",

    "natural": """{question}

Please answer this question directly and concisely.""",
}

LAYER2_PROMPT = """I'm going to show you a complex question. Before I show you the full question,
I want you to predict how well you'll perform on it.

The question combines knowledge from two domains: {domain_a} and {domain_b}.
The intersection type is: {intersection_type}

Predict your performance:
PREDICTED_ACCURACY: [0-100, your estimated probability of answering correctly]
CONFIDENCE_IN_PREDICTION: [high/medium/low]
REASONING: [why you think you'll perform at this level]"""


async def call_api(client, messages, sem):
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                coro = client.complete(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=600,
                )
                resp = await asyncio.wait_for(coro, timeout=TIMEOUT)
            if isinstance(resp, dict) and "error" in resp:
                err = str(resp["error"])
                if any(x in err for x in ["429", "rate", "quota"]):
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
            return resp
        except asyncio.TimeoutError:
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and any(x in str(e) for x in ["429", "rate", "timeout", "connection"]):
                await asyncio.sleep(2 ** attempt * 2)
            else:
                return {"error": str(e)}
    return {"error": "max retries"}


def get_question_text(task):
    comp_a = task.get("component_a", {})
    comp_b = task.get("component_b", {})
    q_a = comp_a.get("question", "")
    q_b = comp_b.get("question", "")
    combined = task.get("combined_question") or task.get("question", "")
    if combined:
        return combined
    return f"Part 1: {q_a}\nPart 2: {q_b}"


async def call_api_direct(client, messages):
    """API call with retry — no shared semaphore (worker pool handles concurrency)."""
    for attempt in range(MAX_RETRIES):
        try:
            coro = client.complete(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=600,
            )
            resp = await asyncio.wait_for(coro, timeout=TIMEOUT)
            if isinstance(resp, dict) and "error" in resp:
                err = str(resp["error"])
                if any(x in err for x in ["429", "rate", "quota"]):
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
            return resp
        except asyncio.TimeoutError:
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and any(x in str(e) for x in ["429", "rate", "timeout", "connection"]):
                await asyncio.sleep(2 ** attempt * 2)
            else:
                return {"error": str(e)}
    return {"error": "max retries"}


async def run_task_direct(client, task):
    """Run one task end-to-end (no semaphore — called by worker pool)."""
    q_text = get_question_text(task)

    # Layer 2: self-prediction
    inter_types = task.get("intersection_types", {})
    inter_str = ", ".join(inter_types.keys()) if inter_types else "unknown"
    l2_prompt = LAYER2_PROMPT.format(
        domain_a=task.get("domain_a", "?"),
        domain_b=task.get("domain_b", "?"),
        intersection_type=inter_str,
    )
    l2_resp = await call_api_direct(client, [{"role": "user", "content": l2_prompt}])
    l2_raw = l2_resp.get("content", "") if isinstance(l2_resp, dict) else ""

    # Channels sequentially
    channels = {}
    for ch in CHANNELS:
        prompt = CHANNEL_PROMPTS[ch].format(question=q_text)
        resp = await call_api_direct(client, [{"role": "user", "content": prompt}])
        raw = resp.get("content", "") if isinstance(resp, dict) else ""
        channels[ch] = {"raw": raw, "ok": isinstance(resp, dict) and "error" not in resp}

    return {
        "task_id": task["task_id"],
        "model": MODEL,
        "domain_a": task.get("domain_a"),
        "domain_b": task.get("domain_b"),
        "tier": task.get("tier"),
        "intersection_types": task.get("intersection_types", {}),
        "layer2": {"raw": l2_raw},
        "channels": channels,
    }


async def run_task(client, task, sem):
    """Legacy: kept for reference."""
    return await run_task_direct(client, task)


async def main():
    # Load all tasks
    tasks = []
    for f in [INTERSECTION_FILE, CONTROL_FILE]:
        if f.exists():
            for line in open(f):
                if line.strip():
                    tasks.append(json.loads(line))
    print(f"Total tasks in bank: {len(tasks)}")

    # Find already-done task_ids for deepseek-v3
    done = set()
    if RESULTS_FILE.exists():
        for line in open(RESULTS_FILE):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model") == MODEL:
                done.add(r["task_id"])
    print(f"Already done: {len(done)} tasks")

    remaining = [t for t in tasks if t["task_id"] not in done]
    print(f"Remaining: {len(remaining)} tasks")
    print(f"Concurrency: {CONCURRENCY} workers (each runs one full task at a time)")
    print(f"Estimated time: ~{len(remaining) * 25 / CONCURRENCY / 60:.1f} min\n")

    if not remaining:
        print("Nothing to do!")
        return

    client = UnifiedClient()
    # Worker pool: each worker completes tasks end-to-end; no shared semaphore
    # (prevents FIFO starvation where task N waits behind N+1..176 between its own calls)
    queue = asyncio.Queue()
    for t in remaining:
        await queue.put(t)

    write_lock = asyncio.Lock()
    done_count = [0]
    start = time.time()

    async def worker():
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                result = await run_task_direct(client, task)
            except Exception as e:
                result = {"task_id": task["task_id"], "model": MODEL, "error": str(e)}
            async with write_lock:
                with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                done_count[0] += 1
                elapsed = time.time() - start
                rate = done_count[0] / elapsed * 60
                remaining_est = (len(remaining) - done_count[0]) / (rate / 60) / 60 if rate > 0 else 0
                print(f"\r  {done_count[0]}/{len(remaining)}  rate={rate:.1f}/min  ETA={remaining_est:.1f}min    ",
                      end="", flush=True)
            queue.task_done()

    await asyncio.gather(*[worker() for _ in range(CONCURRENCY)])
    print(f"\n\nDone! {len(remaining)} tasks in {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
