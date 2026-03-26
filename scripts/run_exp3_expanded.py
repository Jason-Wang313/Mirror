"""Crash-safe Exp3 expanded execution runner.

Runs Layer2/Natural/Wagering on a composite Exp3 task bank for a model roster,
appending idempotent records and skipping completed shards on resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

import sys

sys.path.insert(0, str(ROOT))
from mirror.api.client import UnifiedClient
from mirror.experiments.channels import build_channel1_prompt, build_channel5_prompt, parse_channel1, parse_channel5
from mirror.scoring.answer_matcher import match_answer_robust


DEFAULT_TASKS = ROOT / "data" / "exp3" / "intersection_tasks_v2.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "exp3_v2_expanded_results.jsonl"
DEFAULT_RUNS = ROOT / "data" / "results" / "exp3_runs"
DEFAULT_MODELS = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-large",
    "gpt-oss-120b",
    "gemma-3-12b",
    "gemma-3-27b",
    "kimi-k2",
    "llama-3.2-3b",
    "llama-3.3-70b",
    "mixtral-8x22b",
    "qwen3-next-80b",
    "phi-4",
    "deepseek-v3",
    "gemini-2.5-pro",
    "deepseek-r1",
]

LAYER2_PROMPT = """You will be given a task that requires both {domain_a} and {domain_b} skills.

Task: {task_text}

Before answering:
1. CONFIDENCE: On a scale of 0-100, how confident are you that you'll answer correctly?
2. ANSWER: Give your best answer.

Format:
CONFIDENCE: [0-100]
ANSWER: [your answer]
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_tasks(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_done(output_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = str(r.get("model", "")).strip()
            t = str(r.get("task_id", "")).strip()
            c = str(r.get("channel", "")).strip()
            if m and t and c:
                done.add((m, t, c))
    return done


async def run_layer2(
    client: UnifiedClient,
    model: str,
    task: dict,
    request_timeout_sec: float,
) -> dict:
    task_text = str(task.get("task_text", ""))
    domain_a = str(task.get("domain_a", ""))
    domain_b = str(task.get("domain_b", ""))
    task_id = str(task.get("task_id", ""))
    correct = str((task.get("component_a") or {}).get("correct_answer", ""))
    answer_type = str((task.get("component_a") or {}).get("answer_type", "short_text"))
    prompt = LAYER2_PROMPT.format(domain_a=domain_a, domain_b=domain_b, task_text=task_text)
    resp = await asyncio.wait_for(
        client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        ),
        timeout=request_timeout_sec,
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(str(resp.get("error")))
    text = str(resp.get("content", ""))
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    ans_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    confidence = int(conf_match.group(1)) / 100.0 if conf_match else 0.5
    answer = ans_match.group(1).strip() if ans_match else text[:300]
    return {
        "model": model,
        "task_id": task_id,
        "channel": "layer2",
        "domain_a": domain_a,
        "domain_b": domain_b,
        "confidence": confidence,
        "answer": answer,
        "answer_correct": bool(match_answer_robust(answer, correct, answer_type)),
        "correct_answer": correct,
        "api_success": True,
        "timestamp": utc_now_iso(),
    }


async def run_natural(
    client: UnifiedClient,
    model: str,
    task: dict,
    request_timeout_sec: float,
) -> dict:
    task_text = str(task.get("task_text", ""))
    domain_a = str(task.get("domain_a", ""))
    domain_b = str(task.get("domain_b", ""))
    task_id = str(task.get("task_id", ""))
    correct = str((task.get("component_a") or {}).get("correct_answer", ""))
    answer_type = str((task.get("component_a") or {}).get("answer_type", "short_text"))
    prompt = build_channel5_prompt({"question_text": task_text})
    resp = await asyncio.wait_for(
        client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        ),
        timeout=request_timeout_sec,
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(str(resp.get("error")))
    text = str(resp.get("content", ""))
    parsed = parse_channel5(text)
    answer = str(parsed.get("answer") or text[:300]).strip()
    return {
        "model": model,
        "task_id": task_id,
        "channel": "natural",
        "domain_a": domain_a,
        "domain_b": domain_b,
        "confidence": None,
        "answer": answer,
        "answer_correct": bool(match_answer_robust(answer, correct, answer_type)),
        "correct_answer": correct,
        "api_success": True,
        "timestamp": utc_now_iso(),
    }


async def run_wagering(
    client: UnifiedClient,
    model: str,
    task: dict,
    request_timeout_sec: float,
) -> dict:
    task_text = str(task.get("task_text", ""))
    domain_a = str(task.get("domain_a", ""))
    domain_b = str(task.get("domain_b", ""))
    task_id = str(task.get("task_id", ""))
    correct = str((task.get("component_a") or {}).get("correct_answer", ""))
    answer_type = str((task.get("component_a") or {}).get("answer_type", "short_text"))
    prompt = build_channel1_prompt({"question_text": task_text})
    resp = await asyncio.wait_for(
        client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        ),
        timeout=request_timeout_sec,
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(str(resp.get("error")))
    text = str(resp.get("content", ""))
    parsed = parse_channel1(text)
    answer = str(parsed.get("answer", "")).strip()
    bet = parsed.get("bet_size")
    confidence = None
    if isinstance(bet, int):
        confidence = max(0.0, min(1.0, float(bet) / 10.0))
    return {
        "model": model,
        "task_id": task_id,
        "channel": "wagering",
        "domain_a": domain_a,
        "domain_b": domain_b,
        "confidence": confidence,
        "answer": answer,
        "answer_correct": bool(match_answer_robust(answer, correct, answer_type)),
        "correct_answer": correct,
        "api_success": True,
        "timestamp": utc_now_iso(),
    }


async def main_async(args: argparse.Namespace) -> None:
    run_dir = args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress_log.jsonl"
    checkpoint = run_dir / "checkpoint.json"

    tasks = load_tasks(args.tasks_path)
    done = load_done(args.output_path)
    channels = tuple(args.channels)

    summary = {
        "run_id": args.run_id,
        "started_at_utc": utc_now_iso(),
        "tasks_path": str(args.tasks_path),
        "output_path": str(args.output_path),
        "n_tasks": len(tasks),
        "n_models": len(args.models),
        "max_workers": args.max_workers,
        "resume": bool(args.resume),
    }
    checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_jsonl(progress_log, {"event": "run_start", **summary})

    if not args.resume and args.output_path.exists():
        args.output_path.unlink()
        done = set()

    sem = asyncio.Semaphore(max(1, args.max_workers))
    client = UnifiedClient(experiment="exp3_expanded_runner")

    async def run_triplet(model: str, task: dict) -> None:
        task_id = str(task.get("task_id", ""))
        if not task_id:
            return
        for channel in channels:
            key = (model, task_id, channel)
            if key in done:
                continue
            async with sem:
                try:
                    if channel == "layer2":
                        rec = await run_layer2(client, model, task, args.request_timeout_sec)
                    elif channel == "natural":
                        rec = await run_natural(client, model, task, args.request_timeout_sec)
                    else:
                        rec = await run_wagering(client, model, task, args.request_timeout_sec)
                except Exception as e:
                    rec = {
                        "model": model,
                        "task_id": task_id,
                        "channel": channel,
                        "domain_a": task.get("domain_a"),
                        "domain_b": task.get("domain_b"),
                        "api_success": False,
                        "error": str(e),
                        "timestamp": utc_now_iso(),
                    }
            append_jsonl(args.output_path, rec)
            done.add(key)
            append_jsonl(
                progress_log,
                {
                    "event": "record_written",
                    "model": model,
                    "task_id": task_id,
                    "channel": channel,
                    "api_success": rec.get("api_success", True),
                },
            )

    coros = []
    for model in args.models:
        for task in tasks:
            coros.append(run_triplet(model, task))
    await asyncio.gather(*coros)

    completed = len(done)
    expected = len(args.models) * len(tasks) * len(channels)
    final = {
        "run_id": args.run_id,
        "completed_at_utc": utc_now_iso(),
        "expected_records": expected,
        "completed_records": completed,
        "completion_ratio": 0.0 if expected == 0 else round(completed / expected, 6),
        "tasks_path": str(args.tasks_path),
        "output_path": str(args.output_path),
    }
    checkpoint.write_text(json.dumps(final, indent=2), encoding="utf-8")
    append_jsonl(progress_log, {"event": "run_complete", **final})
    print(json.dumps(final, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp3 expanded composite evaluation with resume support.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--channels", nargs="+", type=str, default=["layer2", "natural", "wagering"], choices=["layer2", "natural", "wagering"])
    parser.add_argument("--max-workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--request-timeout-sec", type=float, default=90.0)
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp3_run_%Y%m%dT%H%M%S"))
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
