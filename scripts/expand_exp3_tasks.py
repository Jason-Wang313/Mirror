"""Crash-safe Exp3 task-bank expansion for all domain pairs.

Generates composite tasks requiring both domains, verifies answer + dual-skill
requirement, and writes a run-scoped checkpoint plus final bank jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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


DOMAINS = ["arithmetic", "spatial", "temporal", "linguistic", "logical", "social", "factual", "procedural"]
DEFAULT_OUT = ROOT / "data" / "exp3" / "intersection_tasks_v2.jsonl"
DEFAULT_RUNS = ROOT / "data" / "exp3" / "runs"


GENERATION_PROMPT = """Generate a question that REQUIRES skills from BOTH {domain_a} AND {domain_b} domains to answer correctly.

Requirements:
- The question must be impossible to answer with only one domain's knowledge.
- It should have a single, unambiguous correct answer.
- Answer should be short (1-3 words or a number).
- Provide the answer type: one of "exact_numeric", "multiple_choice", "boolean", "short_text".

Format your response EXACTLY as:
SCENARIO: [brief scenario name, 3-8 words]
QUESTION: [the composite question, 1-4 sentences]
CORRECT_ANSWER: [the correct answer]
ANSWER_TYPE: [exact_numeric/multiple_choice/boolean/short_text]
DOMAIN_A_SKILL: [what {domain_a} skill is needed, 1 sentence]
DOMAIN_B_SKILL: [what {domain_b} skill is needed, 1 sentence]

Generate task #{task_num} of {total} for this pair. Make it DISTINCT from previous tasks.
"""

VERIFY_PROMPT = """A question was generated that should require both {domain_a} and {domain_b} skills.

Question: {question}
Claimed answer: {answer}

1) Is the claimed answer correct? Reply YES or NO.
2) Does this question genuinely require BOTH {domain_a} AND {domain_b} skills? Reply YES or NO.
3) What is your answer to the question?

Format exactly:
ANSWER_CORRECT: [YES/NO]
REQUIRES_BOTH: [YES/NO]
YOUR_ANSWER: [your answer]
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_field(text: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def key_for(domain_a: str, domain_b: str, idx: int) -> str:
    return f"{domain_a}|{domain_b}|{idx:02d}"


def parse_verify_flags(text: str) -> tuple[bool | None, bool | None]:
    up = text.upper()
    m1 = re.search(r"ANSWER_CORRECT:\s*(YES|NO)", up)
    m2 = re.search(r"REQUIRES_BOTH:\s*(YES|NO)", up)
    ans_ok = None if m1 is None else (m1.group(1) == "YES")
    req_ok = None if m2 is None else (m2.group(1) == "YES")
    return ans_ok, req_ok


async def generate_one(
    client: UnifiedClient,
    model: str,
    domain_a: str,
    domain_b: str,
    task_num: int,
    total: int,
    temperature: float,
    max_tokens: int,
) -> dict | None:
    prompt = GENERATION_PROMPT.format(domain_a=domain_a, domain_b=domain_b, task_num=task_num, total=total)
    try:
        resp = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = ""
        if isinstance(resp, dict):
            text = str(
                resp.get("content")
                or resp.get("text")
                or resp.get("output_text")
                or ""
            ).strip()
            if not text:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        msg = c0.get("message")
                        if isinstance(msg, dict):
                            text = str(msg.get("content") or "").strip()
                        if not text:
                            text = str(c0.get("text") or "").strip()
        if not text:
            text = str(resp).strip()
        if not text:
            return None
        return {
            "domain_a": domain_a,
            "domain_b": domain_b,
            "scenario": parse_field(text, "SCENARIO"),
            "question": parse_field(text, "QUESTION"),
            "correct_answer": parse_field(text, "CORRECT_ANSWER"),
            "answer_type": parse_field(text, "ANSWER_TYPE") or "short_text",
            "domain_a_skill": parse_field(text, "DOMAIN_A_SKILL"),
            "domain_b_skill": parse_field(text, "DOMAIN_B_SKILL"),
            "raw_generation": text,
        }
    except Exception as e:
        return {
            "domain_a": domain_a,
            "domain_b": domain_b,
            "error": str(e),
            "question": "",
            "correct_answer": "",
            "answer_type": "short_text",
        }


async def verify_one(
    client: UnifiedClient,
    model: str,
    rec: dict,
    max_tokens: int,
) -> dict:
    if not rec.get("question"):
        rec["verify_answer_correct"] = None
        rec["verify_requires_both"] = None
        rec["verify_error"] = "empty_question"
        return rec
    prompt = VERIFY_PROMPT.format(
        domain_a=rec["domain_a"],
        domain_b=rec["domain_b"],
        question=rec.get("question", ""),
        answer=rec.get("correct_answer", ""),
    )
    try:
        resp = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = str(resp.get("content", ""))
        ans_ok, req_ok = parse_verify_flags(text)
        rec["verify_answer_correct"] = ans_ok
        rec["verify_requires_both"] = req_ok
        rec["verify_raw"] = text
        return rec
    except Exception as e:
        rec["verify_answer_correct"] = None
        rec["verify_requires_both"] = None
        rec["verify_error"] = str(e)
        return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand Exp3 composite task bank with crash-safe checkpoints.")
    parser.add_argument("--tasks-per-pair", type=int, default=4)
    parser.add_argument("--generator-model", type=str, default="llama-3.1-70b")
    parser.add_argument("--verifier-model", type=str, default="mistral-large")
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--generation-max-tokens", type=int, default=500)
    parser.add_argument("--verify-max-tokens", type=int, default=220)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("exp3_expand_%Y%m%dT%H%M%S"))
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unverified", action="store_true")
    parser.add_argument("--retry-empty-generation", action="store_true")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    run_dir = args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress_log.jsonl"
    generated_path = run_dir / "generated_records.jsonl"
    verified_path = run_dir / "verified_records.jsonl"
    checkpoint_path = run_dir / "checkpoint.json"

    pairs = list(itertools.combinations(DOMAINS, 2))
    targets = [(a, b, idx) for (a, b) in pairs for idx in range(1, args.tasks_per_pair + 1)]

    existing_generated = {row.get("key"): row for row in read_jsonl(generated_path)}
    existing_verified = {row.get("key"): row for row in read_jsonl(verified_path)}
    if not args.resume:
        existing_generated = {}
        existing_verified = {}
        generated_path.write_text("", encoding="utf-8")
        verified_path.write_text("", encoding="utf-8")
    elif args.retry_unverified:
        existing_verified = {
            k: v
            for k, v in existing_verified.items()
            if v.get("verify_answer_correct") is not None and v.get("verify_requires_both") is not None
        }
    if args.resume and args.retry_empty_generation:
        existing_generated = {
            k: v
            for k, v in existing_generated.items()
            if str(v.get("question") or "").strip()
        }

    append_jsonl(
        progress_path,
        {
            "ts_utc": utc_now_iso(),
            "event": "run_start",
            "run_id": args.run_id,
            "tasks_per_pair": args.tasks_per_pair,
            "pairs": len(pairs),
            "targets": len(targets),
            "resume": bool(args.resume),
            "retry_unverified": bool(args.retry_unverified),
            "retry_empty_generation": bool(args.retry_empty_generation),
        },
    )

    gen_client = UnifiedClient(experiment="exp3_expand_generate")
    ver_client = UnifiedClient(experiment="exp3_expand_verify")
    sem = asyncio.Semaphore(max(1, args.max_workers))

    async def do_generate(a: str, b: str, idx: int) -> None:
        k = key_for(a, b, idx)
        if k in existing_generated:
            return
        async with sem:
            rec = await generate_one(
                gen_client,
                model=args.generator_model,
                domain_a=a,
                domain_b=b,
                task_num=idx,
                total=args.tasks_per_pair,
                temperature=float(args.generation_temperature),
                max_tokens=int(args.generation_max_tokens),
            )
        row = rec or {"domain_a": a, "domain_b": b}
        row["key"] = k
        row["task_num"] = idx
        row["generated_at_utc"] = utc_now_iso()
        append_jsonl(generated_path, row)
        existing_generated[k] = row
        append_jsonl(progress_path, {"ts_utc": utc_now_iso(), "event": "generated", "key": k, "ok": bool(row.get("question"))})

    await asyncio.gather(*[do_generate(a, b, idx) for (a, b, idx) in targets])

    async def do_verify(k: str, rec: dict) -> None:
        if k in existing_verified:
            return
        async with sem:
            out = await verify_one(ver_client, model=args.verifier_model, rec=dict(rec), max_tokens=int(args.verify_max_tokens))
        out["key"] = k
        out["verified_at_utc"] = utc_now_iso()
        append_jsonl(verified_path, out)
        existing_verified[k] = out
        append_jsonl(
            progress_path,
            {
                "ts_utc": utc_now_iso(),
                "event": "verified",
                "key": k,
                "answer_ok": out.get("verify_answer_correct"),
                "requires_both": out.get("verify_requires_both"),
            },
        )

    await asyncio.gather(*[do_verify(k, rec) for k, rec in existing_generated.items()])

    verified_rows = list(existing_verified.values())
    verified_rows.sort(key=lambda x: x.get("key", ""))

    # Emit final task bank (keep all rows; verification flags retained).
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(verified_rows, start=1):
            a = str(rec.get("domain_a", "")).strip()
            b = str(rec.get("domain_b", "")).strip()
            q = str(rec.get("question", "")).strip()
            ans = str(rec.get("correct_answer", "")).strip()
            atype = str(rec.get("answer_type", "short_text")).strip() or "short_text"
            payload = {
                "task_id": f"exp3v{args.tasks_per_pair}_{i:04d}",
                "domain_a": a,
                "domain_b": b,
                "tier": 1,
                "scenario": rec.get("scenario", ""),
                "task_text": q,
                "component_a": {
                    "domain": a,
                    "text": q,
                    "correct_answer": ans,
                    "difficulty": "medium",
                    "answer_type": atype,
                },
                "component_b": {
                    "domain": b,
                    "text": q,
                    "correct_answer": ans,
                    "difficulty": "medium",
                    "answer_type": atype,
                },
                "requires_both": True,
                "verified_answer_correct": rec.get("verify_answer_correct"),
                "verified_requires_both": rec.get("verify_requires_both"),
                "domain_a_skill": rec.get("domain_a_skill", ""),
                "domain_b_skill": rec.get("domain_b_skill", ""),
                "generation_key": rec.get("key", ""),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    pass_both = sum(
        1
        for rec in verified_rows
        if rec.get("verify_answer_correct") is True and rec.get("verify_requires_both") is True
    )
    summary = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "tasks_per_pair": args.tasks_per_pair,
        "pairs": len(pairs),
        "targets": len(targets),
        "n_generated_records": len(existing_generated),
        "n_verified_records": len(existing_verified),
        "n_pass_both_verifications": pass_both,
        "generator_model": args.generator_model,
        "verifier_model": args.verifier_model,
        "output_path": str(args.output_path),
    }
    checkpoint_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_jsonl(progress_path, {"ts_utc": utc_now_iso(), "event": "run_complete", **summary})
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
