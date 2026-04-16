"""
Experiment 9: Task Bank Verification
======================================

Two-phase verification of data/exp9_tasks.jsonl:

  Phase 1 — Static (always runs):
    • Schema completeness (all required fields present)
    • Answer type consistency
    • Subcategory membership
    • Domain pair plausibility

  Phase 2 — Cross-LLM (requires --api):
    • Sends each task to 3 verifier models (concurrent)
    • Checks model answers against recorded ground truth
    • Flags tasks with ≥2/3 verifier disagreements
    • Writes detailed report + clean/flagged JSONL

Usage:
    python scripts/verify_exp9_tasks.py                   # static only
    python scripts/verify_exp9_tasks.py --api             # static + LLM
    python scripts/verify_exp9_tasks.py --api --sample 30 # LLM on 30 random tasks
    python scripts/verify_exp9_tasks.py --fixed-only      # fixed tasks only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.scoring.agentic_metrics import SUBCATEGORIES, DOMAINS

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_FILE = Path("data/exp9_tasks.jsonl")
REPORT_FILE = Path("data/exp9_verification_report.json")
FLAGGED_FILE = Path("data/exp9_tasks_flagged.jsonl")
CLEAN_FILE = Path("data/exp9_tasks_clean.jsonl")
DEFAULT_CHECKPOINT_DIR = Path("data/exp9_verification_runs")

REQUIRED_FIELDS = [
    "task_id", "task_type", "circularity_free",
    "domain_a", "domain_b", "subcategory_a", "subcategory_b",
    "difficulty_a", "difficulty_b",
    "correct_answer_a", "correct_answer_b",
    "answer_type_a", "answer_type_b",
    "task_text", "part1_text", "part2_text",
]

VALID_TASK_TYPES = {"fixed", "tailored"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_ANSWER_TYPES = {
    "exact_numeric", "short_text", "boolean", "multiple_choice"
}

# Verifier models (affordable, capable)
VERIFIER_MODELS = [
    "llama-3.1-70b",
    "deepseek-v3",
    "gemini-2.5-pro",
]

VERIFIER_PROMPT = """\
You are verifying answers for a structured two-component task.

Task context: {task_text}

COMPONENT A QUESTION:
{part1_text}

RECORDED ANSWER A: {correct_answer_a}

COMPONENT B QUESTION:
{part2_text}

RECORDED ANSWER B: {correct_answer_b}

For each component, state:
1. Whether the recorded answer is CORRECT, INCORRECT, or AMBIGUOUS.
2. If incorrect, provide the correct answer.

Format your response as JSON:
{{
  "component_a": {{"verdict": "CORRECT"|"INCORRECT"|"AMBIGUOUS", "correct_answer": "...or null if correct"}},
  "component_b": {{"verdict": "CORRECT"|"INCORRECT"|"AMBIGUOUS", "correct_answer": "...or null if correct"}}
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Static verification
# ─────────────────────────────────────────────────────────────────────────────

def load_tasks(fixed_only: bool = False) -> list[dict]:
    tasks = []
    with open(TASK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if fixed_only and t.get("task_type") != "fixed":
                continue
            tasks.append(t)
    return tasks


def static_verify_task(task: dict) -> list[str]:
    """Return list of issue strings (empty = OK)."""
    issues = []

    # 1. Required fields
    missing = [f for f in REQUIRED_FIELDS if f not in task]
    if missing:
        issues.append(f"Missing fields: {missing}")

    # 2. task_type
    if task.get("task_type") not in VALID_TASK_TYPES:
        issues.append(f"Invalid task_type: {task.get('task_type')!r}")

    # 3. circularity_free must be bool
    if not isinstance(task.get("circularity_free"), bool):
        issues.append("circularity_free is not a boolean")

    # 4. Domains valid
    for slot in ("a", "b"):
        d = task.get(f"domain_{slot}")
        if d not in DOMAINS:
            issues.append(f"Invalid domain_{slot}: {d!r}")
        sc = task.get(f"subcategory_{slot}")
        valid_scs = SUBCATEGORIES.get(d, [])
        if sc not in valid_scs:
            issues.append(f"subcategory_{slot} {sc!r} not in {d!r} subcategories {valid_scs}")

    # 5. Difficulties
    for slot in ("a", "b"):
        if task.get(f"difficulty_{slot}") not in VALID_DIFFICULTIES:
            issues.append(f"Invalid difficulty_{slot}: {task.get(f'difficulty_{slot}')!r}")

    # 6. Answer types
    for slot in ("a", "b"):
        if task.get(f"answer_type_{slot}") not in VALID_ANSWER_TYPES:
            issues.append(f"Invalid answer_type_{slot}: {task.get(f'answer_type_{slot}')!r}")

    # 7. Answers non-empty
    for slot in ("a", "b"):
        ans = str(task.get(f"correct_answer_{slot}", "")).strip()
        if not ans:
            issues.append(f"Empty correct_answer_{slot}")

    # 8. Multiple-choice answers must be A/B/C/D or a single letter
    for slot in ("a", "b"):
        if task.get(f"answer_type_{slot}") == "multiple_choice":
            ans = str(task.get(f"correct_answer_{slot}", "")).strip().upper()
            if ans not in {"A", "B", "C", "D"}:
                issues.append(
                    f"answer_{slot} is multiple_choice but value is {ans!r} (expected A/B/C/D)"
                )

    # 9. Boolean answers must be yes/no/true/false
    for slot in ("a", "b"):
        if task.get(f"answer_type_{slot}") == "boolean":
            ans = str(task.get(f"correct_answer_{slot}", "")).strip().lower()
            if ans not in {"yes", "no", "true", "false"}:
                issues.append(
                    f"answer_{slot} is boolean but value is {ans!r} (expected yes/no/true/false)"
                )

    # 10. Numeric answers must parse as float
    for slot in ("a", "b"):
        if task.get(f"answer_type_{slot}") == "exact_numeric":
            raw = str(task.get(f"correct_answer_{slot}", "")).strip()
            try:
                float(raw.replace(",", ""))
            except ValueError:
                issues.append(f"answer_{slot} is exact_numeric but {raw!r} is not parseable as float")

    # 11. Text fields non-empty
    for field in ("task_text", "part1_text", "part2_text"):
        if not str(task.get(field, "")).strip():
            issues.append(f"Empty {field}")

    # 12. Tailored tasks must have target_model
    if task.get("task_type") == "tailored":
        if not task.get("target_model"):
            issues.append("Tailored task missing target_model")

    return issues


def run_static_verification(tasks: list[dict]) -> dict:
    """Run static checks and return summary."""
    ok = []
    flagged = []

    for task in tasks:
        issues = static_verify_task(task)
        if issues:
            flagged.append({"task_id": task["task_id"], "issues": issues})
        else:
            ok.append(task)

    # Distribution stats
    type_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}

    for t in tasks:
        type_counts[t.get("task_type", "?")] = type_counts.get(t.get("task_type", "?"), 0) + 1
        for slot in ("a", "b"):
            d = t.get(f"domain_{slot}", "?")
            domain_counts[d] = domain_counts.get(d, 0) + 1
            diff = t.get(f"difficulty_{slot}", "?")
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

    circularity_free = sum(1 for t in tasks if t.get("circularity_free"))

    return {
        "total_tasks": len(tasks),
        "passed_static": len(ok),
        "failed_static": len(flagged),
        "circularity_free_count": circularity_free,
        "task_type_distribution": type_counts,
        "domain_component_counts": dict(sorted(domain_counts.items())),
        "difficulty_distribution": diff_counts,
        "static_failures": flagged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Cross-LLM verification
# ─────────────────────────────────────────────────────────────────────────────

async def verify_task_with_model(client, model: str, task: dict) -> dict:
    """Call model to verify a single task's ground truth answers."""
    prompt = VERIFIER_PROMPT.format(
        task_text=task["task_text"],
        part1_text=task["part1_text"],
        correct_answer_a=task["correct_answer_a"],
        part2_text=task["part2_text"],
        correct_answer_b=task["correct_answer_b"],
    )
    try:
        response = await client.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        text = response.get("content", "")
        # Extract JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            verdict = json.loads(text[start:end])
        else:
            verdict = {"component_a": {"verdict": "AMBIGUOUS", "correct_answer": None},
                       "component_b": {"verdict": "AMBIGUOUS", "correct_answer": None}}
    except Exception as e:
        verdict = {
            "component_a": {"verdict": "ERROR", "correct_answer": str(e)},
            "component_b": {"verdict": "ERROR", "correct_answer": str(e)},
        }
    if not isinstance(verdict, dict):
        verdict = {
            "component_a": {"verdict": "AMBIGUOUS", "correct_answer": str(verdict)},
            "component_b": {"verdict": "AMBIGUOUS", "correct_answer": str(verdict)},
        }
    for comp in ("component_a", "component_b"):
        comp_v = verdict.get(comp)
        if not isinstance(comp_v, dict):
            verdict[comp] = {"verdict": "AMBIGUOUS", "correct_answer": str(comp_v)}
            continue
        if "verdict" not in comp_v:
            comp_v["verdict"] = "AMBIGUOUS"
        if "correct_answer" not in comp_v:
            comp_v["correct_answer"] = None
    return {"model": model, "verdict": verdict}


async def verify_batch(client, tasks: list[dict], n_verifiers: int = 3) -> list[dict]:
    """Verify a batch of tasks using N verifier models concurrently."""
    results = []
    total = len(tasks)
    for i, task in enumerate(tasks):
        print(f"  Verifying task {i+1}/{total}: {task['task_id']}", flush=True)
        model_verdicts = await asyncio.gather(*[
            verify_task_with_model(client, m, task)
            for m in VERIFIER_MODELS[:n_verifiers]
        ])
        # Aggregate: flag if ≥2 verifiers say INCORRECT for either component
        flag_a = sum(
            1 for v in model_verdicts
            if v["verdict"].get("component_a", {}).get("verdict") == "INCORRECT"
        ) >= 2
        flag_b = sum(
            1 for v in model_verdicts
            if v["verdict"].get("component_b", {}).get("verdict") == "INCORRECT"
        ) >= 2
        results.append({
            "task_id": task["task_id"],
            "flagged_a": flag_a,
            "flagged_b": flag_b,
            "flagged": flag_a or flag_b,
            "model_verdicts": model_verdicts,
        })
    return results


def append_progress(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def verify_batch_resumable(
    client,
    tasks: list[dict],
    n_verifiers: int,
    max_workers: int,
    checkpoint_path: Path,
    progress_log: Path,
    resume_results: dict[str, dict] | None = None,
) -> list[dict]:
    """Task-level resumable verifier with deterministic shard keys (task_id)."""
    existing = dict(resume_results or {})
    tasks_sorted = sorted(tasks, key=lambda t: str(t.get("task_id", "")))
    pending = [t for t in tasks_sorted if t.get("task_id") not in existing]
    sem = asyncio.Semaphore(max(1, max_workers))
    lock = asyncio.Lock()

    append_progress(
        progress_log,
        {
            "event": "resume_state",
            "completed": len(existing),
            "pending": len(pending),
            "total": len(tasks_sorted),
            "timestamp": time.time(),
        },
    )

    async def run_one(task: dict) -> None:
        task_id = task["task_id"]
        async with sem:
            verdicts = await asyncio.gather(
                *[verify_task_with_model(client, m, task) for m in VERIFIER_MODELS[:n_verifiers]]
            )
            flag_a = sum(1 for v in verdicts if v["verdict"].get("component_a", {}).get("verdict") == "INCORRECT") >= 2
            flag_b = sum(1 for v in verdicts if v["verdict"].get("component_b", {}).get("verdict") == "INCORRECT") >= 2
            out = {
                "task_id": task_id,
                "flagged_a": flag_a,
                "flagged_b": flag_b,
                "flagged": flag_a or flag_b,
                "model_verdicts": verdicts,
            }
            async with lock:
                existing[task_id] = out
                ck = {
                    "completed_task_ids": sorted(existing.keys()),
                    "results": [existing[k] for k in sorted(existing.keys())],
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(json.dumps(ck, indent=2), encoding="utf-8")
                append_progress(
                    progress_log,
                    {
                        "event": "task_complete",
                        "task_id": task_id,
                        "completed": len(existing),
                        "total": len(tasks_sorted),
                        "timestamp": time.time(),
                    },
                )

    await asyncio.gather(*[run_one(t) for t in pending])
    return [existing[k] for k in sorted(existing.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify Experiment 9 task bank")
    parser.add_argument("--api", action="store_true",
                        help="Run cross-LLM verification (requires API access)")
    parser.add_argument("--sample", type=int, default=0,
                        help="For --api mode: verify only N random tasks (0 = all)")
    parser.add_argument("--fixed-only", action="store_true",
                        help="Verify fixed (circularity_free) tasks only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for resumable outputs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if it exists.")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max concurrent task verifications in --api mode.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory for run checkpoints and logs.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.utcnow().strftime("exp9_verify_%Y%m%dT%H%M%SZ")
    run_dir = args.checkpoint_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_file = run_dir / "exp9_verification_report.json"
    flagged_file = run_dir / "exp9_tasks_flagged.jsonl"
    clean_file = run_dir / "exp9_tasks_clean.jsonl"
    checkpoint_file = run_dir / "verification_checkpoint.json"
    progress_log = run_dir / "progress_log.jsonl"

    append_progress(progress_log, {"event": "run_start", "run_id": run_id, "timestamp": time.time()})

    print("=" * 70)
    print("EXPERIMENT 9: TASK BANK VERIFICATION")
    print("=" * 70)

    if not TASK_FILE.exists():
        print(f"ERROR: {TASK_FILE} not found. Run generate_exp9_tasks.py first.")
        sys.exit(1)

    # ── Phase 1: Static ──────────────────────────────────────────────────────
    print(f"\nLoading tasks from {TASK_FILE}...")
    tasks = load_tasks(fixed_only=args.fixed_only)
    print(f"  Loaded {len(tasks)} tasks.")

    print("\n[Phase 1] Running static verification...")
    static_report = run_static_verification(tasks)

    print(f"\nStatic verification summary:")
    print(f"  Total tasks      : {static_report['total_tasks']}")
    print(f"  Passed           : {static_report['passed_static']}")
    print(f"  Failed           : {static_report['failed_static']}")
    print(f"  Circularity-free : {static_report['circularity_free_count']}")
    print(f"  Type distribution: {static_report['task_type_distribution']}")
    print(f"  Difficulty dist  : {static_report['difficulty_distribution']}")

    if static_report["static_failures"]:
        print(f"\n  STATIC FAILURES ({len(static_report['static_failures'])}):")
        for fail in static_report["static_failures"][:10]:
            print(f"    {fail['task_id']}: {fail['issues']}")
        if len(static_report["static_failures"]) > 10:
            print(f"    ... and {len(static_report['static_failures']) - 10} more")
    else:
        print("\n  All tasks passed static verification.")

    # Domain component distribution
    print("\n  Domain component counts (both A and B slots combined):")
    total_slots = sum(static_report["domain_component_counts"].values())
    for domain, count in sorted(static_report["domain_component_counts"].items()):
        bar = "█" * (count * 30 // max(1, total_slots))
        print(f"    {domain:12s}: {count:4d}  {bar}")

    full_report = {"static": static_report, "llm_verification": None}

    # ── Phase 2: LLM verification ─────────────────────────────────────────────
    if args.api:
        print("\n[Phase 2] Cross-LLM verification...")
        try:
            from mirror.api.client import UnifiedClient
            client = UnifiedClient()
        except Exception as e:
            print(f"ERROR: Could not initialise API client: {e}")
            print("Skipping LLM verification.")
            args.api = False

    if args.api:
        verify_tasks = [t for t in tasks if static_verify_task(t) == []]
        if args.sample > 0:
            random.seed(args.seed)
            verify_tasks = random.sample(verify_tasks, min(args.sample, len(verify_tasks)))
            print(f"  Sampled {len(verify_tasks)} tasks for LLM verification.")
        else:
            print(f"  Verifying all {len(verify_tasks)} statically-clean tasks.")
        resume_results = {}
        if args.resume and checkpoint_file.exists():
            try:
                ck = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                for r in ck.get("results", []):
                    if isinstance(r, dict) and r.get("task_id"):
                        resume_results[str(r["task_id"])] = r
                print(f"  Resuming with {len(resume_results)} completed tasks from checkpoint.")
            except Exception as e:
                print(f"  WARNING: failed to load checkpoint: {e}")

        llm_results = asyncio.run(
            verify_batch_resumable(
                client=client,
                tasks=verify_tasks,
                n_verifiers=3,
                max_workers=args.max_workers,
                checkpoint_path=checkpoint_file,
                progress_log=progress_log,
                resume_results=resume_results,
            )
        )

        n_flagged = sum(1 for r in llm_results if r["flagged"])
        flagged_a = sum(1 for r in llm_results if r["flagged_a"])
        flagged_b = sum(1 for r in llm_results if r["flagged_b"])

        print(f"\nLLM verification summary:")
        print(f"  Tasks verified: {len(llm_results)}")
        print(f"  Flagged total : {n_flagged}")
        print(f"  Flagged part A: {flagged_a}")
        print(f"  Flagged part B: {flagged_b}")

        if n_flagged > 0:
            print("\n  Flagged tasks:")
            flagged_ids = {r["task_id"] for r in llm_results if r["flagged"]}
            for r in llm_results:
                if r["flagged"]:
                    print(f"    {r['task_id']}: A={r['flagged_a']} B={r['flagged_b']}")

        full_report["llm_verification"] = {
            "tasks_verified": len(llm_results),
            "flagged_count": n_flagged,
            "flagged_part_a": flagged_a,
            "flagged_part_b": flagged_b,
            "results": llm_results,
        }

        # Write clean and flagged splits
        verified_ids = {r["task_id"] for r in llm_results}
        flagged_ids = {r["task_id"] for r in llm_results if r["flagged"]}
        task_by_id = {t["task_id"]: t for t in tasks}

        clean_file.parent.mkdir(parents=True, exist_ok=True)
        with open(clean_file, "w", encoding="utf-8") as fc, \
             open(flagged_file, "w", encoding="utf-8") as ff:
            for t in tasks:
                tid = t["task_id"]
                if tid in verified_ids:
                    target = ff if tid in flagged_ids else fc
                else:
                    fc.write(json.dumps(t) + "\n")  # unverified → clean
                    continue
                target.write(json.dumps(t) + "\n")

        print(f"\n  Clean tasks written to  : {clean_file}")
        print(f"  Flagged tasks written to: {flagged_file}")

    # ── Write report ──────────────────────────────────────────────────────────
    # Include run metadata
    full_report["run_metadata"] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "max_workers": int(args.max_workers),
        "timestamp": time.time(),
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Stable latest aliases for downstream tooling.
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    if clean_file.exists():
        CLEAN_FILE.write_text(clean_file.read_text(encoding="utf-8"), encoding="utf-8")
    if flagged_file.exists():
        FLAGGED_FILE.write_text(flagged_file.read_text(encoding="utf-8"), encoding="utf-8")

    append_progress(progress_log, {"event": "run_complete", "run_id": run_id, "timestamp": time.time()})

    print(f"\nFull report saved to: {report_file}")
    print("\n" + "=" * 70)
    verdict = "PASS" if static_report["failed_static"] == 0 else "FAIL"
    print(f"STATIC VERIFICATION: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
