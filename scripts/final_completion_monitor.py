"""
Final completion monitor for Task 1 (Exp5 clean) + Task 2 (Exp1 questions).

Tracks:
1. Question generation → compiles questions.jsonl when done
2. Launches Exp1 fast for all models on expanded question bank
3. Monitors gemini-2.5-pro exp5 clean (+ qwen if miraculously works)
4. Runs Exp5 analysis once clean data is ready
"""
import glob
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/final_monitor.log")
LOG_PATH.parent.mkdir(exist_ok=True)


def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def count_lines(path):
    try:
        return sum(1 for l in open(path) if l.strip())
    except Exception:
        return 0


def count_model(path, model):
    n = 0
    try:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model:
                    n += 1
    except Exception:
        pass
    return n


def run_cmd(cmd, name, timeout=600):
    log(f"  >> {name}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            for line in (r.stdout or "").strip().split("\n")[-8:]:
                log(f"     {line}")
            log(f"  >> DONE: {name}")
        else:
            log(f"  >> FAILED: {name}: {(r.stderr or '')[-400:]}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  >> TIMEOUT: {name}")
        return False


def check_gen_complete():
    """Return True if all domains have enough unique questions."""
    total = 0
    for domain in ["arithmetic", "factual", "linguistic", "logical",
                   "procedural", "social", "spatial", "temporal"]:
        seed_p = Path(f"data/seeds_v2/{domain}.jsonl")
        gen_p = Path(f"data/generated_v2/{domain}.jsonl")
        seen = set()
        for p in [seed_p, gen_p]:
            if not p.exists():
                continue
            for line in open(p):
                if line.strip():
                    q = json.loads(line)
                    qid = q.get("source_id") or q.get("question_id") or "none"
                    seen.add(qid)
        total += min(len(seen), 625)
        if len(seen) < 625:
            return False, total
    return True, total


def compile_questions():
    log("Compiling questions.jsonl (target=625/domain)...")
    return run_cmd(
        ["python", "scripts/generate_questions_fast.py", "--compile-only", "--target", "625"],
        "compile questions.jsonl"
    )


def launch_exp1(run_id=None):
    """Launch exp1_fast for all models on the expanded question bank."""
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log(f"Launching Exp1 fast for all models (run_id={run_id})...")
    cmd = ["python", "-u", "scripts/run_exp1_fast.py", "--run-id", run_id]
    log_p = Path(f"logs/exp1_fast_{run_id}.log")
    with open(log_p, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    log(f"  Exp1 fast launched PID={proc.pid}, log={log_p}")
    return proc.pid, run_id


def check_exp1_done(run_id):
    """Check if exp1_fast results file has all models."""
    pattern = f"data/results/exp1_{run_id}_results.jsonl"
    files = glob.glob(pattern)
    if not files:
        return False
    counts = Counter()
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                counts[r.get("model", "?")] += 1
    # We consider done if ≥6 models have ≥300 records each
    done_models = [m for m, c in counts.items() if c >= 300]
    return len(done_models) >= 6, counts


def get_latest_exp5_clean_file():
    """Return the most recent exp5_clean results file."""
    files = sorted(
        Path("data/results").glob("exp5_clean_*_results.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


log("=== Final completion monitor starting ===")

# ── Phase 1: Question generation ────────────────────────────────────────────
gen_done = False
gen_compiled = False
questions_target = 5000

# Check if already done
gen_complete, gen_total = check_gen_complete()
if gen_complete:
    log(f"Generation already complete — {gen_total} unique questions available")
    gen_done = True
else:
    log(f"Generation in progress — {gen_total}/5000 unique questions so far")

# ── Phase 2: Gemini exp5 clean monitoring ────────────────────────────────────
EXP5_GEMINI_TARGET = 320

# Find gemini's results file (latest run)
gemini_file = None
for f in sorted(Path("data/results").glob("exp5_clean_2026*_results.jsonl"),
                key=lambda p: p.stat().st_mtime, reverse=True):
    if count_model(f, "gemini-2.5-pro") > 0:
        gemini_file = f
        break

if gemini_file:
    n_gemini = count_model(gemini_file, "gemini-2.5-pro")
    log(f"Gemini exp5 clean: {n_gemini}/{EXP5_GEMINI_TARGET} in {gemini_file.name}")
    gemini_done = n_gemini >= EXP5_GEMINI_TARGET
else:
    # Will be created by the running process
    log("Gemini exp5 clean: file not yet created, scanning...")
    gemini_done = False
    gemini_file = Path("data/results/exp5_clean_20260314T111037_results.jsonl")

exp1_launched = False
exp1_pid = None
exp1_run_id = None
analyses_done = False

log("Entering monitoring loop (30s interval)...")

while True:
    time.sleep(30)

    # ── Check generation ──────────────────────────────────────────────────────
    if not gen_done:
        gen_complete, gen_total = check_gen_complete()
        if gen_complete:
            log(f"✓ Question generation complete: {gen_total} unique questions")
            gen_done = True
        else:
            log(f"  Generation: {gen_total}/5000 unique questions")

    # ── Compile after generation ──────────────────────────────────────────────
    if gen_done and not gen_compiled:
        ok = compile_questions()
        if ok:
            n_q = count_lines("data/questions.jsonl")
            log(f"✓ Compiled questions.jsonl: {n_q} questions")
            gen_compiled = True
        else:
            log("  Compile failed, will retry...")

    # ── Launch Exp1 after compilation ─────────────────────────────────────────
    if gen_compiled and not exp1_launched:
        exp1_pid, exp1_run_id = launch_exp1()
        exp1_launched = True

    # ── Monitor gemini exp5 clean ─────────────────────────────────────────────
    if not gemini_done:
        # Re-scan for the gemini file (in case it was just created)
        for f in sorted(Path("data/results").glob("exp5_clean_2026*_results.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            ng = count_model(f, "gemini-2.5-pro")
            if ng > 0:
                gemini_file = f
                n_gemini = ng
                break
        else:
            n_gemini = 0

        if n_gemini >= EXP5_GEMINI_TARGET:
            log(f"✓ Gemini exp5 clean COMPLETE: {n_gemini}/{EXP5_GEMINI_TARGET}")
            gemini_done = True
        else:
            log(f"  Gemini exp5 clean: {n_gemini}/{EXP5_GEMINI_TARGET}")

    # ── Check Exp1 progress ───────────────────────────────────────────────────
    exp1_done_flag = False
    if exp1_launched and exp1_run_id:
        done_check, counts = check_exp1_done(exp1_run_id)
        if done_check:
            log(f"✓ Exp1 fast complete: {dict(counts)}")
            exp1_done_flag = True

    # ── Run analyses when gemini done ─────────────────────────────────────────
    if gemini_done and not analyses_done:
        log("Running Exp5 analysis with all clean data...")

        # Use the most comprehensive clean file (the main one with 9 models)
        main_clean = Path("data/results/exp5_clean_20260313T205910_results.jsonl")
        adv_file = Path("data/results/exp5_20260313T205347_results.jsonl")

        if adv_file.exists() and main_clean.exists():
            run_cmd(
                ["python", "scripts/analyze_experiment_5.py",
                 str(adv_file),
                 "--clean-baseline", str(main_clean)],
                "Exp5 adversarial analysis (5-model comparison)",
                timeout=300,
            )
        else:
            log(f"  Skipping analysis: adv={adv_file.exists()}, clean={main_clean.exists()}")

        # Also run analysis on the gemini file specifically
        if gemini_file and gemini_file.exists():
            run_cmd(
                ["python", "scripts/analyze_experiment_5.py", str(gemini_file)],
                "Exp5 clean gemini analysis",
                timeout=300,
            )

        analyses_done = True

    # ── Final check ───────────────────────────────────────────────────────────
    if gen_compiled and gemini_done and analyses_done:
        log("=== All immediate tasks complete ===")
        log("  ✓ questions.jsonl compiled to 5000")
        log("  ✓ Gemini exp5 clean done")
        log("  ✓ Exp5 analysis run")
        if exp1_launched:
            log(f"  Exp1 fast still running (PID={exp1_pid}) — will complete in background")
        break

    # Safety: don't loop forever if nothing changes
    all_stuck = gen_done and gen_compiled and gemini_done and analyses_done
    if all_stuck:
        break

log("Final completion monitor finished.")
