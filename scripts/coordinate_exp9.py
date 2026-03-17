"""
Exp9 coordinator v3: monitors shard completion, runs merge/control2/analysis.

Phase 1: wait for 6 fast model shards → merge + control2 + partial analysis
Phase 2: wait for 9 mistral condition×paradigm shards → merge + final analysis

v3 changes:
- Phase 6b: runs mistral control2 as 3 PARALLEL paradigm shards (3x faster)
- Phase 1: exits immediately if fast models already done (shard files deleted)
"""
from __future__ import annotations
import json, subprocess, sys, time, threading
from datetime import datetime
from pathlib import Path

RUN_ID = "20260312T140842"
RESULTS_DIR = Path("data/results")
LOG_FILE = RESULTS_DIR / "pipeline_log.txt"

FAST_MODELS = ["llama-3.1-8b","llama-3.1-70b","llama-3.1-405b","qwen-3-235b","gpt-oss-120b","deepseek-r1"]
TARGET = 327 * 11  # 3597 per model

# 9 mistral shard files (one per remaining condition×paradigm combo)
MISTRAL_SHARDS = [
    RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_{tag}_shard.jsonl"
    for tag in ["c1p3","c2p1","c2p2","c2p3","c3p1","c3p2","c3p3","c4p1","c4p2"]
]

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[COORD {ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def shard_count(model):
    slug = model.replace(".","-").replace("/","-")
    shard = RESULTS_DIR / f"exp9_{RUN_ID}_{slug}_full_shard.jsonl"
    if not shard.exists():
        return -1
    return sum(1 for l in shard.read_text(encoding="utf-8").splitlines() if l.strip())

def merge_into_main(extra_shards):
    """Merge shard files into main results file, deduplicated."""
    main_file = RESULTS_DIR / f"exp9_{RUN_ID}_results.jsonl"
    existing_keys = set()
    if main_file.exists():
        for l in main_file.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            try:
                r = json.loads(l)
                existing_keys.add((r.get("model"),r.get("task_id"),r.get("condition"),r.get("paradigm"),r.get("is_false_score_control")))
            except: pass
    log(f"Merge: {len(existing_keys)} existing keys")
    new_lines = []
    for shard in extra_shards:
        if not shard.exists(): continue
        count = 0
        for l in shard.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            try:
                r = json.loads(l)
                k = (r.get("model"),r.get("task_id"),r.get("condition"),r.get("paradigm"),r.get("is_false_score_control"))
                if k not in existing_keys:
                    existing_keys.add(k); new_lines.append(l); count += 1
            except: pass
        if count: log(f"  {shard.name}: +{count}")
    if new_lines:
        with open(main_file,"a",encoding="utf-8") as f:
            for l in new_lines: f.write(l+"\n")
        log(f"Merge complete: +{len(new_lines)} records")
    return len(new_lines)

def run_cmd(cmd, name, attempts=5):
    for i in range(1, attempts+1):
        log(f"{name}: attempt {i}/{attempts}")
        r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        if r.returncode == 0:
            log(f"{name}: SUCCESS"); return True
        log(f"{name}: exit={r.returncode}")
        if i < attempts: time.sleep(30)
    log(f"{name}: FAILED"); return False

def fast_model_shards():
    return [RESULTS_DIR / f"exp9_{RUN_ID}_{m.replace('.', '-').replace('/', '-')}_full_shard.jsonl"
            for m in FAST_MODELS]

def main():
    log("="*60)
    log(f"COORDINATOR v2  run_id={RUN_ID}")
    log(f"Fast models ({len(FAST_MODELS)}): {', '.join(FAST_MODELS)}")
    log(f"Mistral shards ({len(MISTRAL_SHARDS)}): 9 condition×paradigm shards")
    log("="*60)

    # ── Phase 1: wait for 6 fast models ──────────────────────────────────────
    # If all shard files are gone (deleted after successful merge), skip immediately
    log("Phase 1: Checking fast model shards...")
    counts = {m: shard_count(m) for m in FAST_MODELS}
    done = [m for m,c in counts.items() if c < 0 or c >= TARGET]
    if len(done) == 6:
        log(f"Phase 1: All 6 fast models already done — skipping wait")
    else:
        log(f"Phase 1: {len(done)}/6 done, waiting...")
        prev_total = sum(max(c,0) for c in counts.values())
        stable_rounds = 0
        while True:
            time.sleep(60)
            counts = {m: shard_count(m) for m in FAST_MODELS}
            total = sum(max(c,0) for c in counts.values())
            done = [m for m,c in counts.items() if c < 0 or c >= TARGET]
            log(f"Fast: {len(done)}/6 done  total={total}  +{total-prev_total}/min")
            for m,c in counts.items():
                pct = 100*c/TARGET if c >= 0 else 100
                log(f"  {m}: {c}/{TARGET} ({pct:.1f}%)")
            if total == prev_total:
                stable_rounds += 1
            else:
                stable_rounds = 0
            prev_total = total
            if len(done) == 6 or stable_rounds >= 5:
                log(f"Phase 1 complete (done={len(done)}, stable_rounds={stable_rounds})")
                break

    # ── Phase 2: merge fast model shards ─────────────────────────────────────
    log("\nPhase 2: Merging fast model shards")
    merge_into_main(fast_model_shards())

    # ── Phase 3: control2 for fast models (parallel) ─────────────────────────
    log("\nPhase 3: Control2 for fast models")
    run_cmd([sys.executable,"scripts/launch_exp9_parallel.py",
             "--run-id",RUN_ID,"--start-phase","control2",
             "--models",",".join(FAST_MODELS)], "Control2-Fast")

    # ── Phase 4: partial analysis (6 models) ─────────────────────────────────
    log("\nPhase 4: Partial analysis (6 models)")
    run_cmd([sys.executable,"scripts/analyze_experiment_9.py","--run-id",RUN_ID], "Analysis-Partial")

    # ── Phase 5: wait for 9 mistral shards ───────────────────────────────────
    log("\nPhase 5: Waiting for 9 mistral condition×paradigm shards...")
    last_counts = {}
    stable = {}
    while True:
        time.sleep(120)
        all_stable = True
        for shard in MISTRAL_SHARDS:
            c = sum(1 for l in shard.read_text(encoding="utf-8").splitlines() if l.strip()) if shard.exists() else 0
            prev = last_counts.get(shard.name, -999)
            if c == prev:
                stable[shard.name] = stable.get(shard.name, 0) + 1
            else:
                stable[shard.name] = 0
            last_counts[shard.name] = c
            log(f"  {shard.name}: {c} records (stable={stable[shard.name]})")
            if stable[shard.name] < 3:
                all_stable = False
        if all_stable:
            log("All mistral shards stable (no new writes 6+ min) — merging")
            break

    # ── Phase 6: merge mistral + re-run analysis ──────────────────────────────
    log("\nPhase 6: Merging all mistral shards")
    # Include old shards too
    all_mistral_shards = list(RESULTS_DIR.glob(f"exp9_{RUN_ID}_mistral*.jsonl"))
    merge_into_main(all_mistral_shards)

    log("Phase 6b: Control2 for mistral-large (3 parallel paradigm shards)")
    ctrl2_results: dict = {}

    def _run_ctrl2(paradigm: int) -> None:
        tag = f"ctrl2_c2p{paradigm}"
        shard = RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_{tag}_shard.jsonl"
        cmd = [
            sys.executable, "scripts/run_experiment_9.py",
            "--mode", "control2", "--run-id", RUN_ID,
            "--models", "mistral-large",
            "--output-file", str(shard),
            "--resume", "--concurrency", "12",
            "--conditions", "2", "--paradigms", str(paradigm),
        ]
        for attempt in range(1, 6):
            log(f"  ctrl2 P{paradigm}: attempt {attempt}/5")
            r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
            if r.returncode == 0:
                log(f"  ctrl2 P{paradigm}: DONE"); ctrl2_results[paradigm] = True; return
            log(f"  ctrl2 P{paradigm}: exit={r.returncode}")
            if attempt < 5: time.sleep(15)
        ctrl2_results[paradigm] = False

    threads = [threading.Thread(target=_run_ctrl2, args=(p,)) for p in [1, 2, 3]]
    for t in threads: t.start()
    for t in threads: t.join()
    log(f"Control2-Mistral results: {ctrl2_results}")

    ctrl2_shards = [RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_ctrl2_c2p{p}_shard.jsonl"
                    for p in [1, 2, 3]]
    merge_into_main([s for s in ctrl2_shards if s.exists()])

    log("Phase 6c: Final analysis (all 7 models)")
    run_cmd([sys.executable,"scripts/analyze_experiment_9.py","--run-id",RUN_ID], "Analysis-Final")

    log("="*60)
    log("COORDINATOR COMPLETE")
    log(f"  Results: data/results/exp9_{RUN_ID}_results.jsonl")
    log(f"  Analysis: data/results/exp9_{RUN_ID}_analysis/")
    log("="*60)

if __name__ == "__main__":
    main()
