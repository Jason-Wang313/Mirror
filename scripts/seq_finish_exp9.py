"""
Sequential finisher: wait for full data → ctrl2 (3 parallel) → merge → analysis.
"""
from __future__ import annotations
import json, subprocess, sys, time, threading
from datetime import datetime
from pathlib import Path

RUN_ID = "20260312T140842"
RESULTS_DIR = Path("data/results")
LOG_FILE = RESULTS_DIR / "pipeline_log.txt"

FULL_SHARDS = {
    "c1p3": (297, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c1p3_shard.jsonl"),
    "c2p1": (319, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c2p1_shard.jsonl"),
    "c2p2": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c2p2_shard.jsonl"),
    "c2p3": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c2p3_shard.jsonl"),
    "c3p1": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c3p1_shard.jsonl"),
    "c3p2": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c3p2_shard.jsonl"),
    "c3p3": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c3p3_shard.jsonl"),
    "c4p1": (319, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c4p1_shard.jsonl"),
    "c4p2": (327, RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_c4p2_shard.jsonl"),
}
CWD = str(Path(__file__).parent.parent)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[SEQ {ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    except Exception:
        return 0


def merge_into_main(shards):
    main_file = RESULTS_DIR / f"exp9_{RUN_ID}_results.jsonl"
    existing_keys: set = set()
    if main_file.exists():
        for l in main_file.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            try:
                r = json.loads(l)
                existing_keys.add((r.get("model"), r.get("task_id"), r.get("condition"),
                                   r.get("paradigm"), r.get("is_false_score_control")))
            except Exception:
                pass
    log(f"Merge: {len(existing_keys)} existing keys")
    new_lines = []
    for shard in shards:
        if not shard.exists():
            continue
        count = 0
        for l in shard.read_text(encoding="utf-8").splitlines():
            if not l.strip(): continue
            try:
                r = json.loads(l)
                k = (r.get("model"), r.get("task_id"), r.get("condition"),
                     r.get("paradigm"), r.get("is_false_score_control"))
                if k not in existing_keys:
                    existing_keys.add(k); new_lines.append(l); count += 1
            except Exception:
                pass
        if count:
            log(f"  {shard.name}: +{count}")
    if new_lines:
        with open(main_file, "a", encoding="utf-8") as f:
            for l in new_lines: f.write(l + "\n")
        log(f"Merge: +{len(new_lines)} new records")
    return len(new_lines)


def wait_for_full_data():
    log("Phase 1: Waiting for full data shards to complete...")
    prev = {}
    stable = {}
    while True:
        time.sleep(60)
        curr = {tag: count_lines(path) for tag, (_, path) in FULL_SHARDS.items()}
        total = sum(curr.values())
        done = all(curr[tag] >= target for tag, (target, _) in FULL_SHARDS.items())
        log(f"  Full: {total}/2897 ({100*total/2897:.1f}%) done={done}")

        all_stable = True
        for tag in curr:
            if curr[tag] != prev.get(tag, -1):
                stable[tag] = 0; all_stable = False
            else:
                stable[tag] = stable.get(tag, 0) + 1
                if stable[tag] < 3:
                    all_stable = False
        prev = curr.copy()

        if done or all_stable:
            log(f"Full data complete (done={done}, all_stable={all_stable})")
            return


def run_ctrl2():
    log("Phase 2: Running ctrl2 for mistral-large (3 parallel paradigm shards)...")
    results: dict = {}

    def _run(paradigm: int):
        shard = RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_ctrl2_c2p{paradigm}_shard.jsonl"
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
            r = subprocess.run(cmd, cwd=CWD)
            if r.returncode == 0:
                log(f"  ctrl2 P{paradigm}: DONE"); results[paradigm] = True; return
            log(f"  ctrl2 P{paradigm}: exit={r.returncode}")
            if attempt < 5: time.sleep(15)
        results[paradigm] = False

    threads = [threading.Thread(target=_run, args=(p,)) for p in [1, 2, 3]]
    for t in threads: t.start()
    for t in threads: t.join()
    log(f"ctrl2 results: {results}")
    return results


def main():
    log("=" * 60)
    log(f"SEQ FINISHER  run_id={RUN_ID}")
    log("=" * 60)

    wait_for_full_data()

    # Merge full data shards
    log("\nMerging full data shards...")
    all_full_shards = list(RESULTS_DIR.glob(f"exp9_{RUN_ID}_mistral*.jsonl"))
    merge_into_main(all_full_shards)

    # Run ctrl2
    run_ctrl2()

    # Merge ctrl2 shards
    log("\nMerging ctrl2 shards...")
    ctrl2_shards = [RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_ctrl2_c2p{p}_shard.jsonl"
                    for p in [1, 2, 3]]
    merge_into_main([s for s in ctrl2_shards if s.exists()])

    # Final analysis
    log("\nRunning final analysis...")
    for attempt in range(1, 4):
        log(f"Analysis attempt {attempt}/3")
        r = subprocess.run(
            [sys.executable, "scripts/analyze_experiment_9.py", "--run-id", RUN_ID],
            cwd=CWD
        )
        if r.returncode == 0:
            log("Analysis: SUCCESS"); break
        log(f"Analysis: exit={r.returncode}")
        if attempt < 3: time.sleep(15)

    log("=" * 60)
    log("SEQ FINISHER COMPLETE")
    log(f"  Results: data/results/exp9_{RUN_ID}_results.jsonl")
    log(f"  Analysis: data/results/exp9_{RUN_ID}_analysis/")
    log("=" * 60)


if __name__ == "__main__":
    main()
