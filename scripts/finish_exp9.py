"""
Finish Exp9: wait for full data + ctrl2 to both complete, then merge + analysis.
Run after launching mistral full shards AND ctrl2 shards in parallel.
"""
from __future__ import annotations
import json, subprocess, sys, time
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
CTRL2_SHARDS = [
    RESULTS_DIR / f"exp9_{RUN_ID}_mistral-large_ctrl2_c2p{p}_shard.jsonl"
    for p in [1, 2, 3]
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[FINISH {ts}] {msg}"
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
    log(f"Merge: {len(existing_keys)} existing keys in main file")
    new_lines = []
    for shard in shards:
        if not shard.exists():
            log(f"  {shard.name}: NOT FOUND — skipping")
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
        log(f"  {shard.name}: +{count}")
    if new_lines:
        with open(main_file, "a", encoding="utf-8") as f:
            for l in new_lines: f.write(l + "\n")
        log(f"Merge complete: +{len(new_lines)} records")
    else:
        log("Merge: 0 new records")
    return len(new_lines)


def all_stable(prev: dict, curr: dict, rounds: dict, threshold=2) -> bool:
    """Check if all shards are stable (no change for 'threshold' rounds)."""
    for key in curr:
        if curr[key] != prev.get(key, -1):
            rounds[key] = 0
        else:
            rounds[key] = rounds.get(key, 0) + 1
    return all(rounds.get(k, 0) >= threshold for k in curr)


def main():
    log("=" * 60)
    log(f"FINISHER  run_id={RUN_ID}")
    log("Waiting for full data + ctrl2 to both complete...")
    log("=" * 60)

    full_prev: dict = {}
    ctrl2_prev: dict = {}
    full_rounds: dict = {}
    ctrl2_rounds: dict = {}

    while True:
        time.sleep(60)

        # Full shard status
        full_curr = {tag: count_lines(path) for tag, (_, path) in FULL_SHARDS.items()}
        full_done = all(full_curr[tag] >= target for tag, (target, _) in FULL_SHARDS.items())
        full_pct = 100 * sum(full_curr.values()) / 2897
        log(f"Full: {sum(full_curr.values())}/2897 ({full_pct:.1f}%) done={full_done}")

        # Ctrl2 shard status
        ctrl2_curr = {s.name: count_lines(s) for s in CTRL2_SHARDS}
        ctrl2_total = sum(ctrl2_curr.values())
        ctrl2_done_flag = all_stable(ctrl2_prev, ctrl2_curr, ctrl2_rounds, threshold=2)
        log(f"Ctrl2: {ctrl2_total} records, stable_done={ctrl2_done_flag}")
        for name, c in ctrl2_curr.items():
            log(f"  {name}: {c} (stable={ctrl2_rounds.get(name, 0)})")

        full_stable_done = all_stable(full_prev, full_curr, full_rounds, threshold=2)

        full_prev = full_curr.copy()
        ctrl2_prev = ctrl2_curr.copy()

        if (full_done or full_stable_done) and ctrl2_done_flag:
            log("Both full data and ctrl2 stable/complete — proceeding to merge")
            break

    # Merge ALL mistral shards (full + ctrl2)
    log("\n--- Merging all mistral shards ---")
    all_shards = (
        list(RESULTS_DIR.glob(f"exp9_{RUN_ID}_mistral*.jsonl")) +
        list(CTRL2_SHARDS)
    )
    merge_into_main(list(set(all_shards)))

    # Final analysis
    log("\n--- Running final analysis ---")
    for attempt in range(1, 4):
        log(f"Analysis: attempt {attempt}/3")
        r = subprocess.run(
            [sys.executable, "scripts/analyze_experiment_9.py", "--run-id", RUN_ID],
            cwd=str(Path(__file__).parent.parent)
        )
        if r.returncode == 0:
            log("Analysis: SUCCESS")
            break
        log(f"Analysis: exit={r.returncode}")
        if attempt < 3: time.sleep(15)

    log("=" * 60)
    log("FINISHER COMPLETE")
    log(f"  Results: data/results/exp9_{RUN_ID}_results.jsonl")
    log(f"  Analysis: data/results/exp9_{RUN_ID}_analysis/")
    log("=" * 60)


if __name__ == "__main__":
    main()
