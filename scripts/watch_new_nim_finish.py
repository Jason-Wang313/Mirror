"""
Watches the 5 new NIM model shards until all complete, then merges + analyzes.
Run after launch_new_nim_exp9.py was killed/interrupted.
Targets: full=3267 per model, control2=450 per model.
"""
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

RUN_ID = "20260312T140842"
RESULTS_DIR = Path("data/results")
CWD = str(Path(__file__).parent.parent)

# Expected counts per mode
TARGETS = {"full": 3267, "control2": 450}

MODELS = ["llama-3.3-70b", "kimi-k2", "phi-4", "gemma-3-27b", "qwen3-235b-nim"]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[WATCH {ts}] {msg}", flush=True)


def shard_path(model: str, mode: str) -> Path:
    slug = model.replace(".", "-").replace("/", "-")
    return RESULTS_DIR / f"exp9_{RUN_ID}_{slug}_{mode}_shard.jsonl"


def count_shard(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def merge_into_main(shards: list) -> int:
    main_file = RESULTS_DIR / f"exp9_{RUN_ID}_results.jsonl"
    existing_keys: set = set()
    if main_file.exists():
        for l in main_file.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                existing_keys.add((
                    r.get("model"), r.get("task_id"), r.get("condition"),
                    r.get("paradigm"), r.get("is_false_score_control"),
                ))
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
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                k = (r.get("model"), r.get("task_id"), r.get("condition"),
                     r.get("paradigm"), r.get("is_false_score_control"))
                if k not in existing_keys:
                    existing_keys.add(k)
                    new_lines.append(l)
                    count += 1
            except Exception:
                pass
        log(f"  {shard.name}: +{count}")
    if new_lines:
        with open(main_file, "a", encoding="utf-8") as f:
            for l in new_lines:
                f.write(l + "\n")
        log(f"Merge complete: +{len(new_lines)} records → {main_file.name}")
    else:
        log("Merge: 0 new records")
    return len(new_lines)


def main() -> None:
    log(f"Watching new NIM shards for run {RUN_ID}")
    log(f"Models: {MODELS}")
    log(f"Targets: full={TARGETS['full']}, ctrl2={TARGETS['control2']}")

    prev: dict = {}
    stable_rounds: dict = {}

    while True:
        time.sleep(300)  # check every 5 min

        curr = {}
        lines = []
        all_done = True

        for m in MODELS:
            for mode in ("full", "control2"):
                key = (m, mode)
                n = count_shard(shard_path(m, mode))
                target = TARGETS[mode]
                curr[key] = n
                pct = 100 * n / target
                if n < target:
                    all_done = False
                changed = n != prev.get(key, -1)
                if changed:
                    stable_rounds[key] = 0
                else:
                    stable_rounds[key] = stable_rounds.get(key, 0) + 1
                lines.append(f"{m}/{mode}={n}/{target}({pct:.0f}%,stbl={stable_rounds[key]})")

        log("PROGRESS: " + "  ".join(lines))
        prev = curr.copy()

        # Done if all at target, OR all stable for 6 rounds (30 min stable = truly stalled)
        all_stable = all(stable_rounds.get(k, 0) >= 6 for k in curr)

        if all_done or all_stable:
            log(f"All shards done or stable (done={all_done}, stable={all_stable})")
            break

    # Merge
    log("\n" + "=" * 60)
    log("MERGING ALL NEW NIM SHARDS INTO MAIN RESULTS")
    log("=" * 60)
    all_shards = [shard_path(m, mode) for m in MODELS for mode in ("full", "control2")]
    merge_into_main(all_shards)

    # Analysis
    log("\nRunning final analysis...")
    for attempt in range(1, 4):
        r = subprocess.run(
            [sys.executable, "scripts/analyze_experiment_9.py", "--run-id", RUN_ID],
            cwd=CWD,
        )
        if r.returncode == 0:
            log("Analysis: SUCCESS")
            break
        log(f"Analysis: exit={r.returncode} (attempt {attempt}/3)")
        if attempt < 3:
            time.sleep(15)

    # Summary
    log("\n" + "=" * 60)
    log("COMPLETE")
    for m in MODELS:
        fn = count_shard(shard_path(m, "full"))
        cn = count_shard(shard_path(m, "control2"))
        log(f"  {m:<25}: full={fn}/{TARGETS['full']}  ctrl2={cn}/{TARGETS['control2']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
