"""
Full Exp9 run for the 5 new NIM models.

MAX SPEED: runs full mode AND control2 simultaneously per model = 10 parallel processes.
All writes go to per-model shard files (crash-safe, fsynced). --resume skips done trials.
On any crash: re-run this script — it picks up exactly where it left off.

DO NOT touch deepseek-r1 or mistral-large (retrying concurrently).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

RUN_ID = "20260312T140842"
RESULTS_DIR = Path("data/results")
CWD = str(Path(__file__).parent.parent)
LOG_FILE = RESULTS_DIR / "new_nim_launch_log.txt"

# Per-model, per-mode concurrency tuned for NIM.
# Full + ctrl2 run simultaneously → effective NIM load is sum of both.
# Total across all 10 processes: ~120 concurrent (safe; original fast-model run was 192)
MODEL_CONFIG: dict[str, dict] = {
    "llama-3.3-70b":  {"full": 14, "control2": 10},  # 70B instruct
    "kimi-k2":        {"full": 10, "control2":  6},  # unknown size, moderate
    "phi-4":          {"full": 18, "control2": 12},  # phi-4-mini, small/fast
    "gemma-3-27b":    {"full": 12, "control2":  8},  # 27B
    "qwen3-235b-nim": {"full":  6, "control2":  4},  # 397B large — conservative
}

NEW_MODELS = list(MODEL_CONFIG.keys())


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[NIM {ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def shard_path(model: str, mode: str) -> Path:
    slug = model.replace(".", "-").replace("/", "-")
    return RESULTS_DIR / f"exp9_{RUN_ID}_{slug}_{mode}_shard.jsonl"


def count_shard(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def run_model_mode(
    model: str,
    mode: str,
    concurrency: int,
    results: dict,
    lock: threading.Lock,
    attempts: int = 5,
) -> None:
    shard = shard_path(model, mode)
    cmd = [
        sys.executable, "scripts/run_experiment_9.py",
        "--mode", mode,
        "--run-id", RUN_ID,
        "--models", model,
        "--output-file", str(shard),
        "--resume",
        "--concurrency", str(concurrency),
    ]
    with lock:
        log(f"  [{mode}] {model}: starting c={concurrency} → {shard.name}")

    for attempt in range(1, attempts + 1):
        r = subprocess.run(cmd, cwd=CWD)
        if r.returncode == 0:
            n = count_shard(shard)
            with lock:
                log(f"  [{mode}] {model}: DONE ({n} records)")
            results[(model, mode)] = True
            return
        with lock:
            log(f"  [{mode}] {model}: attempt {attempt}/{attempts} exit={r.returncode}")
        if attempt < attempts:
            time.sleep(20)

    with lock:
        log(f"  [{mode}] {model}: FAILED after {attempts} attempts")
    results[(model, mode)] = False


def merge_into_main(shards: list[Path]) -> int:
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
    new_lines: list[str] = []
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
        log(f"Merge complete: +{len(new_lines)} records")
    else:
        log("Merge: 0 new records")
    return len(new_lines)


def monitor_progress(stop_event: threading.Event) -> None:
    """Background thread: log shard counts every 5 minutes."""
    while not stop_event.is_set():
        time.sleep(300)
        if stop_event.is_set():
            break
        lines = []
        total = 0
        for model in NEW_MODELS:
            for mode in ("full", "control2"):
                n = count_shard(shard_path(model, mode))
                total += n
                lines.append(f"{model}/{mode}={n}")
        log(f"PROGRESS [{total} total]: " + "  ".join(lines))


def main() -> None:
    log("=" * 60)
    log(f"NEW NIM MODELS EXP9  run_id={RUN_ID}")
    log(f"Models ({len(NEW_MODELS)}): {NEW_MODELS}")
    log("MAX SPEED: full + control2 running simultaneously per model")
    log(f"Total NIM threads: {len(NEW_MODELS) * 2} = {len(NEW_MODELS)} full + {len(NEW_MODELS)} ctrl2")
    total_c = sum(v["full"] + v["control2"] for v in MODEL_CONFIG.values())
    log(f"Total NIM concurrent requests: ~{total_c} (+ 8 from mistral retry)")
    log("=" * 60)

    results: dict = {}
    lock = threading.Lock()

    # Start background progress monitor
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_progress, args=(stop_monitor,), daemon=True)
    monitor_thread.start()

    # Launch all 10 threads simultaneously (full + ctrl2 per model)
    threads = []
    for model in NEW_MODELS:
        for mode in ("full", "control2"):
            c = MODEL_CONFIG[model][mode]
            t = threading.Thread(
                target=run_model_mode,
                args=(model, mode, c, results, lock),
                daemon=True,
            )
            threads.append(t)

    log(f"\nLaunching {len(threads)} threads simultaneously...")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stop_monitor.set()

    # ── Merge all shards ────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("MERGING ALL NEW NIM SHARDS INTO MAIN RESULTS")
    log("=" * 60)
    all_shards = [shard_path(m, mode)
                  for m in NEW_MODELS
                  for mode in ("full", "control2")]
    merge_into_main([s for s in all_shards if s.exists()])

    # ── Re-run analysis ─────────────────────────────────────────────
    log("\nRunning final analysis...")
    for attempt in range(1, 4):
        r = subprocess.run(
            [sys.executable, "scripts/analyze_experiment_9.py", "--run-id", RUN_ID],
            cwd=CWD,
        )
        if r.returncode == 0:
            log("Analysis: SUCCESS")
            break
        log(f"Analysis: exit={r.returncode}")
        if attempt < 3:
            time.sleep(15)

    # ── Summary ─────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("COMPLETE")
    log("=" * 60)
    for model in NEW_MODELS:
        full_ok = "✓" if results.get((model, "full")) else "✗"
        ctrl2_ok = "✓" if results.get((model, "control2")) else "✗"
        fn = count_shard(shard_path(model, "full"))
        cn = count_shard(shard_path(model, "control2"))
        log(f"  {model:<25}: full={full_ok}({fn})  ctrl2={ctrl2_ok}({cn})")
    log(f"\nResults: data/results/exp9_{RUN_ID}_results.jsonl")
    log(f"Analysis: data/results/exp9_{RUN_ID}_analysis/")
    log("=" * 60)


if __name__ == "__main__":
    main()
