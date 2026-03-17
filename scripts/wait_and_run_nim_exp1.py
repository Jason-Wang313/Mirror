"""
Polls NIM until it responds, then launches exp1 fast for the 5 NIM models.
Also restarts mistral-large retry (also NIM).
Run in background after NIM quota resets.
"""
import asyncio, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

CWD = str(Path(__file__).parent.parent)
NIM_MODELS = "phi-4,command-r-plus,kimi-k2,gemma-3-27b,qwen3-235b-nim"
RUN_ID = "20260312T140842"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


async def nim_alive() -> bool:
    try:
        from mirror.api.providers.nvidia_nim import NVIDIANIMProvider
        p = NVIDIANIMProvider()
        r = await asyncio.wait_for(
            p.complete(model_id="microsoft/phi-4-mini-instruct",
                       messages=[{"role": "user", "content": "2+2=?"}], max_tokens=5),
            timeout=20,
        )
        return bool(r.get("content"))
    except Exception as e:
        return False


def main():
    log("Waiting for NIM to recover...")
    while True:
        alive = asyncio.run(nim_alive())
        if alive:
            log("NIM is ALIVE — launching exp1 fast for NIM models!")
            break
        log("NIM still unresponsive — checking again in 10 min")
        time.sleep(600)

    # Launch exp1 fast for 5 NIM models
    log(f"Starting: python scripts/run_exp1_fast.py --models {NIM_MODELS}")
    subprocess.run(
        [sys.executable, "scripts/run_exp1_fast.py", "--models", NIM_MODELS, "--resume"],
        cwd=CWD,
    )
    log("Exp1 NIM models done.")

    # Restart mistral retry if incomplete
    from pathlib import Path
    import json
    shard = Path("data/results") / f"exp9_{RUN_ID}_mistral-large_retry_shard.jsonl"
    n = sum(1 for l in shard.read_text(encoding="utf-8").splitlines()
            if l.strip()) if shard.exists() else 0
    if n < 1633:
        log(f"Mistral retry incomplete ({n}/1633) — restarting...")
        subprocess.run([
            sys.executable, "scripts/retry_failed_exp9.py",
            "--run-id", RUN_ID, "--models", "mistral-large",
            "--concurrency", "8",
            "--output-file", str(shard),
            "--no-merge", "--resume",
        ], cwd=CWD)
        log("Merging mistral retry...")
        subprocess.run([
            sys.executable, "scripts/launch_retry_exp9.py",
            "--run-id", RUN_ID, "--merge-only",
        ], cwd=CWD)
        log("Re-running analysis...")
        subprocess.run([
            sys.executable, "scripts/analyze_experiment_9.py", "--run-id", RUN_ID,
        ], cwd=CWD)

    log("ALL DONE.")


if __name__ == "__main__":
    main()
