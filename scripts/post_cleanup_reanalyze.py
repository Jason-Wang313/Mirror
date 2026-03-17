"""
Post-cleanup reanalysis: re-dedup Exp4 and rerun Exp4+Exp6 analyses
after the completion runs for llama-3.3-70b, gemini-2.5-pro, and Exp6 backfill finish.

Run: python scripts/post_cleanup_reanalyze.py
"""
import json, subprocess, sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path("data/results")
RUN_ID  = "20260314T135731"

def check_completion():
    """Return (llama33_done, gemini_done, exp6_done)."""
    a_file = RESULTS / f"exp4_v2_{RUN_ID}_condition_a_results.jsonl"
    b_file = RESULTS / f"exp4_v2_{RUN_ID}_condition_b_results.jsonl"

    def unique_per_model(path):
        counts = defaultdict(set)
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                counts[r["model"]].add(r["trial_id"])
        return {m: len(ids) for m, ids in counts.items()}

    ua = unique_per_model(a_file)
    ub = unique_per_model(b_file)
    llama33_done  = ua.get("llama-3.3-70b", 0) >= 320 and ub.get("llama-3.3-70b", 0) >= 320
    gemini_done   = ua.get("gemini-2.5-pro", 0) >= 320 and ub.get("gemini-2.5-pro", 0) >= 320

    # Exp6: check 4 backfill models have 338 each in master file
    master = RESULTS / "exp6_master_results.jsonl"
    exp6_counts = defaultdict(int)
    with open(master) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            exp6_counts[r["model"]] += 1
    backfill = ["gemma-3-12b", "llama-3.2-3b", "mixtral-8x22b", "qwen3-next-80b"]
    exp6_done = all(exp6_counts.get(m, 0) >= 338 for m in backfill)

    return llama33_done, gemini_done, exp6_done


def rededup():
    """Re-deduplicate Exp4 files now that completion runs are done."""
    for cond in ("condition_a", "condition_b"):
        infile  = RESULTS / f"exp4_v2_{RUN_ID}_{cond}_results.jsonl"
        outfile = RESULTS / f"exp4_v2_deduped_{cond}_results.jsonl"
        seen = {}
        with open(infile) as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r["model"] == "command-r-plus":
                    continue
                seen[(r["model"], r["trial_id"])] = r
        with open(outfile, "w") as f:
            for r in seen.values():
                f.write(json.dumps(r) + "\n")
        print(f"  Re-deduped {cond}: {len(seen)} records -> {outfile.name}")


def run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


if __name__ == "__main__":
    llama33_done, gemini_done, exp6_done = check_completion()
    print(f"llama-3.3-70b complete: {llama33_done}")
    print(f"gemini-2.5-pro complete: {gemini_done}")
    print(f"Exp6 backfill complete:  {exp6_done}")

    if not (llama33_done and gemini_done):
        print("\nExp4 completion runs still in progress. Re-deduplicating with current data...")
    rededup()

    rc = run(f"python scripts/analyze_exp4_expanded.py --run-id {RUN_ID}")
    if rc != 0:
        print("WARNING: Exp4 analysis failed — check script args")

    if exp6_done:
        rc = run("python scripts/analyze_experiment_6.py --latest")
        if rc != 0:
            print("WARNING: Exp6 analysis failed")
    else:
        print("\nExp6 backfill still running — skipping Exp6 reanalysis")

    print("\nDone. Check docs/final_status_report.md for updated tables.")
