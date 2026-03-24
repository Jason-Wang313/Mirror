"""
Crash-safe temperature ablation for gemma-3-27b and kimi-k2.

Every record is saved to JSONL immediately with fsync.
Even if the script crashes, all completed records are preserved.

Usage: python scripts/temp_ablation_crash_safe.py
"""
import json, os, re, random, time, threading, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Setup ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
TEMPERATURE = 0.7
NIM_URL = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"

# Collect ALL NIM keys for maximum throughput
NIM_KEYS = []
base = os.environ.get("NVIDIA_NIM_API_KEY", "")
if base: NIM_KEYS.append(base)
for i in range(2, 19):
    k = os.environ.get(f"NVIDIA_NIM_API_KEY_{i}", "")
    if k: NIM_KEYS.append(k)
print(f"[INIT] {len(NIM_KEYS)} NIM API keys loaded")

_lock = threading.Lock()
_key_idx = [0]
def next_key():
    with _lock:
        _key_idx[0] = (_key_idx[0] + 1) % len(NIM_KEYS)
        return NIM_KEYS[_key_idx[0]]

MODELS = {
    "gemma-3-27b": "google/gemma-3-27b-it",
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
}
CONCURRENCY = 48  # High concurrency with many keys

# ── Output files (append-mode JSONL with fsync) ─────────────────────────────
OUT_DIR = ROOT / "paper" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP1_OUT = OUT_DIR / "temp_ablation_exp1_t07.jsonl"
EXP9_OUT = OUT_DIR / "temp_ablation_exp9_t07.jsonl"

_write_lock = threading.Lock()
def save_record(path, record):
    """Append one JSON record and fsync immediately."""
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

# ── API call with retries ────────────────────────────────────────────────────
def call_nim(model_id, prompt, max_tokens=512, retries=5):
    for attempt in range(retries):
        key = next_key()
        data = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }).encode()
        req = urllib.request.Request(NIM_URL, data=data, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(0.3 * (attempt + 1))
                continue
            if e.code == 403:
                time.sleep(0.2)
                continue
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                return None
    return None

# ── Answer matching ──────────────────────────────────────────────────────────
def simple_match(response, expected, answer_type="short_text"):
    if not response or not expected: return False
    r = str(response).strip().lower()
    e = str(expected).strip().lower()
    if answer_type == "multiple_choice": return e in r[:30]
    if answer_type == "boolean": return e in r[:60]
    if answer_type == "exact_numeric":
        # Extract numbers from response
        nums = re.findall(r'-?\d+\.?\d*', r)
        return e in nums or any(e in n for n in nums)
    return e in r

# ── Load data ────────────────────────────────────────────────────────────────
print("[INIT] Loading questions...")
questions = []
with open(ROOT / "data" / "questions.jsonl") as f:
    for line in f: questions.append(json.loads(line))
by_domain = {}
for q in questions: by_domain.setdefault(q.get("domain", "?"), []).append(q)
random.seed(42)
sampled = []
for d, qs in by_domain.items():
    random.shuffle(qs)
    sampled.extend(qs[:25])
sampled = sampled[:200]
print(f"[INIT] {len(sampled)} Exp1 questions sampled")

print("[INIT] Loading Exp9 tasks...")
exp9_tasks = []
with open(ROOT / "data" / "exp9_tasks.jsonl") as f:
    for line in f:
        t = json.loads(line)
        if t.get("circularity_free", False):
            exp9_tasks.append(t)
random.seed(42)
random.shuffle(exp9_tasks)
exp9_tasks = exp9_tasks[:100]
print(f"[INIT] {len(exp9_tasks)} Exp9 tasks loaded")

# Load domain accuracy from Exp1 at t=0 (for C4 routing decisions)
exp1_acc_files = sorted(ROOT.glob("data/results/exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
exp1_acc = {}
for fp in exp1_acc_files:
    with open(fp) as f:
        exp1_acc.update(json.load(f))
print(f"[INIT] Loaded t=0 accuracy for {len(exp1_acc)} models")

# ── Check for already-completed records (crash recovery) ────────────────────
def load_existing(path):
    """Load already-saved records to skip them on restart."""
    existing = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r.get("model", ""), r.get("question_id", r.get("task_id", "")), r.get("channel", r.get("condition", "")))
                    existing.add(key)
                except: pass
    return existing

exp1_done = load_existing(EXP1_OUT)
exp9_done = load_existing(EXP9_OUT)
print(f"[RECOVERY] {len(exp1_done)} Exp1 records already saved, {len(exp9_done)} Exp9 records already saved")

# ── Experiment 1: Wagering + Natural ─────────────────────────────────────────
def run_exp1_question(model_slug, model_id, q):
    """Run one question through Ch1 (wagering) and Ch5 (natural). Save immediately."""
    qtext = q.get("question_text", q.get("question", ""))
    correct = q.get("correct_answer", q.get("answer", ""))
    atype = q.get("answer_type", "short_text")
    domain = q.get("domain", "?")
    qid = q.get("question_id", qtext[:40])

    results = []

    # Channel 1 (wagering)
    key1 = (model_slug, qid, 1)
    if key1 not in exp1_done:
        prompt = f"{qtext}\n\nAfter answering, place a bet on your answer: 1-10 points.\nIf correct, you gain the points. If wrong, you lose them.\nANSWER: [answer]\nBET: [1-10]"
        resp = call_nim(model_id, prompt)
        if resp:
            am = re.search(r"ANSWER:\s*(.+?)(?:\n|BET:|$)", resp, re.IGNORECASE | re.DOTALL)
            ans = am.group(1).strip() if am else resp[:100]
            bm = re.search(r"BET:\s*(\d+)", resp)
            bet = int(bm.group(1)) if bm else None
            ok = simple_match(ans, correct, atype)
            rec = {"model": model_slug, "question_id": qid, "domain": domain,
                   "channel": 1, "answer_correct": ok, "bet_size": bet,
                   "temperature": TEMPERATURE, "raw_answer": ans[:200]}
            save_record(EXP1_OUT, rec)
            results.append(rec)

    # Channel 5 (natural)
    key5 = (model_slug, qid, 5)
    if key5 not in exp1_done:
        resp = call_nim(model_id, qtext)
        if resp:
            ok = simple_match(resp, correct, atype)
            rec = {"model": model_slug, "question_id": qid, "domain": domain,
                   "channel": 5, "answer_correct": ok, "bet_size": None,
                   "temperature": TEMPERATURE, "raw_answer": resp[:200]}
            save_record(EXP1_OUT, rec)
            results.append(rec)

    return results

# ── Experiment 9: C1 + C4 ───────────────────────────────────────────────────
def run_exp9_task(model_slug, model_id, task, domain_acc):
    """Run one Exp9 task through C1 and C4. Save immediately."""
    domain_a = task.get("domain_a", "")
    part1 = task.get("part1_text", task.get("task_text", ""))
    correct_a = task.get("correct_answer_a", "")
    at_a = task.get("answer_type_a", "short_text")
    acc_a = domain_acc.get(domain_a, 0.5)
    tid = task.get("task_id", "")

    results = []

    # C1: uninformed
    key1 = (model_slug, tid, 1)
    if key1 not in exp9_done:
        resp = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:", max_tokens=256)
        ok = simple_match(resp or "", correct_a, at_a)
        rec = {"model": model_slug, "task_id": tid, "domain": domain_a,
               "condition": 1, "correct": ok, "decision": "proceed",
               "externally_routed": False, "temperature": TEMPERATURE}
        save_record(EXP9_OUT, rec)
        results.append(rec)

    # C4: constrained routing
    key4 = (model_slug, tid, 4)
    if key4 not in exp9_done:
        if acc_a < 0.50:
            # Route externally (model is weak in this domain)
            rec = {"model": model_slug, "task_id": tid, "domain": domain_a,
                   "condition": 4, "correct": True, "decision": "defer",
                   "externally_routed": True, "temperature": TEMPERATURE}
            save_record(EXP9_OUT, rec)
            results.append(rec)
        else:
            resp = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:", max_tokens=256)
            ok = simple_match(resp or "", correct_a, at_a)
            rec = {"model": model_slug, "task_id": tid, "domain": domain_a,
                   "condition": 4, "correct": ok, "decision": "proceed",
                   "externally_routed": False, "temperature": TEMPERATURE}
            save_record(EXP9_OUT, rec)
            results.append(rec)

    return results

# ── Main execution ───────────────────────────────────────────────────────────
def run_model(model_slug, model_id):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{model_slug}] Starting temperature ablation (t={TEMPERATURE})")
    print(f"{'='*60}")

    # ── Exp1 ──
    print(f"[{model_slug}] Exp1: {len(sampled)} questions × 2 channels...")
    exp1_results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(run_exp1_question, model_slug, model_id, q): q for q in sampled}
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res: exp1_results.extend(res)
            done += 1
            if done % 50 == 0:
                ch1 = [r for r in exp1_results if r["channel"] == 1]
                ch5 = [r for r in exp1_results if r["channel"] == 5]
                w = sum(1 for r in ch1 if r["answer_correct"]) / max(len(ch1), 1)
                n = sum(1 for r in ch5 if r["answer_correct"]) / max(len(ch5), 1)
                print(f"  [{model_slug}] Exp1 progress: {done}/{len(sampled)} | wag={w:.3f} nat={n:.3f} gap={abs(w-n):.3f}")

    ch1 = [r for r in exp1_results if r["channel"] == 1]
    ch5 = [r for r in exp1_results if r["channel"] == 5]
    wag = sum(1 for r in ch1 if r["answer_correct"]) / max(len(ch1), 1)
    nat = sum(1 for r in ch5 if r["answer_correct"]) / max(len(ch5), 1)
    gap = abs(wag - nat)
    print(f"  [{model_slug}] Exp1 DONE ({time.time()-t0:.0f}s): wag={wag:.3f}({len(ch1)}) nat={nat:.3f}({len(ch5)}) gap={gap:.3f}")

    # ── Exp9 ──
    model_domain_acc = {}
    if model_slug in exp1_acc:
        for d, m in exp1_acc[model_slug].items():
            if isinstance(m, dict):
                model_domain_acc[d] = m.get("natural_acc", 0.5)

    print(f"[{model_slug}] Exp9: {len(exp9_tasks)} tasks × 2 conditions... (domain_acc for {len(model_domain_acc)} domains)")
    exp9_results = []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(run_exp9_task, model_slug, model_id, t, model_domain_acc): t for t in exp9_tasks}
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res: exp9_results.extend(res)
            done += 1
            if done % 25 == 0:
                c1 = [r for r in exp9_results if r["condition"] == 1]
                c4 = [r for r in exp9_results if r["condition"] == 4]
                cfr1 = sum(1 for r in c1 if not r["correct"]) / max(len(c1), 1)
                cfr4 = sum(1 for r in c4 if not r.get("externally_routed", False) and not r["correct"]) / max(len(c4), 1)
                print(f"  [{model_slug}] Exp9 progress: {done}/{len(exp9_tasks)} | CFR_C1={cfr1:.3f} CFR_C4={cfr4:.3f}")

    c1 = [r for r in exp9_results if r["condition"] == 1]
    c4 = [r for r in exp9_results if r["condition"] == 4]
    cfr1 = sum(1 for r in c1 if not r["correct"]) / max(len(c1), 1)
    cfr4 = sum(1 for r in c4 if not r.get("externally_routed", False) and not r["correct"]) / max(len(c4), 1)
    red = ((cfr1 - cfr4) / cfr1 * 100) if cfr1 > 0 else 0
    print(f"  [{model_slug}] Exp9 DONE ({time.time()-t1:.0f}s): CFR_C1={cfr1:.3f} CFR_C4={cfr4:.3f} reduction={red:.1f}%")

    total_time = time.time() - t0
    print(f"\n[{model_slug}] COMPLETE in {total_time:.0f}s")
    return {
        "model": model_slug,
        "gap_t07": gap, "wag_acc_t07": wag, "nat_acc_t07": nat,
        "cfr_c1_t07": cfr1, "cfr_c4_t07": cfr4, "reduction_t07": red,
        "n_exp1_ch1": len(ch1), "n_exp1_ch5": len(ch5),
        "n_exp9_c1": len(c1), "n_exp9_c4": len(c4),
    }


def main():
    print(f"\n{'#'*60}")
    print(f"# MIRROR TEMPERATURE ABLATION (crash-safe)")
    print(f"# Models: {list(MODELS.keys())}")
    print(f"# Temperature: {TEMPERATURE}")
    print(f"# Concurrency: {CONCURRENCY}")
    print(f"# Output: {EXP1_OUT}")
    print(f"#         {EXP9_OUT}")
    print(f"{'#'*60}\n")

    # Run both models (sequentially to avoid key contention)
    summaries = []
    for slug, model_id in MODELS.items():
        s = run_model(slug, model_id)
        summaries.append(s)

    # ── Merge with existing 3-model data ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("MERGING with existing 3-model temperature ablation data...")
    existing_path = ROOT / "paper" / "supplementary" / "temperature_ablation.json"
    with open(existing_path) as f:
        existing = json.load(f)

    # Read back all saved records
    new_exp1 = []
    if EXP1_OUT.exists():
        with open(EXP1_OUT) as f:
            for line in f:
                try: new_exp1.append(json.loads(line))
                except: pass

    new_exp9 = []
    if EXP9_OUT.exists():
        with open(EXP9_OUT) as f:
            for line in f:
                try: new_exp9.append(json.loads(line))
                except: pass

    merged = {
        "temperature": TEMPERATURE,
        "models": existing.get("models", []) + list(MODELS.keys()),
        "exp1_records": existing.get("exp1_records", []) + new_exp1,
        "exp9_records": existing.get("exp9_records", []) + new_exp9,
    }

    # Compute per-model summaries
    merged["exp1_results"] = {}
    merged["exp9_results"] = {}
    for model_slug in merged["models"]:
        e1 = [r for r in merged["exp1_records"] if r["model"] == model_slug]
        ch1 = [r for r in e1 if r.get("channel") == 1]
        ch5 = [r for r in e1 if r.get("channel") == 5]
        if ch1 and ch5:
            w = sum(1 for r in ch1 if r["answer_correct"]) / len(ch1)
            n = sum(1 for r in ch5 if r["answer_correct"]) / len(ch5)
            merged["exp1_results"][model_slug] = {
                "wagering_acc": round(w, 3), "natural_acc": round(n, 3),
                "mirror_gap": round(abs(w - n), 3), "n_ch1": len(ch1), "n_ch5": len(ch5),
            }

        e9 = [r for r in merged["exp9_records"] if r["model"] == model_slug]
        c1 = [r for r in e9 if r.get("condition") == 1]
        c4 = [r for r in e9 if r.get("condition") == 4]
        if c1 and c4:
            f1 = sum(1 for r in c1 if not r["correct"]) / len(c1) if c1 else 0
            f4 = sum(1 for r in c4 if not r.get("externally_routed", False) and not r["correct"]) / len(c4) if c4 else 0
            rd = ((f1 - f4) / f1 * 100) if f1 > 0 else 0
            merged["exp9_results"][model_slug] = {
                "cfr_c1": round(f1, 3), "cfr_c4": round(f4, 3),
                "reduction_pct": round(rd, 1), "n_c1": len(c1), "n_c4": len(c4),
            }

    # Save merged
    out_path = OUT_DIR / "temperature_ablation_5models.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged data: {out_path}")

    # ── Print final summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEMPERATURE ABLATION SUMMARY (t=0.7)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Gap(t=.7)':>10} {'CFR_C1':>8} {'CFR_C4':>8} {'Reduct':>8}")
    print("-" * 56)
    for ms in merged["models"]:
        e1r = merged["exp1_results"].get(ms, {})
        e9r = merged["exp9_results"].get(ms, {})
        g = e1r.get("mirror_gap", "?")
        f1 = e9r.get("cfr_c1", "?")
        f4 = e9r.get("cfr_c4", "?")
        rd = e9r.get("reduction_pct", "?")
        print(f"  {ms:<20} {str(g):>10} {str(f1):>8} {str(f4):>8} {str(rd)+'%' if isinstance(rd, (int,float)) else '?':>8}")

    print(f"\nDONE! All data saved to:")
    print(f"  Exp1 records: {EXP1_OUT}")
    print(f"  Exp9 records: {EXP9_OUT}")
    print(f"  Merged JSON:  {out_path}")


if __name__ == "__main__":
    main()
