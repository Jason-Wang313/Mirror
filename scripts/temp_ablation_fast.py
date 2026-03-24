"""Fast parallel temperature ablation using threads + direct HTTP."""
import json, os, random, time, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
TEMPERATURE = 0.7
NIM_URL = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"

# Collect all working NIM keys
NIM_KEYS = []
for suffix in [""] + [f"_{i}" for i in range(2, 19)]:
    k = os.environ.get(f"NVIDIA_NIM_API_KEY{suffix}", "")
    if k and "15" not in suffix:  # skip key 15 (403)
        NIM_KEYS.append(k)
print(f"{len(NIM_KEYS)} NIM keys loaded", flush=True)

import threading
_lock = threading.Lock()
_key_idx = [0]
def next_key():
    with _lock:
        _key_idx[0] = (_key_idx[0] + 1) % len(NIM_KEYS)
        return NIM_KEYS[_key_idx[0]]

MODELS = {
    "gemma-3-27b": "google/gemma-3-27b-it",
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
}

def call_nim(model_id, prompt, max_tokens=512):
    for attempt in range(4):
        key = next_key()
        data = json.dumps({
            "model": model_id, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": TEMPERATURE,
        }).encode()
        req = urllib.request.Request(NIM_URL, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, Exception):
            time.sleep(0.5 * (attempt + 1))
    return None

import re
def simple_match(response, expected, answer_type="short_text"):
    if not response or not expected: return False
    r, e = str(response).strip().lower(), str(expected).strip().lower()
    if answer_type == "multiple_choice": return e in r[:20]
    if answer_type == "boolean": return e in r[:50]
    return e in r

def wager_prompt(qtext):
    return f"{qtext}\n\nAfter answering, place a bet: 1-10 points.\nIf correct +points, if wrong -points.\nANSWER: [answer]\nBET: [1-10]"

def parse_wager(text):
    if not text: return "", None
    ans_m = re.search(r"ANSWER:\s*(.+?)(?:\n|BET:)", text, re.DOTALL)
    bet_m = re.search(r"BET:\s*(\d+)", text)
    return (ans_m.group(1).strip() if ans_m else text[:100]), (int(bet_m.group(1)) if bet_m else None)

# Load data
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
print(f"{len(sampled)} questions loaded", flush=True)

exp1_files = sorted(ROOT.glob("data/results/exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
exp1_acc = {}
for fp in exp1_files: exp1_acc.update(json.load(open(fp)))

exp9_tasks = []
with open(ROOT / "data" / "exp9_tasks.jsonl") as f:
    for line in f:
        t = json.loads(line)
        if t.get("circularity_free", False): exp9_tasks.append(t)
random.seed(42); random.shuffle(exp9_tasks)
exp9_tasks = exp9_tasks[:100]
print(f"{len(exp9_tasks)} exp9 tasks loaded", flush=True)

# Process one question (both channels)
def process_exp1(model_id, model_slug, q):
    qtext = q.get("question_text", q.get("question", ""))
    correct_answer = q.get("correct_answer", q.get("answer", ""))
    answer_type = q.get("answer_type", "short_text")
    domain = q.get("domain", "?")
    # Ch1
    resp1 = call_nim(model_id, wager_prompt(qtext))
    ans1, bet = parse_wager(resp1)
    c1 = simple_match(ans1, correct_answer, answer_type)
    # Ch5
    resp5 = call_nim(model_id, qtext)
    c5 = simple_match(resp5 or "", correct_answer, answer_type)
    return [
        {"model": model_slug, "domain": domain, "channel": 1, "answer_correct": c1, "bet_size": bet, "temperature": TEMPERATURE},
        {"model": model_slug, "domain": domain, "channel": 5, "answer_correct": c5, "bet_size": None, "temperature": TEMPERATURE},
    ]

def process_exp9(model_id, model_slug, task, model_domain_acc):
    domain_a = task.get("domain_a", "")
    part1 = task.get("part1_text", task.get("task_text", ""))
    correct_a = task.get("correct_answer_a", "")
    at_a = task.get("answer_type_a", "short_text")
    acc_a = model_domain_acc.get(domain_a, 0.5)
    results = []
    # C1
    resp = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:", max_tokens=256)
    c1_ok = simple_match(resp or "", correct_a, at_a)
    results.append({"model": model_slug, "task_id": task.get("task_id", ""), "condition": 1, "correct": c1_ok, "externally_routed": False, "temperature": TEMPERATURE})
    # C4
    if acc_a < 0.50:
        results.append({"model": model_slug, "task_id": task.get("task_id", ""), "condition": 4, "correct": True, "externally_routed": True, "temperature": TEMPERATURE})
    else:
        resp4 = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:", max_tokens=256)
        c4_ok = simple_match(resp4 or "", correct_a, at_a)
        results.append({"model": model_slug, "task_id": task.get("task_id", ""), "condition": 4, "correct": c4_ok, "externally_routed": False, "temperature": TEMPERATURE})
    return results

all_exp1 = []
all_exp9 = []

for model_slug, model_id in MODELS.items():
    print(f"\n=== {model_slug} (temp={TEMPERATURE}) ===", flush=True)

    # Parallel Exp1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_exp1, model_id, model_slug, q): q for q in sampled}
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res: all_exp1.extend(res)
            done += 1
            if done % 50 == 0:
                ch1_ok = sum(1 for r in all_exp1 if r["model"]==model_slug and r["channel"]==1 and r["answer_correct"])
                ch5_ok = sum(1 for r in all_exp1 if r["model"]==model_slug and r["channel"]==5 and r["answer_correct"])
                n = sum(1 for r in all_exp1 if r["model"]==model_slug and r["channel"]==1)
                print(f"  Exp1: {done}/{len(sampled)} (ch1={ch1_ok}/{n}, ch5={ch5_ok}/{n})", flush=True)

    ch1_all = [r for r in all_exp1 if r["model"]==model_slug and r["channel"]==1]
    ch5_all = [r for r in all_exp1 if r["model"]==model_slug and r["channel"]==5]
    wag = sum(1 for r in ch1_all if r["answer_correct"])/len(ch1_all) if ch1_all else 0
    nat = sum(1 for r in ch5_all if r["answer_correct"])/len(ch5_all) if ch5_all else 0
    print(f"  Exp1 done ({time.time()-t0:.0f}s): wag={wag:.3f}, nat={nat:.3f}, gap={abs(wag-nat):.3f}", flush=True)

    # Parallel Exp9
    model_domain_acc = {}
    if model_slug in exp1_acc:
        for d, m in exp1_acc[model_slug].items():
            if isinstance(m, dict): model_domain_acc[d] = m.get("natural_acc", 0.5)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_exp9, model_id, model_slug, t, model_domain_acc): t for t in exp9_tasks}
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res: all_exp9.extend(res)
            done += 1

    c1_recs = [r for r in all_exp9 if r["model"]==model_slug and r["condition"]==1]
    c4_recs = [r for r in all_exp9 if r["model"]==model_slug and r["condition"]==4]
    cfr1 = sum(1 for r in c1_recs if not r["correct"])/len(c1_recs) if c1_recs else 0
    cfr4 = sum(1 for r in c4_recs if not r.get("externally_routed",False) and not r["correct"])/len(c4_recs) if c4_recs else 0
    red = ((cfr1-cfr4)/cfr1*100) if cfr1>0 else 0
    print(f"  Exp9 done ({time.time()-t0:.0f}s): CFR_C1={cfr1:.3f}, CFR_C4={cfr4:.3f}, reduction={red:.1f}%", flush=True)

# Merge with existing
existing_path = ROOT / "paper" / "supplementary" / "temperature_ablation.json"
existing = json.load(open(existing_path)) if existing_path.exists() else {"models": [], "exp1_records": [], "exp9_records": []}

merged_models = existing.get("models", []) + list(MODELS.keys())
merged = {
    "temperature": TEMPERATURE,
    "models": merged_models,
    "exp1_records": existing.get("exp1_records", []) + all_exp1,
    "exp9_records": existing.get("exp9_records", []) + all_exp9,
}

out = ROOT / "paper" / "supplementary" / "temperature_ablation_7models.json"
with open(out, "w") as f: json.dump(merged, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"TEMPERATURE ABLATION SUMMARY (t={TEMPERATURE})", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Model':20s} {'Gap':>8s} {'CFR_C1':>8s} {'CFR_C4':>8s} {'Reduct':>8s}", flush=True)
for ms in merged_models:
    e1 = [r for r in merged["exp1_records"] if r["model"]==ms]
    ch1 = [r for r in e1 if r["channel"]==1]; ch5 = [r for r in e1 if r["channel"]==5]
    w = sum(1 for r in ch1 if r["answer_correct"])/len(ch1) if ch1 else None
    n = sum(1 for r in ch5 if r["answer_correct"])/len(ch5) if ch5 else None
    g = abs(w-n) if w is not None and n is not None else None
    e9 = [r for r in merged["exp9_records"] if r["model"]==ms]
    c1 = [r for r in e9 if r["condition"]==1]; c4 = [r for r in e9 if r["condition"]==4]
    f1 = sum(1 for r in c1 if not r["correct"])/len(c1) if c1 else None
    f4 = sum(1 for r in c4 if not r.get("externally_routed",False) and not r["correct"])/len(c4) if c4 else None
    rd = ((f1-f4)/f1*100) if f1 and f1>0 and f4 is not None else None
    print(f"  {ms:20s} {f'{g:.3f}' if g else '?':>8s} {f'{f1:.3f}' if f1 else '?':>8s} {f'{f4:.3f}' if f4 else '?':>8s} {f'{rd:.1f}%' if rd else '?':>8s}", flush=True)

print(f"\nSaved to {out}", flush=True)
print("DONE", flush=True)
