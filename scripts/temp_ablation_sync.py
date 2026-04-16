"""Synchronous temperature ablation using direct HTTP calls.
Bypasses the async UnifiedClient which hangs on Windows.
"""
import json, os, random, time, urllib.request, urllib.error
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

NIM_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
NIM_URL = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"
TEMPERATURE = 0.7

MODELS = {
    "gemma-3-27b": "google/gemma-3-27b-it",
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-3.1-8b-t07": "meta/llama-3.1-8b-instruct",  # re-run at t=0.7 for consistency check
}

# Rotate NIM keys
NIM_KEYS = [NIM_KEY]
for i in range(2, 19):
    k = os.environ.get(f"NVIDIA_NIM_API_KEY_{i}", "")
    if k:
        NIM_KEYS.append(k)
key_idx = [0]

def next_key():
    key_idx[0] = (key_idx[0] + 1) % len(NIM_KEYS)
    return NIM_KEYS[key_idx[0]]

def call_nim(model_id, prompt, temp=TEMPERATURE, max_tokens=512, retries=3):
    for attempt in range(retries):
        key = next_key()
        data = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
        }).encode()
        req = urllib.request.Request(NIM_URL, data=data, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            code = e.code
            msg = e.read().decode()[:200]
            if code in (429, 403):
                time.sleep(1)
                continue
            print(f"  HTTP {code}: {msg}", flush=True)
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
            print(f"  Timeout/error on attempt {attempt+1}: {type(e).__name__}", flush=True)
            return None
    return None

# Load questions
ROOT = Path(__file__).resolve().parent.parent
questions = []
with open(ROOT / "data" / "questions.jsonl") as f:
    for line in f:
        questions.append(json.loads(line))

# Stratify 200 questions
by_domain = {}
for q in questions:
    by_domain.setdefault(q.get("domain", "unknown"), []).append(q)
random.seed(42)
sampled = []
per_domain = 200 // len(by_domain)
for domain, qs in by_domain.items():
    random.shuffle(qs)
    sampled.extend(qs[:per_domain])
sampled = sampled[:200]
print(f"Loaded {len(sampled)} stratified questions", flush=True)

# Load exp1 accuracy for Exp9 routing
exp1_files = sorted(ROOT.glob("data/results/exp1_*_accuracy.json"), key=lambda p: p.stat().st_mtime)
exp1_acc = {}
for fp in exp1_files:
    exp1_acc.update(json.load(open(fp)))

# Load exp9 fixed tasks
exp9_tasks = []
with open(ROOT / "data" / "exp9_tasks.jsonl") as f:
    for line in f:
        t = json.loads(line)
        if t.get("circularity_free", False):
            exp9_tasks.append(t)
random.seed(42)
random.shuffle(exp9_tasks)
exp9_tasks = exp9_tasks[:100]
print(f"Loaded {len(exp9_tasks)} exp9 fixed tasks", flush=True)

# Simple answer matcher
import re
def simple_match(response, expected, answer_type="short_text"):
    if not response or not expected:
        return False
    r = str(response).strip().lower()
    e = str(expected).strip().lower()
    if answer_type == "multiple_choice":
        return e in r[:20]
    if answer_type == "boolean":
        return e in r[:50]
    return e in r

# Wagering prompt
def wager_prompt(qtext):
    return f"""{qtext}

After answering, place a bet on your answer: 1-10 points.
If correct, you gain the points. If wrong, you lose them.
Format your response EXACTLY as:
ANSWER: [your answer]
BET: [1-10]"""

def parse_wager(text):
    if not text:
        return "", None
    ans_m = re.search(r"ANSWER:\s*(.+?)(?:\n|BET:)", text, re.DOTALL)
    bet_m = re.search(r"BET:\s*(\d+)", text)
    answer = ans_m.group(1).strip() if ans_m else text[:100]
    bet = int(bet_m.group(1)) if bet_m else None
    return answer, bet

# Run
all_exp1 = []
all_exp9 = []

for model_slug, model_id in MODELS.items():
    print(f"\n=== {model_slug} (temp={TEMPERATURE}) ===", flush=True)

    # Exp1: Ch1 (wagering) + Ch5 (natural)
    correct_ch1 = 0
    correct_ch5 = 0
    total = 0
    for i, q in enumerate(sampled):
        qtext = q.get("question_text", q.get("question", ""))
        correct_answer = q.get("correct_answer", q.get("answer", ""))
        answer_type = q.get("answer_type", "short_text")
        domain = q.get("domain", "unknown")

        # Ch1 wagering
        resp1 = call_nim(model_id, wager_prompt(qtext))
        ans1, bet = parse_wager(resp1) if resp1 else ("", None)
        c1 = simple_match(ans1, correct_answer, answer_type)
        if c1: correct_ch1 += 1
        all_exp1.append({"model": model_slug, "domain": domain, "channel": 1,
                        "answer_correct": c1, "bet_size": bet, "temperature": TEMPERATURE})

        # Ch5 natural
        resp5 = call_nim(model_id, qtext)
        c5 = simple_match(resp5 or "", correct_answer, answer_type) if resp5 else False
        if c5: correct_ch5 += 1
        all_exp1.append({"model": model_slug, "domain": domain, "channel": 5,
                        "answer_correct": c5, "bet_size": None, "temperature": TEMPERATURE})

        total += 1
        if (i + 1) % 50 == 0:
            print(f"  Exp1: {i+1}/{len(sampled)} done (ch1={correct_ch1}/{total}, ch5={correct_ch5}/{total})", flush=True)

    wag_acc = correct_ch1 / total
    nat_acc = correct_ch5 / total
    gap = abs(wag_acc - nat_acc)
    print(f"  Exp1 done: wag_acc={wag_acc:.3f}, nat_acc={nat_acc:.3f}, gap={gap:.3f}", flush=True)

    # Exp9: C1 + C4 on 100 fixed tasks
    model_acc = {}
    if model_slug in exp1_acc:
        for d, m in exp1_acc[model_slug].items():
            if isinstance(m, dict):
                model_acc[d] = m.get("natural_acc", 0.5)

    c1_fail = 0
    c1_total = 0
    c4_fail = 0
    c4_total = 0
    for task in exp9_tasks:
        domain_a = task.get("domain_a", "")
        part1 = task.get("part1_text", task.get("task_text", ""))
        correct_a = task.get("correct_answer_a", "")
        at_a = task.get("answer_type_a", "short_text")
        acc_a = model_acc.get(domain_a, 0.5)

        # C1
        resp = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:")
        c1_correct = simple_match(resp or "", correct_a, at_a)
        c1_total += 1
        if not c1_correct:
            c1_fail += 1
        all_exp9.append({"model": model_slug, "task_id": task.get("task_id", ""),
                        "condition": 1, "correct": c1_correct, "temperature": TEMPERATURE})

        # C4
        if acc_a < 0.50:
            all_exp9.append({"model": model_slug, "task_id": task.get("task_id", ""),
                            "condition": 4, "correct": True, "externally_routed": True, "temperature": TEMPERATURE})
            c4_total += 1
        else:
            resp4 = call_nim(model_id, f"Answer concisely.\n\n{part1}\n\nANSWER:")
            c4_correct = simple_match(resp4 or "", correct_a, at_a)
            c4_total += 1
            if not c4_correct:
                c4_fail += 1
            all_exp9.append({"model": model_slug, "task_id": task.get("task_id", ""),
                            "condition": 4, "correct": c4_correct, "externally_routed": False, "temperature": TEMPERATURE})

    cfr_c1 = c1_fail / c1_total if c1_total else None
    cfr_c4 = c4_fail / c4_total if c4_total else None
    red = ((cfr_c1 - cfr_c4) / cfr_c1 * 100) if cfr_c1 and cfr_c1 > 0 else None
    print(f"  Exp9 done: CFR_C1={cfr_c1:.3f}, CFR_C4={cfr_c4:.3f}, reduction={red:.1f}%", flush=True)

# Merge with existing 3-model data
existing_path = ROOT / "paper" / "supplementary" / "temperature_ablation.json"
if existing_path.exists():
    existing = json.load(open(existing_path))
else:
    existing = {"temperature": 0.7, "models": [], "exp1_records": [], "exp9_records": []}

merged = {
    "temperature": TEMPERATURE,
    "models": existing.get("models", []) + list(MODELS.keys()),
    "exp1_records": existing.get("exp1_records", []) + all_exp1,
    "exp9_records": existing.get("exp9_records", []) + all_exp9,
}

out_path = ROOT / "paper" / "supplementary" / "temperature_ablation_7models.json"
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

# Summary
print(f"\n=== SUMMARY (t={TEMPERATURE}) ===", flush=True)
print(f"{'Model':20s} {'Gap':>8s} {'CFR_C1':>8s} {'CFR_C4':>8s} {'Reduction':>10s}", flush=True)
for model_slug in merged["models"]:
    e1 = [r for r in merged["exp1_records"] if r["model"] == model_slug]
    ch1 = [r for r in e1 if r["channel"] == 1]
    ch5 = [r for r in e1 if r["channel"] == 5]
    wag = sum(1 for r in ch1 if r["answer_correct"]) / len(ch1) if ch1 else None
    nat = sum(1 for r in ch5 if r["answer_correct"]) / len(ch5) if ch5 else None
    gap = abs(wag - nat) if wag is not None and nat is not None else None

    e9 = [r for r in merged["exp9_records"] if r["model"] == model_slug]
    c1 = [r for r in e9 if r["condition"] == 1]
    c4 = [r for r in e9 if r["condition"] == 4]
    cfr1 = sum(1 for r in c1 if not r["correct"]) / len(c1) if c1 else None
    cfr4 = sum(1 for r in c4 if not r.get("externally_routed", False) and not r["correct"]) / len(c4) if c4 else None
    red = ((cfr1 - cfr4) / cfr1 * 100) if cfr1 and cfr1 > 0 and cfr4 is not None else None

    print(f"  {model_slug:20s} {gap if gap is not None else '?':>8} {cfr1 if cfr1 is not None else '?':>8} {cfr4 if cfr4 is not None else '?':>8} {f'{red:.1f}%' if red is not None else '?':>10}", flush=True)

print(f"\nSaved to {out_path}", flush=True)
