#!/usr/bin/env python3
"""
MIRROR Benchmark — Human Annotation Tool
Run: python annotate.py items.jsonl
"""

import json, sys, os
from datetime import datetime

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def wrap(text, w=70):
    lines = []
    for p in text.split('\n'):
        if len(p) <= w:
            lines.append(p)
            continue
        cur = ""
        for word in p.split():
            if cur and len(cur) + len(word) + 1 > w:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        if cur:
            lines.append(cur)
    return '\n'.join(lines)

def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items

def load_progress(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                out[rec['item_id']] = rec
    return out

def save_one(path, rec):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()

def show_item(item, idx, total):
    clear()
    exp = item.get('experiment', '?')
    iid = item.get('item_id', '?')
    
    print("=" * 70)
    print(f"  ITEM {idx+1} / {total}   |   {exp}   |   {iid}")
    print("=" * 70)
    
    # Experiment-specific reminder
    if '4' in str(exp):
        print()
        print("  ** Exp 4 — Judge ONLY the AI's FINAL answer. Ignore feedback. **")
    elif '9' in str(exp):
        print()
        print("  ** Exp 9 — Multi-part task. ALL parts must be correct. **")
        print("  ** If the AI deferred/refused instead of answering → incorrect **")
    
    # Question
    print()
    print("-" * 70)
    print("QUESTION:")
    print("-" * 70)
    q = item.get('question', item.get('task', item.get('prompt', 'N/A')))
    print(wrap(str(q)))
    
    # Response
    print()
    print("-" * 70)
    print("AI RESPONSE:")
    print("-" * 70)
    r = str(item.get('model_response', item.get('response', 'N/A')))
    if len(r) > 2500:
        print(wrap(r[:2000]))
        print(f"\n  [...{len(r)} chars total, showing first 2000 + last 300...]\n")
        print(wrap(r[-300:]))
    else:
        print(wrap(r))
    print()

def get_input():
    print("-" * 70)
    print("  [c] correct   [i] incorrect   [s] skip   [q] save & quit")
    print("-" * 70)
    
    while True:
        v = input("\n  Label (c/i/s/q): ").strip().lower()
        if v in ('c', 'correct'):  label = 'correct'; break
        elif v in ('i', 'incorrect'): label = 'incorrect'; break
        elif v in ('s', 'skip'):   return None
        elif v in ('q', 'quit'):   return 'QUIT'
        print("  Type c, i, s, or q")
    
    print("\n  Confidence:  [1] certain  [2] likely  [3] unsure")
    while True:
        v = input("  (1/2/3): ").strip()
        if v == '1': conf = 'certain'; break
        elif v == '2': conf = 'likely'; break
        elif v == '3': conf = 'unsure'; break
        print("  Type 1, 2, or 3")
    
    notes = input("\n  Notes (optional, Enter to skip): ").strip()
    return {'label': label, 'confidence': conf, 'notes': notes}

def main():
    if len(sys.argv) < 2:
        print("Usage: python annotate.py items.jsonl")
        sys.exit(1)
    
    src = sys.argv[1]
    prog_path = src.replace('.jsonl', '') + '_annotations.jsonl'
    
    items = load_jsonl(src)
    done = load_progress(prog_path)
    total = len(items)
    n_done = sum(1 for it in items if it['item_id'] in done and done[it['item_id']]['label'] in ('correct','incorrect'))
    
    clear()
    print()
    print("=" * 70)
    print("  MIRROR — Human Annotation Tool")
    print("=" * 70)
    print(f"""
  {total} items loaded.  {n_done} already labeled.

  YOUR JOB: Read the question, read the AI's answer, judge if it's
  correct or incorrect. No answer key is provided — use your own
  knowledge and reasoning.

  RULES:
  • 'correct'   = the AI's core answer is right (verbose/hedgy is OK)
  • 'incorrect'  = wrong answer, refusal, too vague, or missing
  • For Exp 4: judge the FINAL answer only, ignore feedback
  • For Exp 9: ALL parts must be right for 'correct'
  • Don't Google answers — use your own judgment
  • If you genuinely can't tell, pick your best guess + confidence 'unsure'
  • Use a calculator for math if needed — that's fine
  • Progress auto-saves. Quit anytime with 'q', resume later.
  • Take a break every 25 items!
""")
    input("  Press Enter to start...")
    
    remaining = [(i, it) for i, it in enumerate(items)
                 if it['item_id'] not in done or done[it['item_id']].get('label') == 'skip']
    
    session_count = 0
    for _, (idx, item) in enumerate(remaining):
        if session_count > 0 and session_count % 25 == 0:
            clear()
            print(f"\n  ☕ BREAK — {session_count} done this session. Stretch for 5 min.\n")
            input("  Press Enter when ready...")
        
        show_item(item, idx, total)
        result = get_input()
        
        if result == 'QUIT':
            n = sum(1 for it in items if it['item_id'] in done and done[it['item_id']]['label'] in ('correct','incorrect'))
            print(f"\n  Saved. {n}/{total} labeled. Run same command to resume.\n")
            return
        
        if result is None:  # skip
            continue
        
        rec = {
            'item_id': item['item_id'],
            'experiment': item.get('experiment', ''),
            'label': result['label'],
            'confidence': result['confidence'],
            'notes': result['notes'],
            'timestamp': datetime.now().isoformat()
        }
        done[item['item_id']] = rec
        save_one(prog_path, rec)
        session_count += 1
    
    clear()
    n = sum(1 for v in done.values() if v['label'] in ('correct','incorrect'))
    c = sum(1 for v in done.values() if v['label'] == 'correct')
    print(f"""
  ✓ ALL DONE!

  {n}/{total} labeled  |  {c} correct  |  {n-c} incorrect
  
  Results saved to: {prog_path}
  
  You're finished — thank you!
""")

if __name__ == '__main__':
    main()
