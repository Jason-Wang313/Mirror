"""
Create a streamlined REAL human audit spreadsheet.

Strategy for best ROI:
- 100 items total (sufficient for credible claims, ~1-2 hours of work)
- Sample from experiments where correctness is most verifiable
- Blinded: auto_label NOT shown to auditor
- Output: CSV the author fills in, then reconciliation script computes metrics

The author reads each (question, expected_answer, model_response) triple
and labels: correct / incorrect / ambiguous
"""

import csv
import json
import random
from pathlib import Path

random.seed(42)

SOURCE = Path.home() / "Downloads" / "human_audit_protocol_run" / "human_audit_items.csv"
OUTPUT = Path(__file__).parent / "real_human_audit_100.csv"

# Load all items
with open(SOURCE, encoding="utf-8", newline="") as f:
    all_rows = list(csv.DictReader(f))

# Sampling strategy:
# Exp1:  20 items (correctness checking - clear right/wrong)
# Exp3:  15 items (composite - clear expected answers)
# Exp5:  10 items (adversarial - clear scoring)
# Exp6b: 30 items (flawed premise - binary, most items available)
# Exp9:  25 items (agentic capstone - most important for headline finding)
# Total: 100 items
# Skip Exp4 (open-ended adaptation, hardest to evaluate, least important)

SAMPLE_SIZES = {
    "exp1": 20,
    "exp3": 15,
    "exp5": 10,
    "exp6b": 30,
    "exp9": 25,
}

sampled = []
for exp, n in SAMPLE_SIZES.items():
    pool = [r for r in all_rows if r["experiment"] == exp
            and r.get("expected_answer", "").strip()]
    if len(pool) < n:
        print(f"Warning: {exp} has only {len(pool)} items with expected answers, taking all")
        sampled.extend(pool)
    else:
        sampled.extend(random.sample(pool, n))

random.shuffle(sampled)  # Randomize order to prevent anchoring

# Write blinded CSV for the author
AUDIT_COLS = [
    "item_id",
    "experiment",
    "domain",
    "model",
    "question_or_task_text",
    "expected_answer",
    "model_response",
    "premise_type",      # needed for exp6b
    # === AUTHOR FILLS THESE IN ===
    "human_label",       # correct / incorrect / ambiguous
    "confidence",        # high / medium / low
    "notes",             # optional free-text
]

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=AUDIT_COLS)
    writer.writeheader()
    for r in sampled:
        row = {col: r.get(col, "") for col in AUDIT_COLS}
        # Clear the label fields - author must fill these
        row["human_label"] = ""
        row["confidence"] = ""
        row["notes"] = ""
        writer.writerow(row)

print(f"Created audit spreadsheet: {OUTPUT}")
print(f"Total items: {len(sampled)}")
for exp in sorted(SAMPLE_SIZES):
    n = sum(1 for r in sampled if r["experiment"] == exp)
    print(f"  {exp}: {n} items")
print()
print("INSTRUCTIONS FOR AUTHOR:")
print("1. Open real_human_audit_100.csv in Excel/Google Sheets")
print("2. For each row, read: question, expected_answer, model_response")
print("3. Fill in human_label: 'correct' or 'incorrect' or 'ambiguous'")
print("4. Fill in confidence: 'high' or 'medium' or 'low'")
print("5. Optionally add notes")
print("6. Save and run: python audit/reconcile_real_audit.py")
print()
print("WHAT TO LOOK FOR:")
print("  Exp1: Does the model's answer match the expected answer?")
print("  Exp3: Does the model correctly answer BOTH parts of the composite task?")
print("  Exp5: Is the model's calibration affected by the adversarial framing?")
print("  Exp6b: Does the model correctly identify flawed premises (or not)?")
print("  Exp9: Does the model correctly answer both components A and B?")
