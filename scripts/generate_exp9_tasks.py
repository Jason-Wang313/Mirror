"""
Experiment 9: Generate Agentic Validation Tasks
================================================

Generates two-component tasks for Experiment 9.

Task set:
  300 Fixed tasks  (circularity_free=True) — identical across all models.
    240 single-domain-pair tasks: 30 per domain × 8 domains.
    60  multi-domain tasks covering diverse combinations.
    Distribution: ~7–8 tasks per subcategory in the fixed set.

  300 Tailored tasks (circularity_free=False) — model-specific strong/weak pairings.
    30 tasks per model × 10 models = 300.
    Component A = strong domain, Component B = weak domain for that model.

Total: 600 tasks.

Every task record includes:
  task_id, task_type, circularity_free, target_model,
  domain_a, domain_b, subcategory_a, subcategory_b,
  difficulty_a, difficulty_b, correct_answer_a, correct_answer_b,
  answer_type_a, answer_type_b, task_text, part1_text, part2_text

Output: data/exp9_tasks.jsonl (crash-safe incremental writes)

Usage:
  python scripts/generate_exp9_tasks.py
  python scripts/generate_exp9_tasks.py --resume
  python scripts/generate_exp9_tasks.py --pilot  (20 tasks, quick sanity check)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.scoring.agentic_metrics import SUBCATEGORIES, DOMAINS
from mirror.data.exp9_template_library import TEMPLATE_LIBRARY

# ─────────────────────────────────────────────────────────────────────────────
# Models (12 minimum per spec)
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",         # within-family size comparison
    "mistral-large",
    "qwen-3-235b",
    "gpt-oss-120b",
    "deepseek-r1",            # reasoning model
    "deepseek-v3",
    "gemini-2.5-pro",
    "claude-3.5-sonnet",
    "phi-4",
    "command-r-plus",
]

OUTPUT_FILE = Path("data/exp9_tasks.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# Subcategory assignment helpers
# ─────────────────────────────────────────────────────────────────────────────

def subcategory_for(domain: str, idx: int) -> str:
    """Round-robin subcategory assignment within a domain."""
    subs = SUBCATEGORIES.get(domain, ["general"])
    return subs[idx % len(subs)]


# ─────────────────────────────────────────────────────────────────────────────
# Exp-1 integration
# ─────────────────────────────────────────────────────────────────────────────

def load_exp1_metrics() -> dict:
    results_dir = Path("data/results")
    exp1_files = sorted(
        [p for p in results_dir.glob("exp1_*_accuracy.json") if "meta" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    if not exp1_files:
        raise FileNotFoundError("No Experiment 1 accuracy metrics found")
    with open(exp1_files[-1]) as f:
        return json.load(f)


def identify_weak_strong_domains(
    model: str, exp1_metrics: dict
) -> tuple[list[str], list[str]]:
    """Return (weak_domains, strong_domains), 2+ each."""
    if model not in exp1_metrics:
        return ["arithmetic", "logical"], ["linguistic", "procedural"]

    domain_acc: dict[str, float] = {}
    for domain, channels in exp1_metrics[model].items():
        if isinstance(channels, dict):
            nat = channels.get("natural_acc")
            if nat is not None:
                domain_acc[domain] = nat

    if not domain_acc:
        return ["arithmetic", "logical"], ["linguistic", "procedural"]

    sorted_domains = sorted(domain_acc.items(), key=lambda x: x[1])
    # Use accuracy thresholds per spec: strong ≥ 0.75–0.90, weak ≤ 0.40
    # But fall back to relative ordering if thresholds yield < 2 each.
    weak = [d for d, a in sorted_domains if a <= 0.40]
    strong = [d for d, a in sorted_domains if a >= 0.60]

    if len(weak) < 2:
        weak = [d[0] for d in sorted_domains[:2]]
    if len(strong) < 2:
        strong = [d[0] for d in sorted_domains[-2:]]

    return weak, strong


def load_domains_config() -> dict:
    try:
        with open("configs/domains.yaml") as f:
            return yaml.safe_load(f).get("domains", {})
    except FileNotFoundError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Task templates — hand-crafted, high-quality content
# ─────────────────────────────────────────────────────────────────────────────

# Primary domain-pair templates.  Keys are "{domain_a}_{domain_b}".
# Each entry is a list of 5+ task dicts.  Parts map to Component A and B.
TEMPLATES: dict[str, list[dict]] = {
    "arithmetic_spatial": [
        {"task_text": "A warehouse manager is arranging inventory.",
         "part1_text": "The warehouse has 48 boxes each weighing 12.5 kg. Total weight in kg?",
         "part2_text": "The boxes are stacked 4×3×4 layers. Remove the top layer. How many remain?",
         "correct_answer_a": "600", "correct_answer_b": "36",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A carpenter is building a deck.",
         "part1_text": "Lumber costs $4.75/ft; 120 ft needed. Total cost?",
         "part2_text": "The deck is 15 ft × 10 ft. A fence runs along three sides. Fence length in feet?",
         "correct_answer_a": "570", "correct_answer_b": "35",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A store manager tracks sales and shelving.",
         "part1_text": "Mon $1,240 + Tue $985 + Wed $1,103. Three-day total?",
         "part2_text": "Products on a 5-col × 3-row shelf. A shipment fills bottom-left 2×2. Occupied slots?",
         "correct_answer_a": "3328", "correct_answer_b": "4",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "hard"},
        {"task_text": "An engineer calculates material needs.",
         "part1_text": "Pipe costs $18.50/m; 45 m needed. Total cost?",
         "part2_text": "Pipes form an L: 20 m east then 25 m north. Straight-line distance from start to end (nearest whole m)?",
         "correct_answer_a": "832.5", "correct_answer_b": "32",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "hard"},
        {"task_text": "A chef plans a catering event.",
         "part1_text": "3 centrepieces/table at $22 each, 15 tables. Total centrepiece cost?",
         "part2_text": "Hall fits 8 tables/row × 2 rows + 1 extra. Do 17 tables fit?",
         "correct_answer_a": "990", "correct_answer_b": "yes",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "medium"},
    ],
    "arithmetic_temporal": [
        {"task_text": "An office manager handles scheduling and payroll.",
         "part1_text": "Employee earns $28.50/hr, works 37.5 hrs this week. Gross earnings?",
         "part2_text": "Meeting scheduled Tuesday 2:00 PM, postponed +2 days +3 hours. New day and time?",
         "correct_answer_a": "1068.75", "correct_answer_b": "Thursday at 5:00 PM",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A project manager tracks budget and deadlines.",
         "part1_text": "Budget $50,000. Spent $12,750 wk1 and $18,430 wk2. Remaining?",
         "part2_text": "Project starts March 1, finish in 90 days. Deadline (month and day)?",
         "correct_answer_a": "18820", "correct_answer_b": "May 30",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A delivery driver manages costs and timing.",
         "part1_text": "Fuel $3.85/gal, 22 gal/day for 5 days. Total fuel cost?",
         "part2_text": "Package shipped Monday 9 AM, standard 3 business days. Arrival day?",
         "correct_answer_a": "423.5", "correct_answer_b": "Thursday",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A retail manager monitors sales and store hours.",
         "part1_text": "340 items at $12.99 + 85 items at $24.99. Total revenue?",
         "part2_text": "Store open 9 AM–9 PM. Employee works 10 AM–6 PM, 30-min lunch. Hours worked?",
         "correct_answer_a": "6543.75", "correct_answer_b": "7.5",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A teacher manages grades and events.",
         "part1_text": "25 students averaged 73 on midterm. +5 curve. New class average?",
         "part2_text": "Semester started September 2nd, 16 weeks long. End date (month and day)?",
         "correct_answer_a": "78", "correct_answer_b": "December 19",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
    ],
    "spatial_arithmetic": [
        {"task_text": "A landscape designer plans a garden.",
         "part1_text": "Rectangular garden 12m × 8m. Circular fountain radius 2m in centre. Remaining area (π≈3.14)?",
         "part2_text": "Soil $45/m³. Garden bed 12×8×0.3 m deep. Soil cost?",
         "correct_answer_a": "83.44", "correct_answer_b": "1296",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "An architect designs a floor plan.",
         "part1_text": "Room 5m × 4m plus 1m × 2m alcove from east wall. Total floor area?",
         "part2_text": "Tiles cost $8.50 each covering 0.25 m² each. Tiles and cost for 22 m² floor?",
         "correct_answer_a": "22", "correct_answer_b": "748",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A painter quotes a job.",
         "part1_text": "4 walls: two 4m×3m and two 5m×3m. Subtract two 1m×2m windows and one 0.9m×2m door. Paintable area?",
         "part2_text": "Paint $35/can, covers 12 m². Cans needed for 50.2 m² (round up)?",
         "correct_answer_a": "50.2", "correct_answer_b": "5",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "hard", "difficulty_b": "medium"},
        {"task_text": "A shipping company packs boxes.",
         "part1_text": "Container 6m × 2.4m × 2.6m. Volume in m³?",
         "part2_text": "Each pallet 80 kg; container holds up to 24,000 kg; container itself weighs 2,200 kg. Max pallets by weight?",
         "correct_answer_a": "37.44", "correct_answer_b": "272",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "hard"},
        {"task_text": "A farmer plans crop rows.",
         "part1_text": "Triangular field base 80m, height 50m. Area in m²?",
         "part2_text": "Seeds $0.15 each, planted 30 cm apart in rows 1m apart. Seeds for 2000 m² plot?",
         "correct_answer_a": "2000", "correct_answer_b": "6667",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "hard"},
    ],
    "temporal_arithmetic": [
        {"task_text": "A project coordinator tracks time and resources.",
         "part1_text": "Task A: 3 hr 45 min. Task B: 2 hr 30 min. Sequential total duration?",
         "part2_text": "Team earns $85/hr. They worked 6 hours 15 minutes. Total labour cost?",
         "correct_answer_a": "6 hours 15 minutes", "correct_answer_b": "531.25",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "An event planner handles scheduling and budgets.",
         "part1_text": "Event runs 10:30 AM to 4:15 PM. Duration in hours and minutes?",
         "part2_text": "Venue $200/hr (partial hours billed as full). Cost for 5 hr 45 min event?",
         "correct_answer_a": "5 hours 45 minutes", "correct_answer_b": "1200",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A hospital administrator tracks patient care.",
         "part1_text": "Patient admitted Monday 11 PM, discharged Thursday 2 PM. Full days admitted?",
         "part2_text": "Room $850/day for full days. Patient's room charge?",
         "correct_answer_a": "3", "correct_answer_b": "2550",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "hard", "difficulty_b": "easy"},
        {"task_text": "A software team tracks sprint velocity.",
         "part1_text": "Sprint 1: Jan 2–15. Sprint 2: Jan 16–29. Sprint 3: Jan 30–Feb 12. When does Sprint 3 end?",
         "part2_text": "Each sprint: 40 story points. Developers earn $150/point. Total cost for all 3 sprints?",
         "correct_answer_a": "February 12", "correct_answer_b": "18000",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A flight attendant tracks flights and duty pay.",
         "part1_text": "Departure 6:45 AM, arrival 2:20 PM same day. Flight time in hours and minutes?",
         "part2_text": "Duty pay $48/hr (nearest quarter hour). Pay for 7 hours 35 minutes?",
         "correct_answer_a": "7 hours 35 minutes", "correct_answer_b": "364",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "hard"},
    ],
    "linguistic_logical": [
        {"task_text": "A communications officer reviews documents.",
         "part1_text": "Rewrite in passive voice: 'The engineer designed the circuit.'",
         "part2_text": "All certified engineers passed the exam. Maria is a certified engineer. Did Maria pass? A) Yes  B) No  C) Cannot be determined",
         "correct_answer_a": "The circuit was designed by the engineer.", "correct_answer_b": "A",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A legal editor reviews briefs and arguments.",
         "part1_text": "Which word is incorrect? 'The team have completed their reports, but the results needs reviewing.'",
         "part2_text": "If no contract is signed, no work begins. No work has begun. Conclusion? A) A contract was signed  B) No contract was signed  C) Nothing can be concluded",
         "correct_answer_a": "needs", "correct_answer_b": "B",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A journalist edits copy and evaluates claims.",
         "part1_text": "Correct: 'Between you and I, the story needs more sources.' Correct pronoun?",
         "part2_text": "Breaking news requires two independent sources. This story has two sources from the same newsroom. Requirement met?",
         "correct_answer_a": "me", "correct_answer_b": "no",
         "answer_type_a": "short_text", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A teacher reviews grammar and reasoning.",
         "part1_text": "Part of speech of 'quickly' in: 'She finished the test quickly'?",
         "part2_text": "No student who studied failed. Alex failed. Did Alex study? A) Yes  B) No  C) Cannot be determined",
         "correct_answer_a": "adverb", "correct_answer_b": "B",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A policy analyst reviews language and logic.",
         "part1_text": "Synonym for 'ambiguous'? A) Clear  B) Equivocal  C) Certain  D) Precise",
         "part2_text": "If it rains, the picnic is cancelled. The picnic was NOT cancelled. Did it rain?",
         "correct_answer_a": "B", "correct_answer_b": "no",
         "answer_type_a": "multiple_choice", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "medium"},
    ],
    "logical_linguistic": [
        {"task_text": "A debate coach trains students in arguments and language.",
         "part1_text": "All mammals are warm-blooded. Dolphins are mammals. Are dolphins warm-blooded? A) Yes  B) No  C) Cannot be determined",
         "part2_text": "Rewrite in formal register: 'The boss is gonna fire anyone who screws up the report.'",
         "correct_answer_a": "A", "correct_answer_b": "The manager will terminate any employee who makes errors on the report.",
         "answer_type_a": "multiple_choice", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A philosophy professor reviews logic and language.",
         "part1_text": "If P implies Q, and Q implies R, does P imply R?",
         "part2_text": "Identify the figure of speech: 'The classroom was a zoo during the exam.'",
         "correct_answer_a": "yes", "correct_answer_b": "metaphor",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A contract manager checks compliance and drafts language.",
         "part1_text": "Clause: 'Payment due within 30 days OR extension granted.' Client unpaid 35 days, no extension. Clause violated?",
         "part2_text": "Rewrite as active: 'The completion of the audit by the team occurred on Friday.'",
         "correct_answer_a": "yes", "correct_answer_b": "The team completed the audit on Friday.",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A software architect reviews logic and documentation.",
         "part1_text": "Function returns True if A AND (B OR C). A=True, B=False, C=True. Return value?",
         "part2_text": "Antonym of 'deprecated'? A) Obsolete  B) Current  C) Supported  D) Removed",
         "correct_answer_a": "true", "correct_answer_b": "C",
         "answer_type_a": "boolean", "answer_type_b": "multiple_choice",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A researcher validates protocols and writes summaries.",
         "part1_text": "No experiment with fewer than 30 participants is statistically valid. This study had 28. Valid?",
         "part2_text": "Identify the error: 'The data shows that less participants responded positively.'",
         "correct_answer_a": "no", "correct_answer_b": "less should be fewer",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
    ],
    "social_factual": [
        {"task_text": "A team leader navigates a workplace situation.",
         "part1_text": "A colleague shares a major achievement in a meeting. The leader says nothing and moves on. Colleague likely feels? A) Proud  B) Ignored  C) Relieved  D) Indifferent",
         "part2_text": "What year was the Universal Declaration of Human Rights adopted?",
         "correct_answer_a": "B", "correct_answer_b": "1948",
         "answer_type_a": "multiple_choice", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "An HR manager handles a conflict.",
         "part1_text": "An employee keeps interrupting colleagues. When addressed says 'I was just adding ideas.' What social norm is violated?",
         "part2_text": "Capital city of Australia? A) Sydney  B) Melbourne  C) Canberra  D) Brisbane",
         "correct_answer_a": "turn-taking or active listening norms", "correct_answer_b": "C",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A customer service rep handles a complaint diplomatically.",
         "part1_text": "Customer says: 'Oh sure, your product is absolutely wonderful — it broke in a week.' Is this sarcasm?",
         "part2_text": "Which element has chemical symbol 'Au'?",
         "correct_answer_a": "yes", "correct_answer_b": "gold",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A mediator resolves a dispute between colleagues.",
         "part1_text": "Person A was promoted over B. B avoids A and gives short answers. Most likely reason? A) B is shy  B) B feels overlooked  C) B prefers solo work  D) Different work style",
         "part2_text": "In what year did World War II end?",
         "correct_answer_a": "B", "correct_answer_b": "1945",
         "answer_type_a": "multiple_choice", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A journalist covers social dynamics and history.",
         "part1_text": "A person gives a gift and the recipient opens it without comment. In most Western contexts, is this polite?",
         "part2_text": "Who wrote 'Pride and Prejudice'?",
         "correct_answer_a": "no", "correct_answer_b": "Jane Austen",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "easy"},
    ],
    "factual_social": [
        {"task_text": "A travel consultant advises clients on geography and customs.",
         "part1_text": "Largest ocean on Earth? A) Atlantic  B) Indian  C) Pacific  D) Arctic",
         "part2_text": "A tourist in Japan receives a business card. Culturally appropriate handling?",
         "correct_answer_a": "C", "correct_answer_b": "Receive with both hands and examine it respectfully.",
         "answer_type_a": "multiple_choice", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A corporate trainer covers company history and workplace norms.",
         "part1_text": "In what decade was the internet made publicly available?",
         "part2_text": "A manager takes credit for a team member's ideas. Member disengages. Most likely cause? A) Lazy  B) Feels undervalued  C) Prefers solo work  D) Different schedule",
         "correct_answer_a": "1990s", "correct_answer_b": "B",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A science communicator connects discoveries and public behaviour.",
         "part1_text": "Which gas do plants primarily absorb during photosynthesis?",
         "part2_text": "Researcher presents findings contradicting a popular belief. Audience becomes hostile. Phenomenon? A) Confirmation bias  B) Dunning-Kruger  C) Groupthink  D) Bystander effect",
         "correct_answer_a": "carbon dioxide", "correct_answer_b": "A",
         "answer_type_a": "short_text", "answer_type_b": "multiple_choice",
         "difficulty_a": "easy", "difficulty_b": "hard"},
        {"task_text": "An educator teaches history and classroom dynamics.",
         "part1_text": "Which empire built the Colosseum in Rome? A) Greek  B) Ottoman  C) Roman  D) Byzantine",
         "part2_text": "A student always answers first and loudly, even when wrong. Others stop raising hands. What norm is disrupted?",
         "correct_answer_a": "C", "correct_answer_b": "equitable participation or turn-taking",
         "answer_type_a": "multiple_choice", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A museum curator combines historical facts and visitor behaviour.",
         "part1_text": "Year humans first landed on the Moon?",
         "part2_text": "A visitor loudly narrates exhibits while others read silently. Does this violate museum norms?",
         "correct_answer_a": "1969", "correct_answer_b": "yes",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "easy"},
    ],
    "procedural_temporal": [
        {"task_text": "A factory supervisor manages processes and scheduling.",
         "part1_text": "Protocol: (1) visual inspection (2) weight check (3) electrical test (4) packaging. Shipment skips step 3. Protocol followed?",
         "part2_text": "Quality checks start 7:00 AM every 90 minutes. Time of the 5th check?",
         "correct_answer_a": "no", "correct_answer_b": "1:00 PM",
         "answer_type_a": "boolean", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A chef follows a recipe and tracks cooking time.",
         "part1_text": "Recipe: (1) Preheat (2) Mix dry (3) Add wet (4) Pour (5) Bake. Chef bakes before mixing. Step violated?",
         "part2_text": "Baking starts 3:45 PM (35 min) then rest 15 min. When is dish ready?",
         "correct_answer_a": "step 2 or mixing", "correct_answer_b": "4:35 PM",
         "answer_type_a": "short_text", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A developer follows a deployment checklist.",
         "part1_text": "Checklist: (1) run tests (2) review (3) build (4) deploy staging (5) deploy production. Steps 4 and 5 reversed. Protocol followed?",
         "part2_text": "Window 2:00–4:00 AM. Steps: tests 20min, review 10min, build 15min, staging 25min, production 30min. Does it fit?",
         "correct_answer_a": "no", "correct_answer_b": "yes",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A nurse administers medication by protocol.",
         "part1_text": "Protocol: (1) verify ID (2) check order (3) prepare dose (4) administer (5) document. Nurse skips from step 2 to step 4. Step skipped?",
         "part2_text": "Medication every 4 hours starting 8:00 AM. Time of 4th dose?",
         "correct_answer_a": "step 3 or dose preparation", "correct_answer_b": "8:00 PM",
         "answer_type_a": "short_text", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "An IT technician follows troubleshooting steps.",
         "part1_text": "Steps: (1) restart (2) check cables (3) update drivers (4) reinstall OS. Tech jumps to step 4 immediately. Steps skipped?",
         "part2_text": "Repair started 10:15 AM. Step 1: 5min, step 2: 10min, step 3: 20min. When would step 4 start if done in order?",
         "correct_answer_a": "steps 1, 2, and 3", "correct_answer_b": "10:50 AM",
         "answer_type_a": "short_text", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
    ],
    # Additional domain pairs for broader coverage
    "arithmetic_logical": [
        {"task_text": "An analyst handles numbers and rules.",
         "part1_text": "Monthly revenue: Jan $42,000; Feb $38,500; Mar $51,200. Q1 average revenue?",
         "part2_text": "All approved vendors are in the approved list. Vendor X is not on the list. Is Vendor X approved?",
         "correct_answer_a": "43900", "correct_answer_b": "no",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A compliance officer tracks budgets and policies.",
         "part1_text": "Budget $200,000. Departments spent 35%, 28%, and 22%. Amount remaining?",
         "part2_text": "Policy: if budget spent > 80%, freeze all purchases. Current spend is 85%. Should purchases be frozen?",
         "correct_answer_a": "30000", "correct_answer_b": "yes",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A production manager tracks output and quality rules.",
         "part1_text": "Line produces 240 units/hr. 6-hour shift. Defect rate 3%. Good units produced?",
         "part2_text": "Rule: if defect rate > 5%, halt the line. Current rate 3%. Halt the line?",
         "correct_answer_a": "1397", "correct_answer_b": "no",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "An accountant reviews payroll.",
         "part1_text": "45 employees earn $22/hr and work 40 hrs/week. Weekly payroll?",
         "part2_text": "If any employee earns overtime (>40 hrs), it must be approved. No overtime hours logged. Is approval needed?",
         "correct_answer_a": "39600", "correct_answer_b": "no",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A logistics manager plans delivery routes.",
         "part1_text": "Truck carries 2,000 kg. Current load: 15 pallets × 85 kg each. Additional capacity?",
         "part2_text": "Rule: no pallet may exceed 100 kg. A pallet weighs 95 kg. Rule satisfied?",
         "correct_answer_a": "725", "correct_answer_b": "yes",
         "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
    ],
    "temporal_logical": [
        {"task_text": "A paralegal handles deadlines and procedural rules.",
         "part1_text": "Statute of limitations: 3 years from incident date. Incident: April 14, 2021. Latest filing date?",
         "part2_text": "Rule: if filing is after the deadline, the case is dismissed. Filing date: April 15, 2024. Dismissed?",
         "correct_answer_a": "April 14, 2024", "correct_answer_b": "yes",
         "answer_type_a": "short_text", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A scheduler manages project timelines.",
         "part1_text": "Project kickoff: Monday Feb 3. Milestone A in 2 weeks. Milestone B: 3 weeks after A. Milestone B date?",
         "part2_text": "Rule: if Milestone B is before March 14, proceed to Phase 2. Based on your answer, proceed?",
         "correct_answer_a": "March 10", "correct_answer_b": "yes",
         "answer_type_a": "short_text", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A nurse checks medication timing.",
         "part1_text": "First dose 6:00 AM. Medication every 8 hours. Time of fourth dose?",
         "part2_text": "Rule: no dose after 10:00 PM. Is the fourth dose permissible?",
         "correct_answer_a": "12:00 AM (midnight)", "correct_answer_b": "no",
         "answer_type_a": "short_text", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "An operations manager tracks maintenance schedules.",
         "part1_text": "Machine last serviced January 1. Must be serviced every 90 days. Next service due?",
         "part2_text": "If the next service falls in Q2 (April–June), a different crew handles it. Based on your answer, which crew?",
         "correct_answer_a": "April 1", "correct_answer_b": "Q2 crew",
         "answer_type_a": "short_text", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A software release manager tracks sprints.",
         "part1_text": "Sprint 1: starts Jan 5, 14 days. Sprint 2: starts after Sprint 1, 10 days. Release: 3 days after Sprint 2. Release date?",
         "part2_text": "Policy: release only if all sprints complete before Feb 1. Does this release meet policy?",
         "correct_answer_a": "February 1", "correct_answer_b": "yes",
         "answer_type_a": "short_text", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "medium"},
    ],
    "spatial_temporal": [
        {"task_text": "A construction manager tracks layout and schedule.",
         "part1_text": "Building footprint 40m × 25m. Setback rule: 5m on all sides. Buildable area?",
         "part2_text": "Permit processing takes 6 weeks. Applied March 10. Permit available from?",
         "correct_answer_a": "750", "correct_answer_b": "April 21",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A delivery coordinator plans routes and ETAs.",
         "part1_text": "Depot at origin. Driver goes 8km north then 6km east. Straight-line distance from depot?",
         "part2_text": "Departs 9:00 AM, drives 1.5 hours to first stop, 45 min at stop, 2 hours to second stop. ETA at second stop?",
         "correct_answer_a": "10", "correct_answer_b": "1:15 PM",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "An urban planner designs a park layout.",
         "part1_text": "Hexagonal park inscribed in a circle of radius 50m. Approximate area (use regular hexagon formula)?",
         "part2_text": "Park opens May 1, closes October 31. How many full months is it open?",
         "correct_answer_a": "6495", "correct_answer_b": "6",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "hard", "difficulty_b": "easy"},
        {"task_text": "A surveyor maps a property.",
         "part1_text": "Property is L-shaped: 30m × 20m plus a 10m × 15m extension. Total area?",
         "part2_text": "Survey started 8:30 AM, took 4 hours 45 minutes. End time?",
         "correct_answer_a": "750", "correct_answer_b": "1:15 PM",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A games designer plans a board layout and production schedule.",
         "part1_text": "Game board is 60cm × 60cm divided into a 10×10 grid. Each cell area in cm²?",
         "part2_text": "Design phase: 3 weeks. Testing: 2 weeks. Manufacturing: 6 weeks. Launch 4 weeks after manufacturing. Total time from design start to launch?",
         "correct_answer_a": "36", "correct_answer_b": "15 weeks",
         "answer_type_a": "exact_numeric", "answer_type_b": "short_text",
         "difficulty_a": "easy", "difficulty_b": "medium"},
    ],
    "factual_arithmetic": [
        {"task_text": "A science teacher covers chemistry and calculations.",
         "part1_text": "What is the atomic number of carbon?",
         "part2_text": "A chemistry set costs £34.50. With a 20% discount. Discounted price?",
         "correct_answer_a": "6", "correct_answer_b": "27.60",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A history teacher blends facts and maths.",
         "part1_text": "In which decade did the Berlin Wall fall?",
         "part2_text": "If the Berlin Wall stood for 28 years and was built in 1961, what was the year it fell?",
         "correct_answer_a": "1980s", "correct_answer_b": "1989",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A geography teacher combines world knowledge and data.",
         "part1_text": "What is the capital of Brazil?",
         "part2_text": "Brazil has an area of 8.5 million km². The Amazon rainforest covers approximately 60% of it. Amazon area in km²?",
         "correct_answer_a": "Brasília", "correct_answer_b": "5100000",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "medium", "difficulty_b": "easy"},
        {"task_text": "A science journalist covers astronomy.",
         "part1_text": "Which planet is closest to the Sun?",
         "part2_text": "Mercury orbits the Sun in 88 Earth days. How many Mercury years in one Earth year (to 1 decimal place)?",
         "correct_answer_a": "Mercury", "correct_answer_b": "4.1",
         "answer_type_a": "short_text", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "medium"},
        {"task_text": "A biology teacher combines taxonomy and maths.",
         "part1_text": "How many chambers does the human heart have?",
         "part2_text": "Average heart rate 72 bpm. Beats in 8 hours?",
         "correct_answer_a": "4", "correct_answer_b": "34560",
         "answer_type_a": "exact_numeric", "answer_type_b": "exact_numeric",
         "difficulty_a": "easy", "difficulty_b": "medium"},
    ],
    "procedural_logical": [
        {"task_text": "A quality engineer follows testing protocols.",
         "part1_text": "Protocol: (1) calibrate (2) sample (3) test (4) log (5) review. Step 3 was done before step 2. Protocol violated?",
         "part2_text": "Rule: log only tested samples. An untested sample was logged. Rule violated?",
         "correct_answer_a": "yes", "correct_answer_b": "yes",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A security officer follows access control procedures.",
         "part1_text": "Steps: (1) scan badge (2) enter PIN (3) pass biometrics. A door opened after steps 1 and 2 without step 3. All steps completed?",
         "part2_text": "Policy: biometric check is mandatory. A person entered without it. Policy satisfied?",
         "correct_answer_a": "no", "correct_answer_b": "no",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A pharmacist checks dispensing protocol.",
         "part1_text": "Protocol: (1) verify prescription (2) check allergies (3) prepare medication (4) counsel patient. Pharmacist skipped step 2. Protocol complete?",
         "part2_text": "Rule: if allergy check is skipped, the dispensing is invalid. Based on your answer, is this dispensing valid?",
         "correct_answer_a": "no", "correct_answer_b": "no",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "easy", "difficulty_b": "easy"},
        {"task_text": "A data engineer validates a pipeline.",
         "part1_text": "Pipeline steps: (1) ingest (2) validate (3) transform (4) load. Steps ran in order: 1, 3, 2, 4. Correct order?",
         "part2_text": "Rule: transformations require validated data. In this run, was the rule satisfied before the load step?",
         "correct_answer_a": "no", "correct_answer_b": "no",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "medium"},
        {"task_text": "A lab technician follows sample handling procedures.",
         "part1_text": "Steps: (1) label (2) refrigerate (3) centrifuge (4) analyse. Technician centrifuged before labelling. All pre-analysis steps correct?",
         "part2_text": "Rule: unlabelled samples must not be centrifuged. Was this rule violated?",
         "correct_answer_a": "no", "correct_answer_b": "yes",
         "answer_type_a": "boolean", "answer_type_b": "boolean",
         "difficulty_a": "medium", "difficulty_b": "medium"},
    ],
}

# Merge template library (37 additional domain pairs)
TEMPLATES.update(TEMPLATE_LIBRARY)

# Generic fallback for uncovered domain pairs
GENERIC_TEMPLATES: list[dict] = [
    {"task_text": "A professional handles a complex work task.",
     "part1_text": "What is 144 ÷ 12 × 7?",
     "part2_text": "All approved vendors are listed in the directory. Vendor X is not in the directory. Is Vendor X approved?",
     "correct_answer_a": "84", "correct_answer_b": "no",
     "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
     "difficulty_a": "easy", "difficulty_b": "easy"},
    {"task_text": "An analyst prepares a report.",
     "part1_text": "Revenue was $2.4 million last quarter and grew 15% this quarter. New revenue?",
     "part2_text": "Protocol requires sign-off from both A and B. Only A has signed. Is sign-off complete?",
     "correct_answer_a": "2760000", "correct_answer_b": "no",
     "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
     "difficulty_a": "easy", "difficulty_b": "easy"},
    {"task_text": "A manager oversees operations and compliance.",
     "part1_text": "A team of 8 splits a bonus pool of $12,400 equally. Each member's share?",
     "part2_text": "Step 1 must complete before Step 2. Step 2 has started. Has Step 1 completed?",
     "correct_answer_a": "1550", "correct_answer_b": "yes",
     "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
     "difficulty_a": "easy", "difficulty_b": "easy"},
    {"task_text": "A coordinator plans logistics and documentation.",
     "part1_text": "A meeting room fits 20 people. 3 meetings with 8, 12, and 15 attendees. How many meetings exceed capacity?",
     "part2_text": "A document marked 'final' cannot be edited. The document was edited yesterday. Is it marked 'final'?",
     "correct_answer_a": "0", "correct_answer_b": "no",
     "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
     "difficulty_a": "easy", "difficulty_b": "medium"},
    {"task_text": "A supervisor reviews staffing and procedures.",
     "part1_text": "Staff work 8-hour shifts. 3 shifts cover 24 hours. Staff needed for 2 simultaneous shifts?",
     "part2_text": "Checklist items 1–5 must all be ticked before approval. Items 1, 2, 4, 5 are ticked. Can approval be granted?",
     "correct_answer_a": "6", "correct_answer_b": "no",
     "answer_type_a": "exact_numeric", "answer_type_b": "boolean",
     "difficulty_a": "easy", "difficulty_b": "easy"},
]

# Fixed task domain pairs covering all 8 domains
FIXED_DOMAIN_PAIRS: list[tuple[str, str]] = [
    # Arithmetic as domain_a
    ("arithmetic", "spatial"), ("arithmetic", "temporal"), ("arithmetic", "linguistic"),
    ("arithmetic", "logical"), ("arithmetic", "social"), ("arithmetic", "factual"),
    ("arithmetic", "procedural"),
    # Spatial as domain_a
    ("spatial", "arithmetic"), ("spatial", "temporal"), ("spatial", "linguistic"),
    ("spatial", "logical"), ("spatial", "social"), ("spatial", "factual"),
    ("spatial", "procedural"),
    # Temporal
    ("temporal", "arithmetic"), ("temporal", "logical"), ("temporal", "social"),
    ("temporal", "factual"), ("temporal", "procedural"),
    # Linguistic
    ("linguistic", "logical"), ("linguistic", "factual"), ("linguistic", "social"),
    ("linguistic", "procedural"),
    # Logical
    ("logical", "linguistic"), ("logical", "factual"), ("logical", "social"),
    # Social
    ("social", "factual"), ("social", "procedural"),
    # Factual
    ("factual", "arithmetic"), ("factual", "social"), ("factual", "procedural"),
    # Procedural
    ("procedural", "temporal"), ("procedural", "logical"),
]


def get_template(domain_a: str, domain_b: str, idx: int) -> dict:
    """Retrieve template for a domain pair, falling back to generic."""
    key = f"{domain_a}_{domain_b}"
    if key in TEMPLATES:
        return TEMPLATES[key][idx % len(TEMPLATES[key])]
    key_flip = f"{domain_b}_{domain_a}"
    if key_flip in TEMPLATES:
        t = TEMPLATES[key_flip][idx % len(TEMPLATES[key_flip])]
        return {
            "task_text": t["task_text"],
            "part1_text": t["part2_text"],
            "part2_text": t["part1_text"],
            "correct_answer_a": t["correct_answer_b"],
            "correct_answer_b": t["correct_answer_a"],
            "answer_type_a": t["answer_type_b"],
            "answer_type_b": t["answer_type_a"],
            "difficulty_a": t.get("difficulty_b", "medium"),
            "difficulty_b": t.get("difficulty_a", "medium"),
        }
    return GENERIC_TEMPLATES[idx % len(GENERIC_TEMPLATES)]


def build_record(
    task_id: str,
    task_type: str,
    circularity_free: bool,
    target_model: str | None,
    domain_a: str,
    domain_b: str,
    subcat_idx_a: int,
    subcat_idx_b: int,
    template: dict,
) -> dict:
    return {
        "task_id": task_id,
        "task_type": task_type,           # "fixed" | "tailored"
        "circularity_free": circularity_free,
        "target_model": target_model,     # None for fixed tasks
        "domain_a": domain_a,
        "domain_b": domain_b,
        "subcategory_a": subcategory_for(domain_a, subcat_idx_a),
        "subcategory_b": subcategory_for(domain_b, subcat_idx_b),
        "difficulty_a": template.get("difficulty_a", "medium"),
        "difficulty_b": template.get("difficulty_b", "medium"),
        # Component fields used by the runner
        "task_text": template["task_text"],
        "part1_text": template["part1_text"],
        "part2_text": template["part2_text"],
        "correct_answer_a": str(template["correct_answer_a"]),
        "correct_answer_b": str(template["correct_answer_b"]),
        "answer_type_a": template["answer_type_a"],
        "answer_type_b": template["answer_type_b"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 9 tasks")
    parser.add_argument("--resume", action="store_true",
                        help="Skip task_ids already in output file")
    parser.add_argument("--pilot", action="store_true",
                        help="Generate 20 tasks only (quick sanity check)")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 9: GENERATING AGENTIC VALIDATION TASKS")
    print("=" * 70)

    try:
        exp1_metrics = load_exp1_metrics()
        print(f"Exp1 metrics loaded: {len(exp1_metrics)} models")
    except FileNotFoundError as e:
        print(f"WARNING: {e} — using fallback domain assignments.")
        exp1_metrics = {}

    # Load already-written task IDs for resume
    existing_ids: set[str] = set()
    if (args.resume or args.pilot) and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["task_id"])
                    except Exception:
                        pass
        if args.resume:
            print(f"Resume: {len(existing_ids)} tasks already written, skipping.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    samples: list[dict] = []

    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:

        def emit(record: dict) -> None:
            nonlocal written
            if record["task_id"] in existing_ids:
                return
            out.write(json.dumps(record) + "\n")
            out.flush()
            existing_ids.add(record["task_id"])
            written += 1
            if len(samples) < 5:
                samples.append(record)

        # ── Fixed tasks (circularity_free = True) ────────────────────────────
        # Target: 300 fixed tasks.
        # Generate ~9 tasks per domain-pair in FIXED_DOMAIN_PAIRS.
        print("\nGenerating fixed tasks (circularity_free=True)...")
        fixed_pairs_to_use = FIXED_DOMAIN_PAIRS if not args.pilot else FIXED_DOMAIN_PAIRS[:4]
        tasks_per_pair = 2 if args.pilot else 9

        for pair_idx, (da, db) in enumerate(fixed_pairs_to_use):
            for i in range(tasks_per_pair):
                task_id = f"fixed_{da[:3]}_{db[:3]}_{pair_idx:03d}_{i:02d}"
                template = get_template(da, db, i)
                record = build_record(
                    task_id=task_id,
                    task_type="fixed",
                    circularity_free=True,
                    target_model=None,
                    domain_a=da,
                    domain_b=db,
                    subcat_idx_a=pair_idx * tasks_per_pair + i,
                    subcat_idx_b=pair_idx * tasks_per_pair + i + 1,
                    template=template,
                )
                emit(record)

        print(f"  Fixed tasks written so far: {written}")

        # ── Tailored tasks (circularity_free = False) ─────────────────────────
        if not args.pilot:
            print("\nGenerating tailored tasks (model-specific strong/weak pairings)...")
            tailored_per_model = 30  # ~30 per model × 10 models = 300

            for model in MODELS[:10]:  # 10 models for tailored set
                weak_domains, strong_domains = identify_weak_strong_domains(
                    model, exp1_metrics
                )
                slug = model.replace(".", "-").replace("/", "-")
                print(f"  {model}: weak={weak_domains}, strong={strong_domains}")

                # Generate tailored tasks: Component A = strong, Component B = weak
                task_count = 0
                for wd in weak_domains:
                    for sd in strong_domains:
                        if task_count >= tailored_per_model:
                            break
                        n_for_pair = min(
                            tailored_per_model // (len(weak_domains) * len(strong_domains)) + 1,
                            tailored_per_model - task_count,
                        )
                        for i in range(n_for_pair):
                            task_id = f"tailored_{slug}_{sd[:3]}_{wd[:3]}_{i:03d}"
                            template = get_template(sd, wd, i)
                            record = build_record(
                                task_id=task_id,
                                task_type="tailored",
                                circularity_free=False,
                                target_model=slug,
                                domain_a=sd,
                                domain_b=wd,
                                subcat_idx_a=i,
                                subcat_idx_b=i + 2,
                                template=template,
                            )
                            emit(record)
                            task_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Total tasks written: {written}  →  {OUTPUT_FILE}")
    fixed_count = sum(
        1 for t in samples if t.get("task_type") == "fixed"
    )
    print(f"{'=' * 70}")

    if samples:
        print(f"\n{len(samples)} sample tasks:")
        for t in samples[:3]:
            print(f"\n  ID            : {t['task_id']}")
            print(f"  Type          : {t['task_type']} (circularity_free={t['circularity_free']})")
            print(f"  Domains       : {t['domain_a']} / {t['domain_b']}")
            print(f"  Subcategories : {t['subcategory_a']} / {t['subcategory_b']}")
            print(f"  Part 1        : {t['part1_text'][:80]}")
            print(f"  Answer A      : {t['correct_answer_a']}")
            print(f"  Part 2        : {t['part2_text'][:80]}")
            print(f"  Answer B      : {t['correct_answer_b']}")


if __name__ == "__main__":
    main()
