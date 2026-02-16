"""
BIG-Bench dataset loader - Multi-domain reasoning tasks.

Source: https://github.com/google/BIG-bench
Note: HuggingFace loading can be inconsistent. Falls back to minimal set if unavailable.
"""

import json
from pathlib import Path

from datasets import load_dataset


# Selected tasks per domain
TASK_MAP = {
    "arithmetic": ["elementary_math_qa", "simple_arithmetic_json"],
    "spatial": ["geometric_shapes", "navigate"],
    "temporal": ["date_understanding", "temporal_sequences"],
    "linguistic": ["grammar_correctness", "language_identification"],
    "logical": ["logical_deduction", "boolean_expressions", "formal_fallacies_syllogisms_negation"],
    "social": ["sarcasm_detection", "intent_recognition"],
    "factual": ["qa_wikidata", "known_unknowns"],
    "procedural": ["mult Stepistep_arithmetic_two"],
}


def download() -> None:
    """Download selected BIG-Bench tasks to data/raw/bigbench/"""
    output_dir = Path("data/raw/bigbench")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading BIG-Bench tasks...")

        # Try to load some key tasks
        successful_tasks = []
        for domain, tasks in TASK_MAP.items():
            for task in tasks:
                try:
                    dataset = load_dataset("google/bigbench", task, trust_remote_code=True)
                    task_dir = output_dir / task
                    task_dir.mkdir(exist_ok=True)

                    for split in dataset.keys():
                        with open(task_dir / f"{split}.json", "w", encoding="utf-8") as f:
                            json.dump([dict(item) for item in dataset[split]], f, indent=2)

                    successful_tasks.append(task)
                except Exception as e:
                    print(f"  ⚠️  Task '{task}' failed: {e}")

        if successful_tasks:
            print(f"✅ BIG-Bench downloaded: {len(successful_tasks)} tasks")
        else:
            print("⚠️  No BIG-Bench tasks downloaded (this is non-critical)")

    except Exception as e:
        print(f"⚠️  BIG-Bench download failed: {e} (this is non-critical)")


def load_and_normalize() -> list[dict]:
    """
    Load BIG-Bench and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/bigbench")

    if not output_dir.exists():
        print("BIG-Bench not downloaded, skipping...")
        return []

    questions = []

    # Reverse map: task -> domain
    task_to_domain = {}
    task_to_subcategory = {
        "elementary_math_qa": ("arithmetic", "word_problems"),
        "simple_arithmetic_json": ("arithmetic", "mental_calculation"),
        "geometric_shapes": ("spatial", "2d_geometry"),
        "navigate": ("spatial", "relative_positioning"),
        "date_understanding": ("temporal", "ordering_events"),
        "temporal_sequences": ("temporal", "ordering_events"),
        "grammar_correctness": ("linguistic", "grammar_correction"),
        "language_identification": ("linguistic", "morphology"),
        "logical_deduction": ("logical", "conditional_reasoning"),
        "boolean_expressions": ("logical", "propositional_logic"),
        "formal_fallacies_syllogisms_negation": ("logical", "syllogisms"),
        "sarcasm_detection": ("social", "sarcasm_detection"),
        "intent_recognition": ("social", "pragmatic_inference"),
        "qa_wikidata": ("factual", "cross_domain_trivia"),
        "known_unknowns": ("factual", "cross_domain_trivia"),
        "multistep_arithmetic_two": ("procedural", "multistep_planning"),
    }

    for task_dir in output_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        domain, subcategory = task_to_subcategory.get(task_name, ("factual", "cross_domain_trivia"))

        for split_file in task_dir.glob("*.json"):
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for idx, item in enumerate(data):
                    # BIG-Bench format varies by task
                    question_text = item.get("input", item.get("question", ""))
                    target = item.get("target", item.get("answer", ""))

                    if not question_text:
                        continue

                    # Determine answer type
                    answer_type = "short_text"
                    if "multiple_choice" in item or "choices" in item:
                        answer_type = "multiple_choice"
                    elif task_name in ["simple_arithmetic_json", "elementary_math_qa"]:
                        answer_type = "exact_numeric"

                    questions.append({
                        "question_text": str(question_text),
                        "correct_answer": str(target),
                        "answer_type": answer_type,
                        "source": "bigbench",
                        "source_id": f"{task_name}_{idx}",
                        "domain": domain,
                        "subcategory": subcategory,
                        "difficulty": None,
                        "metadata": {
                            "task": task_name,
                            "split": split_file.stem
                        }
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading {split_file}: {e}")

    print(f"✅ BIG-Bench normalized: {len(questions)} questions")
    return questions
