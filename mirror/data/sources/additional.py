"""
Additional datasets loader - ToMi, TimeQA, GLUE subtasks, etc.

These are supplementary datasets that may or may not be available.
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download additional datasets to data/raw/additional/"""
    output_dir = Path("data/raw/additional")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_try = [
        ("facebook/tomi", "tomi"),  # Theory of Mind
        ("nyu-mll/glue", "cola", "glue_cola"),  # Grammar acceptability
    ]

    for dataset_info in datasets_to_try:
        try:
            if len(dataset_info) == 2:
                dataset_name, short_name = dataset_info
                dataset = load_dataset(dataset_name, trust_remote_code=True)
            else:
                dataset_name, subset, short_name = dataset_info
                dataset = load_dataset(dataset_name, subset, trust_remote_code=True)

            task_dir = output_dir / short_name
            task_dir.mkdir(exist_ok=True)

            for split in dataset.keys():
                with open(task_dir / f"{split}.json", "w", encoding="utf-8") as f:
                    json.dump([dict(item) for item in dataset[split]], f, indent=2)

            print(f"✅ Downloaded {short_name}")

        except Exception as e:
            print(f"  ⚠️  {dataset_info[0] if len(dataset_info) == 2 else dataset_info[2]} not available: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load additional datasets and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/additional")

    if not output_dir.exists():
        print("Additional datasets not downloaded, skipping...")
        return []

    questions = []

    # ToMi - Theory of Mind
    tomi_dir = output_dir / "tomi"
    if tomi_dir.exists():
        for split_file in tomi_dir.glob("*.json"):
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for idx, item in enumerate(data):
                    story = item.get("story", item.get("context", ""))
                    question = item.get("question", "")
                    answer = item.get("answer", "")

                    questions.append({
                        "question_text": f"{story}\n\n{question}",
                        "correct_answer": str(answer),
                        "answer_type": "short_text",
                        "source": "tomi",
                        "source_id": f"{split_file.stem}_{idx}",
                        "domain": "social",
                        "subcategory": "theory_of_mind",
                        "difficulty": None,
                        "metadata": {"split": split_file.stem}
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading ToMi: {e}")

    # GLUE CoLA - Grammar acceptability
    cola_dir = output_dir / "glue_cola"
    if cola_dir.exists():
        for split_file in cola_dir.glob("*.json"):
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for idx, item in enumerate(data):
                    sentence = item.get("sentence", "")
                    label = item.get("label", 0)

                    question_text = f"Is this sentence grammatically acceptable?\n\n\"{sentence}\""

                    questions.append({
                        "question_text": question_text,
                        "correct_answer": "Yes" if label == 1 else "No",
                        "answer_type": "boolean",
                        "source": "glue_cola",
                        "source_id": f"{split_file.stem}_{idx}",
                        "domain": "linguistic",
                        "subcategory": "grammar_correction",
                        "difficulty": None,
                        "metadata": {"split": split_file.stem}
                    })
            except Exception as e:
                print(f"  ⚠️  Error loading GLUE CoLA: {e}")

    print(f"✅ Additional datasets normalized: {len(questions)} questions")
    return questions
