"""
LogiQA dataset loader - Logical reasoning questions.

Source: https://huggingface.co/datasets/lucasmccabe/logiqa
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download LogiQA dataset to data/raw/logiqa/"""
    output_dir = Path("data/raw/logiqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading LogiQA dataset...")
        dataset = load_dataset("lucasmccabe/logiqa", trust_remote_code=True)

        for split in dataset.keys():
            with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump([dict(item) for item in dataset[split]], f, indent=2)

        print(f"✅ LogiQA downloaded")

    except Exception as e:
        print(f"⚠️  LogiQA download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load LogiQA and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/logiqa")

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            # Format: context + question + options (A/B/C/D)
            context = item.get("context", "")
            question = item.get("query", item.get("question", ""))
            options = item.get("options", [])
            label = item.get("correct_option", item.get("label", ""))

            # Build full question text
            question_text = f"{context}\n\n{question}\n\n"
            for i, opt in enumerate(options):
                question_text += f"{chr(65+i)}. {opt}\n"

            questions.append({
                "question_text": question_text.strip(),
                "correct_answer": label if label else "A",
                "answer_type": "multiple_choice",
                "source": "logiqa",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "logical",
                "subcategory": "conditional_reasoning",
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "options": options
                }
            })

    print(f"✅ LogiQA normalized: {len(questions)} questions")
    return questions
