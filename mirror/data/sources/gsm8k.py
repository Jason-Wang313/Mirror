"""
GSM8K dataset loader - Grade school math word problems.

Source: https://huggingface.co/datasets/openai/gsm8k
"""

import json
import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def download() -> None:
    """Download GSM8K dataset to data/raw/gsm8k/"""
    output_dir = Path("data/raw/gsm8k")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", trust_remote_code=True)

        # Save to JSON
        train_data = dataset["train"]
        test_data = dataset["test"]

        with open(output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump([dict(item) for item in train_data], f, indent=2)

        with open(output_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump([dict(item) for item in test_data], f, indent=2)

        print(f"✅ GSM8K downloaded: {len(train_data)} train + {len(test_data)} test")

    except Exception as e:
        print(f"⚠️  GSM8K download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load GSM8K and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/gsm8k")

    if not (output_dir / "train.json").exists():
        print("GSM8K not downloaded, skipping...")
        return []

    questions = []

    # Load train and test
    for split in ["train", "test"]:
        with open(output_dir / f"{split}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            question_text = item["question"]
            answer_text = item["answer"]

            # Extract numeric answer after ####
            match = re.search(r"####\s*(.+)", answer_text)
            if match:
                correct_answer = match.group(1).strip()
            else:
                correct_answer = answer_text.strip()

            # Remove commas from numbers
            correct_answer = correct_answer.replace(",", "")

            questions.append({
                "question_text": question_text,
                "correct_answer": correct_answer,
                "answer_type": "exact_numeric",
                "source": "gsm8k",
                "source_id": f"{split}_{idx}",
                "domain": "arithmetic",
                "subcategory": "word_problems",
                "difficulty": None,  # Assigned later
                "metadata": {
                    "split": split,
                    "full_solution": answer_text
                }
            })

    print(f"✅ GSM8K normalized: {len(questions)} questions")
    return questions
