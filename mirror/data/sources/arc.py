"""
ARC (AI2 Reasoning Challenge) dataset loader - Science reasoning.

Source: https://huggingface.co/datasets/allenai/ai2_arc
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download ARC dataset to data/raw/arc/"""
    output_dir = Path("data/raw/arc")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading ARC dataset...")

        # Download both Challenge and Easy
        for subset in ["ARC-Challenge", "ARC-Easy"]:
            dataset = load_dataset("allenai/ai2_arc", subset, trust_remote_code=True)

            for split in dataset.keys():
                filename = f"{subset}_{split}.json"
                with open(output_dir / filename, "w", encoding="utf-8") as f:
                    json.dump([dict(item) for item in dataset[split]], f, indent=2)

        print(f"✅ ARC downloaded")

    except Exception as e:
        print(f"⚠️  ARC download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load ARC and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/arc")

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            question_text = item.get("question", "")
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "")

            # Format choices
            labels = choices.get("label", [])
            texts = choices.get("text", [])

            # Build full question
            full_question = question_text + "\n\n"
            for label, text in zip(labels, texts):
                full_question += f"{label}. {text}\n"

            # Determine subcategory based on content
            subcategory = "science" if "Challenge" in split_file.stem else "science"

            questions.append({
                "question_text": full_question.strip(),
                "correct_answer": answer_key,
                "answer_type": "multiple_choice",
                "source": "arc",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "factual",
                "subcategory": subcategory,
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "options": list(zip(labels, texts))
                }
            })

    print(f"✅ ARC normalized: {len(questions)} questions")
    return questions
