"""
SocialIQA dataset loader - Social reasoning questions.

Source: https://huggingface.co/datasets/allenai/social_i_qa
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download SocialIQA dataset to data/raw/social_iqa/"""
    output_dir = Path("data/raw/social_iqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading SocialIQA dataset...")
        dataset = load_dataset("allenai/social_i_qa", trust_remote_code=True)

        for split in dataset.keys():
            with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump([dict(item) for item in dataset[split]], f, indent=2)

        print(f"✅ SocialIQA downloaded")

    except Exception as e:
        print(f"⚠️  SocialIQA download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load SocialIQA and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/social_iqa")

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            context = item.get("context", "")
            question = item.get("question", "")
            answer_a = item.get("answerA", "")
            answer_b = item.get("answerB", "")
            answer_c = item.get("answerC", "")
            label = item.get("label", "1")

            # Build question text
            question_text = f"{context} {question}\n\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}"

            # Convert label (1/2/3) to A/B/C
            label_map = {"1": "A", "2": "B", "3": "C"}
            correct_answer = label_map.get(str(label), "A")

            questions.append({
                "question_text": question_text,
                "correct_answer": correct_answer,
                "answer_type": "multiple_choice",
                "source": "social_iqa",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "social",
                "subcategory": "pragmatic_inference",
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "options": [answer_a, answer_b, answer_c]
                }
            })

    print(f"✅ SocialIQA normalized: {len(questions)} questions")
    return questions
