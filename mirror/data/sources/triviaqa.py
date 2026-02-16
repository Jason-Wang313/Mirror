"""
TriviaQA dataset loader - Factual recall questions.

Source: https://huggingface.co/datasets/mandarjoshi/trivia_qa
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download TriviaQA dataset to data/raw/triviaqa/"""
    output_dir = Path("data/raw/triviaqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading TriviaQA dataset...")
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", trust_remote_code=True)

        for split in dataset.keys():
            # Limit to reasonable size
            subset = list(dataset[split])[:5000] if len(dataset[split]) > 5000 else list(dataset[split])
            with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump([dict(item) for item in subset], f, indent=2)

        print(f"✅ TriviaQA downloaded")

    except Exception as e:
        print(f"⚠️  TriviaQA download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load TriviaQA and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/triviaqa")

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            question_text = item.get("question", "")
            answer = item.get("answer", {})

            # Get primary answer and aliases
            if isinstance(answer, dict):
                correct_answer = answer.get("value", "")
                aliases = answer.get("aliases", [])
            else:
                correct_answer = str(answer)
                aliases = []

            questions.append({
                "question_text": question_text,
                "correct_answer": correct_answer,
                "answer_type": "short_text",
                "source": "triviaqa",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "factual",
                "subcategory": "cross_domain_trivia",
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "aliases": aliases
                }
            })

    print(f"✅ TriviaQA normalized: {len(questions)} questions")
    return questions
