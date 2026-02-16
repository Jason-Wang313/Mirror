"""
ReClor dataset loader - Reading comprehension logical reasoning.

Source: https://huggingface.co/datasets/metaeval/reclor (or alternatives)
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download ReClor dataset to data/raw/reclor/"""
    output_dir = Path("data/raw/reclor")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading ReClor dataset...")
        # Try multiple possible dataset names
        dataset = None
        for name in ["metaeval/reclor", "reclor"]:
            try:
                dataset = load_dataset(name, trust_remote_code=True)
                break
            except:
                continue

        if dataset:
            for split in dataset.keys():
                with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                    json.dump([dict(item) for item in dataset[split]], f, indent=2)
            print(f"✅ ReClor downloaded")
        else:
            print("⚠️  ReClor not available (this is non-critical)")

    except Exception as e:
        print(f"⚠️  ReClor download failed: {e} (this is non-critical)")


def load_and_normalize() -> list[dict]:
    """
    Load ReClor and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/reclor")

    if not output_dir.exists() or not list(output_dir.glob("*.json")):
        print("ReClor not downloaded, skipping...")
        return []

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            context = item.get("context", "")
            question = item.get("question", "")
            answers = item.get("answers", [])
            label = item.get("label", 0)

            # Build full question
            question_text = f"{context}\n\n{question}\n\n"
            for i, ans in enumerate(answers):
                question_text += f"{chr(65+i)}. {ans}\n"

            correct_answer = chr(65 + label) if label < len(answers) else "A"

            questions.append({
                "question_text": question_text.strip(),
                "correct_answer": correct_answer,
                "answer_type": "multiple_choice",
                "source": "reclor",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "logical",
                "subcategory": "conditional_reasoning",
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "options": answers
                }
            })

    print(f"✅ ReClor normalized: {len(questions)} questions")
    return questions
