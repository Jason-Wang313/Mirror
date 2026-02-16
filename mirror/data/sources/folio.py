"""
FOLIO dataset loader - First-order logic reasoning.

Source: https://huggingface.co/datasets/yale-nlp/FOLIO
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download FOLIO dataset to data/raw/folio/"""
    output_dir = Path("data/raw/folio")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading FOLIO dataset...")
        dataset = load_dataset("yale-nlp/FOLIO", trust_remote_code=True)

        for split in dataset.keys():
            with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump([dict(item) for item in dataset[split]], f, indent=2)

        print(f"✅ FOLIO downloaded")

    except Exception as e:
        print(f"⚠️  FOLIO download failed: {e} (this is non-critical)")


def load_and_normalize() -> list[dict]:
    """
    Load FOLIO and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/folio")

    if not output_dir.exists() or not list(output_dir.glob("*.json")):
        print("FOLIO not downloaded, skipping...")
        return []

    questions = []

    for split_file in output_dir.glob("*.json"):
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            premises = item.get("premises", "")
            conclusion = item.get("conclusion", "")
            label = item.get("label", True)

            # Build question
            question_text = f"Given the following premises:\n{premises}\n\nIs this conclusion valid?\n{conclusion}"

            questions.append({
                "question_text": question_text,
                "correct_answer": "True" if label else "False",
                "answer_type": "boolean",
                "source": "folio",
                "source_id": f"{split_file.stem}_{idx}",
                "domain": "logical",
                "subcategory": "proof_evaluation",
                "difficulty": None,
                "metadata": {
                    "split": split_file.stem,
                    "premises": premises,
                    "conclusion": conclusion
                }
            })

    print(f"✅ FOLIO normalized: {len(questions)} questions")
    return questions
