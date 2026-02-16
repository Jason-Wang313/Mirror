"""
MATH dataset loader - Competition mathematics problems.

Source: https://huggingface.co/datasets/hendrycks/competition_math
"""

import json
from pathlib import Path

from datasets import load_dataset


def download() -> None:
    """Download MATH dataset to data/raw/math_dataset/"""
    output_dir = Path("data/raw/math_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Downloading MATH dataset...")
        dataset = load_dataset("hendrycks/competition_math", trust_remote_code=True)

        # Save train and test
        for split in ["train", "test"]:
            if split in dataset:
                with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                    json.dump([dict(item) for item in dataset[split]], f, indent=2)

        print(f"✅ MATH dataset downloaded")

    except Exception as e:
        print(f"⚠️  MATH dataset download failed: {e}")


def load_and_normalize() -> list[dict]:
    """
    Load MATH dataset and normalize to standard format.

    Returns:
        List of normalized question dicts
    """
    output_dir = Path("data/raw/math_dataset")

    if not (output_dir / "train.json").exists():
        print("MATH dataset not downloaded, skipping...")
        return []

    questions = []

    # Type to domain mapping
    type_domain_map = {
        "Algebra": "arithmetic",
        "Number Theory": "arithmetic",
        "Counting & Probability": "arithmetic",
        "Geometry": "spatial",
        "Intermediate Algebra": "arithmetic",
        "Prealgebra": "arithmetic",
        "Precalculus": "arithmetic",
    }

    type_subcategory_map = {
        "Algebra": "numerical_reasoning",
        "Number Theory": "numerical_reasoning",
        "Counting & Probability": "numerical_reasoning",
        "Geometry": "2d_geometry",
        "Intermediate Algebra": "numerical_reasoning",
        "Prealgebra": "mental_calculation",
        "Precalculus": "numerical_reasoning",
    }

    for split in ["train", "test"]:
        filepath = output_dir / f"{split}.json"
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            problem_type = item.get("type", "Algebra")
            domain = type_domain_map.get(problem_type, "arithmetic")
            subcategory = type_subcategory_map.get(problem_type, "numerical_reasoning")

            questions.append({
                "question_text": item["problem"],
                "correct_answer": item["solution"].split("\\boxed{")[-1].rstrip("}").strip() if "\\boxed{" in item["solution"] else item["solution"],
                "answer_type": "exact_numeric" if domain == "arithmetic" else "short_text",
                "source": "math_dataset",
                "source_id": f"{split}_{idx}",
                "domain": domain,
                "subcategory": subcategory,
                "difficulty": None,
                "metadata": {
                    "split": split,
                    "type": problem_type,
                    "level": item.get("level", "unknown"),
                    "full_solution": item["solution"]
                }
            })

    print(f"✅ MATH dataset normalized: {len(questions)} questions")
    return questions
