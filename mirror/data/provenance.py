"""
Provenance tracker - builds full traceability table for all questions.

Tracks every question back to its source dataset and transformation history.
"""

import csv
import json
from pathlib import Path

import yaml


def build_provenance_table(output_path: str = "data/provenance.csv") -> dict:
    """
    Build full provenance table for all questions.

    Args:
        output_path: Path to save CSV

    Returns:
        Stats dict
    """
    print("="*60)
    print("Building Provenance Table")
    print("="*60)

    # Load all questions from verified files
    all_questions = []

    verified_dir = Path("data/verified")
    if verified_dir.exists():
        for verified_file in verified_dir.glob("*.jsonl"):
            with open(verified_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_questions.append(json.loads(line))

    # Also load counterfactuals
    cf_dir = Path("data/counterfactual")
    if cf_dir.exists():
        for cf_file in cf_dir.glob("*.jsonl"):
            with open(cf_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_questions.append(json.loads(line))

    print(f"Total questions: {len(all_questions)}")

    # Build CSV
    fieldnames = [
        "question_id",
        "domain",
        "subcategory",
        "difficulty",
        "source_dataset",
        "source_question_id",
        "transformation_type",
        "parent_question_id",
        "cross_llm_agreement",
        "difficulty_validated",
        "answer_type",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in all_questions:
            verification = q.get("verification", {})

            row = {
                "question_id": q.get("question_id", ""),
                "domain": q.get("domain", ""),
                "subcategory": q.get("subcategory", ""),
                "difficulty": q.get("difficulty", ""),
                "source_dataset": q.get("source", ""),
                "source_question_id": q.get("source_id", ""),
                "transformation_type": q.get("transformation", ""),
                "parent_question_id": q.get("parent_id", ""),
                "cross_llm_agreement": verification.get("cross_llm_agreement", ""),
                "difficulty_validated": verification.get("difficulty_validated", ""),
                "answer_type": q.get("answer_type", ""),
            }

            writer.writerow(row)

    print(f"✅ Provenance table saved to {output_path}")

    # Stats
    stats = {
        "total_questions": len(all_questions),
        "by_domain": {},
        "by_transformation": {},
        "by_source": {},
    }

    for q in all_questions:
        domain = q.get("domain", "unknown")
        transformation = q.get("transformation", "unknown")
        source = q.get("source", "unknown")

        stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
        stats["by_transformation"][transformation] = stats["by_transformation"].get(transformation, 0) + 1
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

    return stats


def compile_final_question_bank(output_path: str = "data/questions.jsonl") -> int:
    """
    Compile final question bank from all verified questions.

    Args:
        output_path: Path to save final JSONL

    Returns:
        Total number of questions
    """
    print("="*60)
    print("Compiling Final Question Bank")
    print("="*60)

    all_questions = []

    # Load from verified
    verified_dir = Path("data/verified")
    if verified_dir.exists():
        for verified_file in verified_dir.glob("*.jsonl"):
            with open(verified_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_questions.append(json.loads(line))

    # Load counterfactuals
    cf_dir = Path("data/counterfactual")
    if cf_dir.exists():
        for cf_file in cf_dir.glob("*.jsonl"):
            with open(cf_file, "r", encoding="utf-8") as f:
                for line in f:
                    all_questions.append(json.loads(line))

    # Save final bank
    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"✅ Final question bank: {len(all_questions)} questions")
    print(f"   Saved to {output_path}")

    return len(all_questions)
