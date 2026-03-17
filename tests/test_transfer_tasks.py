"""
Tests for transfer task generation and validation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.experiments.transfer_tasks import (
    DOMAIN_PAIRS,
    load_transfer_tasks,
)


def test_domain_pairs_structure():
    """Test that DOMAIN_PAIRS has correct structure."""
    assert len(DOMAIN_PAIRS) == 8, "Should have 8 domain pairs"

    for pair in DOMAIN_PAIRS:
        # Required fields
        assert "source_domain" in pair
        assert "surface_domain" in pair
        assert "hidden_dependency" in pair
        assert "description" in pair
        assert "examples" in pair

        # Examples structure
        assert len(pair["examples"]) >= 3, f"Each pair should have >= 3 examples, got {len(pair['examples'])}"

        for example in pair["examples"]:
            assert "task_text" in example
            assert "correct_answer" in example
            assert "requires_from_source" in example

    print("✓ Domain pairs structure valid")


def test_task_loading():
    """Test that tasks can be loaded from JSONL."""
    # Create dummy task file
    dummy_tasks = [
        {
            "task_id": "transfer_test_001",
            "source_domain": "arithmetic",
            "surface_domain": "business",
            "hidden_dependency": "calculation",
            "task_text": "Test task",
            "correct_answer": "42",
            "requires_from_source": "math",
            "difficulty": "medium",
            "is_adversarial_disguise": False,
        }
    ]

    test_file = Path("data/test_transfer_tasks.jsonl")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, "w") as f:
        for task in dummy_tasks:
            f.write(json.dumps(task) + "\n")

    # Load
    loaded = load_transfer_tasks(str(test_file))
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "transfer_test_001"
    assert loaded[0]["source_domain"] == "arithmetic"

    # Cleanup
    test_file.unlink()

    print("✓ Task loading works")


def test_task_id_uniqueness():
    """Test that generated task IDs would be unique."""
    # Simulate ID generation
    task_ids = set()
    for pair in DOMAIN_PAIRS:
        for i in range(25):
            task_id = f"transfer_{pair['source_domain']}_{pair['surface_domain']}_{i+1:03d}"
            assert task_id not in task_ids, f"Duplicate task ID: {task_id}"
            task_ids.add(task_id)

    assert len(task_ids) == 8 * 25  # 8 pairs × 25 tasks each

    print("✓ Task ID uniqueness verified")


def test_adversarial_task_structure():
    """Test adversarial task has source != surface."""
    adversarial_task = {
        "task_id": "transfer_adv_001",
        "source_domain": "logical",
        "surface_domain": "business_evaluation",
        "hidden_dependency": "fallacy_detection",
        "task_text": "Evaluate this business claim...",
        "correct_answer": "The claim commits affirming the consequent fallacy",
        "is_adversarial_disguise": True,
    }

    assert adversarial_task["is_adversarial_disguise"] == True
    assert adversarial_task["source_domain"] != adversarial_task["surface_domain"]

    print("✓ Adversarial task structure valid")


def test_all_domains_covered():
    """Test that all 8 reasoning domains appear as source domains."""
    expected_domains = {
        "arithmetic", "spatial", "temporal", "linguistic",
        "logical", "social", "factual", "procedural",
    }

    source_domains = {pair["source_domain"] for pair in DOMAIN_PAIRS}

    assert source_domains == expected_domains, \
        f"Missing domains: {expected_domains - source_domains}"

    print("✓ All 8 domains covered")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running transfer_tasks tests...")
    print("="*60 + "\n")

    test_domain_pairs_structure()
    test_task_loading()
    test_task_id_uniqueness()
    test_adversarial_task_structure()
    test_all_domains_covered()

    print("\n" + "="*60)
    print("All tests passed ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
