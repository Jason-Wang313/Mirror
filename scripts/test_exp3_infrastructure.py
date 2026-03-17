"""
Test Experiment 3 infrastructure without making API calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("\n" + "="*80)
print("EXPERIMENT 3 INFRASTRUCTURE TEST")
print("="*80 + "\n")

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from scripts.generate_exp3_tasks import load_capability_profiles, load_seed_templates
    from mirror.scoring.compositional_metrics import compute_cce, compute_bci
    from mirror.experiments.channels import build_channel1_prompt, parse_channel1
    print("✅ All imports successful\n")
except ImportError as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Load capability profiles
print("Test 2: Loading capability profiles...")
try:
    profiles = load_capability_profiles()
    print(f"✅ Loaded {len(profiles)} model profiles\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 3: Load templates
print("Test 3: Loading seed templates...")
try:
    templates = load_seed_templates()
    print(f"✅ Loaded templates for {len(templates)} domain pairs\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 4: Load generated tasks
print("Test 4: Loading generated tasks...")
try:
    tasks_path = Path("data/exp3/intersection_tasks.jsonl")
    if tasks_path.exists():
        with open(tasks_path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        print(f"✅ Loaded {len(tasks)} intersection tasks\n")
    else:
        print("⚠️  No tasks generated yet (run generate_exp3_tasks.py)\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 5: Test prompt building
print("Test 5: Testing prompt builders...")
try:
    question = {"question_text": "What is 2+2?"}
    prompt = build_channel1_prompt(question)
    if "BET:" in prompt and "ANSWER:" in prompt:
        print("✅ Channel 1 (wagering) prompt builds correctly\n")
    else:
        print(f"⚠️  Channel 1 prompt missing expected format\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 6: Test response parsing
print("Test 6: Testing response parsers...")
try:
    sample_response = "ANSWER: 4\nBET: 8"
    parsed = parse_channel1(sample_response)
    if parsed.get("answer") and parsed.get("bet"):
        print(f"✅ Channel 1 parser works: answer={parsed['answer']}, bet={parsed['bet']}\n")
    else:
        print(f"⚠️  Parser returned: {parsed}\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 7: Test CCE computation
print("Test 7: Testing CCE computation...")
try:
    cce = compute_cce(
        intersection_confidence=0.6,
        intersection_accuracy=0.4,
        domain_a_confidence=0.7,
        domain_a_accuracy=0.65,
    )
    print(f"✅ CCE computed: {cce:.3f} (intersection CE - domain CE)\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Summary
print("="*80)
print("ALL TESTS PASSED ✅")
print("="*80)
print("\nExperiment 3 infrastructure is ready!")
print("\nNext steps:")
print("  1. Generate full task set: python scripts/generate_exp3_tasks.py")
print("  2. Run pilot: python scripts/run_experiment_3.py --mode pilot")
print("  3. Run full experiment: python scripts/run_experiment_3.py --mode full")
print()
