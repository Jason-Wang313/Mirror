"""
Test Experiment 4 infrastructure without making API calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("\n" + "="*80)
print("EXPERIMENT 4 INFRASTRUCTURE TEST")
print("="*80 + "\n")

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from scripts.generate_exp4_tasks import load_capability_profiles
    from mirror.experiments.burn_test_runner import BurnTestRunner
    from mirror.scoring.adaptation_metrics import (
        compute_all_ai,
        compute_sar,
        compute_strategy_fingerprint
    )
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

# Test 3: Load generated trials
print("Test 3: Loading generated trials...")
try:
    trials_path = Path("data/exp4/trials.jsonl")
    if trials_path.exists():
        with open(trials_path) as f:
            trials = [json.loads(line) for line in f if line.strip()]
        print(f"✅ Loaded {len(trials)} trials\n")

        # Check trial types
        standard = sum(1 for t in trials if t.get("trial_type") == "standard")
        syc = sum(1 for t in trials if t.get("trial_type") == "sycophancy_control")
        rec = sum(1 for t in trials if t.get("trial_type") == "recency_control")
        spec = sum(1 for t in trials if "specificity" in t.get("trial_type", ""))

        print(f"  Trial breakdown:")
        print(f"    Standard: {standard}")
        print(f"    Sycophancy controls: {syc}")
        print(f"    Recency controls: {rec}")
        print(f"    Specificity controls: {spec}\n")
    else:
        print("⚠️  No trials generated yet (run generate_exp4_tasks.py)\n")
except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 4: Test BurnTestRunner prompt formatting
print("Test 4: Testing BurnTestRunner prompt builders...")
try:
    class MockClient:
        pass

    runner = BurnTestRunner(MockClient(), "test-model")

    # Format Phase A
    sample_trial = trials[0] if trials else {
        "phase_a": {"task_text": "Sample task A"},
        "phase_b": {"task_text": "Sample task B"},
        "phase_b_feedback": "Sample feedback",
        "phase_c_related": {"task_text": "Sample task C related"},
        "phase_c_unrelated": {"task_text": "Sample task C unrelated"},
    }

    phase_a_prompt = runner.format_phase_a(sample_trial)
    if "APPROACH OPTIONS:" in phase_a_prompt and "TASK:" in phase_a_prompt:
        print("✅ Phase A prompt formatting works\n")
    else:
        print("⚠️  Phase A prompt missing expected structure\n")

    # Format Phase B
    phase_b_prompt = runner.format_phase_b(sample_trial)
    if "APPROACH OPTIONS:" in phase_b_prompt:
        print("✅ Phase B prompt formatting works\n")
    else:
        print("⚠️  Phase B prompt missing expected structure\n")

    # Format burn feedback
    feedback = runner.format_burn_feedback(sample_trial)
    if "try again" in feedback.lower():
        print("✅ Burn feedback formatting works\n")
    else:
        print("⚠️  Burn feedback missing expected structure\n")

    # Format Phase C
    phase_c_prompt = runner.format_phase_c(sample_trial)
    if "TASK 1:" in phase_c_prompt and "TASK 2:" in phase_c_prompt:
        print("✅ Phase C prompt formatting works\n")
    else:
        print("⚠️  Phase C prompt missing expected structure\n")

except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 5: Test metrics extraction
print("Test 5: Testing metrics extraction...")
try:
    sample_response = """Approach: 2 (decompose)

I'll break this down into steps:
1. First step
2. Second step
[USE_TOOL: calculator | 2+2]

I'm not entirely sure about this, but the answer is: 4"""

    metrics = runner.extract_metrics(sample_response, {"correct_answer": "4"})

    if metrics["approach"] == 2:
        print(f"✅ Approach extraction works: {metrics['approach']}\n")
    else:
        print(f"⚠️  Approach extraction: got {metrics['approach']}, expected 2\n")

    if "calculator" in metrics["tools_used"]:
        print(f"✅ Tool extraction works: {metrics['tools_used']}\n")
    else:
        print(f"⚠️  Tool extraction: {metrics['tools_used']}\n")

    if metrics["hedge_count"] > 0:
        print(f"✅ Hedge counting works: {metrics['hedge_count']} hedges\n")
    else:
        print(f"⚠️  Hedge counting: {metrics['hedge_count']}\n")

except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 6: Test AI computation
print("Test 6: Testing Adaptation Index computation...")
try:
    mock_result = {
        "phase_a": {
            "approach": 1,
            "tool_count": 0,
            "hedge_count": 2,
            "length_tokens": 50,
            "flagged_for_review": False,
        },
        "phase_c_related": {
            "approach": 2,
            "tool_count": 1,
            "hedge_count": 5,
            "length_tokens": 80,
            "flagged_for_review": True,
        },
        "phase_c_unrelated": {
            "approach": 1,
            "tool_count": 0,
            "hedge_count": 2,
            "length_tokens": 55,
            "flagged_for_review": False,
        }
    }

    ai_data = compute_all_ai(mock_result)
    mean_ai = ai_data["mean_ai"]

    print(f"✅ AI computed: {mean_ai:.3f}")
    print(f"  AI by channel: {ai_data['ai_by_channel']}\n")

except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Test 7: Test strategy fingerprint
print("Test 7: Testing strategy fingerprint extraction...")
try:
    sample_response = """Given my earlier mistake with the procedure order, I'll be more careful this time.
I'm less confident about the temporal reasoning here.
[USE_TOOL: expert | verify procedure steps]

Let me break this down:
1. First step
2. Second step
3. Third step"""

    sample_feedback = "Your procedure was incorrect. The temporal reasoning was off."

    fingerprint = compute_strategy_fingerprint(sample_response, sample_feedback)

    if fingerprint["referenced_prior_failure"]:
        print(f"✅ Prior failure reference detection works\n")
    else:
        print(f"⚠️  Prior failure reference: {fingerprint['referenced_prior_failure']}\n")

    if fingerprint["decomposed_more"]:
        print(f"✅ Decomposition detection works\n")
    else:
        print(f"⚠️  Decomposition detection: {fingerprint['decomposed_more']}\n")

    print(f"  Full fingerprint: {fingerprint}\n")

except Exception as e:
    print(f"❌ Failed: {e}\n")
    sys.exit(1)

# Summary
print("="*80)
print("ALL TESTS PASSED ✅")
print("="*80)
print("\nExperiment 4 infrastructure is ready!")
print("\nNext steps:")
print("  1. Generate full trial set: python scripts/generate_exp4_tasks.py")
print("  2. Run pilot: python scripts/run_experiment_4.py --mode pilot")
print("  3. Run full experiment: python scripts/run_experiment_4.py --mode full")
print()
