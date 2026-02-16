"""
API integration test script.

Tests all three providers with a simple query and verifies:
1. Successful API calls
2. Valid response format
3. Token usage tracking
4. Latency measurement
5. JSONL logging

Run with: python tests/test_api.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from mirror.api import UnifiedClient


def test_provider(client: UnifiedClient, model: str, test_name: str):
    """
    Test a single model.

    Args:
        client: UnifiedClient instance
        model: Model name to test
        test_name: Descriptive name for the test
    """
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": "What is 2+2? Answer in one sentence."}]

    try:
        response = client.complete_sync(model=model, messages=messages)

        # Check if error occurred
        if "error" in response:
            print(f"❌ FAILED: {response['error']}")
            return False

        # Verify required fields
        required_fields = [
            "content",
            "model",
            "provider",
            "model_id",
            "usage",
            "latency_ms",
            "timestamp",
        ]

        missing_fields = [field for field in required_fields if field not in response]

        if missing_fields:
            print(f"❌ FAILED: Missing fields: {missing_fields}")
            return False

        # Print results
        print(f"\n✅ SUCCESS")
        print(f"\nResponse: {response['content'][:200]}")
        if len(response["content"]) > 200:
            print("...")

        print(f"\nProvider: {response['provider']}")
        print(f"Model ID: {response['model_id']}")
        print(f"Latency: {response['latency_ms']:.2f}ms")
        print(f"\nToken Usage:")
        print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"  Completion tokens: {response['usage']['completion_tokens']}")
        print(f"  Total tokens: {response['usage']['total_tokens']}")

        # Check for reasoning content (DeepSeek R1)
        if "reasoning_content" in response:
            print(f"\nReasoning content length: {len(response['reasoning_content'])} chars")

        return True

    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all API tests."""
    print("="*60)
    print("MIRROR API Integration Test")
    print("="*60)

    # Load environment variables
    load_dotenv()

    # Check for API keys
    required_keys = ["NVIDIA_NIM_API_KEY", "GOOGLE_AI_API_KEY", "DEEPSEEK_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"\n❌ Missing API keys in .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease add these keys to your .env file.")
        return False

    # Initialize client
    client = UnifiedClient(log_dir="results/api_logs", experiment="test_run")

    # Test models (one from each provider)
    test_cases = [
        ("llama-3.1-8b", "NVIDIA NIM - Llama 3.1 8B"),
        ("gemini-2.5-pro", "Google AI - Gemini 2.5 Pro"),
        ("deepseek-r1", "DeepSeek - R1 Reasoner"),
    ]

    results = []
    for model, test_name in test_cases:
        success = test_provider(client, model, test_name)
        results.append((test_name, success))

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False

    # Check log file
    log_files = list(Path("results/api_logs").glob("*_test_run.jsonl"))
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"\n📝 Logs written to: {latest_log}")

        # Count log entries
        with open(latest_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"   Total log entries: {len(lines)}")
    else:
        print("\n⚠️  No log files found")

    if all_passed:
        print(f"\n{'='*60}")
        print("🎉 All tests passed!")
        print(f"{'='*60}")
        return True
    else:
        print(f"\n{'='*60}")
        print("❌ Some tests failed. Check the output above.")
        print(f"{'='*60}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
