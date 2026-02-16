"""
Quick test for Gemini 2.5 Pro model ID.

Run with: python tests/test_gemini_quick.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from mirror.api import UnifiedClient


def main():
    """Test Gemini 2.5 Pro."""
    print("="*60)
    print("Testing Gemini 2.5 Pro Model ID")
    print("="*60)

    # Load environment variables
    load_dotenv()

    if not os.getenv("GOOGLE_AI_API_KEY"):
        print("\n❌ GOOGLE_AI_API_KEY not found in .env file")
        return False

    try:
        # Initialize client
        client = UnifiedClient(experiment="gemini_test")

        # Test the model
        print("\nTesting model: gemini-2.5-pro")
        print("Model ID in registry: gemini-2.5-pro")

        response = client.complete_sync(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one sentence."}]
        )

        if "error" in response:
            print(f"\n❌ FAILED: {response['error']}")
            return False

        print(f"\n✅ SUCCESS!")
        print(f"Response: {response['content'][:200]}")
        print(f"Latency: {response['latency_ms']:.2f}ms")
        print(f"Tokens used: {response['usage']['total_tokens']}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
