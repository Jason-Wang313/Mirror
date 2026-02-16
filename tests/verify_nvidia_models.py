"""
Script to verify NVIDIA NIM model IDs.

Lists all available models from NVIDIA NIM API and compares them
against the model IDs in our MODEL_REGISTRY.

Run with: python tests/verify_nvidia_models.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from mirror.api.models import MODEL_REGISTRY
from mirror.api.providers.nvidia_nim import NVIDIANIMProvider


def main():
    """Verify NVIDIA NIM model IDs."""
    print("="*60)
    print("NVIDIA NIM Model Verification")
    print("="*60)

    # Load environment variables
    load_dotenv()

    if not os.getenv("NVIDIA_NIM_API_KEY"):
        print("\n❌ NVIDIA_NIM_API_KEY not found in .env file")
        return False

    try:
        # Initialize provider
        provider = NVIDIANIMProvider()

        # Get available models
        print("\nFetching available models from NVIDIA NIM API...")
        import asyncio
        available_models = asyncio.run(provider.list_models())

        print(f"\n✅ Found {len(available_models)} models")

        # Get our registered NVIDIA NIM models
        our_models = {
            k: v["model_id"]
            for k, v in MODEL_REGISTRY.items()
            if v["provider"] == "nvidia_nim"
        }

        print(f"\nRegistered NVIDIA NIM models in MODEL_REGISTRY: {len(our_models)}")

        # Check each registered model
        print(f"\n{'='*60}")
        print("Verification Results")
        print(f"{'='*60}")

        all_valid = True
        for friendly_name, model_id in our_models.items():
            if model_id in available_models:
                print(f"✅ {friendly_name}")
                print(f"   Model ID: {model_id}")
            else:
                print(f"❌ {friendly_name}")
                print(f"   Model ID: {model_id}")
                print(f"   Status: NOT FOUND in NVIDIA NIM API")
                all_valid = False

        # Suggest alternatives for failed models
        if not all_valid:
            print(f"\n{'='*60}")
            print("Available NVIDIA NIM Models (sample)")
            print(f"{'='*60}")
            print("\nShowing first 20 models (sorted):")
            for model in sorted(available_models)[:20]:
                print(f"  - {model}")

            print(f"\n...and {len(available_models) - 20} more models")

            # Search for relevant models
            print(f"\n{'='*60}")
            print("Searching for similar models...")
            print(f"{'='*60}")

            search_terms = ["llama", "mistral", "qwen", "gpt"]
            for term in search_terms:
                matches = [m for m in available_models if term.lower() in m.lower()]
                if matches:
                    print(f"\nModels containing '{term}':")
                    for model in matches[:10]:  # Show first 10 matches
                        print(f"  - {model}")

        if all_valid:
            print(f"\n{'='*60}")
            print("🎉 All model IDs verified successfully!")
            print(f"{'='*60}")
            return True
        else:
            print(f"\n{'='*60}")
            print("⚠️  Some model IDs need updating in mirror/api/models.py")
            print(f"{'='*60}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
