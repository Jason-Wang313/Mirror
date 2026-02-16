"""
Script to list available Gemini models from Google AI Studio.

Run with: python tests/list_gemini_models.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google import genai


def main():
    """List all available Gemini models."""
    print("="*60)
    print("Google AI Studio - Available Models")
    print("="*60)

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("\n❌ GOOGLE_AI_API_KEY not found in .env file")
        return False

    try:
        # Initialize client
        client = genai.Client(api_key=api_key)

        # List models
        print("\nFetching models from Google AI Studio...")
        models = client.models.list()

        # Filter for Gemini models
        gemini_models = []
        for model in models:
            if hasattr(model, 'name') and 'gemini' in model.name.lower():
                gemini_models.append(model)

        print(f"\n✅ Found {len(gemini_models)} Gemini models\n")

        # Print details
        print(f"{'Model Name':<50} {'Display Name':<30}")
        print("="*80)

        for model in sorted(gemini_models, key=lambda m: m.name):
            display_name = getattr(model, 'display_name', 'N/A')
            print(f"{model.name:<50} {display_name:<30}")

        # Look for Gemini 2.5 Pro specifically
        print(f"\n{'='*60}")
        print("Gemini 2.5 Pro variants:")
        print(f"{'='*60}")

        gemini_25_models = [m for m in gemini_models if '2.5' in m.name or '25' in m.name]
        if gemini_25_models:
            for model in gemini_25_models:
                print(f"\n✅ {model.name}")
                if hasattr(model, 'display_name'):
                    print(f"   Display Name: {model.display_name}")
                if hasattr(model, 'description'):
                    print(f"   Description: {model.description[:100]}...")
        else:
            print("\n⚠️  No Gemini 2.5 models found. Showing latest Gemini Pro:")
            gemini_pro = [m for m in gemini_models if 'pro' in m.name.lower()]
            if gemini_pro:
                latest = gemini_pro[-1]
                print(f"\n✅ {latest.name}")
                if hasattr(latest, 'display_name'):
                    print(f"   Display Name: {latest.display_name}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
