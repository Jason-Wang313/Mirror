"""
Quick test to verify multi-provider routing with Groq.
"""

import asyncio
from mirror.api.client import UnifiedClient
from mirror.api.models import MODEL_REGISTRY


async def test_routing():
    """Test that models route to the correct providers."""
    print("=" * 80)
    print("MULTI-PROVIDER ROUTING TEST")
    print("=" * 80)
    print()

    # Show routing configuration
    print("Current routing configuration:")
    print()

    groq_models = [k for k, v in MODEL_REGISTRY.items() if v["provider"] == "groq"]
    nim_models = [k for k, v in MODEL_REGISTRY.items() if v["provider"] == "nvidia_nim"]
    deepseek_models = [k for k, v in MODEL_REGISTRY.items() if v["provider"] == "deepseek"]

    print(f"Groq models ({len(groq_models)}):")
    for model in groq_models:
        config = MODEL_REGISTRY[model]
        print(f"  {model:20s} → {config['model_id']}")
    print()

    print(f"NVIDIA NIM models ({len(nim_models)}):")
    for model in nim_models:
        config = MODEL_REGISTRY[model]
        print(f"  {model:20s} → {config['model_id']}")
    print()

    print(f"DeepSeek models ({len(deepseek_models)}):")
    for model in deepseek_models:
        config = MODEL_REGISTRY[model]
        print(f"  {model:20s} → {config['model_id']}")
    print()

    # Test a simple request to Groq
    print("=" * 80)
    print("Testing Groq provider with llama-3.1-8b...")
    print("=" * 80)
    print()

    client = UnifiedClient(log_dir="results/test_logs", experiment="routing_test")

    try:
        response = await client.complete(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            temperature=0.0,
            max_tokens=10,
        )

        print(f"✅ Request successful!")
        print(f"   Model: {response.get('model')}")
        print(f"   Provider: {response.get('provider')}")
        print(f"   Model ID: {response.get('model_id')}")
        print(f"   Response: {response.get('content', 'N/A')}")
        print(f"   Latency: {response.get('latency_ms', 0):.0f}ms")
        print(f"   Tokens: {response.get('usage', {}).get('total_tokens', 'N/A')}")
        print()

    except Exception as e:
        print(f"❌ Request failed: {e}")
        print()

    print("=" * 80)
    print("Routing test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_routing())
