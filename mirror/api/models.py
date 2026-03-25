"""
Model registry and routing configuration for all supported LLMs.
"""

MODEL_REGISTRY = {
    # --- Llama 3.1 Scaling Family (NVIDIA NIM) ---
    "llama-3.1-8b": {
        "provider": "nvidia_nim",
        "model_id": "meta/llama-3.1-8b-instruct",
        "display_name": "Llama 3.1 8B",
        "lab": "Meta",
        "open_weight": True,
        "role": "scaling_small",
    },
    "llama-3.1-70b": {
        "provider": "nvidia_nim",
        "model_id": "meta/llama-3.1-70b-instruct",
        "display_name": "Llama 3.1 70B",
        "lab": "Meta",
        "open_weight": True,
        "role": "scaling_medium",
    },
    "llama-3.1-405b": {
        "provider": "nvidia_nim",
        "model_id": "meta/llama-3.1-405b-instruct",
        "display_name": "Llama 3.1 405B",
        "lab": "Meta",
        "open_weight": True,
        "role": "scaling_large",
    },
    # --- Frontier Models ---
    "gemini-2.5-pro": {
        "provider": "google_ai",
        "model_id": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "lab": "Google",
        "open_weight": False,
        "role": "frontier",
    },
    "deepseek-r1": {
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "display_name": "DeepSeek R1",
        "lab": "DeepSeek",
        "open_weight": False,
        "role": "reasoning",
    },
    # --- Diversity Models (NVIDIA NIM) ---
    "mistral-large": {
        "provider": "nvidia_nim",
        "model_id": "mistralai/mistral-large-3-675b-instruct-2512",
        "display_name": "Mistral Large 3",
        "lab": "Mistral",
        "open_weight": True,
        "role": "diversity",
    },
    "qwen-3-235b": {
        "provider": "nvidia_nim",
        "model_id": "qwen/qwen3-235b-a22b",
        "display_name": "Qwen 3 235B",
        "lab": "Alibaba",
        "open_weight": True,
        "role": "diversity",
    },
    "gpt-oss-120b": {
        "provider": "nvidia_nim",
        "model_id": "openai/gpt-oss-120b",
        "display_name": "GPT OSS 120B",
        "lab": "OpenAI",
        "open_weight": True,
        "role": "openai_lineage",
    },
    # --- Additional models for expanded experiments ---
    "deepseek-v3": {
        # Route through NIM mirror to avoid DeepSeek-credit outages during reruns.
        "provider": "nvidia_nim",
        "model_id": "deepseek-ai/deepseek-v3.1",
        "display_name": "DeepSeek V3",
        "lab": "DeepSeek",
        "open_weight": False,
        "role": "diversity",
    },
    "gemma-3-12b": {
        "provider": "google_ai",
        "model_id": "gemma-3-12b-it",
        "display_name": "Gemma 3 12B",
        "lab": "Google",
        "open_weight": True,
        "role": "diversity",
    },
    "gemma-3-27b": {
        "provider": "nvidia_nim",
        "model_id": "google/gemma-3-27b-it",
        "display_name": "Gemma 3 27B",
        "lab": "Google",
        "open_weight": True,
        "role": "diversity",
    },
    "kimi-k2": {
        "provider": "nvidia_nim",
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "display_name": "Kimi K2",
        "lab": "Moonshot AI",
        "open_weight": True,
        "role": "diversity",
    },
    "llama-3.2-3b": {
        "provider": "nvidia_nim",
        "model_id": "meta/llama-3.2-3b-instruct",
        "display_name": "Llama 3.2 3B",
        "lab": "Meta",
        "open_weight": True,
        "role": "scaling_tiny",
    },
    "llama-3.3-70b": {
        "provider": "nvidia_nim",
        "model_id": "meta/llama-3.3-70b-instruct",
        "display_name": "Llama 3.3 70B",
        "lab": "Meta",
        "open_weight": True,
        "role": "generation_comparison",
    },
    "mixtral-8x22b": {
        "provider": "nvidia_nim",
        "model_id": "mistralai/mixtral-8x22b-instruct-v0.1",
        "display_name": "Mixtral 8x22B",
        "lab": "Mistral",
        "open_weight": True,
        "role": "diversity",
    },
    "phi-4": {
        "provider": "nvidia_nim",
        "model_id": "microsoft/phi-4-mini-instruct",
        "display_name": "Phi-4",
        "lab": "Microsoft",
        "open_weight": True,
        "role": "diversity",
    },
    "qwen3-next-80b": {
        "provider": "nvidia_nim",
        "model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "display_name": "Qwen3 Next 80B",
        "lab": "Alibaba",
        "open_weight": True,
        "role": "diversity",
    },
}


def get_model(name: str) -> dict:
    """
    Get model config by friendly name.

    Args:
        name: Friendly model name (e.g., "llama-3.1-8b")

    Returns:
        Model configuration dictionary

    Raises:
        KeyError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models(role: str = None, provider: str = None) -> list[str]:
    """
    List model names, optionally filtered by role or provider.

    Args:
        role: Filter by model role (e.g., "scaling_small", "frontier")
        provider: Filter by provider (e.g., "nvidia_nim", "google_ai")

    Returns:
        Sorted list of model names matching the filters
    """
    models = MODEL_REGISTRY
    if role:
        models = {k: v for k, v in models.items() if v["role"] == role}
    if provider:
        models = {k: v for k, v in models.items() if v["provider"] == provider}
    return sorted(models.keys())
