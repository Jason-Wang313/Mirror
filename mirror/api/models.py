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
        "model_id": "gemini-2.5-pro-preview-05-06",
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
