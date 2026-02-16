"""
Unified API client that routes requests to appropriate providers.

Single interface for all LLM providers with rate limiting, retries, and logging.
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from ..utils.logging import APILogger
from .models import get_model
from .providers.deepseek import DeepSeekProvider
from .providers.google_ai import GoogleAIProvider
from .providers.nvidia_nim import NVIDIANIMProvider
from .rate_limiter import ProviderRateLimiter


class UnifiedClient:
    """
    Single interface for all LLM providers.

    Routes to the correct provider based on model name.
    Handles rate limiting, retries, and structured logging.
    """

    def __init__(self, log_dir: str = "results/api_logs", experiment: str = "default"):
        """
        Initialize unified client.

        Args:
            log_dir: Directory for API logs
            experiment: Experiment name for log files
        """
        # Load environment variables
        load_dotenv()

        # Initialize logger
        self.logger = APILogger(log_dir=log_dir, experiment=experiment)

        # Initialize rate limiter
        self.rate_limiter = ProviderRateLimiter()

        # Initialize providers (lazy loading)
        self._providers = {}

    def _get_provider(self, provider_name: str):
        """
        Get or initialize a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance
        """
        if provider_name not in self._providers:
            if provider_name == "nvidia_nim":
                self._providers[provider_name] = NVIDIANIMProvider()
            elif provider_name == "google_ai":
                self._providers[provider_name] = GoogleAIProvider()
            elif provider_name == "deepseek":
                self._providers[provider_name] = DeepSeekProvider()
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        return self._providers[provider_name]

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Send a completion request to the appropriate provider.

        Args:
            model: Friendly model name from MODEL_REGISTRY (e.g., "llama-3.1-8b")
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (default 0 for deterministic)
            max_tokens: Maximum tokens to generate
            metadata: Optional experiment-specific metadata for logging

        Returns:
            {
                "content": str,           # The model's response text
                "model": str,             # Friendly model name
                "provider": str,          # Provider name
                "model_id": str,          # Provider-specific model ID
                "usage": {                # Token usage
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                },
                "latency_ms": float,      # Request latency
                "timestamp": str,         # ISO timestamp
                "metadata": dict,         # Pass-through metadata
            }
        """
        # Get model config
        model_config = get_model(model)
        provider_name = model_config["provider"]
        model_id = model_config["model_id"]

        # Get provider
        provider = self._get_provider(provider_name)

        # Apply rate limiting
        limiter = self.rate_limiter.get_limiter(provider_name)
        await limiter.acquire()

        try:
            # Make API call
            response = await provider.complete(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Add friendly model name and metadata
            response["model"] = model
            response["metadata"] = metadata or {}

            # Log successful request
            self.logger.log_request(
                model=model, messages=messages, response=response, metadata=metadata
            )

            return response

        except Exception as e:
            # Log error
            error_msg = str(e)
            self.logger.log_error(
                model=model, messages=messages, error=error_msg, metadata=metadata
            )

            # Return error dict instead of crashing
            return {
                "error": error_msg,
                "model": model,
                "provider": provider_name,
                "model_id": model_id,
                "metadata": metadata or {},
            }

    async def complete_batch(
        self,
        model: str,
        messages_list: list[list[dict]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        metadata_list: Optional[list[dict]] = None,
        max_concurrent: int = 5,
    ) -> list[dict]:
        """
        Run multiple completions concurrently with rate limiting.

        Args:
            model: Friendly model name
            messages_list: List of message lists
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            metadata_list: Optional list of metadata dicts (one per request)
            max_concurrent: Maximum concurrent requests

        Returns:
            List of response dicts in same order as input.
            Failed requests return {"error": str, ...} instead.
        """
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [None] * len(messages_list)

        # Create tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_complete(messages, metadata):
            async with semaphore:
                return await self.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                )

        tasks = [
            bounded_complete(messages, metadata)
            for messages, metadata in zip(messages_list, metadata_list)
        ]

        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {
                    "error": str(result),
                    "model": model,
                    "metadata": metadata_list[i] or {},
                }

        return results

    def complete_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Synchronous wrapper around complete() for simple scripts.

        Args:
            model: Friendly model name
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            metadata: Optional experiment metadata

        Returns:
            Response dict
        """
        # Get model config
        model_config = get_model(model)
        provider_name = model_config["provider"]
        model_id = model_config["model_id"]

        # Get provider
        provider = self._get_provider(provider_name)

        try:
            # Make synchronous API call
            response = provider.complete_sync(
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Add friendly model name and metadata
            response["model"] = model
            response["metadata"] = metadata or {}

            # Log successful request
            self.logger.log_request(
                model=model, messages=messages, response=response, metadata=metadata
            )

            return response

        except Exception as e:
            # Log error
            error_msg = str(e)
            self.logger.log_error(
                model=model, messages=messages, error=error_msg, metadata=metadata
            )

            # Return error dict
            return {
                "error": error_msg,
                "model": model,
                "provider": provider_name,
                "model_id": model_id,
                "metadata": metadata or {},
            }
