"""
Groq provider using OpenAI-compatible API.

Groq provides high-speed inference for open-weight models through an OpenAI-compatible interface.
"""

import os
import time
from typing import Optional

from openai import AsyncOpenAI, OpenAI

from ..rate_limiter import with_retry


class GroqProvider:
    """
    Provider for Groq API.

    Uses OpenAI SDK pointed at Groq's base URL.
    """

    def __init__(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        base_url = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url)

    @with_retry
    async def complete(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Send completion request to Groq.

        Args:
            model_id: Groq model ID (e.g., "llama-3.1-8b-instant")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (default 0 for deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict with content, usage, latency
        """
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            return {
                "content": response.choices[0].message.content,
                "model_id": model_id,
                "provider": "groq",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        except Exception as e:
            error_msg = str(e)
            # Re-raise with provider context
            raise Exception(f"Groq API error: {error_msg}") from e

    def complete_sync(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Synchronous completion request.

        Args:
            model_id: Groq model ID
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict
        """
        start_time = time.time()

        try:
            response = self.sync_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            return {
                "content": response.choices[0].message.content,
                "model_id": model_id,
                "provider": "groq",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        except Exception as e:
            error_msg = str(e)
            raise Exception(f"Groq API error: {error_msg}") from e

    async def list_models(self) -> list[str]:
        """
        List available models from Groq.

        Returns:
            List of model IDs available
        """
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise Exception(f"Failed to list Groq models: {e}") from e
