"""
DeepSeek provider using OpenAI-compatible API.

DeepSeek R1 includes reasoning content in a separate field.
"""

import os
import time
from typing import Optional

from openai import AsyncOpenAI, OpenAI

from ..rate_limiter import with_retry


class DeepSeekProvider:
    """
    Provider for DeepSeek API.

    Uses OpenAI SDK pointed at DeepSeek's base URL.
    Handles special reasoning_content field for DeepSeek R1.
    """

    def __init__(self):
        """Initialize DeepSeek client."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        base_url = "https://api.deepseek.com"

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
        Send completion request to DeepSeek.

        Args:
            model_id: DeepSeek model ID (e.g., "deepseek-reasoner")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (default 0 for deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict with content, usage, latency
            For DeepSeek R1, includes reasoning_content field
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

            # Extract main content
            message = response.choices[0].message
            content = message.content

            # Check for reasoning_content (DeepSeek R1 specific)
            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            result = {
                "content": content,
                "model_id": model_id,
                "provider": "deepseek",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            # Add reasoning content if present
            if reasoning_content:
                result["reasoning_content"] = reasoning_content

            return result

        except Exception as e:
            error_msg = str(e)
            raise Exception(f"DeepSeek API error: {error_msg}") from e

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
            model_id: DeepSeek model ID
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

            message = response.choices[0].message
            content = message.content

            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            result = {
                "content": content,
                "model_id": model_id,
                "provider": "deepseek",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            if reasoning_content:
                result["reasoning_content"] = reasoning_content

            return result

        except Exception as e:
            error_msg = str(e)
            raise Exception(f"DeepSeek API error: {error_msg}") from e
