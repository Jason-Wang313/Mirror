"""
MiniMax provider using OpenAI-compatible API.

Required env var: MINIMAX_API_KEY
Optional env var: MINIMAX_BASE_URL (default: https://api.minimax.chat/v1)
"""

import os
import time

from openai import AsyncOpenAI, OpenAI

from ..rate_limiter import with_retry


class MiniMaxProvider:
    """
    Provider for MiniMax API.

    Uses OpenAI SDK pointed at MiniMax's base URL.
    """

    def __init__(self):
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable not set")

        base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    @with_retry
    async def complete(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
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
                "provider": "minimax",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:
            raise Exception(f"MiniMax API error: {e}") from e

    def complete_sync(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
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
                "provider": "minimax",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:
            raise Exception(f"MiniMax API error: {e}") from e
