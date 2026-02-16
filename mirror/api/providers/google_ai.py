"""
Google AI Studio provider for Gemini models.

Uses the google-genai SDK and normalizes responses to unified format.
"""

import os
import time
from typing import Optional

from google import genai
from google.genai import types

from ..rate_limiter import with_retry


class GoogleAIProvider:
    """
    Provider for Google AI Studio (Gemini).

    Handles message format conversion and response normalization.
    """

    def __init__(self):
        """Initialize Google AI client."""
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[types.Content]]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses a different message format with system instructions separate.

        Args:
            messages: OpenAI-style messages [{"role": "user", "content": "..."}]

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System messages become system_instruction
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        return system_instruction, contents

    @with_retry
    async def complete(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Send completion request to Google AI.

        Args:
            model_id: Gemini model ID (e.g., "gemini-2.5-pro-preview-05-06")
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict
        """
        start_time = time.time()

        try:
            system_instruction, contents = self._convert_messages(messages)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction if system_instruction else None,
            )

            response = await self.client.aio.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from response
            content = response.text if hasattr(response, 'text') else ""

            # Extract usage metadata
            usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None

            usage = {
                "prompt_tokens": usage_metadata.prompt_token_count if usage_metadata else 0,
                "completion_tokens": usage_metadata.candidates_token_count if usage_metadata else 0,
                "total_tokens": usage_metadata.total_token_count if usage_metadata else 0,
            }

            return {
                "content": content,
                "model_id": model_id,
                "provider": "google_ai",
                "usage": usage,
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        except Exception as e:
            error_msg = str(e)
            raise Exception(f"Google AI API error: {error_msg}") from e

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
            model_id: Gemini model ID
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Normalized response dict
        """
        start_time = time.time()

        try:
            system_instruction, contents = self._convert_messages(messages)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction if system_instruction else None,
            )

            response = self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.text if hasattr(response, 'text') else ""
            usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None

            usage = {
                "prompt_tokens": usage_metadata.prompt_token_count if usage_metadata else 0,
                "completion_tokens": usage_metadata.candidates_token_count if usage_metadata else 0,
                "total_tokens": usage_metadata.total_token_count if usage_metadata else 0,
            }

            return {
                "content": content,
                "model_id": model_id,
                "provider": "google_ai",
                "usage": usage,
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        except Exception as e:
            error_msg = str(e)
            raise Exception(f"Google AI API error: {error_msg}") from e
