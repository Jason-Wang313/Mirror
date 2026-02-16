"""
Rate limiting and retry logic for API calls.

Implements per-provider rate limits with exponential backoff retry.
"""

import asyncio
import time
from typing import Any, Callable, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Limits requests per minute per provider.
    """

    def __init__(self, requests_per_minute: int = 30):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests allowed per minute
        """
        self.rpm = requests_per_minute
        self.semaphore = asyncio.Semaphore(requests_per_minute)
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        Blocks if rate limit exceeded, refills tokens every minute.
        """
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill

            # Refill tokens based on time passed
            if time_passed >= 60.0:
                self.tokens = self.rpm
                self.last_refill = now
            elif time_passed > 0:
                tokens_to_add = (time_passed / 60.0) * self.rpm
                self.tokens = min(self.rpm, self.tokens + tokens_to_add)
                self.last_refill = now

            # Wait if no tokens available
            while self.tokens < 1:
                await asyncio.sleep(1)
                now = time.time()
                time_passed = now - self.last_refill
                tokens_to_add = (time_passed / 60.0) * self.rpm
                self.tokens = min(self.rpm, self.tokens + tokens_to_add)
                self.last_refill = now

            self.tokens -= 1


class ProviderRateLimiter:
    """
    Manages rate limiters for multiple providers.
    """

    def __init__(self):
        """Initialize with default rate limits per provider."""
        self.limiters = {
            "nvidia_nim": RateLimiter(requests_per_minute=30),
            "google_ai": RateLimiter(requests_per_minute=30),
            "deepseek": RateLimiter(requests_per_minute=30),
        }

    def get_limiter(self, provider: str) -> RateLimiter:
        """
        Get rate limiter for a provider.

        Args:
            provider: Provider name

        Returns:
            RateLimiter instance for the provider
        """
        if provider not in self.limiters:
            # Default limiter for unknown providers
            self.limiters[provider] = RateLimiter(requests_per_minute=30)
        return self.limiters[provider]


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class APIError(Exception):
    """Raised for API errors that should be retried."""

    pass


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.

    Args:
        exception: Exception to check

    Returns:
        True if the error should be retried
    """
    # Retry on rate limit errors, server errors, and timeout errors
    if isinstance(exception, (RateLimitError, APIError)):
        return True

    # Check for OpenAI SDK specific errors
    error_msg = str(exception).lower()
    retryable_codes = ["429", "500", "503", "timeout", "rate limit"]
    return any(code in error_msg for code in retryable_codes)


def with_retry(func: Callable) -> Callable:
    """
    Decorator to add retry logic to async functions.

    Retries up to 3 times with exponential backoff (2, 4, 8 seconds).
    """
    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )(func)
