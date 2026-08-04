"""Retry policy with exponential backoff, jitter, and predicate-based retry."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Type

logger = logging.getLogger(__name__)


def _default_retry_predicate(exc: Exception) -> bool:
    """Retry on timeout, rate-limit, and server errors; not on client errors."""
    import asyncio

    # Check by type first (more reliable than message matching)
    if isinstance(exc, (
        TimeoutError,
        asyncio.TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
    )):
        return True

    # Then check by message for HTTP/API-level errors
    retryable = (
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "server error",
        "service unavailable",
        "internal server error",
        "connection",
    )
    msg = str(exc).lower()
    return any(token in msg for token in retryable)


@dataclass
class RetryPolicy:
    """Declarative retry policy.

    Fields:
        max_attempts: Total attempts including the original call. Min 1.
        backoff: Strategy — ``"exponential"`` (2ⁿ sec), ``"linear"`` (n sec),
            or ``"fixed"`` (constant).
        base_delay: Base delay in seconds.
        max_delay: Cap on backoff delay in seconds.
        jitter: If True, add ±25% random jitter to avoid thundering herd.
        retry_on: Optional predicate ``(Exception) -> bool``. If None, a
            built-in heuristic retries on timeout / rate-limit / 5xx.
    """

    max_attempts: int = 3
    backoff: str = "exponential"
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    retry_on: Callable[[Exception], bool] | None = None

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff not in ("exponential", "linear", "fixed"):
            raise ValueError(f"Unknown backoff strategy: {self.backoff}")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")

    def delay(self, attempt: int) -> float:
        """Compute the delay before the *attempt*-th retry (1-indexed)."""
        if self.backoff == "fixed":
            d = self.base_delay
        elif self.backoff == "linear":
            d = self.base_delay * attempt
        else:  # exponential
            d = self.base_delay * (2 ** (attempt - 1))

        d = min(d, self.max_delay)

        if self.jitter:
            d = d * (0.75 + random.random() * 0.5)  # ±25%

        return d


async def retry(
    fn: Callable[[], Awaitable],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> object:
    """Execute *fn* with retry according to *policy*.

    Returns the result of *fn* on success.
    Raises the last exception if all attempts are exhausted.
    """
    predicate = policy.retry_on or _default_retry_predicate
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= policy.max_attempts:
                break
            if not predicate(exc):
                raise

            delay = policy.delay(attempt)
            logger.debug(
                "Retry attempt %d/%d after %.1fs (error: %s)",
                attempt + 1, policy.max_attempts, delay, exc,
            )
            if on_retry:
                on_retry(attempt, exc)
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]
