"""Fallback policies for when all retries are exhausted."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Awaitable


class FallbackPolicy(str, Enum):
    RAISE = "raise"     # Re-raise the original exception
    SKIP = "skip"       # Return None / skip the operation
    DEFAULT = "default" # Return a pre-configured default value
    CALL = "call"       # Call a fallback function


@dataclass
class FallbackConfig:
    policy: FallbackPolicy = FallbackPolicy.RAISE
    default_value: Any = None
    fallback_fn: Callable[[], Awaitable[Any]] | None = None


async def execute_fallback(
    config: FallbackConfig,
    error: Exception,
) -> Any:
    """Execute the fallback and return a result (or re-raise)."""
    if config.policy == FallbackPolicy.RAISE:
        raise error
    elif config.policy == FallbackPolicy.SKIP:
        return None
    elif config.policy == FallbackPolicy.DEFAULT:
        return config.default_value
    elif config.policy == FallbackPolicy.CALL:
        if config.fallback_fn is None:
            raise ValueError("FallbackPolicy.CALL requires fallback_fn")
        return await config.fallback_fn()
    raise ValueError(f"Unknown fallback policy: {config.policy}")
