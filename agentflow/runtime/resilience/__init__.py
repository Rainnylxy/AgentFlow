"""AgentFlow Resilience — circuit breaker, retry, and fallback for LLM calls.

Decorate any async callable with ``@resilient(...)`` to compose all three
mechanisms without modifying the underlying function::

    from agentflow.runtime.resilience import resilient, CircuitBreaker, RetryPolicy

    @resilient(
        circuit_breaker=CircuitBreaker(failure_threshold=5, recovery_timeout=30),
        retry=RetryPolicy(max_attempts=3, backoff="exponential"),
    )
    async def chat(messages, **kwargs):
        return await openai.chat.completions.create(...)
"""

from __future__ import annotations

from typing import Optional

from agentflow.runtime.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitStats,
)
from agentflow.runtime.resilience.retry import RetryPolicy, retry
from agentflow.runtime.resilience.fallback import FallbackPolicy, FallbackConfig, execute_fallback

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CircuitStats",
    "FallbackConfig",
    "FallbackPolicy",
    "RetryPolicy",
    "execute_fallback",
    "resilient",
    "retry",
]


def resilient(
    *,
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_policy: Optional[RetryPolicy] = None,
    fallback: Optional[FallbackConfig] = None,
):
    """Decorator that composes CircuitBreaker → Retry → Fallback.

    Execution order for each call::

        1. cb.allow() — fast-fail if circuit is OPEN
        2. retry(fn, retry_policy) — retry with backoff
        3. on final failure, execute fallback (or re-raise)

    Args:
        circuit_breaker: Optional CircuitBreaker instance.
        retry_policy: Optional RetryPolicy. If None, failures propagate after
            circuit breaker check.
        fallback: Optional FallbackConfig for graceful degradation.
    """

    def decorator(fn):
        async def wrapper(*args, **kwargs):
            cb = circuit_breaker

            # 1. Circuit breaker gate
            if cb is not None and not cb.allow():
                if fallback is not None:
                    return await execute_fallback(
                        fallback,
                        CircuitOpenError(f"Circuit is {cb.state.value} — request rejected"),
                    )
                raise CircuitOpenError(f"Circuit is {cb.state.value}")

            # 2. Execute with retry
            try:
                if retry_policy is not None:
                    async def _call():
                        return await fn(*args, **kwargs)
                    result = await retry(_call, retry_policy)
                else:
                    result = await fn(*args, **kwargs)

                if cb is not None:
                    cb.on_success()
                return result

            except CircuitOpenError:
                raise
            except Exception as exc:
                if cb is not None:
                    cb.on_failure()
                # Fallback only applies when retries were configured and exhausted.
                # Without a retry policy, the first failure should propagate
                # (possibly tripping the circuit breaker for the next caller).
                if fallback is not None and retry_policy is not None:
                    return await execute_fallback(fallback, exc)
                raise

        # Preserve metadata
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        return wrapper

    return decorator
