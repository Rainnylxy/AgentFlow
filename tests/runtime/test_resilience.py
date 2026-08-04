"""Tests for agentflow.runtime.resilience — CircuitBreaker, RetryPolicy, Fallback, and resilient() integration."""

from __future__ import annotations

import asyncio
import time

import pytest

from agentflow.runtime.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from agentflow.runtime.resilience.retry import RetryPolicy, retry
from agentflow.runtime.resilience.fallback import (
    FallbackConfig,
    FallbackPolicy,
    execute_fallback,
)
from agentflow.runtime.resilience import resilient


# ==============================================================================
# CircuitBreaker
# ==============================================================================

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        cb.on_success()
        assert cb.state == CircuitState.CLOSED
        cb.on_failure()
        cb.on_failure()
        # still at 2 failures after success reset
        assert cb.state == CircuitState.CLOSED

    def test_open_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

        time.sleep(0.06)
        # Next call to allow() should transition to HALF_OPEN
        assert cb.allow() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.on_failure()
        time.sleep(0.06)
        assert cb.allow() is True  # enters HALF_OPEN
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.on_failure()
        time.sleep(0.06)
        assert cb.allow() is True  # enters HALF_OPEN
        cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_only_allows_one_probe(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05,
                            half_open_max_requests=1)
        cb.on_failure()
        time.sleep(0.06)
        assert cb.allow() is True   # probe
        assert cb.allow() is False  # rejected — only 1 probe allowed

    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_stats_reflects_current_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        s = cb.stats
        assert s.state == CircuitState.CLOSED
        assert s.failure_count == 2

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_recovery_timeout_raises(self):
        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=0)


# ==============================================================================
# RetryPolicy
# ==============================================================================

class TestRetryPolicy:
    def test_exponential_backoff_delays(self):
        p = RetryPolicy(backoff="exponential", base_delay=1.0, jitter=False)
        assert p.delay(1) == 1.0   # 2^0
        assert p.delay(2) == 2.0   # 2^1
        assert p.delay(3) == 4.0   # 2^2

    def test_linear_backoff_delays(self):
        p = RetryPolicy(backoff="linear", base_delay=2.0, jitter=False)
        assert p.delay(1) == 2.0
        assert p.delay(2) == 4.0
        assert p.delay(3) == 6.0

    def test_fixed_backoff_delays(self):
        p = RetryPolicy(backoff="fixed", base_delay=3.0, jitter=False)
        assert p.delay(1) == 3.0
        assert p.delay(5) == 3.0

    def test_max_delay_cap(self):
        p = RetryPolicy(backoff="exponential", base_delay=1.0,
                        max_delay=2.0, jitter=False)
        assert p.delay(3) == 2.0  # would be 4.0, capped at 2.0

    def test_jitter_adds_randomness(self):
        p = RetryPolicy(backoff="exponential", base_delay=1.0, jitter=True)
        # jitter ±25% around 1.0 → 0.75 ~ 1.25
        delays = [p.delay(1) for _ in range(50)]
        assert any(d != 1.0 for d in delays)  # at least some jitter

    def test_invalid_backoff_raises(self):
        with pytest.raises(ValueError):
            RetryPolicy(backoff="fibonacci")

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_try(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry(fn, RetryPolicy(max_attempts=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timed out")
            return "recovered"

        result = await retry(fn, RetryPolicy(max_attempts=5, base_delay=0.01))
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_last_error(self):
        async def fn():
            raise TimeoutError("always fails")

        with pytest.raises(TimeoutError):
            await retry(fn, RetryPolicy(max_attempts=2, base_delay=0.01))

    @pytest.mark.asyncio
    async def test_retry_does_not_retry_non_retryable(self):
        """401 / 403 / 400 should not be retried."""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad request")

        with pytest.raises(ValueError):
            await retry(fn, RetryPolicy(max_attempts=3, base_delay=0.01))
        assert call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_custom_retry_predicate(self):
        """retry_on can override the default heuristic."""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("custom retry")
            return "ok"

        result = await retry(
            fn,
            RetryPolicy(
                max_attempts=5, base_delay=0.01,
                retry_on=lambda e: isinstance(e, ValueError),
            ),
        )
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        attempts = []

        async def fn():
            raise TimeoutError("fail")

        with pytest.raises(TimeoutError):
            await retry(
                fn,
                RetryPolicy(max_attempts=3, base_delay=0.01),
                on_retry=lambda attempt, exc: attempts.append(attempt),
            )
        assert attempts == [1, 2]  # called on retry 2 and 3 (1-indexed)


# ==============================================================================
# Fallback
# ==============================================================================

class TestFallback:
    @pytest.mark.asyncio
    async def test_raise_re_raises(self):
        with pytest.raises(ValueError, match="boom"):
            await execute_fallback(FallbackConfig(policy=FallbackPolicy.RAISE), ValueError("boom"))

    @pytest.mark.asyncio
    async def test_skip_returns_none(self):
        result = await execute_fallback(
            FallbackConfig(policy=FallbackPolicy.SKIP), ValueError("boom"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_default_returns_value(self):
        result = await execute_fallback(
            FallbackConfig(policy=FallbackPolicy.DEFAULT, default_value="fallback answer"),
            ValueError("boom"),
        )
        assert result == "fallback answer"

    @pytest.mark.asyncio
    async def test_call_invokes_function(self):
        async def backup():
            return "from backup"

        result = await execute_fallback(
            FallbackConfig(policy=FallbackPolicy.CALL, fallback_fn=backup),
            ValueError("boom"),
        )
        assert result == "from backup"

    @pytest.mark.asyncio
    async def test_call_without_fn_raises(self):
        with pytest.raises(ValueError, match="requires fallback_fn"):
            await execute_fallback(
                FallbackConfig(policy=FallbackPolicy.CALL, fallback_fn=None),
                ValueError("boom"),
            )


# ==============================================================================
# resilient() decorator — full integration
# ==============================================================================

class TestResilientDecorator:
    @pytest.mark.asyncio
    async def test_no_policies_passthrough(self):
        @resilient()
        async def greet(name):
            return f"Hello, {name}"

        result = await greet("World")
        assert result == "Hello, World"

    @pytest.mark.asyncio
    async def test_retry_only(self):
        call_count = 0

        @resilient(retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01))
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("transient timeout")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_and_fast_fails(self):
        call_count = 0

        @resilient(
            circuit_breaker=CircuitBreaker(failure_threshold=2, recovery_timeout=60),
            retry_policy=RetryPolicy(max_attempts=1, base_delay=0.01),  # no retry
        )
        async def doomed():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("fail")

        for _ in range(2):
            with pytest.raises(TimeoutError):
                await doomed()

        # Circuit should now be OPEN
        with pytest.raises(CircuitOpenError):
            await doomed()

        assert call_count == 2  # circuit opened, no more calls

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_fallback(self):
        call_count = 0

        @resilient(
            circuit_breaker=CircuitBreaker(failure_threshold=1, recovery_timeout=60),
            fallback=FallbackConfig(
                policy=FallbackPolicy.DEFAULT,
                default_value="fallback response",
            ),
        )
        async def risky():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("fail")

        # First call trips the breaker
        with pytest.raises(TimeoutError):
            await risky()

        # Second call hits open circuit → fallback
        result = await risky()
        assert result == "fallback response"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_preserves_function_name(self):
        @resilient(circuit_breaker=CircuitBreaker())
        async def my_func():
            return "ok"

        assert my_func.__name__ == "my_func"

    @pytest.mark.asyncio
    async def test_fallback_after_retry_exhausted(self):
        call_count = 0

        @resilient(
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.01),
            fallback=FallbackConfig(
                policy=FallbackPolicy.DEFAULT,
                default_value="degraded",
            ),
        )
        async def unreliable():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        result = await unreliable()
        assert result == "degraded"
        assert call_count == 2  # original + 1 retry
