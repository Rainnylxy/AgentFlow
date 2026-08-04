"""Circuit breaker with CLOSED → OPEN → HALF_OPEN state machine.

Classic implementation modeled after Netflix Hystrix / resilience4j.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"           # Normal operation — requests pass through
    OPEN = "open"               # Circuit tripped — requests fail fast
    HALF_OPEN = "half_open"     # Probing — one trial request allowed


@dataclass
class CircuitStats:
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float
    opened_at: float


class CircuitBreaker:
    """A thread-safe circuit breaker for async operations.

    Usage::

        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        async def call_api():
            if not cb.allow():
                raise CircuitOpenError("Circuit is open")

            try:
                result = await api.call()
                cb.on_success()
                return result
            except Exception:
                cb.on_failure()
                raise

    The breaker can wrap any async callable via the :func:`resilient` decorator
    — manual allow/on_success/on_failure is usually unnecessary.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 1,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self._half_open_requests = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self) -> bool:
        """Return True if a request should be allowed through."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_requests = 0
            else:
                return False

        # HALF_OPEN — allow one probe request
        if self._half_open_requests < self.half_open_max_requests:
            self._half_open_requests += 1
            return True
        return False

    def on_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
        else:
            self._failure_count = 0

    def on_failure(self) -> None:
        """Record a failed request."""
        self._last_failure_time = time.time()
        self._failure_count += 1

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitStats:
        return CircuitStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            opened_at=self._opened_at,
        )

    def reset(self) -> None:
        """Force the breaker back to CLOSED (e.g. for testing)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self._half_open_requests = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._failure_count = 0
        self._half_open_requests = 0
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()


from agentflow.errors import AgentFlowCircuitOpenError


class CircuitOpenError(AgentFlowCircuitOpenError):
    """Raised (or used internally) when the circuit is open."""
