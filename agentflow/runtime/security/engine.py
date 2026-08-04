"""PolicyEngine — pre-execution check + post-execution audit for tools."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from agentflow.runtime.security.policy import (
    AuditEntry,
    CallContext,
    PolicyResult,
    PolicyVerdict,
    SecurityPolicy,
)
from agentflow.runtime.security.sanitizer import sanitize


class PolicyEngine:
    """Evaluates security policies before tool execution and records audits.

    Usage::

        engine = PolicyEngine(default_deny=False)
        engine.register(SecurityPolicy(
            tool_name="delete_file",
            require_approval=True,
            sensitive_params=["path"],
        ))

        # Before tool execution
        result = engine.check("delete_file", context, params)
        if not result.allowed:
            raise PermissionError(result.reason)

        # After tool execution
        engine.audit("delete_file", context, params, tool_result)
    """

    def __init__(self, default_deny: bool = False):
        self._policies: dict[str, SecurityPolicy] = {}
        self.default_deny = default_deny

        # Rate limiting state
        self._call_timestamps: dict[str, list[float]] = defaultdict(list)

        # Per-session call counters: (tool_name, session_id) -> count
        self._session_counters: dict[tuple[str, str], int] = defaultdict(int)

        # Pending approvals: approval_id -> (tool_name, params)
        self._pending: dict[str, tuple[str, dict]] = {}

        # Audit log
        self._audit_log: list[AuditEntry] = []

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def register(self, policy: SecurityPolicy) -> None:
        self._policies[policy.tool_name] = policy

    def unregister(self, tool_name: str) -> None:
        self._policies.pop(tool_name, None)

    def get_policy(self, tool_name: str) -> SecurityPolicy | None:
        return self._policies.get(tool_name)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        tool_name: str,
        context: CallContext,
        params: dict,
    ) -> PolicyResult:
        """Evaluate all applicable policies before tool execution.

        Returns a PolicyResult. If *verdict* is ALLOW, the caller should
        proceed; if DENY or PENDING_APPROVAL it should block.
        """
        policy = self._policies.get(tool_name)

        # --- No policy registered ---
        if policy is None:
            if self.default_deny:
                return PolicyResult(
                    verdict=PolicyVerdict.DENY,
                    reason=f"Tool '{tool_name}' has no security policy (default-deny enabled)",
                )
            return PolicyResult(verdict=PolicyVerdict.ALLOW)

        # --- Blocked agents ---
        if context.agent_id and context.agent_id in policy.blocked_agents:
            return PolicyResult(
                verdict=PolicyVerdict.DENY,
                reason=f"Agent '{context.agent_id}' is blocked from '{tool_name}'",
            )

        # --- Allowed agents ---
        if policy.allowed_agents is not None and context.agent_id not in policy.allowed_agents:
            return PolicyResult(
                verdict=PolicyVerdict.DENY,
                reason=f"Agent '{context.agent_id}' is not allowed to use '{tool_name}'",
            )

        # --- Per-session call cap ---
        if policy.max_calls_per_session is not None and context.session_id:
            key = (tool_name, context.session_id)
            if self._session_counters[key] >= policy.max_calls_per_session:
                return PolicyResult(
                    verdict=PolicyVerdict.DENY,
                    reason=(
                        f"Tool '{tool_name}' exceeded session limit "
                        f"({policy.max_calls_per_session} calls)"
                    ),
                )

        # --- Rate limit (per minute) ---
        if policy.max_calls_per_minute is not None:
            now = time.time()
            window_start = now - 60.0
            timestamps = self._call_timestamps[tool_name]
            # Prune expired entries
            self._call_timestamps[tool_name] = [
                ts for ts in timestamps if ts > window_start
            ]
            if len(self._call_timestamps[tool_name]) >= policy.max_calls_per_minute:
                return PolicyResult(
                    verdict=PolicyVerdict.DENY,
                    reason=(
                        f"Tool '{tool_name}' rate limit exceeded "
                        f"({policy.max_calls_per_minute} calls/min)"
                    ),
                )

        # --- Approval required ---
        if policy.require_approval:
            import uuid
            approval_id = uuid.uuid4().hex[:8]
            self._pending[approval_id] = (tool_name, params)
            return PolicyResult(
                verdict=PolicyVerdict.PENDING_APPROVAL,
                reason=f"Tool '{tool_name}' requires human approval",
                approval_id=approval_id,
            )

        # --- Sanitize parameters ---
        cleaned_params = None
        if policy.sensitive_params:
            cleaned_params = sanitize(params, policy.sensitive_params)

        # --- Update counters ---
        if context.session_id:
            key = (tool_name, context.session_id)
            self._session_counters[key] += 1
        self._call_timestamps[tool_name].append(time.time())

        return PolicyResult(
            verdict=PolicyVerdict.ALLOW,
            sanitized_params=cleaned_params,
        )

    # ------------------------------------------------------------------
    # Approval flow
    # ------------------------------------------------------------------

    def approve(self, approval_id: str) -> tuple[str, dict] | None:
        """Approve a pending tool call. Returns (tool_name, params)."""
        entry = self._pending.pop(approval_id, None)
        if entry is None:
            return None
        tool_name, params = entry
        policy = self._policies.get(tool_name)
        if policy and policy.sensitive_params:
            params = sanitize(params, policy.sensitive_params)
        return tool_name, params

    def deny(self, approval_id: str) -> bool:
        """Reject a pending tool call."""
        if approval_id in self._pending:
            del self._pending[approval_id]
            return True
        return False

    @property
    def pending_approvals(self) -> dict[str, str]:
        """Return a dict of pending approval_id → tool_name."""
        return {aid: tn for aid, (tn, _) in self._pending.items()}

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(
        self,
        tool_name: str,
        context: CallContext,
        params: dict,
        *,
        success: bool = True,
        output: str = "",
        error: str = "",
    ) -> None:
        """Record an audit entry after tool execution."""
        policy = self._policies.get(tool_name)
        if policy is not None and not policy.audit:
            return
        entry = AuditEntry(
            tool_name=tool_name,
            context=context,
            params=params,
            result_success=success,
            result_output=output,
            result_error=error,
            timestamp=time.time(),
        )
        self._audit_log.append(entry)

    def audit_log(self) -> list[AuditEntry]:
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        self._audit_log.clear()

    # ------------------------------------------------------------------
    # State reset (for testing)
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear rate-limit counters, session counters, and pending approvals."""
        self._call_timestamps.clear()
        self._session_counters.clear()
        self._pending.clear()
        self._audit_log.clear()
