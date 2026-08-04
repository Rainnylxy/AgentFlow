"""Security policy data model for tool execution control."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PolicyVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    PENDING_APPROVAL = "pending_approval"


@dataclass
class CallContext:
    """Context passed to PolicyEngine.check() for each tool invocation."""

    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    user_id: str = ""


@dataclass
class PolicyResult:
    """Result of a policy check before tool execution."""

    verdict: PolicyVerdict
    reason: str = ""
    approval_id: str = ""
    sanitized_params: dict | None = None


@dataclass
class AuditEntry:
    """A single audit record for a tool invocation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str = ""
    context: CallContext = field(default_factory=CallContext)
    params: dict = field(default_factory=dict)
    result_success: bool = True
    result_output: str = ""
    result_error: str = ""
    timestamp: float = 0.0


@dataclass
class SecurityPolicy:
    """Per-tool security policy.

    Fields:
        tool_name: The tool this policy applies to.
        allowed_agents: If set, only these agent IDs may call the tool.
            None means any agent may call it.
        blocked_agents: Agent IDs explicitly denied access.
        max_calls_per_session: Per-session call cap.
        max_calls_per_minute: Rate limit (sliding window).
        require_approval: If True, calls return PENDING_APPROVAL until
            PolicyEngine.approve() is invoked.
        sensitive_params: Parameter names to sanitize before execution.
            Mapping is looked up in the sanitizer registry.
        audit: Whether to record an audit entry for every invocation.
    """

    tool_name: str
    allowed_agents: list[str] | None = None
    blocked_agents: list[str] = field(default_factory=list)
    max_calls_per_session: int | None = None
    max_calls_per_minute: int | None = None
    require_approval: bool = False
    sensitive_params: list[str] = field(default_factory=list)
    audit: bool = True
