"""AgentFlow structured error hierarchy.

All AgentFlow exceptions inherit from AgentFlowError. This gives callers
a single root to catch, plus granular subtypes for specific handling.

Usage::

    try:
        result = await agent.run("do something")
    except AgentFlowTimeoutError:
        # handle timeout specifically
    except AgentFlowToolError as e:
        # e.tool_name, e.tool_input are available
    except AgentFlowError:
        # catch-all for any agentflow failure
"""

from __future__ import annotations


# =============================================================================
# Root
# =============================================================================


class AgentFlowError(Exception):
    """Base for all AgentFlow exceptions.

    Attributes:
        message: Human-readable error description.
        context: Optional dict with structured error metadata.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


# =============================================================================
# LLM errors
# =============================================================================


class AgentFlowLLMError(AgentFlowError):
    """LLM API call failed after all retries exhausted."""

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        status_code: int | None = None,
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"model": model, "status_code": status_code})
        super().__init__(message, ctx)
        self.model = model
        self.status_code = status_code


class AgentFlowTokenLimitError(AgentFlowLLMError):
    """Input or output exceeded token limits."""

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        max_tokens: int = 0,
        actual_tokens: int = 0,
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"max_tokens": max_tokens, "actual_tokens": actual_tokens})
        super().__init__(message, model=model, context=ctx)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


# =============================================================================
# Tool errors
# =============================================================================


class AgentFlowToolError(AgentFlowError):
    """Tool execution failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        tool_input: dict | None = None,
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"tool_name": tool_name, "tool_input": tool_input})
        super().__init__(message, ctx)
        self.tool_name = tool_name
        self.tool_input = tool_input or {}


class AgentFlowToolNotFoundError(AgentFlowToolError):
    """Referenced tool is not registered."""

    pass


class AgentFlowToolParamError(AgentFlowToolError):
    """Tool parameter validation failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        tool_input: dict | None = None,
        param: str = "",
        expected: str = "",
        got: str = "",
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"param": param, "expected": expected, "got": got})
        super().__init__(message, tool_name=tool_name, tool_input=tool_input, context=ctx)
        self.param = param
        self.expected = expected
        self.got = got


# =============================================================================
# Security / policy errors
# =============================================================================


class AgentFlowSecurityError(AgentFlowError):
    """Tool execution blocked by security policy."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        policy_verdict: str = "",
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"tool_name": tool_name, "verdict": policy_verdict})
        super().__init__(message, ctx)
        self.tool_name = tool_name
        self.policy_verdict = policy_verdict


# =============================================================================
# Resilience errors
# =============================================================================


class AgentFlowCircuitOpenError(AgentFlowError):
    """Circuit breaker is open — request rejected without execution."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        *,
        context: dict | None = None,
    ):
        super().__init__(message, context)


class AgentFlowRetryExhaustedError(AgentFlowError):
    """All retry attempts exhausted."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        last_error: str = "",
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"attempts": attempts, "last_error": last_error})
        super().__init__(message, ctx)
        self.attempts = attempts
        self.last_error = last_error


# =============================================================================
# Timeout errors
# =============================================================================


class AgentFlowTimeoutError(AgentFlowError):
    """Operation timed out."""

    def __init__(
        self,
        message: str,
        *,
        timeout_ms: int = 0,
        node_id: str = "",
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"timeout_ms": timeout_ms, "node_id": node_id})
        super().__init__(message, ctx)
        self.timeout_ms = timeout_ms
        self.node_id = node_id


# =============================================================================
# Workflow / DSL errors
# =============================================================================


class AgentFlowWorkflowError(AgentFlowError):
    """Workflow definition or execution error."""

    def __init__(
        self,
        message: str,
        *,
        workflow_name: str = "",
        node_id: str = "",
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"workflow_name": workflow_name, "node_id": node_id})
        super().__init__(message, ctx)
        self.workflow_name = workflow_name
        self.node_id = node_id


class AgentFlowDSLError(AgentFlowWorkflowError):
    """DSL validation or parse error."""

    pass


# =============================================================================
# Budget errors
# =============================================================================


class AgentFlowBudgetError(AgentFlowError):
    """Budget cap exceeded."""

    def __init__(
        self,
        message: str,
        *,
        limit_usd: float = 0.0,
        spent_usd: float = 0.0,
        context: dict | None = None,
    ):
        ctx = context or {}
        ctx.update({"limit_usd": limit_usd, "spent_usd": spent_usd})
        super().__init__(message, ctx)
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
