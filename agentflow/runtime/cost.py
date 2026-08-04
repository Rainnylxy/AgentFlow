"""Cost Management — token-to-cost tracking, budget caps, and attribution.

Prices are per 1M tokens (input / output). Add models as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Model pricing per 1M tokens (USD, as of 2026-08).
# Key: model id prefix → (input_price, output_price)
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-haiku-4.5": (0.80, 4.00),
    "claude-fable-5": (15.00, 75.00),
    "deepseek-v4": (0.27, 1.10),
    "deepseek-r1": (0.55, 2.19),
    "qwen3-8b": (0.02, 0.06),
}


def get_model_price(model_id: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a model.

    Falls back to (1.00, 4.00) for unknown models.
    """
    lowered = model_id.lower()
    for prefix, prices in _MODEL_PRICES.items():
        if lowered.startswith(prefix):
            return prices
    return (1.00, 4.00)


def _tokens_to_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
) -> float:
    in_price, out_price = get_model_price(model_id)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return round(cost, 6)


@dataclass
class CostEntry:
    """Cost for a single unit of work (agent / workflow / session)."""
    category: str          # "agent", "workflow", "session"
    name: str              # agent_id, workflow_name, or session_id
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class CostTracker:
    """Accumulate token usage and compute costs.

    Usage::

        tracker = CostTracker()
        tracker.record("agent", "support_agent", "gpt-4o",
                       input_tokens=500, output_tokens=200)

        total = tracker.total_cost()
        breakdown = tracker.by_category()
    """

    def __init__(self):
        self._entries: list[CostEntry] = []

    def record(
        self,
        category: str,
        name: str,
        model_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> CostEntry:
        cost = _tokens_to_cost(input_tokens, output_tokens, model_id)
        entry = CostEntry(
            category=category,
            name=name,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._entries.append(entry)
        return entry

    def total_cost(self) -> float:
        return round(sum(e.cost_usd for e in self._entries), 6)

    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(e.input_tokens for e in self._entries),
            "output": sum(e.output_tokens for e in self._entries),
        }

    def by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            result[e.category] = round(result.get(e.category, 0.0) + e.cost_usd, 6)
        return result

    def by_name(self, category: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            if e.category == category:
                result[e.name] = round(result.get(e.name, 0.0) + e.cost_usd, 6)
        return result

    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    def reset(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------
    # Trace integration helper
    # ------------------------------------------------------------------

    @classmethod
    def from_trace(cls, trace, model_id: str = "unknown") -> CostTracker:
        """Build a CostTracker from a WorkflowTrace or AgentTrace.

        Extracts token counts from each node's turns and computes costs.
        """
        tracker = cls()
        # Works with both WorkflowTrace and AgentTrace
        node_traces = getattr(trace, "node_traces", None)
        if node_traces:
            for agent_id, at in node_traces.items():
                in_tok = at.total_tokens.get("input", 0)
                out_tok = at.total_tokens.get("output", 0)
                if in_tok or out_tok:
                    tracker.record("agent", agent_id, model_id,
                                   input_tokens=in_tok, output_tokens=out_tok)
        elif hasattr(trace, "total_tokens"):
            in_tok = trace.total_tokens.get("input", 0)  # type: ignore[union-attr]
            out_tok = trace.total_tokens.get("output", 0)  # type: ignore[union-attr]
            agent_id = getattr(trace, "agent_id", "agent")  # type: ignore[union-attr]
            if in_tok or out_tok:
                tracker.record("agent", agent_id, model_id,
                               input_tokens=in_tok, output_tokens=out_tok)
        return tracker


from agentflow.errors import AgentFlowBudgetError


class BudgetExceededError(AgentFlowBudgetError):
    """Raised when a budget cap is exceeded."""


@dataclass
class BudgetCap:
    """Track spending against a budget limit.

    Usage::

        budget = BudgetCap(limit_usd=5.00)
        budget.spend(0.032)  # OK
        budget.spend(5.00)   # raises BudgetExceededError
    """

    limit_usd: float
    spent_usd: float = 0.0
    unit: str = ""           # optional label: "workflow", "session"

    def spend(self, cost_usd: float) -> None:
        self.spent_usd = round(self.spent_usd + cost_usd, 6)
        if self.spent_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.spent_usd:.4f} > ${self.limit_usd:.4f}"
                + (f" for {self.unit}" if self.unit else "")
            )

    @property
    def remaining(self) -> float:
        return round(max(0.0, self.limit_usd - self.spent_usd), 6)

    def reset(self) -> None:
        self.spent_usd = 0.0
