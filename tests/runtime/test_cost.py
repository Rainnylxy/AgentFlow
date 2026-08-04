"""Tests for agentflow.runtime.cost — CostTracker, BudgetCap, model pricing."""

from __future__ import annotations

import pytest

from agentflow.runtime.cost import (
    BudgetCap,
    BudgetExceededError,
    CostTracker,
    get_model_price,
)
from agentflow.trace.tracer import AgentTrace


class TestModelPricing:
    def test_known_model_returns_prices(self):
        in_price, out_price = get_model_price("gpt-4o")
        assert in_price == 2.50
        assert out_price == 10.00

    def test_prefix_match(self):
        in_price, out_price = get_model_price("gpt-4o-2024-08-06")
        assert in_price == 2.50

    def test_case_insensitive(self):
        in_price, out_price = get_model_price("GPT-4O")
        assert in_price == 2.50

    def test_claude_model(self):
        in_price, out_price = get_model_price("claude-sonnet-5-20251001")
        assert in_price == 3.00
        assert out_price == 15.00

    def test_unknown_model_fallback(self):
        in_price, out_price = get_model_price("custom-llm-v1")
        assert in_price == 1.00
        assert out_price == 4.00


class TestCostTracker:
    def test_record_single_entry(self):
        tracker = CostTracker()
        e = tracker.record("agent", "support", "gpt-4o",
                           input_tokens=1000, output_tokens=500)

        # 1000/1e6 * 2.50 + 500/1e6 * 10.00 = 0.0025 + 0.005 = 0.0075
        assert e.cost_usd == 0.0075
        assert e.model_id == "gpt-4o"

    def test_total_cost(self):
        tracker = CostTracker()
        tracker.record("agent", "a", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        tracker.record("agent", "b", "gpt-4o-mini", input_tokens=0, output_tokens=500_000)

        # 1M input * 0.15 + 500k output * 0.60 = 0.15 + 0.30 = 0.45
        assert tracker.total_cost() == 0.45

    def test_total_tokens(self):
        tracker = CostTracker()
        tracker.record("agent", "a", "gpt-4o", input_tokens=300, output_tokens=200)
        tracker.record("agent", "b", "gpt-4o", input_tokens=100, output_tokens=50)
        assert tracker.total_tokens() == {"input": 400, "output": 250}

    def test_by_category(self):
        tracker = CostTracker()
        tracker.record("agent", "a", "gpt-4o", input_tokens=1_000_000, output_tokens=0)
        tracker.record("workflow", "wf1", "gpt-4o", input_tokens=500_000, output_tokens=0)

        cats = tracker.by_category()
        assert "agent" in cats
        assert "workflow" in cats

    def test_by_name(self):
        tracker = CostTracker()
        tracker.record("agent", "alice", "gpt-4o", input_tokens=1_000_000, output_tokens=0)
        tracker.record("agent", "bob", "gpt-4o", input_tokens=500_000, output_tokens=0)
        tracker.record("agent", "alice", "gpt-4o", input_tokens=0, output_tokens=1_000_000)

        by_agent = tracker.by_name("agent")
        assert "alice" in by_agent
        assert "bob" in by_agent
        assert by_agent["alice"] > by_agent["bob"]

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("agent", "a", "gpt-4o", input_tokens=100, output_tokens=0)
        tracker.reset()
        assert tracker.total_cost() == 0.0
        assert len(tracker.entries()) == 0

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker()
        e = tracker.record("agent", "a", "gpt-4o", input_tokens=0, output_tokens=0)
        assert e.cost_usd == 0.0

    def test_from_agent_trace(self):
        at = AgentTrace(agent_id="test_agent")
        at.total_tokens = {"input": 1_000_000, "output": 500_000}

        tracker = CostTracker.from_trace(at, model_id="gpt-4o-mini")
        assert tracker.total_cost() > 0
        assert len(tracker.entries()) == 1
        assert tracker.entries()[0].name == "test_agent"

    def test_from_workflow_trace(self):
        from agentflow.trace.tracer import WorkflowTrace

        wt = WorkflowTrace(workflow_name="test_wf")
        at = AgentTrace(agent_id="node_a")
        at.total_tokens = {"input": 1_000_000, "output": 0}
        wt.node_traces["node_a"] = at

        tracker = CostTracker.from_trace(wt, model_id="claude-sonnet-5")
        assert tracker.total_cost() > 0
        assert tracker.entries()[0].name == "node_a"


class TestBudgetCap:
    def test_within_budget(self):
        budget = BudgetCap(limit_usd=1.00)
        budget.spend(0.50)
        assert budget.spent_usd == 0.50
        assert budget.remaining == 0.50

    def test_exceeded_raises(self):
        budget = BudgetCap(limit_usd=0.10)
        with pytest.raises(BudgetExceededError, match="Budget exceeded"):
            budget.spend(0.20)

    def test_exact_limit(self):
        budget = BudgetCap(limit_usd=1.00)
        budget.spend(1.00)  # exact limit, should not raise

    def test_unit_label_in_error(self):
        budget = BudgetCap(limit_usd=0.01, unit="session")
        with pytest.raises(BudgetExceededError, match="session"):
            budget.spend(0.02)

    def test_reset(self):
        budget = BudgetCap(limit_usd=1.00)
        budget.spend(0.80)
        budget.reset()
        assert budget.spent_usd == 0.0
        assert budget.remaining == 1.00

    def test_multiple_spends(self):
        budget = BudgetCap(limit_usd=1.00)
        budget.spend(0.30)
        budget.spend(0.30)
        budget.spend(0.30)
        assert budget.spent_usd == 0.90
        with pytest.raises(BudgetExceededError):
            budget.spend(0.30)
