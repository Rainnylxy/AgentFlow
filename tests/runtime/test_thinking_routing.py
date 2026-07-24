"""Tests for RoutingStrategy — dynamic expert routing with handoff loop."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentflow.runtime.agent import AgentResult
from agentflow.runtime.agent_registry import AgentCapability, AgentRegistry
from agentflow.runtime.thinking.base import ThinkContext, ThinkResult
from agentflow.runtime.thinking.routing import RoutingStrategy
from agentflow.runtime.memory.manager import MemoryManager


def _llm_response(content: str):
    """Build a mock LLM response with the given text content."""
    return MagicMock(
        content=content,
        role="assistant",
        tool_calls=[],
        reasoning_content="",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _make_expert(name: str, output: str = "Done."):
    """Create a mock expert that returns a fixed AgentResult."""
    expert = AsyncMock()
    expert.name = name

    async def run(user_input, stream=None, agent_trace=None):
        return AgentResult(output=output, tool_calls=[], steps=[])

    expert.run.side_effect = run
    return expert


class TestRoutingStrategy:
    """RoutingStrategy test suite."""

    def test_simple_route_no_handoff(self):
        """Router selects an expert, expert completes -- no handoff."""
        registry = AgentRegistry()
        registry.register(AgentCapability(
            agent_id="billing_expert",
            description="Handles billing and payment issues",
        ))
        registry.register(AgentCapability(
            agent_id="refund_expert",
            description="Handles refund requests and disputes",
        ))

        experts = {
            "billing_expert": _make_expert("billing_expert", "Billing info: paid."),
            "refund_expert": _make_expert("refund_expert", "Refund processed."),
        }

        # LLM selects refund_expert
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = _llm_response(
            '{"agent_id": "refund_expert", "reason": "handles refunds"}'
        )

        ctx = ThinkContext(
            user_input="I need a refund",
            system_prompt="You route tasks.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = RoutingStrategy(registry=registry, experts=experts)
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "routing"
        assert result.output == "Refund processed."
        assert len(result.steps) == 1
        assert result.steps[0]["agent_id"] == "refund_expert"
        # Verify LLM was called for routing
        mock_llm.chat.assert_called_once()

    def test_route_with_one_handoff(self):
        """Expert A hands off to Expert B who completes the task."""
        registry = AgentRegistry()
        registry.register(AgentCapability(
            agent_id="query_expert",
            description="Handles database queries and lookups",
        ))
        registry.register(AgentCapability(
            agent_id="report_expert",
            description="Generates reports and summaries",
        ))

        handoff_output = (
            "I can only look up data.\n"
            "---HANDOFF---\n"
            "reason: Need report generation\n"
            "suggest: report_expert\n"
            "context: Found the sales data\n"
            "---END---\n"
        )

        expert_a = AsyncMock()
        expert_a.name = "query_expert"
        async def run_a(user_input, stream=None, agent_trace=None):
            return AgentResult(output=handoff_output, tool_calls=[], steps=[])
        expert_a.run.side_effect = run_a

        expert_b = AsyncMock()
        expert_b.name = "report_expert"
        async def run_b(user_input, stream=None, agent_trace=None):
            assert "Found the sales data" in user_input or "sales data" in user_input.lower()
            return AgentResult(output="Report: sales data analyzed.", tool_calls=[], steps=[])
        expert_b.run.side_effect = run_b

        experts = {"query_expert": expert_a, "report_expert": expert_b}

        # First route -> query_expert, second route -> report_expert
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            _llm_response('{"agent_id": "query_expert", "reason": "can query data"}'),
            _llm_response('{"agent_id": "report_expert", "reason": "generates reports"}'),
        ]

        ctx = ThinkContext(
            user_input="Query sales data and generate a report",
            system_prompt="You route tasks.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = RoutingStrategy(registry=registry, experts=experts)
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "routing"
        assert "Report: sales data analyzed" in result.output
        assert len(result.steps) == 2
        assert result.steps[0]["agent_id"] == "query_expert"
        assert result.steps[1]["agent_id"] == "report_expert"
        assert mock_llm.chat.call_count == 2

    def test_no_expert_matches(self):
        """Empty registry returns a graceful error message."""
        registry = AgentRegistry()
        experts = {}

        mock_llm = AsyncMock()

        ctx = ThinkContext(
            user_input="Do something I cannot do",
            system_prompt="You route.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = RoutingStrategy(registry=registry, experts=experts)
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "routing"
        assert "don't have an expert" in result.output.lower()
        assert len(result.steps) == 0
        # LLM should never be called when no candidates match
        mock_llm.chat.assert_not_called()

    def test_handoff_loop_exhausted(self):
        """Experts keep handing off; stops after max_handoffs cycles."""
        # Need more agents than max_handoffs+1 so we can exhaust
        # the handoff limit before running out of available agents.
        # Use descriptions that all match the task input.
        registry = AgentRegistry()
        agents_def = [
            ("agent_a", "Handles complex analytical tasks"),
            ("agent_b", "Handles complex billing tasks"),
            ("agent_c", "Handles complex reporting tasks"),
            ("agent_d", "Handles complex query tasks"),
        ]
        for agent_id, desc in agents_def:
            registry.register(AgentCapability(agent_id=agent_id, description=desc))

        handoff_text = (
            "---HANDOFF---\n"
            "reason: Cannot complete, need another agent\n"
            "suggest: another agent\n"
            "context: partial work\n"
            "---END---\n"
        )

        def make_handoff_expert(name):
            expert = AsyncMock()
            expert.name = name
            async def run(user_input, stream=None, agent_trace=None):
                return AgentResult(output=handoff_text, tool_calls=[], steps=[])
            expert.run.side_effect = run
            return expert

        experts = {f"agent_{c}": make_handoff_expert(f"agent_{c}")
                   for c in ["a", "b", "c", "d"]}

        mock_llm = AsyncMock()
        # max_handoffs=2 => 3 cycles (0, 1, 2), each picks a different agent
        mock_llm.chat.side_effect = [
            _llm_response('{"agent_id": "agent_a", "reason": "first"}'),
            _llm_response('{"agent_id": "agent_b", "reason": "second"}'),
            _llm_response('{"agent_id": "agent_c", "reason": "third"}'),
        ]

        ctx = ThinkContext(
            user_input="Do a complex task",
            system_prompt="You route.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = RoutingStrategy(registry=registry, experts=experts, max_handoffs=2)
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "routing"
        assert "handoff limit" in result.output.lower()
        assert "agent_a" in result.output
        assert "agent_b" in result.output
        assert len(result.steps) == 3  # 3 cycles with max_handoffs=2
        assert mock_llm.chat.call_count == 3
