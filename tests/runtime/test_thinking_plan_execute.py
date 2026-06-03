import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.memory.manager import MemoryManager


class TestPlanExecuteStrategy:
    def test_generates_plan_then_executes(self):
        """首先生成计划，然后逐步执行，最后汇总。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="PLAN:\n1. Search for refund policy\n2. Apply to user case\n3. Summarize", role="assistant", tool_calls=[]),
            MagicMock(content="EXECUTE: Found refund policy - 30 days unconditional.", role="assistant", tool_calls=[]),
            MagicMock(content="FINAL: You are eligible for a refund within 30 days.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="Can I get a refund?",
            system_prompt="You handle refunds.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=10,
        )

        strategy = PlanExecuteStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "plan_execute"
        assert "refund" in result.output.lower()
        assert len(result.steps) >= 2

    def test_records_all_phases(self):
        """验证三个阶段都被记录。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Plan step 1, step 2", role="assistant", tool_calls=[]),
            MagicMock(content="Executed step 1 and 2", role="assistant", tool_calls=[]),
            MagicMock(content="Final summary", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="Do something",
            system_prompt="You help.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=10,
        )

        strategy = PlanExecuteStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        phases = {s["phase"] for s in result.steps}
        assert "plan" in phases
        assert "execute" in phases
        assert "finalize" in phases
