import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.memory.manager import MemoryManager


class TestPlanExecuteStrategy:
    def test_generates_plan_then_executes(self):
        """生成计划后，每步独立 query，最后汇总。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            # Phase 1: Plan
            MagicMock(content="1. Search for refund policy\n2. Apply to user case\n3. Summarize", role="assistant", tool_calls=[]),
            # Phase 2: Execute step 1
            MagicMock(content="Found refund policy - 30 days unconditional.", role="assistant", tool_calls=[]),
            # Phase 2: Execute step 2
            MagicMock(content="User case matches refund policy.", role="assistant", tool_calls=[]),
            # Phase 2: Execute step 3
            MagicMock(content="Summary: user is eligible.", role="assistant", tool_calls=[]),
            # Phase 3: Finalize
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
        assert len(result.steps) == 5  # plan + 3 executes + finalize

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

    def test_each_step_independent_query(self):
        """每个步骤有独立的上下文窗口，上一步结果传给下一步。"""
        mock_llm = AsyncMock()
        steps_seen = []

        async def side_effect(messages, tools=None, **kwargs):
            user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            steps_seen.append(user_msg)
            return MagicMock(content="Done.", role="assistant", tool_calls=[])

        mock_llm.chat.side_effect = side_effect

        ctx = ThinkContext(
            user_input="Research AI testing",
            system_prompt="You do research.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = PlanExecuteStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        # Phase 1 (plan) + 1 step (fallback, no numbered format) + Phase 3 (finalize) = 3 queries
        assert len(steps_seen) == 3

        # Step 2 (execute) should see previous step info
        execute_call = steps_seen[1]
        assert "Now execute Step 1/1" in execute_call
        assert "Research AI testing" in execute_call

        # Finalize should see execute result
        finalize_call = steps_seen[2]
        assert "Synthesize the final result" in finalize_call
