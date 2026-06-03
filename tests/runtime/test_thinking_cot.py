import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.memory.manager import MemoryManager


class TestCoTStrategy:
    def test_think_then_answer(self):
        """先深度推理，再给出答案。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Let me think step by step...\nGiven: 10 apples, eat 3, buy 5.\n10 - 3 + 5 = 12.", role="assistant", tool_calls=[]),
            MagicMock(content="FINAL ANSWER: You have 12 apples.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="I have 10 apples, eat 3, buy 5. How many left?",
            system_prompt="You solve math problems.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = CoTStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "cot"
        assert "12" in result.output

    def test_records_both_phases(self):
        """验证 think 和 answer 两个阶段都被记录。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Reasoning...", role="assistant", tool_calls=[]),
            MagicMock(content="Answer.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="Why is the sky blue?",
            system_prompt="You explain things.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
        )

        strategy = CoTStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        phases = {s["phase"] for s in result.steps}
        assert "think" in phases
        assert "answer" in phases
