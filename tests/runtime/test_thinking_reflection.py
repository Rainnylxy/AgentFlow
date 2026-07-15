import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper
from agentflow.runtime.memory.manager import MemoryManager


class TestReflectionWrapper:
    def test_passes_when_first_attempt_ok(self):
        """当第一次反思就通过时，返回原始结果。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="The answer is 42.", role="assistant", tool_calls=[]),
            MagicMock(content="PASS", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="What is the answer?",
            system_prompt="You are helpful.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        wrapped = ReflectionWrapper(ReActStrategy(), max_reflections=2)
        import asyncio
        result = asyncio.run(wrapped.run(ctx))

        assert "42" in result.output
        assert len(result.reflection_notes) > 0

    def test_retries_on_failure(self):
        """反思发现 FAIL 后重试。"""
        call_count = [0]
        mock_llm = AsyncMock()

        async def side_effect(messages, tools=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(content="Wrong answer 100.", role="assistant", tool_calls=[])
            elif call_count[0] == 2:
                return MagicMock(content="FAIL: answer seems wrong, recalculate.", role="assistant", tool_calls=[])
            elif call_count[0] == 3:
                return MagicMock(content="Correct answer: 42.", role="assistant", tool_calls=[])
            else:
                return MagicMock(content="PASS", role="assistant", tool_calls=[])

        mock_llm.chat.side_effect = side_effect

        ctx = ThinkContext(
            user_input="What is 6*7?",
            system_prompt="You do math.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        wrapped = ReflectionWrapper(ReActStrategy(), max_reflections=3)
        import asyncio
        result = asyncio.run(wrapped.run(ctx))

        # Should eventually pass
        assert len(result.reflection_notes) >= 1

    def test_exhausts_reflections(self):
        """反思次数用尽时返回最后的结果。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Answer.", role="assistant", tool_calls=[]),
            MagicMock(content="FAIL: still wrong.", role="assistant", tool_calls=[]),
            MagicMock(content="Answer 2.", role="assistant", tool_calls=[]),
            MagicMock(content="FAIL: nope.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="Hard question",
            system_prompt="You help.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
        )

        wrapped = ReflectionWrapper(ReActStrategy(), max_reflections=2)
        import asyncio
        result = asyncio.run(wrapped.run(ctx))

        # 2 reflections used, returns last result
        assert len(result.reflection_notes) == 2
