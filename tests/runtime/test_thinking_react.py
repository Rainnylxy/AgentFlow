import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.toolkit import ToolKit, tool
from agentflow.runtime.memory.manager import MemoryManager


class TestReActStrategy:
    def test_simple_answer_no_tools(self):
        """无工具调用的简单问答。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="The answer is 42.", role="assistant", tool_calls=[],
        )

        ctx = ThinkContext(
            user_input="What is the meaning of life?",
            system_prompt="You are helpful.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=10,
        )

        strategy = ReActStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "42" in result.output
        assert result.mode_used == "react"
        assert len(result.steps) == 1

    def test_uses_tool_then_answers(self):
        """Agent 先调用工具，再给出最终答案。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None, role="assistant",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }],
            ),
            MagicMock(content="2 + 3 = 5", role="assistant", tool_calls=[]),
        ]

        kit = ToolKit()

        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        kit.add(add)

        ctx = ThinkContext(
            user_input="What is 2+3?",
            system_prompt="You do math.",
            messages=[],
            tools=kit.list_for_llm(),
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = ReActStrategy(toolkit=kit)
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "5" in result.output
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "add"

    def test_stops_at_max_iterations(self):
        """达到最大迭代次数时强制停止。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content=None, role="assistant",
            tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "ping"}'},
            }],
        )

        kit = ToolKit()

        @tool
        def echo(text: str) -> str:
            """Echo."""
            return text

        kit.add(echo)

        ctx = ThinkContext(
            user_input="ping", system_prompt="You loop.",
            messages=[], tools=kit.list_for_llm(),
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=2,
        )

        strategy = ReActStrategy(toolkit=kit)
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "maximum iterations" in result.output.lower()
