"""测试 ReAct 思考模式（通过 AgentBuilder + ThinkingMode.REACT）。

原 tests/runtime/test_react_agent.py 直接测试已废弃的 ReActAgent 类。
现在改为测试 AgentBuilder + ThinkingMode.REACT，行为等价。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.memory.manager import MemoryProfile


class TestReActViaBuilder:
    """通过 AgentBuilder + ThinkingMode.REACT 测试 ReAct 行为。"""

    def test_agent_simple_answer_no_tools(self):
        """最简单的问答：无需工具调用。"""
        mock_client = AsyncMock()
        mock_client.chat.return_value = MagicMock(
            content="The answer is 42.", role="assistant", tool_calls=[],
        )

        agent = (
            AgentBuilder("test")
            .with_llm(mock_client)
            .with_prompt("You are helpful.")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .build_sync()
        )

        result = asyncio.run(agent.run("What is the meaning of life?"))

        assert "42" in result.output
        assert len(result.steps) == 1  # 一次调用直接完成

    def test_agent_uses_tool(self):
        """Agent 需要调用工具获取信息。"""
        mock_client = AsyncMock()
        mock_client.chat.side_effect = [
            MagicMock(
                content=None, role="assistant",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
                }],
            ),
            MagicMock(content="2+2 equals 4", role="assistant", tool_calls=[]),
        ]

        @tool
        def calculator(expression: str) -> str:
            """Calculate a math expression."""
            return str(eval(expression))

        agent = (
            AgentBuilder("math")
            .with_llm(mock_client)
            .with_prompt("You do math.")
            .with_tools(calculator)
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(5)
            .build_sync()
        )

        result = asyncio.run(agent.run("What is 2+2?"))

        assert "4" in result.output
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "calculator"

    def test_agent_stops_at_max_iterations(self):
        """超过最大迭代次数时停止，不无限循环。"""
        mock_client = AsyncMock()
        mock_client.chat.return_value = MagicMock(
            content=None, role="assistant",
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "ping"}'},
            }],
        )

        @tool
        def echo(text: str) -> str:
            """Echo back the text."""
            return text

        agent = (
            AgentBuilder("loop")
            .with_llm(mock_client)
            .with_prompt("You loop forever.")
            .with_tools(echo)
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(2)
            .build_sync()
        )

        result = asyncio.run(agent.run("ping"))

        assert "maximum iterations" in result.output.lower()
        assert len(result.tool_calls) == 2  # 两次 tool calls 后强制停止
