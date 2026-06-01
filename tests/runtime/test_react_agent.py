import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentflow.runtime.react_agent import ReActAgent
from agentflow.runtime.tool_registry import ToolRegistry, Tool, ToolType
from agentflow.runtime.memory import MemoryManager


class TestReActAgent:
    def test_agent_simple_answer_no_tools(self):
        """最简单的问答：无需工具调用。"""
        mock_client = AsyncMock()
        mock_client.chat.return_value = MagicMock(
            content="The answer is 42.", role="assistant", tool_calls=[],
        )

        agent = ReActAgent(
            name="test",
            llm_client=mock_client,
            system_prompt="You are helpful.",
            tool_registry=ToolRegistry(),
            memory_manager=MemoryManager(),
        )

        import asyncio
        result = asyncio.run(agent.run("What is the meaning of life?"))

        assert "42" in result.output
        assert len(result.steps) == 1  # 一次调用直接完成

    def test_agent_uses_tool(self):
        """Agent 需要调用工具获取信息。"""
        mock_client = AsyncMock()
        # 第一次返回 tool_call，第二次返回最终答案
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

        reg = ToolRegistry()
        reg.register(Tool(
            name="calculator", description="Calculate",
            tool_type=ToolType.LOCAL,
            func=lambda expression: str(eval(expression)),
        ))

        agent = ReActAgent(
            name="math", llm_client=mock_client,
            system_prompt="You do math.",
            tool_registry=reg, memory_manager=MemoryManager(),
            max_iterations=5,
        )

        import asyncio
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

        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="Echo",
                          tool_type=ToolType.LOCAL, func=lambda text: text))

        agent = ReActAgent(
            name="loop", llm_client=mock_client,
            system_prompt="You loop forever.",
            tool_registry=reg, memory_manager=MemoryManager(),
            max_iterations=2,
        )

        import asyncio
        result = asyncio.run(agent.run("ping"))

        assert "maximum iterations" in result.output.lower()
        assert len(result.steps) == 2  # 两次 tool calls 后强制停止
