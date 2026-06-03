import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.memory.manager import MemoryProfile
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.thinking import ThinkingMode


class TestAgentBuilder:
    def test_minimal_build(self):
        """最简 Builder：仅名称 + mock LLM。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Hello!", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("minimal")
            .with_llm(mock_llm)
            .build())

        assert agent.name == "minimal"
        import asyncio
        result = asyncio.run(agent.run("Hi"))
        assert "Hello" in result.output

    def test_with_tools(self):
        """Builder 集成 ToolKit。"""
        kit = __import__('agentflow.runtime.toolkit', fromlist=['ToolKit']).ToolKit()

        @tool
        def echo(text: str) -> str:
            """Echo text back."""
            return text

        kit.add(echo)

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content=None, tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "hello world"}'},
            }]),
            MagicMock(content="You said: hello world", tool_calls=[]),
        ]

        agent = (AgentBuilder("tool-agent")
            .with_llm(mock_llm)
            .with_tools(echo)
            .build())

        import asyncio
        result = asyncio.run(agent.run("Echo hello world"))
        assert len(result.tool_calls) == 1

    def test_with_memory_profile(self):
        """Builder 集成 MemoryProfile。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Got it.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("mem-agent")
            .with_llm(mock_llm)
            .with_memory(MemoryProfile.light())
            .build())

        assert agent.memory.profile.working.max_turns == 10

    def test_with_prompt_template(self):
        """Builder 集成 PromptTemplate。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="I am a support agent.", role="assistant", tool_calls=[],
        )

        template = PromptTemplate.preset("customer_support")
        agent = (AgentBuilder("prompt-agent")
            .with_llm(mock_llm)
            .with_prompt(template)
            .build())

        import asyncio
        result = asyncio.run(agent.run("Help!"))
        assert result.output is not None

    def test_with_thinking_mode(self):
        """Builder 集成 ThinkingMode。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Answer.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("thinker")
            .with_llm(mock_llm)
            .with_thinking(ThinkingMode.COT)
            .build())

        assert agent.thinking_engine.mode == ThinkingMode.COT

    def test_full_build(self):
        """全组件 Builder 集成测试。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content=None, tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "refund"}'},
            }]),
            MagicMock(content="Based on our policy, you can refund within 30 days.", tool_calls=[]),
        ]

        @tool
        def lookup(query: str) -> str:
            """Search for info."""
            return f"Result for {query}"

        agent = (AgentBuilder("full-agent")
            .with_llm(mock_llm)
            .with_tools(lookup)
            .with_memory(MemoryProfile.standard())
            .with_prompt(PromptTemplate.preset("customer_support"))
            .with_thinking(ThinkingMode.REACT)
            .with_max_iterations(3)
            .build())

        import asyncio
        result = asyncio.run(agent.run("I want a refund"))
        assert "refund" in result.output.lower()

    def test_builder_from_string_prompt(self):
        """向后兼容：纯字符串 System Prompt。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Yes.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("compat")
            .with_llm(mock_llm)
            .with_prompt("You are helpful.")
            .build())

        import asyncio
        result = asyncio.run(agent.run("Hello"))
        assert result.output is not None

    def test_builder_missing_llm_raises(self):
        """未提供 LLM 客户端时抛错。"""
        with pytest.raises(ValueError, match="llm"):
            AgentBuilder("no-llm").build()
