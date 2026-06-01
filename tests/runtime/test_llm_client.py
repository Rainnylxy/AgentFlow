import pytest
from agentflow.runtime.llm_client import LLMClient, OpenAIClient, LLMResponse


class TestOpenAIClient:
    def test_client_stores_config(self):
        """客户端正确保存配置。"""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        assert client.model == "gpt-4o"
        assert client.base_url == "https://api.openai.com/v1"

    def test_custom_base_url(self):
        """支持切换不同 LLM 提供商。"""
        client = OpenAIClient(
            api_key="key",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
        )
        assert client.base_url == "https://api.deepseek.com/v1"

    def test_empty_api_key_raises(self):
        """空 API key 初始化即报错。"""
        with pytest.raises(ValueError, match="api_key is required"):
            OpenAIClient(api_key="", model="gpt-4o")

    def test_chat_returns_response(self):
        """chat 调用返回 LLMResponse。"""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="42", role="assistant"))],
            usage=MagicMock(total_tokens=10),
        )
        mock_client.chat.completions.create = mock_create

        import asyncio

        async def run():
            client = OpenAIClient(api_key="test", model="gpt-4o")
            # 替换内部 AsyncOpenAI client
            client.client = mock_client
            resp = await client.chat(messages=[{"role": "user", "content": "Hi"}])
            return resp

        resp = asyncio.run(run())
        assert resp.content == "42"
        assert resp.role == "assistant"

    def test_chat_with_tools(self):
        """带 tool 定义的 chat 调用。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=None,
                role="assistant",
                tool_calls=[
                    MagicMock(
                        id="call_1",
                        function=MagicMock(name="calc", arguments='{"expr":"2+2"}'),
                    )
                ],
            ))],
            usage=MagicMock(total_tokens=20),
        )
        mock_client.chat.completions.create = mock_create

        import asyncio

        async def run():
            client = OpenAIClient(api_key="test", model="gpt-4o")
            client.client = mock_client
            resp = await client.chat(
                messages=[{"role": "user", "content": "2+2?"}],
                tools=[{"type": "function", "function": {"name": "calc"}}],
            )
            return resp

        resp = asyncio.run(run())
        assert resp.tool_calls is not None
