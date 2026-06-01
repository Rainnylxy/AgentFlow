"""LLM Client 抽象层：统一的 LLM 调用接口，支持 OpenAI-compatible API"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import httpx
from openai import AsyncOpenAI


@dataclass
class LLMResponse:
    """LLM 调用的统一返回格式。"""
    content: str
    role: str = "assistant"
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    async def chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        """发送消息并返回 LLM 响应。"""
        ...


class OpenAIClient(LLMClient):
    """OpenAI-compatible API 客户端。

    支持所有兼容 OpenAI API 格式的提供商（OpenAI、DeepSeek、Qwen 等）。
    只需在初始化时设置不同的 base_url 和 api_key。
    支持通过 proxy 参数或 AGENTFLOW_PROXY 环境变量设置 HTTP 代理。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        proxy: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self.base_url = base_url

        # 构建 httpx 客户端（支持代理）
        import os
        proxy = proxy or os.getenv("AGENTFLOW_PROXY", "")
        http_client = None
        if proxy:
            http_client = httpx.AsyncClient(proxy=proxy)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    async def chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        completion = await self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        # 处理 tool_calls
        tool_calls = []
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            content=msg.content or "",
            role=msg.role,
            tool_calls=tool_calls,
            usage={
                "total_tokens": completion.usage.total_tokens,
            } if completion.usage else {},
        )
