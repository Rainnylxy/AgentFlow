"""LLM Client 抽象层：统一的 LLM 调用接口，支持 OpenAI-compatible API。

提供指数退避重试、超时控制、可配置重试条件。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# 可重试的异常类型
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,  # 5xx 状态码
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


@dataclass
class LLMResponse:
    """LLM 调用的统一返回格式。"""
    content: str
    role: str = "assistant"
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    """LLM 客户端抽象基类。

    提供指数退避重试和超时控制。子类只需实现 _do_chat()。
    调用方统一使用 chat() —— 基类自动应用重试策略。

    配置参数:
        max_retries: 最大重试次数（默认 3）
        base_delay: 首次重试延迟秒数（默认 1.0）
        max_delay: 重试延迟上限秒数（默认 60.0）
        timeout: 单次请求超时秒数（默认 120.0）
        retry_on_status: 额外可重试的 HTTP 状态码集合
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        retry_on_status: Optional[set[int]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self._retry_on_status = retry_on_status or {429, 500, 502, 503, 504}

    async def chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        """发送消息并返回 LLM 响应，自动重试。

        子类不应重写此方法——重写 _do_chat() 即可。
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._do_chat(messages, tools),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                last_exception = TimeoutError(
                    f"LLM request timed out after {self.timeout}s"
                )
                logger.warning(
                    "LLM timeout (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, last_exception,
                )
            except _RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                logger.warning(
                    "LLM retryable error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, e,
                )
            except Exception as e:
                # HTTPStatusError 可能是 5xx，检查状态码
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if e.response.status_code in self._retry_on_status:
                        last_exception = e
                        logger.warning(
                            "LLM HTTP %d (attempt %d/%d)",
                            e.response.status_code,
                            attempt + 1, self.max_retries + 1,
                        )
                        if attempt < self.max_retries:
                            delay = self._compute_delay(attempt)
                            await asyncio.sleep(delay)
                            continue
                raise  # 不可重试异常，直接抛出

            if attempt < self.max_retries:
                delay = self._compute_delay(attempt)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        raise last_exception  # type: ignore[misc]

    def _compute_delay(self, attempt: int) -> float:
        """计算指数退避延迟: base_delay * 2^attempt，上限 max_delay。"""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

    @abstractmethod
    async def _do_chat(
        self, messages: list[dict], tools: Optional[list[dict]] = None
    ) -> LLMResponse:
        """子类实现实际的 LLM API 调用（不含重试逻辑）。"""
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
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
    ):
        super().__init__(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout,
        )
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

    async def _do_chat(
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
