"""Layer 1: Working Memory — 当前对话的完整消息窗口。

支持滑动窗口（按轮数截断）、token 限制（精确计数）、
以及 LLM 摘要压缩（旧消息压缩为摘要注入窗口头部）。
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from agentflow.runtime.memory.token_counter import TokenCounter, AdaptiveCounter


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str = ""
    tool_calls: list = field(default_factory=list)


class WorkingMemory:
    """当前对话窗口，支持滑动窗口、token 限制和 LLM 摘要压缩。

    token 计数的准确性由 TokenCounter 决定，默认使用 AdaptiveCounter
    （基于字符类型的启发式估算），可选 TiktokenCounter。

    压缩策略：当消息被 max_turns 从窗口挤出时，不直接丢弃，
    而是缓存到 overflow 区。调用 compress() 时用 LLM 将溢出
    消息压缩为一段摘要，注入到上下文窗口头部，保留关键信息。
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int = 8000,
        summarizer: Optional[Callable[[list[Message]], Awaitable[str]]] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._summarizer = summarizer
        self._counter = token_counter or AdaptiveCounter()
        self._messages: list[Message] = []
        self._overflow: list[Message] = []   # 被 max_turns 挤出的旧消息
        self._compressed_summary: str = ""   # 旧消息的 LLM 摘要

    def add(self, message: Message) -> None:
        self._messages.append(message)
        while len(self._messages) > self.max_turns:
            self._overflow.append(self._messages.pop(0))

    def clear(self) -> None:
        self._messages.clear()
        self._overflow.clear()
        self._compressed_summary = ""

    @property
    def needs_compression(self) -> bool:
        """是否有未压缩的溢出消息。"""
        return bool(self._overflow) and self._summarizer is not None

    async def compress(self) -> Optional[str]:
        """将溢出消息压缩为摘要。

        调用 LLM 将 _overflow 中的旧消息总结为一段简洁摘要，
        摘要存储在 _compressed_summary 中，后续 get_context_window()
        会自动将其注入到窗口头部。

        Returns:
            生成的摘要文本，如果没有溢出或没有摘要器则返回 None
        """
        if not self._overflow or self._summarizer is None:
            return None
        summary = await self._summarizer(self._overflow)
        if self._compressed_summary:
            self._compressed_summary = (
                f"{self._compressed_summary}\n---\n{summary}"
            )
        else:
            self._compressed_summary = summary
        self._overflow.clear()
        return summary

    def get_context_window(self, overhead_tokens: int = 0) -> list[Message]:
        """获取对话窗口。

        结构：
          1. [system] 压缩摘要（如果有）
          2. 按 token 限制从后向前截取的消息，使用 TokenCounter 精确计数

        Args:
            overhead_tokens: 系统区（system prompt + references + tools）的预估 token 数。
                             从 max_tokens 中扣除，确保总量不超过 LLM 上下文窗口上限。
        """
        result = []
        total_tokens = 0
        effective_limit = max(1, self.max_tokens - overhead_tokens)

        if self._compressed_summary:
            summary_msg = Message(role="system", content=self._compressed_summary)
            total_tokens += self._counter.count(summary_msg.content)
            result.append(summary_msg)

        has_data = bool(self._compressed_summary)
        for msg in reversed(self._messages):
            msg_tokens = self._counter.count(msg.content)
            total_tokens += msg_tokens
            if total_tokens > effective_limit and has_data:
                break
            insert_at = 0 if not self._compressed_summary else 1
            result.insert(insert_at, msg)
            has_data = True

        return result

    def __len__(self):
        return len(self._messages)
