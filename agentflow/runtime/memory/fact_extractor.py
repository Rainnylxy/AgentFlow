"""Fact Extractor — 从对话中提取结构化事实的策略接口。

提供可插拔的事实提取策略，当前内置基于关键词的提取器。
后续可扩展为 LLM 驱动的提取器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from agentflow.runtime.memory.episodic import MemoryFact


class BaseFactExtractor(ABC):
    """事实提取策略抽象基类。

    每个 turn 结束后，MemoryManager.post_turn() 调用此策略
    从工作记忆中提取结构化事实，存入 EpisodicMemory。
    """

    @abstractmethod
    async def extract(self, messages: list, turn_count: int) -> list[MemoryFact]:
        """从对话消息中提取事实。

        Args:
            messages: 当前工作记忆中的消息列表（Message 对象）。
            turn_count: 当前对话轮次。

        Returns:
            MemoryFact 列表，将添加到 EpisodicMemory 中。
        """
        ...


class KeywordFactExtractor(BaseFactExtractor):
    """基于关键词的事实提取器（默认实现）。

    通过预定义的关键词模式匹配来检测事实。
    适用于简单场景；复杂场景建议替换为 LLMFactExtractor。
    """

    # 可配置的关键词模式
    DEFAULT_KEYWORDS = [
        "live in", "住在", "from", "city is", "location",
        " in ", "at ",
    ]
    DEFAULT_TEMPERATURE_KEYWORDS = [
        "temperature", "weather", "度", "°",
    ]

    def __init__(
        self,
        keywords: list[str] | None = None,
        temperature_keywords: list[str] | None = None,
    ):
        self.keywords = keywords or self.DEFAULT_KEYWORDS
        self.temperature_keywords = temperature_keywords or self.DEFAULT_TEMPERATURE_KEYWORDS

    async def extract(self, messages: list, turn_count: int) -> list[MemoryFact]:
        """从消息中提取事实。"""
        facts = []
        for msg in messages:
            content_lower = msg.content.lower()

            # 位置类事实
            if self._matches(content_lower, self.keywords) or \
               self._matches(content_lower, self.temperature_keywords):
                fact = MemoryFact(
                    fact_type="preference",
                    subject=msg.role,
                    predicate="location",
                    object=msg.content[:100],
                    confidence=0.6,
                    timestamp=datetime.now(),
                    source_turn=turn_count,
                    ttl=86400 * 7,
                )
                facts.append(fact)

        return facts

    def _matches(self, text: str, patterns: list[str]) -> bool:
        """检查文本是否匹配任意模式。"""
        return any(pattern in text for pattern in patterns)


class NoOpFactExtractor(BaseFactExtractor):
    """空事实提取器：不提取任何事实。用于 MemoryProfile.light() 场景。"""

    async def extract(self, messages: list, turn_count: int) -> list[MemoryFact]:
        return []
