"""Layer 2: Episodic Memory — 跨会话的结构化事实存储。

基于内存列表的简单实现。
内置自动淘汰和过期清理机制。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class MemoryFact:
    """结构化记忆事实：主体-谓词-客体 三元组。"""

    fact_type: str  # "entity" | "decision" | "event" | "preference"
    subject: str
    predicate: str
    object: str
    confidence: float  # 0.0 - 1.0
    timestamp: datetime
    source_turn: int
    ttl: int  # 生存时间（秒）

    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)

    def decay(self, factor: float) -> None:
        """衰减置信度，factor 为衰减比例（0~1），置信度乘以 (1 - factor)。"""
        self.confidence = max(0.0, self.confidence * (1.0 - factor))


class EpisodicMemory:
    """Layer 2: 情节记忆 — 跨会话的结构化事实存储。"""

    def __init__(self, max_facts: int = 200, backend: str = "memory"):
        self.max_facts = max_facts
        self._facts: list[MemoryFact] = []

    def add(self, fact: MemoryFact) -> None:
        self._facts.append(fact)
        self._evict_if_needed()

    def get_all(self) -> list[MemoryFact]:
        return list(self._facts)

    def get_by_subject(self, subject: str) -> list[MemoryFact]:
        return [f for f in self._facts if f.subject == subject]

    def get_by_type(self, fact_type: str) -> list[MemoryFact]:
        return [f for f in self._facts if f.fact_type == fact_type]

    def count(self) -> int:
        return len(self._facts)

    def forget_expired(self) -> int:
        """遗忘门：删除所有已过期的事实，返回删除数量。"""
        before = len(self._facts)
        self._facts = [f for f in self._facts if not f.is_expired()]
        return before - len(self._facts)

    def _evict_if_needed(self) -> None:
        """容量超限时淘汰低质量条目。

        按 (confidence / 新鲜度) 排序，保留高质量条目。
        """
        if len(self._facts) <= self.max_facts:
            return
        now = datetime.now()
        scored = sorted(
            self._facts,
            key=lambda f: f.confidence / max(1.0, (now - f.timestamp).total_seconds() / 86400),
        )
        self._facts = scored[-(self.max_facts):]
