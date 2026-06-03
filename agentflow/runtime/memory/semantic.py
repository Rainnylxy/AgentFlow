"""Layer 3: Semantic Memory — 长期知识库。

默认使用关键词匹配（零依赖），可选升级为 Chroma 向量检索。
"""

from collections import OrderedDict
from typing import Optional


class SemanticMemory:
    """长期记忆 — 默认关键词检索，可选向量嵌入检索。"""

    def __init__(self, embedder: Optional[str] = None, top_k_default: int = 5):
        self.embedder = embedder
        self.top_k_default = top_k_default
        self._store: OrderedDict[str, dict] = OrderedDict()

    def store(self, key: str, content: str, metadata: Optional[dict] = None) -> None:
        self._store[key] = {"content": content, "metadata": metadata or {}}

    def get(self, key: str) -> Optional[dict]:
        return self._store.get(key)

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """关键词检索（fallback 实现）。

        将 query 拆分为单词，与每条内容的 content 做关键词匹配。
        返回匹配条目按命中数降序排列。
        """
        k = top_k or self.top_k_default
        query_words = query.lower().split()
        scored = []

        for key, entry in self._store.items():
            content = entry["content"].lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored.append((score, {"key": key, **entry}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]
