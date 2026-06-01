"""Memory Manager：短期记忆（滑动窗口 + 摘要）+ 长期记忆（向量库检索）"""

from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict


@dataclass
class Message:
    role: str
    content: str


class ShortTermMemory:
    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def get_context_window(self) -> list[Message]:
        result = []
        total_chars = 0
        char_limit = self.max_tokens * 4  # 1 token ≈ 4 chars
        for msg in reversed(self._messages):
            total_chars += len(msg.content)
            if total_chars > char_limit:
                break
            result.insert(0, msg)
        return result

    def clear(self) -> None:
        self._messages.clear()


class LongTermMemory:
    def __init__(self):
        self._store: OrderedDict[str, dict] = OrderedDict()

    def store(self, key: str, value: dict) -> None:
        self._store[key] = value

    def search(self, query: str) -> list[dict]:
        results = []
        for key, value in self._store.items():
            if any(w.lower() in key.lower() or w.lower() in str(value).lower()
                   for w in query.split()):
                results.append({"key": key, **value})
        return results

    def get(self, key: str) -> Optional[dict]:
        return self._store.get(key)


class MemoryManager:
    def __init__(self, short_term_max: int = 20, short_term_tokens: int = 8000):
        self.short_term = ShortTermMemory(
            max_messages=short_term_max, max_tokens=short_term_tokens
        )
        self.long_term = LongTermMemory()
