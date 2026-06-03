from agentflow.runtime.memory.working import WorkingMemory, Message
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact
from agentflow.runtime.memory.semantic import SemanticMemory
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile, WorkingConfig


class ShortTermMemory(WorkingMemory):
    """Backward-compatible wrapper — old API used ``max_messages``, ``get_messages()``."""

    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        super().__init__(max_turns=max_messages, max_tokens=max_tokens)

    def get_messages(self) -> list[Message]:
        return list(self._messages)


class LongTermMemory(SemanticMemory):
    """Backward-compatible wrapper — old API used ``store(key, dict)``."""

    def store(self, key: str, value: dict) -> None:
        content = str(value)
        super().store(key, content=content, metadata=value)

    def search(self, query: str) -> list[dict]:
        results = []
        query_words = query.lower().split()
        for key, entry in self._store.items():
            key_match = any(w in key.lower() for w in query_words)
            content_text = entry.get("content", "")
            content_match = any(w in content_text.lower() for w in query_words)
            if key_match or content_match:
                results.append({"key": key, **entry.get("metadata", {})})
        return results


__all__ = [
    "WorkingMemory", "Message",
    "EpisodicMemory", "MemoryFact",
    "SemanticMemory",
    "MemoryManager", "MemoryProfile", "WorkingConfig",
    "ShortTermMemory", "LongTermMemory",
]
