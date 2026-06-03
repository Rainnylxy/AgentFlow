"""Memory Manager — backward-compatible re-exports.

This module is kept for backward compatibility.
New code should import from agentflow.runtime.memory directly.
"""

from typing import Optional
from agentflow.runtime.memory.working import Message, WorkingMemory
from agentflow.runtime.memory.semantic import SemanticMemory
from agentflow.runtime.memory.manager import MemoryManager


class ShortTermMemory(WorkingMemory):
    """Backward-compatible wrapper for WorkingMemory.

    Old API used ``max_messages`` (instead of ``max_turns``)
    and exposed ``get_messages()``.
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        super().__init__(max_turns=max_messages, max_tokens=max_tokens)

    def get_messages(self) -> list[Message]:
        """Return the full internal message list (old API)."""
        return list(self._messages)


class LongTermMemory(SemanticMemory):
    """Backward-compatible wrapper for SemanticMemory.

    Old API used ``store(key, value_dict)`` with a dict as second
    argument, and the search scanned keys + values.
    """

    def store(self, key: str, value: dict) -> None:
        """Accept old ``(key, dict)`` signature.

        Converts the dict to a string for the new content field
        and keeps the dict as metadata.
        """
        content = str(value)
        super().store(key, content=content, metadata=value)

    def search(self, query: str) -> list[dict]:
        """Old-style search that also matches against the key."""
        results = []
        query_words = query.lower().split()
        for key, entry in self._store.items():
            key_match = any(w in key.lower() for w in query_words)
            content_text = entry.get("content", "")
            content_match = any(w in content_text.lower() for w in query_words)
            if key_match or content_match:
                results.append({"key": key, **entry.get("metadata", {})})
        return results


__all__ = ["Message", "ShortTermMemory", "LongTermMemory", "MemoryManager"]
