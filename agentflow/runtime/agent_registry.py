"""Agent capability registry with keyword-based expert matching."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentCapability:
    """Declares what an agent can do."""

    agent_id: str
    description: str  # "Handles refund, billing, payment disputes"
    tools: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    priority: int = 0


class AgentRegistry:
    """Registry that maps task descriptions to capable agent experts."""

    def __init__(self) -> None:
        self._capabilities: dict[str, AgentCapability] = {}

    def register(self, cap: AgentCapability) -> None:
        """Register or overwrite an agent capability."""
        self._capabilities[cap.agent_id] = cap

    def unregister(self, agent_id: str) -> None:
        """Remove a capability by agent_id. No-op if not found."""
        self._capabilities.pop(agent_id, None)

    def get(self, agent_id: str) -> AgentCapability | None:
        """Look up a capability by agent_id. Returns None if not found."""
        return self._capabilities.get(agent_id)

    def match(
        self, task: str, top_k: int = 3
    ) -> list[tuple[AgentCapability, float]]:
        """Return top-k capabilities whose descriptions best match *task*.

        Matching algorithm (v1):
        1. Tokenize *task* into a word set (lowercase split).
        2. For each capability, tokenize ``description + " " + " ".join(examples)``
           into a word set.
        3. Score = Jaccard similarity on word sets * (1 + 0.01 × priority).
        4. Return top-k candidates sorted by score descending, filtering out zero
           scores.
        """
        if not self._capabilities:
            return []

        task_tokens = set(task.lower().split())
        if not task_tokens:
            return []

        scored: list[tuple[AgentCapability, float]] = []
        for cap in self._capabilities.values():
            corpus = " ".join([cap.description] + cap.examples)
            corpus_tokens = set(corpus.lower().split())
            if not corpus_tokens:
                continue

            intersection = task_tokens & corpus_tokens
            union = task_tokens | corpus_tokens
            jaccard = len(intersection) / len(union)
            score = jaccard * (1 + 0.01 * cap.priority)
            if score > 0:
                scored.append((cap, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
