import pytest
from datetime import datetime, timedelta
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact


class TestMemoryFact:
    def test_creation(self):
        fact = MemoryFact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            object="quick replies",
            confidence=0.9,
            timestamp=datetime.now(),
            source_turn=3,
            ttl=86400,
        )
        assert fact.fact_type == "preference"
        assert not fact.is_expired()

    def test_expiry(self):
        past = datetime.now() - timedelta(hours=25)
        fact = MemoryFact(
            fact_type="event", subject="user", predicate="did", object="login",
            confidence=0.8, timestamp=past, source_turn=1, ttl=3600,
        )
        assert fact.is_expired()

    def test_decay(self):
        fact = MemoryFact(
            fact_type="entity", subject="tool:weather", predicate="returned", object="22C",
            confidence=0.8, timestamp=datetime.now(), source_turn=1, ttl=86400,
        )
        fact.decay(0.5)
        assert fact.confidence == 0.4


class TestEpisodicMemory:
    def test_store_and_retrieve_by_subject(self):
        mem = EpisodicMemory(max_facts=100)
        f1 = MemoryFact("event", "user", "asked", "refund", 0.9, datetime.now(), 1, 86400)
        f2 = MemoryFact("event", "agent", "responded", "policy", 0.9, datetime.now(), 1, 86400)
        mem.add(f1)
        mem.add(f2)
        user_facts = mem.get_by_subject("user")
        assert len(user_facts) == 1
        assert user_facts[0].object == "refund"

    def test_capacity_eviction(self):
        """容量超限时淘汰低质量条目。"""
        mem = EpisodicMemory(max_facts=3)
        now = datetime.now()

        mem.add(MemoryFact("event", "u", "did", "a", 0.9, now, 1, 86400))
        mem.add(MemoryFact("event", "u", "did", "b", 0.5, now, 2, 86400))
        old = now - timedelta(days=30)
        mem.add(MemoryFact("event", "u", "did", "c", 0.9, old, 3, 86400))
        mem.add(MemoryFact("event", "u", "did", "d", 0.7, now, 4, 86400))

        assert mem.count() == 3
        subjects = [f.object for f in mem.get_all()]
        assert "a" in subjects  # highest confidence + newest

    def test_forget_expired(self):
        """遗忘门：清除所有过期事实。"""
        mem = EpisodicMemory(max_facts=100)
        now = datetime.now()
        past = now - timedelta(hours=25)

        mem.add(MemoryFact("event", "u", "did", "fresh", 0.9, now, 1, ttl=86400))
        mem.add(MemoryFact("event", "u", "did", "stale", 0.9, past, 2, ttl=3600))

        removed = mem.forget_expired()
        assert removed >= 1
        assert mem.count() == 1

    def test_get_by_type(self):
        mem = EpisodicMemory(max_facts=100)
        mem.add(MemoryFact("event", "u", "did", "a", 0.9, datetime.now(), 1, 86400))
        mem.add(MemoryFact("preference", "u", "likes", "b", 0.9, datetime.now(), 1, 86400))
        events = mem.get_by_type("event")
        assert len(events) == 1
        assert events[0].fact_type == "event"
