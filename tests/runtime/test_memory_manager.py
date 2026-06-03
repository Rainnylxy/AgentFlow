import pytest
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile
from agentflow.runtime.memory.working import Message


class TestMemoryProfile:
    def test_light_profile(self):
        p = MemoryProfile.light()
        assert p.working.max_turns == 10
        assert p.episodic_max == 0

    def test_standard_profile(self):
        p = MemoryProfile.standard()
        assert p.working.max_turns == 20
        assert p.episodic_max == 200

    def test_deep_profile(self):
        p = MemoryProfile.deep()
        assert p.working.max_turns == 40
        assert p.episodic_max == 500
        assert p.semantic_enabled is True


class TestMemoryManager:
    def test_pre_turn_retrieves_from_semantic(self):
        mgr = MemoryManager(verbose=True)
        mgr.semantic.store("kb_1", "Refund policy: 30 days unconditional")
        mgr.semantic.store("kb_2", "Contact support: email support@example.com")

        facts = mgr.pre_turn("I want a refund")
        assert len(facts) > 0
        assert any("refund" in str(f).lower() for f in facts)

    def test_post_turn_extracts_facts(self):
        mgr = MemoryManager(verbose=True)
        mgr.working.add(Message(role="user", content="I live in Beijing"))
        mgr.working.add(Message(role="assistant", content="Got it, Beijing it is."))

        mgr.post_turn()
        facts = mgr.episodic.get_all()
        assert any("Beijing" in str(f) for f in facts)

    def test_full_cycle(self):
        mgr = MemoryManager()
        mgr.pre_turn("What's the weather?")
        mgr.working.add(Message(role="user", content="What's the weather?"))
        mgr.working.add(Message(role="assistant", content="It's 22C in Beijing."))
        mgr.post_turn()
        assert mgr.episodic.count() > 0

    def test_working_memory_integration(self):
        mgr = MemoryManager()
        mgr.working.add(Message(role="user", content="Hello"))
        msgs = mgr.working.get_context_window()
        assert len(msgs) == 1
        assert msgs[0].content == "Hello"
