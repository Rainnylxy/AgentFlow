from agentflow.runtime.memory import MemoryManager, ShortTermMemory, LongTermMemory, Message


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_messages=5)
        mem.add(Message(role="user", content="Hello"))
        mem.add(Message(role="assistant", content="Hi!"))
        assert len(mem.get_messages()) == 2

    def test_sliding_window(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add(Message(role="user", content=f"msg-{i}"))
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0].content == "msg-2"  # msg-0, msg-1 被挤出

    def test_context_window_token_limit(self):
        mem = ShortTermMemory(max_messages=100, max_tokens=50)
        # 多条短消息——全在 token 限制内
        mem.add(Message(role="user", content="hello"))
        mem.add(Message(role="assistant", content="world"))
        mem.add(Message(role="user", content="ok"))
        msgs = mem.get_context_window()
        assert len(msgs) == 3


class TestLongTermMemory:
    def test_store_and_search(self):
        mem = LongTermMemory()
        mem.store("k1", {"fact": "AgentFlow is Go+Python"})
        mem.store("k2", {"fact": "AgentFlow uses LangGraph"})
        results = mem.search("Go")
        assert len(results) == 1
        assert "AgentFlow is Go+Python" in str(results[0])

    def test_get_missing_returns_none(self):
        mem = LongTermMemory()
        assert mem.get("nonexistent") is None


class TestMemoryManager:
    def test_integration(self):
        mgr = MemoryManager()
        assert mgr.short_term is not None
        assert mgr.long_term is not None
