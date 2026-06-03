from agentflow.runtime.memory.working import WorkingMemory, Message


class TestWorkingMemory:
    def test_add_and_retrieve(self):
        wm = WorkingMemory(max_turns=20)
        wm.add(Message(role="user", content="Hello"))
        wm.add(Message(role="assistant", content="Hi!"))
        msgs = wm.get_context_window()
        assert len(msgs) == 2

    def test_sliding_window_by_turns(self):
        wm = WorkingMemory(max_turns=3)
        for i in range(6):
            wm.add(Message(role="user", content=f"msg-{i}"))
        msgs = wm.get_context_window()
        assert len(msgs) == 3
        assert msgs[0].content == "msg-3"

    def test_token_limit(self):
        wm = WorkingMemory(max_turns=100, max_tokens=30)
        wm.add(Message(role="user", content="x" * 500))  # ~125 tokens worth of chars
        msgs = wm.get_context_window()
        total_chars = sum(len(m.content) for m in msgs)
        # rough 1 token ≈ 4 chars, plus some tolerance
        assert total_chars <= 30 * 4 + 50

    def test_clear(self):
        wm = WorkingMemory(max_turns=10)
        wm.add(Message(role="user", content="test"))
        wm.clear()
        assert len(wm.get_context_window()) == 0

    def test_len(self):
        wm = WorkingMemory(max_turns=10)
        wm.add(Message(role="user", content="a"))
        wm.add(Message(role="assistant", content="b"))
        assert len(wm) == 2
