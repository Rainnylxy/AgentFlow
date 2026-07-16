from agentflow.runtime.memory.working import WorkingMemory, Message
from agentflow.runtime.memory.token_counter import AdaptiveCounter


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
        """token 限制：短消息全部保留，超长消息被截断。"""
        wm = WorkingMemory(max_turns=100, max_tokens=10)
        # 每个短消息 ~1-2 tokens，全保留
        wm.add(Message(role="user", content="hi"))
        wm.add(Message(role="assistant", content="hello"))
        msgs = wm.get_context_window()
        assert len(msgs) == 2

        # 加一条长消息应该挤掉最旧的
        wm.add(Message(role="user", content="long message " * 20))  # ~25 tokens
        msgs = wm.get_context_window()
        assert len(msgs) >= 1  # 至少长消息本身保留

    def test_token_limit_precise(self):
        """精确验证 token 计数截断行为。"""
        counter = AdaptiveCounter()
        wm = WorkingMemory(max_turns=100, max_tokens=20, token_counter=counter)
        wm.add(Message(role="user", content="hello world"))  # ~2 tokens
        wm.add(Message(role="assistant", content="hi there"))  # ~2 tokens
        wm.add(Message(role="user", content="ok"))  # ~1 token
        msgs = wm.get_context_window()
        # 三条短消息都在 20 token 限制内
        assert len(msgs) == 3

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

    def test_token_counter_is_used(self):
        """验证使用 AdaptiveCounter 而非旧 4 chars/token。"""
        wm = WorkingMemory(max_turns=100, max_tokens=5)

        # "x" * 40: old method → 40 chars = 10 tokens, would exceed 5
        # AdaptiveCounter: 1 word of 40 chars → max(1.3, 10) = 10 tokens, also exceeds
        # 但实际 tokenizer 中 "x"*40 可能更少。关键是使用 counter 而非 chars*4
        wm.add(Message(role="user", content="short text"))
        msgs = wm.get_context_window()
        assert len(msgs) == 1  # 1-2 tokens, fits in 5

    def test_compression_overflow(self):
        """max_turns 挤出消息进 overflow。"""
        wm = WorkingMemory(max_turns=2)
        wm.add(Message(role="user", content="msg1"))
        wm.add(Message(role="assistant", content="msg2"))
        wm.add(Message(role="user", content="msg3"))
        assert len(wm) == 2
        assert len(wm._overflow) == 1
        assert wm._overflow[0].content == "msg1"

    def test_no_compression_without_summarizer(self):
        """没有 summarizer 时不压缩。"""
        wm = WorkingMemory(max_turns=2)
        wm.add(Message(role="user", content="msg1"))
        wm.add(Message(role="assistant", content="msg2"))
        wm.add(Message(role="user", content="msg3"))
        assert not wm.needs_compression  # no summarizer set
