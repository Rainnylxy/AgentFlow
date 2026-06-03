"""Layer 1: Working Memory — 当前对话的完整消息窗口。

支持滑动窗口（按轮数截断）和 token 限制（按字符估算截断）。
"""

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str = ""
    tool_calls: list = field(default_factory=list)


class WorkingMemory:
    """当前对话窗口，支持滑动窗口和 token 限制。"""

    def __init__(self, max_turns: int = 20, max_tokens: int = 8000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        while len(self._messages) > self.max_turns:
            self._messages.pop(0)

    def clear(self) -> None:
        self._messages.clear()

    def get_context_window(self) -> list[Message]:
        """获取对话窗口，按 token 限制从后向前截取。

        使用粗略估算：1 token ≈ 4 chars。
        从最新消息向旧消息累积，超出 token 限制时停止。
        """
        result = []
        total_chars = 0
        char_limit = self.max_tokens * 4

        for msg in reversed(self._messages):
            total_chars += len(msg.content)
            if total_chars > char_limit:
                break
            result.insert(0, msg)
        return result

    def __len__(self):
        return len(self._messages)
