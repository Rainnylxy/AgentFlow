"""Agent 间消息传递 — 显式的 Agent-to-Agent 通信。

与共享记忆的区别：
    共享记忆 — 所有 Agent 隐式读写同一池子（被动、全局）
    消息传递 — Agent A 显式发给 Agent B（主动、定向）

两者共存：记忆存上下文，消息做协调。

用法:
    bus = MessageBus()
    bus.send(AgentMessage(from_agent="planner", to_agent="worker",
                          intent="delegate", payload={"task": "查库存"}))
    msgs = bus.receive("worker")  # → [AgentMessage(...)]
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """消息意图类型。"""
    DELEGATE = "delegate"           # 委派任务
    TASK_COMPLETE = "task_complete"  # 任务完成 + 结果
    NEED_CLARIFICATION = "need_clarification"  # 需要澄清
    HANDOFF = "handoff"             # 转交（把会话交给另一个 Agent）
    ERROR = "error"                 # 执行出错
    INFO = "info"                   # 一般通知


@dataclass
class AgentMessage:
    """一条 Agent 间消息。

    to_agent 可以是:
        - 具体 Agent id: "fraud_check"
        - "orchestrator": 发给编排器
        - "broadcast": 广播给所有下游 Agent
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str = ""
    to_agent: str = "broadcast"
    intent: str = "info"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "intent": self.intent,
            "payload": self.payload,
        }


class MessageBus:
    """Workflow 内的消息总线。

    每个 Workflow 实例化一个 MessageBus。
    Agent 通过它发送定向消息、接收自己的未读消息。
    """

    def __init__(self):
        self._messages: list[AgentMessage] = []
        self._read_ids: set[str] = set()  # 已读消息 id

    def send(self, msg: AgentMessage) -> None:
        """发送一条消息。"""
        self._messages.append(msg)

    def broadcast(self, from_agent: str, intent: str, payload: dict) -> None:
        """广播一条消息给所有 Agent。"""
        self.send(AgentMessage(
            from_agent=from_agent,
            to_agent="broadcast",
            intent=intent,
            payload=payload,
        ))

    def receive(self, agent_id: str) -> list[AgentMessage]:
        """获取发给该 Agent 的未读消息（含广播 + orchestrator）。

        定向消息读取后标记为已读；
        广播消息不标记——所有 Agent 都能读到。
        """
        unread = []
        for m in self._messages:
            key = (m.id, agent_id)
            if key in self._read_ids:
                continue
            matches = (
                m.to_agent == agent_id
                or m.to_agent == "broadcast"
                or m.to_agent == "orchestrator"
            )
            if matches:
                unread.append(m)
                if m.to_agent == agent_id:
                    self._read_ids.add(key)  # 定向消息标记已读
        return unread

    def peek(self, agent_id: str) -> list[AgentMessage]:
        """查看发给该 Agent 的消息（不标记已读）。"""
        result = []
        for m in self._messages:
            key = (m.id, agent_id)
            if key in self._read_ids:
                continue
            if m.to_agent == agent_id or m.to_agent == "broadcast" or m.to_agent == "orchestrator":
                result.append(m)
        return result

    def all_messages(self) -> list[AgentMessage]:
        """返回所有消息（调试用）。"""
        return list(self._messages)

    def clear(self) -> None:
        """清空消息总线。"""
        self._messages.clear()
        self._read_ids.clear()
