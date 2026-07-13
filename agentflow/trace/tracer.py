"""统一 Trace 模型 — 单 Agent 和多 Agent 一个结构。

WorkflowTrace
├── dag_groups: 并行分组
├── node_traces: {node_id: AgentTrace}    ← 每节点内部 Trace
├── message_flow: [MessageRecord]         ← 消息时间线
├── summary: 总耗时 / 瓶颈 / 关键路径
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 单 Agent — 每轮 Tracing
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """一次工具调用的记录。"""
    tool: str
    input: dict = field(default_factory=dict)
    output: str = ""
    success: bool = True
    error: str = ""
    duration_ms: int = 0


@dataclass
class AgentTurn:
    """单轮思考-行动-观察的完整记录。"""
    turn: int
    thinking: str = ""                    # 思考内容
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_answer: str = ""                # 如果这轮给出了最终答案
    tokens: dict = field(default_factory=dict)  # {input, output}
    duration_ms: int = 0
    messages_snapshot: list[dict] = field(default_factory=list)  # 本轮 LLM 调用前的完整 messages


@dataclass
class AgentTrace:
    """单个 Agent 节点的完整执行轨迹。"""
    agent_id: str
    turns: list[AgentTurn] = field(default_factory=list)

    # 输入上下文（Agent 看到了什么）
    upstream_visible: list[str] = field(default_factory=list)  # 可见的上游节点
    incoming_messages: list[dict] = field(default_factory=list)  # 收到的消息摘要
    memory_scope: str = "inherit"

    # 汇总
    total_turns: int = 0
    total_tool_calls: int = 0
    total_tokens: dict = field(default_factory=dict)  # {input, output}
    total_duration_ms: int = 0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# 消息流记录
# ---------------------------------------------------------------------------

@dataclass
class MessageRecord:
    """消息时间线中的一条。"""
    timestamp: float = 0.0
    from_agent: str = ""
    to_agent: str = ""
    intent: str = ""
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 多 Agent Workflow Trace
# ---------------------------------------------------------------------------

@dataclass
class WorkflowSummary:
    """Workflow 执行汇总。"""
    total_duration_ms: int = 0
    total_nodes: int = 0
    nodes_executed: int = 0
    nodes_skipped: int = 0
    nodes_failed: int = 0
    total_turns: int = 0
    total_tool_calls: int = 0
    total_messages: int = 0
    bottleneck: str = ""                  # 耗时最长的节点
    critical_path: list[str] = field(default_factory=list)  # 关键路径


@dataclass
class WorkflowTrace:
    """单/多 Agent 的统一 Trace。

    单个 Agent 就是一个只有 1 个节点的 WorkflowTrace。
    """
    workflow_id: str = ""
    workflow_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    # DAG 结构
    dag_groups: list[list[str]] = field(default_factory=list)

    # 每个节点的内部 Trace
    node_traces: dict[str, AgentTrace] = field(default_factory=dict)

    # 消息时间线
    message_flow: list[MessageRecord] = field(default_factory=list)

    # 汇总
    summary: WorkflowSummary = field(default_factory=WorkflowSummary)

    @classmethod
    def start(cls, workflow_id: str = "", workflow_name: str = "") -> "WorkflowTrace":
        return cls(
            workflow_id=workflow_id or hex(int(time.time() * 1000))[2:],
            workflow_name=workflow_name,
            start_time=time.time(),
        )

    def finish(self) -> None:
        self.end_time = time.time()
        self._compute_summary()

    def _compute_summary(self) -> None:
        """自动计算瓶颈、关键路径等。"""
        traces = self.node_traces
        s = self.summary
        s.total_nodes = len(traces)
        s.nodes_failed = sum(1 for t in traces.values() if not t.success)
        s.total_messages = len(self.message_flow)

        # 瓶颈：耗时最长的节点
        max_dur = 0
        for agent_id, at in traces.items():
            s.total_turns += at.total_turns
            s.total_tool_calls += at.total_tool_calls
            if at.total_duration_ms > max_dur:
                max_dur = at.total_duration_ms
                s.bottleneck = agent_id

        # 关键路径：每层挑最慢的
        s.critical_path = []
        for group in self.dag_groups:
            slowest = ""
            slowest_dur = 0
            for nid in group:
                if nid in traces:
                    dur = traces[nid].total_duration_ms
                    if dur > slowest_dur:
                        slowest_dur = dur
                        slowest = nid
            if slowest:
                s.critical_path.append(slowest)

        s.total_duration_ms = int((self.end_time - self.start_time) * 1000)

    def to_dict(self) -> dict:
        """导出为可序列化的 dict（日志/存储/API）。"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "dag_groups": self.dag_groups,
            "node_traces": {
                nid: {
                    "agent_id": at.agent_id,
                    "total_turns": at.total_turns,
                    "total_tool_calls": at.total_tool_calls,
                    "total_tokens": at.total_tokens,
                    "total_duration_ms": at.total_duration_ms,
                    "success": at.success,
                    "error": at.error,
                    "memory_scope": at.memory_scope,
                    "upstream_visible": at.upstream_visible,
                    "turns": [
                        {
                            "turn": t.turn,
                            "thinking": t.thinking[:200],
                            "tool_calls": [
                                {"tool": tc.tool, "input": tc.input,
                                 "output": tc.output[:100], "duration_ms": tc.duration_ms}
                                for tc in t.tool_calls
                            ],
                            "tokens": t.tokens,
                            "duration_ms": t.duration_ms,
                            "messages_snapshot": t.messages_snapshot,
                        }
                        for t in at.turns
                    ],
                }
                for nid, at in self.node_traces.items()
            },
            "message_flow": [
                {"from": m.from_agent, "to": m.to_agent,
                 "intent": m.intent, "payload": m.payload}
                for m in self.message_flow
            ],
            "summary": {
                "total_duration_ms": self.summary.total_duration_ms,
                "nodes_executed": self.summary.nodes_executed,
                "nodes_skipped": self.summary.nodes_skipped,
                "nodes_failed": self.summary.nodes_failed,
                "bottleneck": self.summary.bottleneck,
                "critical_path": self.summary.critical_path,
                "total_messages": self.summary.total_messages,
            },
        }

    def diff(self, other: "WorkflowTrace") -> dict:
        """A/B 对比：两次执行的差异。"""
        changes = []
        all_nodes = set(self.node_traces) | set(other.node_traces)

        for nid in sorted(all_nodes):
            a = self.node_traces.get(nid)
            b = other.node_traces.get(nid)
            if a is None:
                changes.append({"node": nid, "change": "added"})
            elif b is None:
                changes.append({"node": nid, "change": "removed"})
            else:
                if a.total_duration_ms != b.total_duration_ms:
                    changes.append({
                        "node": nid, "change": "duration",
                        "old_ms": a.total_duration_ms,
                        "new_ms": b.total_duration_ms,
                    })
                if a.total_tool_calls != b.total_tool_calls:
                    changes.append({
                        "node": nid, "change": "tool_calls",
                        "old": a.total_tool_calls, "new": b.total_tool_calls,
                    })
                if a.success != b.success:
                    changes.append({
                        "node": nid, "change": "status",
                        "old": "OK" if a.success else "FAIL",
                        "new": "OK" if b.success else "FAIL",
                    })

        return {
            "trace_old": self.workflow_id,
            "trace_new": other.workflow_id,
            "changes": changes,
            "summary_diff": {
                "duration_delta_ms": other.summary.total_duration_ms - self.summary.total_duration_ms,
                "failed_delta": other.summary.nodes_failed - self.summary.nodes_failed,
            },
        }
