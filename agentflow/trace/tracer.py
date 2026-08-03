"""统一 Trace 模型 — 单 Agent 和多 Agent 一个结构。

WorkflowTrace
├── dag_groups: 并行分组
├── node_traces: {node_id: AgentTrace}    ← 每节点内部 Trace
├── message_flow: [MessageRecord]         ← 消息时间线
├── summary: 总耗时 / 瓶颈 / 关键路径
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    finish_reason: str = ""               # LLM 结束原因: stop, length, tool_calls, content_filter
    reasoning: str = ""                   # 模型的思考/推理过程（DeepSeek R1, OpenAI o1）
    tokens: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}
    duration_ms: int = 0
    llm_call_duration_ms: int = 0       # 本轮 LLM API 调用耗时（不含工具执行）
    start_time: float = 0.0              # turn 开始的绝对时间戳
    messages_snapshot: list[dict] = field(default_factory=list)  # 本轮 LLM 调用前的完整 messages
    tools_snapshot: list[dict] = field(default_factory=list)     # 本轮 LLM 可用的工具定义


@dataclass
class AgentTrace:
    """单个 Agent 节点的完整执行轨迹。"""
    agent_id: str
    turns: list[AgentTurn] = field(default_factory=list)

    # 输入上下文（Agent 看到了什么）
    upstream_visible: list[str] = field(default_factory=list)  # 可见的上游节点
    incoming_messages: list[dict] = field(default_factory=list)  # 收到的消息摘要
    memory_scope: str = "inherit"

    # Memory 操作记录
    memory_retrieved: list[dict] = field(default_factory=list)   # pre_turn 检索到的记忆
    memory_stored: list[dict] = field(default_factory=list)      # post_turn 提取并存储的事实
    memory_forgotten: int = 0                                     # post_turn 遗忘的过期事实数

    # 汇总
    total_turns: int = 0
    total_tool_calls: int = 0
    total_tokens: dict = field(default_factory=dict)  # {input, output}
    total_duration_ms: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        """导出为可序列化的 dict（单 Agent 场景直接调用）。"""
        return {
            "agent_id": self.agent_id,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "error": self.error,
            "memory_scope": self.memory_scope,
            "upstream_visible": self.upstream_visible,
            "memory_retrieved": self.memory_retrieved,
            "memory_stored": self.memory_stored,
            "memory_forgotten": self.memory_forgotten,
            "turns": [
                {
                    "turn": t.turn,
                    "thinking": t.thinking,
                    "tool_calls": [
                        {"tool": tc.tool, "input": tc.input,
                         "output": tc.output, "duration_ms": tc.duration_ms,
                         "success": tc.success, "error": tc.error}
                        for tc in t.tool_calls
                    ],
                    "finish_reason": t.finish_reason,
                    "reasoning": t.reasoning,
                    "tokens": t.tokens,
                    "duration_ms": t.duration_ms,
                    "llm_call_duration_ms": t.llm_call_duration_ms,
                    "messages_snapshot": t.messages_snapshot,
                    "tools_snapshot": t.tools_snapshot,
                }
                for t in self.turns
            ],
        }


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
    turn_number: int = 0  # 消息发送时 from_agent 所在的 turn 序号


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
    total_llm_duration_ms: int = 0        # 所有 LLM 调用总耗时
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
            s.total_llm_duration_ms += sum(
                t.llm_call_duration_ms for t in at.turns
            )
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
                nid: at.to_dict()
                for nid, at in self.node_traces.items()
            },
            "message_flow": [
                {"from": m.from_agent, "to": m.to_agent,
                 "intent": m.intent, "payload": m.payload,
                 "turn_number": m.turn_number}
                for m in self.message_flow
            ],
            "summary": {
                "total_duration_ms": self.summary.total_duration_ms,
                "total_llm_duration_ms": self.summary.total_llm_duration_ms,
                "nodes_executed": self.summary.nodes_executed,
                "nodes_skipped": self.summary.nodes_skipped,
                "nodes_failed": self.summary.nodes_failed,
                "bottleneck": self.summary.bottleneck,
                "critical_path": self.summary.critical_path,
                "total_messages": self.summary.total_messages,
            },
        }

    def diff(self, other: "WorkflowTrace") -> dict:
        """A/B 对比：两次执行的差异，含节点级和 turn 级。"""
        changes = []
        all_nodes = set(self.node_traces) | set(other.node_traces)

        for nid in sorted(all_nodes):
            a = self.node_traces.get(nid)
            b = other.node_traces.get(nid)
            if a is None:
                changes.append({"node": nid, "level": "node", "change": "added"})
                continue
            if b is None:
                changes.append({"node": nid, "level": "node", "change": "removed"})
                continue

            # -- 节点级别 --
            if a.total_duration_ms != b.total_duration_ms:
                changes.append({
                    "node": nid, "level": "node", "change": "duration",
                    "old_ms": a.total_duration_ms, "new_ms": b.total_duration_ms,
                    "delta_ms": b.total_duration_ms - a.total_duration_ms,
                })
            if a.total_tool_calls != b.total_tool_calls:
                changes.append({
                    "node": nid, "level": "node", "change": "tool_calls",
                    "old": a.total_tool_calls, "new": b.total_tool_calls,
                })
            if a.success != b.success:
                changes.append({
                    "node": nid, "level": "node", "change": "status",
                    "old": "OK" if a.success else "FAIL",
                    "new": "OK" if b.success else "FAIL",
                })
            # Token 变化
            old_tok = a.total_tokens.get("total_tokens", 0)
            new_tok = b.total_tokens.get("total_tokens", 0)
            if old_tok != new_tok:
                changes.append({
                    "node": nid, "level": "node", "change": "tokens",
                    "old": old_tok, "new": new_tok,
                    "delta": new_tok - old_tok,
                })

            # -- Turn 级别 --
            if a.total_turns != b.total_turns:
                changes.append({
                    "node": nid, "level": "turn", "change": "turns_count",
                    "old": a.total_turns, "new": b.total_turns,
                })

            for i in range(min(len(a.turns), len(b.turns))):
                ta, tb = a.turns[i], b.turns[i]
                turn_label = f"turn_{i + 1}"

                if ta.finish_reason != tb.finish_reason:
                    changes.append({
                        "node": nid, "level": "turn", "change": "finish_reason",
                        "turn": turn_label,
                        "old": ta.finish_reason, "new": tb.finish_reason,
                    })

                ta_tools = [tc.tool for tc in ta.tool_calls]
                tb_tools = [tc.tool for tc in tb.tool_calls]
                if ta_tools != tb_tools:
                    changes.append({
                        "node": nid, "level": "turn", "change": "tool_sequence",
                        "turn": turn_label,
                        "old": ta_tools, "new": tb_tools,
                    })

                if len(ta.tool_calls) != len(tb.tool_calls):
                    changes.append({
                        "node": nid, "level": "turn", "change": "tool_calls_count",
                        "turn": turn_label,
                        "old": len(ta.tool_calls), "new": len(tb.tool_calls),
                    })

                ta_tok = ta.tokens.get("total_tokens", 0)
                tb_tok = tb.tokens.get("total_tokens", 0)
                if ta_tok != tb_tok:
                    changes.append({
                        "node": nid, "level": "turn", "change": "tokens",
                        "turn": turn_label,
                        "old": ta_tok, "new": tb_tok,
                        "delta": tb_tok - ta_tok,
                    })

        # 汇总 token delta
        total_tok_old = sum(
            t.total_tokens.get("total_tokens", 0)
            for t in self.node_traces.values()
        )
        total_tok_new = sum(
            t.total_tokens.get("total_tokens", 0)
            for t in other.node_traces.values()
        )

        return {
            "trace_old": self.workflow_id,
            "trace_new": other.workflow_id,
            "changes": changes,
            "summary_diff": {
                "duration_delta_ms": other.summary.total_duration_ms - self.summary.total_duration_ms,
                "failed_delta": other.summary.nodes_failed - self.summary.nodes_failed,
                "tokens_delta": total_tok_new - total_tok_old,
            },
        }


# ---------------------------------------------------------------------------
# Trace 持久化
# ---------------------------------------------------------------------------

class TraceStore:
    """文件持久化存储：保存、加载、列表、删除 WorkflowTrace。"""

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".agentflow" / "traces"
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, workflow_id: str) -> Path:
        return self._dir / f"{workflow_id}.json"

    def save(self, trace: WorkflowTrace) -> str:
        """持久化一条 Trace，返回 workflow_id。"""
        path = self._path(trace.workflow_id)
        data = trace.to_dict()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace.workflow_id

    def load(self, workflow_id: str) -> Optional[WorkflowTrace]:
        """从文件加载 WorkflowTrace。"""
        path = self._path(workflow_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._from_dict(data)

    def list(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """列出最近的 Trace 摘要（不含完整 turn 数据）。"""
        files = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        results = []
        for f in files[offset:offset + limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append({
                    "workflow_id": data.get("workflow_id", ""),
                    "workflow_name": data.get("workflow_name", ""),
                    "summary": data.get("summary", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return results

    def delete(self, workflow_id: str) -> bool:
        """删除一条 Trace。返回 True 表示成功删除。"""
        path = self._path(workflow_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _from_dict(data: dict) -> WorkflowTrace:
        """从 dict 反序列化 WorkflowTrace。"""
        trace = WorkflowTrace(
            workflow_id=data.get("workflow_id", ""),
            workflow_name=data.get("workflow_name", ""),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            dag_groups=data.get("dag_groups", []),
        )
        # 还原 summary
        s = data.get("summary", {})
        trace.summary.total_duration_ms = s.get("total_duration_ms", 0)
        trace.summary.nodes_executed = s.get("nodes_executed", 0)
        trace.summary.nodes_skipped = s.get("nodes_skipped", 0)
        trace.summary.nodes_failed = s.get("nodes_failed", 0)
        trace.summary.bottleneck = s.get("bottleneck", "")
        trace.summary.critical_path = s.get("critical_path", [])
        trace.summary.total_messages = s.get("total_messages", 0)
        trace.summary.total_turns = s.get("total_turns", 0)
        trace.summary.total_tool_calls = s.get("total_tool_calls", 0)

        # 还原 node_traces
        for nid, nd in data.get("node_traces", {}).items():
            at = AgentTrace(agent_id=nd.get("agent_id", nid))
            at.total_turns = nd.get("total_turns", 0)
            at.total_tool_calls = nd.get("total_tool_calls", 0)
            at.total_tokens = nd.get("total_tokens", {})
            at.total_duration_ms = nd.get("total_duration_ms", 0)
            at.success = nd.get("success", True)
            at.error = nd.get("error", "")
            for td in nd.get("turns", []):
                turn = AgentTurn(turn=td.get("turn", 0))
                turn.thinking = td.get("thinking", "")
                turn.finish_reason = td.get("finish_reason", "")
                turn.reasoning = td.get("reasoning", "")
                turn.tokens = td.get("tokens", {})
                turn.duration_ms = td.get("duration_ms", 0)
                turn.final_answer = td.get("final_answer", "")
                for tcd in td.get("tool_calls", []):
                    turn.tool_calls.append(ToolCallRecord(
                        tool=tcd.get("tool", ""),
                        input=tcd.get("input", {}),
                        output=tcd.get("output", ""),
                        success=tcd.get("success", True),
                        error=tcd.get("error", ""),
                        duration_ms=tcd.get("duration_ms", 0),
                    ))
                at.turns.append(turn)
            trace.node_traces[nid] = at

        return trace
