"""Trace Client：记录每次 Workflow 执行的完整轨迹，支持 A/B 对比"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TraceSpan:
    """单次 Span —— 对应 DAG 中一个节点的执行记录。"""
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    status: str = "running"
    input: str = ""
    output: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: int = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)

    def end(self, status: str = "success", output: str = ""):
        self.status = status
        self.output = output
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)


@dataclass
class Trace:
    """一次完整的 Workflow 执行轨迹。"""
    trace_id: str
    workflow_id: str
    status: str = "running"
    spans: list[TraceSpan] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def start_span(self, name: str) -> TraceSpan:
        span = TraceSpan(name=name, trace_id=self.trace_id)
        self.spans.append(span)
        return span

    def end(self, status: str = "completed"):
        self.status = status
        self.end_time = time.time()


class TraceClient:
    def start_trace(self, workflow_id: str) -> Trace:
        return Trace(trace_id=uuid.uuid4().hex, workflow_id=workflow_id)

    def diff(self, t1: Trace, t2: Trace) -> dict:
        """对比两个 Trace，输出差异报告（用于 A/B testing）。"""
        changes = []
        s1 = {s.name: s for s in t1.spans}
        s2 = {s.name: s for s in t2.spans}

        for name, span in s1.items():
            other = s2.get(name)
            if other is None:
                changes.append({"span": name, "change": "removed"})
            else:
                if span.output != other.output:
                    changes.append({"span": name, "change": "output",
                                    "old": span.output, "new": other.output})
                if abs(span.duration_ms - other.duration_ms) > 100:
                    changes.append({"span": name, "change": "duration_ms",
                                    "old": span.duration_ms, "new": other.duration_ms})

        for name in set(s2) - set(s1):
            changes.append({"span": name, "change": "added"})

        return {"trace1": t1.trace_id, "trace2": t2.trace_id, "changes": changes}
