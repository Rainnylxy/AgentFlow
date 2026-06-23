"""Eval Suite Runner：批量执行 EvalCase，生成对比报告。

支持可选的 Trace 联动——在评测时自动采集执行轨迹，
事后可按评测维度反向定位到 Trace 中的具体节点。
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional
from agentflow.eval.base import BaseEvaluator, EvalResult


@dataclass
class EvalCase:
    id: str
    input: str
    expected: str
    evaluator: BaseEvaluator


@dataclass
class SuiteReport:
    name: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    details: list = field(default_factory=list)


class EvalSuite:
    def __init__(
        self,
        name: str,
        cases: list[EvalCase],
        trace_client: Optional[object] = None,
    ):
        self.name = name
        self.cases = cases
        self._trace_client = trace_client  # TraceClient 实例（可选）

    async def run(self, agent_fn: Callable[[str], Awaitable[str]]) -> SuiteReport:
        """异步执行评测套件。

        agent_fn: 接受 str 输入、返回 str 输出的异步函数（通常是 agent.run()）。

        如果配置了 trace_client，每个 case 执行时会采集一条 Trace，
        trace_id 会写入详情中，供事后 diagnose() 使用。
        """
        if not self.cases:
            raise ValueError("No cases in suite")
        passed = 0
        failed = 0
        details = []
        for case in self.cases:
            # 可选：为该 case 开启一条 Trace
            trace_id = None
            if self._trace_client:
                trace = self._trace_client.start_trace(f"eval:{self.name}:{case.id}")
                trace_id = trace.trace_id
                # 为 agent 调用创建一个 span
                span = trace.start_span("agent_run")
                actual = await agent_fn(case.input)
                span.end(status="success", output=actual[:200])
                trace.end("completed")
            else:
                actual = await agent_fn(case.input)

            result = await case.evaluator.evaluate(case.expected, actual)
            detail = {
                "case_id": case.id, "input": case.input,
                "expected": case.expected, "actual": actual,
                "score": result.score, "passed": result.passed,
                "reason": result.reason,
            }
            if trace_id:
                detail["trace_id"] = trace_id
            details.append(detail)

            if result.passed:
                passed += 1
            else:
                failed += 1
        return SuiteReport(name=self.name, total=len(self.cases), passed=passed,
                           failed=failed, pass_rate=passed / len(self.cases), details=details)

    def diagnose(self, report: SuiteReport, min_score: float = 0.5) -> list[dict]:
        """关联分析：找出低分案例及其 Trace 信息。

        向后追溯：当某个 case 得分 < min_score 时，
        可通过 trace_id 定位到 Trace，查看具体哪个 step/tool 出了问题。

        Returns:
            [{case_id, score, reason, trace_id}, ...] 按 score 升序排列
        """
        low_scoring = [
            d for d in report.details
            if d["score"] < min_score
        ]
        low_scoring.sort(key=lambda d: d["score"])
        return [
            {
                "case_id": d["case_id"],
                "score": d["score"],
                "reason": d["reason"],
                "trace_id": d.get("trace_id", "N/A"),
            }
            for d in low_scoring
        ]

    def compare(self, old: SuiteReport, new: SuiteReport) -> dict:
        old_map = {d["case_id"]: d for d in old.details}
        new_map = {d["case_id"]: d for d in new.details}
        improved = sum(1 for cid in old_map if new_map[cid]["score"] > old_map[cid]["score"])
        regressed = sum(1 for cid in old_map if new_map[cid]["score"] < old_map[cid]["score"])
        unchanged = len(old_map) - improved - regressed
        return {"total": len(old_map), "improved": improved, "regressed": regressed,
                "unchanged": unchanged, "old_pass_rate": old.pass_rate, "new_pass_rate": new.pass_rate}
