"""Eval Suite Runner：批量执行 EvalCase，生成对比报告。

每条 case 执行时自动创建 WorkflowTrace 记录完整执行轨迹，
事后可通过 diagnose() 按评测维度反向定位到 Trace 中的具体 turn。
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional
from agentflow.eval.base import BaseEvaluator, EvalResult
from agentflow.trace.tracer import WorkflowTrace, AgentTrace


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
    ):
        self.name = name
        self.cases = cases

    async def run(self, agent_fn: Callable[[str], Awaitable[str]]) -> SuiteReport:
        """异步执行评测套件。

        agent_fn: 接受 str 输入、返回 str 输出的异步函数（通常是 agent.run()）。

        每个 case 执行时会创建一条 WorkflowTrace，workflow_id 写入详情中，
        供事后 diagnose() 定位到具体 turn/tool。
        """
        if not self.cases:
            raise ValueError("No cases in suite")
        passed = 0
        failed = 0
        details = []
        for case in self.cases:
            trace = WorkflowTrace.start(
                workflow_id=f"eval:{self.name}:{case.id}",
                workflow_name=self.name,
            )
            agent_trace = AgentTrace(agent_id="eval_agent")
            trace.node_traces["eval_agent"] = agent_trace

            actual = await agent_fn(case.input)

            agent_trace.success = True
            trace.finish()

            result = await case.evaluator.evaluate(case.expected, actual)
            detail = {
                "case_id": case.id, "input": case.input,
                "expected": case.expected, "actual": actual,
                "score": result.score, "passed": result.passed,
                "reason": result.reason,
                "workflow_id": trace.workflow_id,
            }
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
        可通过 workflow_id 定位到 WorkflowTrace，查看具体哪个 step/tool 出了问题。

        Returns:
            [{case_id, score, reason, workflow_id}, ...] 按 score 升序排列
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
                "workflow_id": d.get("workflow_id", "N/A"),
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
