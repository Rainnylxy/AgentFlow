"""Eval Suite Runner：批量执行 EvalCase，生成对比报告"""

from dataclasses import dataclass, field
from typing import Callable
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
    def __init__(self, name: str, cases: list[EvalCase]):
        self.name = name
        self.cases = cases

    def run(self, agent_fn: Callable[[str], str]) -> SuiteReport:
        if not self.cases:
            raise ValueError("No cases in suite")
        passed = 0
        failed = 0
        details = []
        for case in self.cases:
            actual = agent_fn(case.input)
            result = case.evaluator.evaluate(case.expected, actual)
            details.append({
                "case_id": case.id, "input": case.input,
                "expected": case.expected, "actual": actual,
                "score": result.score, "passed": result.passed,
                "reason": result.reason,
            })
            if result.passed:
                passed += 1
            else:
                failed += 1
        return SuiteReport(name=self.name, total=len(self.cases), passed=passed,
                           failed=failed, pass_rate=passed / len(self.cases), details=details)

    def compare(self, old: SuiteReport, new: SuiteReport) -> dict:
        old_map = {d["case_id"]: d for d in old.details}
        new_map = {d["case_id"]: d for d in new.details}
        improved = sum(1 for cid in old_map if new_map[cid]["score"] > old_map[cid]["score"])
        regressed = sum(1 for cid in old_map if new_map[cid]["score"] < old_map[cid]["score"])
        unchanged = len(old_map) - improved - regressed
        return {"total": len(old_map), "improved": improved, "regressed": regressed,
                "unchanged": unchanged, "old_pass_rate": old.pass_rate, "new_pass_rate": new.pass_rate}
