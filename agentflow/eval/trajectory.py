"""Trajectory Scoring Evaluator"""

from agentflow.eval.base import BaseEvaluator, EvalResult


class TrajectoryEvaluator(BaseEvaluator):
    """评估 Agent 推理轨迹的质量。

    评分维度：
    - 是否有思考步骤（Thought）
    - 工具调用是否冗余（重复调用相同工具）
    - 是否有最终答案
    """

    def __init__(self, max_redundant_calls: int = 3):
        self.max_redundant_calls = max_redundant_calls

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_quality(trajectory) for structured scoring")

    def evaluate_quality(self, trajectory: dict) -> EvalResult:
        steps = trajectory.get("steps", [])
        if not steps:
            return EvalResult(score=0.0, passed=False, reason="Empty trajectory")

        has_thought = any(s.get("type") == "thought" for s in steps)
        has_final = any(s.get("type") == "final" for s in steps)
        tool_calls = [s for s in steps if s.get("type") == "tool_call"]
        unique_tools = len(set(s.get("tool", "") for s in tool_calls))
        redundancy = max(0, len(tool_calls) - unique_tools) / max(1, len(tool_calls))

        score = 0.5 + (0.2 if has_thought else 0) + (0.2 if has_final else 0) - redundancy * 0.5
        score = max(0.0, min(1.0, score))

        return EvalResult(score=score, passed=score >= 0.5,
                          reason=f"Thought:{has_thought} Final:{has_final} Redundancy:{redundancy:.2f}")
