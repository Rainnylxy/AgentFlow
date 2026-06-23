"""Plan Quality Evaluator —— Agent 的分步计划是否合理"""

from agentflow.eval.base import BaseEvaluator, EvalResult


class PlanQualityEvaluator(BaseEvaluator):
    """评估 Agent 的规划质量。

    维度：
    - 计划完整性：是否覆盖了所有必要的子任务
    - 步骤效率：实际步数 vs 最优步数
    - 依赖正确性：步骤顺序是否符合逻辑依赖
    """

    def __init__(self, expected_min_steps: int = 1, expected_max_steps: int = 10, required_tools: list = None):
        self.expected_min_steps = expected_min_steps
        self.expected_max_steps = expected_max_steps
        self.required_tools = required_tools or []

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_plan(trajectory, task_complexity) for structured scoring")

    def evaluate_plan(self, trajectory: dict, task_complexity: str = "medium") -> EvalResult:
        """评估 Agent 的执行计划。

        Args:
            trajectory: {steps: [...], tool_calls: [...]}
            task_complexity: "simple" | "medium" | "complex"
        """
        steps = trajectory.get("steps", [])
        tool_calls = trajectory.get("tool_calls", [])

        if not steps:
            return EvalResult(score=0.0, passed=False, reason="Empty plan")

        complexity_factors = {"simple": 0.7, "medium": 1.0, "complex": 1.5}
        factor = complexity_factors.get(task_complexity, 1.0)

        # 1. 步骤效率（实际步数 vs 期望步数）
        actual_steps = len(steps)
        if actual_steps < self.expected_min_steps:
            step_score = 0.5  # 太少了，可能跳过了必要步骤
        elif actual_steps <= self.expected_max_steps * factor:
            step_score = 1.0
        else:
            step_score = max(0.0, 1.0 - (actual_steps - self.expected_max_steps) / 10.0)

        # 2. 工具覆盖度（是否用了所有必要的工具）
        actual_tools = {tc.get("tool", "") for tc in tool_calls}
        required_set = set(self.required_tools)
        if required_set:
            coverage = len(required_set & actual_tools) / len(required_set)
        else:
            coverage = 1.0

        # 3. 是否有最终答案
        has_final = any(s.get("type") == "final" for s in steps)

        score = 0.4 * step_score + 0.4 * coverage + (0.2 if has_final else 0)
        reasons = []
        if step_score < 1.0:
            reasons.append(f"Steps: {actual_steps} (expected {self.expected_min_steps}-{self.expected_max_steps})")
        if coverage < 1.0:
            missing = required_set - actual_tools
            reasons.append(f"Missing tools: {missing}")
        if not has_final:
            reasons.append("No final answer")

        return EvalResult(
            score=score,
            passed=score >= 0.7,
            reason="; ".join(reasons) if reasons else "Plan is well-structured",
        )
