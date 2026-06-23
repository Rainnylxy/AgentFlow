"""Token Efficiency Evaluator —— Agent 花了多少 Token 完成任务"""

from agentflow.eval.base import BaseEvaluator, EvalResult


class TokenEfficiencyEvaluator(BaseEvaluator):
    """评估 Agent 的 Token 使用效率。

    维度：
    - 总 Token 消耗（输入 + 输出）
    - 每步平均 Token
    - 与 baseline 的比值（越低越好）
    """

    def __init__(self, baseline_tokens: int = 500, max_acceptable: int = 3000):
        self.baseline_tokens = baseline_tokens
        self.max_acceptable = max_acceptable

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_efficiency(usage_stats) for structured scoring")

    def evaluate_efficiency(self, usage_stats: dict) -> EvalResult:
        """根据使用统计计算效率分数。

        Args:
            usage_stats: {total_tokens, prompt_tokens, completion_tokens, steps}
        """
        total = usage_stats.get("total_tokens", 0)
        steps = max(usage_stats.get("steps", 1), 1)

        if total == 0:
            return EvalResult(score=0.0, passed=False, reason="No token data available")

        # 效率分 = baseline / actual（超过 baseline 则按比例扣分）
        efficiency_ratio = self.baseline_tokens / max(total, 1)

        # 超过最大可接受值 → 严重扣分
        if total > self.max_acceptable:
            return EvalResult(
                score=max(0.0, 1.0 - total / self.max_acceptable),
                passed=False,
                reason=f"Token waste: {total} tokens for {steps} steps (baseline: {self.baseline_tokens})"
            )

        score = min(1.0, efficiency_ratio)
        return EvalResult(
            score=score,
            passed=total <= self.baseline_tokens * 3,
            reason=f"{total} tokens / {steps} steps = {total//steps} tokens/step (ratio: {efficiency_ratio:.2f})"
        )
