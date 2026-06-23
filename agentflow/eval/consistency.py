"""Consistency Evaluator —— Agent 对同一问题的回答是否稳定"""

import asyncio
from agentflow.eval.base import BaseEvaluator, EvalResult


class ConsistencyEvaluator(BaseEvaluator):
    """评估 Agent 在多次运行中输出的一致性和方差。

    高方差 = Agent 的输出不稳定 = 不可用于生产环境。

    使用方法：
    1. 给定一个 agent_fn 和一个输入
    2. 调用 evaluate_consistency() 跑 N 次
    3. 输出：语义方差、工具调用一致性、得分方差
    """

    def __init__(self, runs: int = 3, max_deviation: float = 0.3):
        self.runs = runs
        self.max_deviation = max_deviation

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_consistency(agent_fn, input) for multi-run scoring")

    async def evaluate_consistency(self, agent_fn, user_input: str) -> EvalResult:
        """跑 N 次，分析输出一致性。

        Args:
            agent_fn: async callable，接受 str 输入，返回 AgentResult
            user_input: 用户输入
        """
        results = []
        for _ in range(self.runs):
            result = await agent_fn(user_input)
            results.append(result)

        if len(results) < 2:
            return EvalResult(score=1.0, passed=True, reason="Single run, consistency N/A")

        # 1. 工具调用一致性：所有运行中调用的工具集合是否一致
        tool_sets = [
            frozenset(tc.get("tool", "") for tc in r.tool_calls)
            for r in results
        ]
        tool_consistency = 1.0 if len(set(tool_sets)) == 1 else len(set(tool_sets)) / len(tool_sets)

        # 2. 输出长度方差
        output_lengths = [len(r.output) for r in results]
        avg_len = sum(output_lengths) / len(output_lengths)
        if avg_len > 0:
            length_variance = sum(abs(l - avg_len) / avg_len for l in output_lengths) / len(output_lengths)
        else:
            length_variance = 0.0

        # 3. 步骤数一致性
        step_counts = [len(r.steps) for r in results]
        step_consistency = 1.0 if len(set(step_counts)) == 1 else 1.0 / len(set(step_counts))

        # 综合得分
        score = 0.4 * tool_consistency + 0.3 * (1.0 - min(length_variance, 1.0)) + 0.3 * step_consistency
        deviation = 1.0 - score

        reasons = []
        if tool_consistency < 1.0:
            reasons.append(f"Tool sets differ across runs ({len(set(tool_sets))}/{self.runs} unique)")
        if length_variance > self.max_deviation:
            reasons.append(f"High output length variance ({length_variance:.2f})")
        if step_consistency < 1.0:
            reasons.append(f"Step counts vary ({set(step_counts)})")

        if not reasons:
            reasons.append("Consistent across all runs")

        return EvalResult(
            score=score,
            passed=deviation <= self.max_deviation,
            reason="; ".join(reasons),
        )
