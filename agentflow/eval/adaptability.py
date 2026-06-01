"""Adaptability Evaluator —— 工具失败时 Agent 能否自适应换策略"""

from agentflow.eval.base import BaseEvaluator, EvalResult


class AdaptabilityEvaluator(BaseEvaluator):
    """评估 Agent 在工具返回错误时的自适应能力。

    维度：
    - 策略切换：失败后是否尝试了不同的工具/参数
    - 重复避免：是否重复调用已失败的工具超过 2 次
    - 优雅降级：是否告知用户失败并给出替代方案
    """

    def __init__(self, failure_patterns: dict = None):
        self.failure_patterns = failure_patterns or {}

    def evaluate(self, expected: str, actual: str) -> EvalResult:
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_adaptability(trajectory, injected_failures) for structured scoring")

    def evaluate_adaptability(
        self,
        trajectory: dict,
        injected_failures: list[dict] = None,
    ) -> EvalResult:
        """评估 Agent 对注入故障的响应。

        Args:
            trajectory: {steps, tool_calls}
            injected_failures: [{tool_name, error_type}, ...]

        Returns:
            EvalResult with adaption score
        """
        failures = injected_failures or []
        steps = trajectory.get("steps", [])
        tool_calls = trajectory.get("tool_calls", [])

        if not failures:
            # 没有注入故障 → 检查是否有失败后的自适应行为
            failed_calls = [tc for tc in tool_calls if tc.get("output", "") and "error" in str(tc.get("output", "")).lower()]
            if not failed_calls:
                return EvalResult(score=1.0, passed=True, reason="No failures injected, no errors observed")

        # 1. 策略多样化：失败后是否用了不同的工具
        all_tool_names = [tc.get("tool", "") for tc in tool_calls]
        unique_tools = set(all_tool_names)
        strategy_diversity = min(1.0, len(unique_tools) / max(len(failures) + 1, 1))

        # 2. 重复检查：同一个失败工具被调用了多少次
        redundant_calls = 0
        tool_call_counts = {}
        for tc in tool_calls:
            name = tc.get("tool", "")
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        for name, count in tool_call_counts.items():
            if count > 2:
                redundant_calls += count - 2

        redundancy_penalty = min(0.5, redundant_calls * 0.1)

        # 3. 是否有最终答案（而不是卡死在工具循环里）
        has_final = any(s.get("type") == "final" for s in steps)
        has_recovery = (strategy_diversity > 0.5) and has_final

        # 4. 是否告知用户失败
        agent_output = ""
        for s in steps:
            if s.get("type") == "final":
                agent_output = s.get("output", "")
        informed_user = any(w in agent_output.lower()
                          for w in ["unable", "cannot", "unfortunately", "try", "alternative", "instead"])

        score = 0.35 * strategy_diversity + 0.25 * (1.0 if has_final else 0) + \
                0.25 * (1.0 if has_recovery else 0) + 0.15 * (1.0 if informed_user else 0) - redundancy_penalty
        score = max(0.0, min(1.0, score))

        reasons = []
        if strategy_diversity < 1.0:
            reasons.append(f"Low strategy diversity ({len(unique_tools)} unique tools)")
        if not has_final:
            reasons.append("No final answer after failures")
        if redundant_calls > 0:
            reasons.append(f"{redundant_calls} redundant retries")
        if not informed_user:
            reasons.append("Did not inform user of limitations")

        return EvalResult(
            score=score,
            passed=score >= 0.6,
            reason="; ".join(reasons) if reasons else "Agent adapted well to failures",
        )
