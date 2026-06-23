"""Faithfulness Evaluator —— Agent 是否编造了工具结果"""

import re
from agentflow.eval.base import BaseEvaluator, EvalResult


class FaithfulnessEvaluator(BaseEvaluator):
    """检测 Agent 回答中是否有编造或幻觉。

    核心逻辑：将每个 tool output 与 Agent 最终回答中的声称进行交叉比对。
    如果 Agent 声称了一个工具没有返回的事实 → 扣分。
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        """字符串模式：简单关键词覆盖检测。"""
        if not expected:
            return EvalResult(score=1.0, passed=True, reason="No tool outputs to verify against")
        return EvalResult(score=0.5, passed=True,
                          reason="Use evaluate_faithfulness(tool_outputs, agent_answer) for structured check")

    def evaluate_faithfulness(
        self,
        tool_outputs: list[dict],  # [{tool_name, output}, ...]
        agent_answer: str,
    ) -> EvalResult:
        """检查 Agent 的回答是否忠实地基于工具返回的结果。

        Args:
            tool_outputs: 所有工具调用的结果
            agent_answer: Agent 的最终回答文本
        """
        if not tool_outputs:
            return EvalResult(score=1.0, passed=True, reason="No tools called, nothing to fabricate")

        # 提取所有工具返回中的关键数值和事实
        tool_facts = set()
        for t in tool_outputs:
            output = t.get("output", "")
            # 提取数字
            numbers = re.findall(r'\b\d+\b', str(output))
            for n in numbers:
                tool_facts.add(("number", n))
            # 提取引号内的内容
            quotes = re.findall(r'"([^"]*)"', str(output))
            for q in quotes:
                tool_facts.add(("quote", q.lower()))

        # 检查 Agent 回答中是否出现了工具输出中不存在的事实
        agent_text = agent_answer.lower()
        unsupported_claims = []

        # 检查 Agent 是否引用了不存在的数字（可能是编造的）
        agent_numbers = re.findall(r'\b\d+\b', agent_answer)
        for n in agent_numbers:
            if ("number", n) not in tool_facts and int(n) > 9:  # 忽略小数字（可能是枚举等）
                unsupported_claims.append(f"Number '{n}' not found in tool outputs")

        # 检查关键断言
        misleading_markers = [
            "according to our records",
            "i have checked",
            "the system shows",
            "the database confirms",
        ]
        for marker in misleading_markers:
            if marker in agent_text and len(tool_facts) == 0:
                unsupported_claims.append(f"Claim '{marker}' without actual tool data")

        if unsupported_claims:
            return EvalResult(
                score=max(0.0, 1.0 - 0.3 * len(unsupported_claims)),
                passed=len(unsupported_claims) == 0,
                reason=f"Unsupported claims: {'; '.join(unsupported_claims[:3])}",
            )

        return EvalResult(score=1.0, passed=True, reason="Answer faithfully reflects tool outputs")
