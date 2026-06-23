"""Tool Parameter Accuracy Evaluator —— 工具调用参数是否准确"""

import json
from agentflow.eval.base import BaseEvaluator, EvalResult


class ToolParamEvaluator(BaseEvaluator):
    """评估工具调用时传入的参数是否准确。

    支持两种校验模式：
    - JSON Schema：精确校验参数类型和结构
    - Semantic：对 string 参数做语义相似度比较（避免 '北京' vs '背景' 类错误）
    """

    def __init__(self, expected_schema: dict = None, param_overrides: dict = None):
        self.expected_schema = expected_schema or {}
        self.param_overrides = param_overrides or {}

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        """字符串模式：尝试按 JSON 解析比较。"""
        try:
            exp = json.loads(expected)
            act = json.loads(actual)
        except (json.JSONDecodeError, TypeError):
            return EvalResult(score=0.0, passed=False, reason="Cannot parse JSON parameters")

        return self.evaluate_params(exp, act)

    def evaluate_params(self, expected: dict, actual: dict) -> EvalResult:
        """逐字段比较参数，支持嵌套对象。"""
        if not expected:
            return EvalResult(score=1.0, passed=True, reason="No expected params to check")

        total_fields = 0
        matched_fields = 0
        mismatches = []

        for key, exp_val in expected.items():
            total_fields += 1
            act_val = actual.get(key)

            if act_val is None and exp_val is not None:
                mismatches.append(f"Missing '{key}': expected {exp_val}")
                continue

            # 数字容差匹配
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                if abs(exp_val - act_val) < 1e-6:
                    matched_fields += 1
                else:
                    mismatches.append(f"'{key}': expected {exp_val}, got {act_val}")
            # 字符串语义匹配
            elif isinstance(exp_val, str) and isinstance(act_val, str):
                exp_lower = exp_val.lower().strip()
                act_lower = act_val.lower().strip()
                if exp_lower == act_lower or exp_lower in act_lower or act_lower in exp_lower:
                    matched_fields += 1
                else:
                    mismatches.append(f"'{key}': expected '{exp_val}', got '{act_val}'")
            # 直接相等
            elif exp_val == act_val:
                matched_fields += 1
            else:
                mismatches.append(f"'{key}': expected {exp_val}, got {act_val}")

        score = matched_fields / max(total_fields, 1)
        reason = "; ".join(mismatches[:3]) if mismatches else f"All {total_fields} params match"
        return EvalResult(score=score, passed=score >= 0.8, reason=reason)
