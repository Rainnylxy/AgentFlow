"""Exact Match Evaluator"""

import json
from agentflow.eval.base import BaseEvaluator, EvalResult


class ExactMatchEvaluator(BaseEvaluator):
    def __init__(self, case_sensitive: bool = True, normalize_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def evaluate(self, expected: str, actual: str) -> EvalResult:
        exp, act = expected, actual
        if self.normalize_whitespace:
            exp, act = " ".join(exp.split()), " ".join(act.split())
        if not self.case_sensitive:
            exp, act = exp.lower(), act.lower()
        try:
            if json.loads(exp) == json.loads(act):
                return EvalResult(score=1.0, passed=True, reason="JSON match")
        except (json.JSONDecodeError, TypeError):
            pass
        if exp == act:
            return EvalResult(score=1.0, passed=True, reason="Exact match")
        return EvalResult(score=0.0, passed=False, reason=f"Expected '{exp[:50]}', got '{act[:50]}'")
