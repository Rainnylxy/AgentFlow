"""LLM-as-Judge Evaluator"""

import json
from agentflow.eval.base import BaseEvaluator, EvalResult
from agentflow.runtime.llm_client import LLMClient


class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self, llm_client: LLMClient, rubric: str = "Evaluate accuracy and completeness."):
        self.llm_client = llm_client
        self.rubric = rubric

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        prompt = (
            f"Rubric: {self.rubric}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}\n"
            f'Return JSON: {{"score": float 0-1, "reason": "string"}}'
        )
        resp = await self.llm_client.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp.content)
            return EvalResult(score=float(data["score"]), passed=data["score"] >= 0.7,
                              reason=data.get("reason", ""))
        except (json.JSONDecodeError, KeyError):
            return EvalResult(score=0.0, passed=False, reason="Judge parse error")
