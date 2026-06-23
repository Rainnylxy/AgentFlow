"""Semantic Similarity Evaluator"""

from agentflow.eval.base import BaseEvaluator, EvalResult

try:
    from sentence_transformers import SentenceTransformer, util
    _MODEL = None

    def _get_model():
        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL
    HAS_ST = True
except Exception:
    HAS_ST = False


class SemanticEvaluator(BaseEvaluator):
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    async def evaluate(self, expected: str, actual: str) -> EvalResult:
        if not HAS_ST:
            return EvalResult(score=0.0, passed=False, reason="sentence-transformers not installed")
        model = _get_model()
        e1 = model.encode(expected, convert_to_tensor=True)
        e2 = model.encode(actual, convert_to_tensor=True)
        score = float(util.cos_sim(e1, e2).item())
        return EvalResult(score=score, passed=score >= self.threshold,
                          reason=f"Semantic similarity: {score:.3f}")
