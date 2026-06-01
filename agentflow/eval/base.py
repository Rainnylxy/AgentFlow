"""Evaluator 基类"""

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class EvalResult:
    score: float      # 0.0 ~ 1.0
    passed: bool
    reason: str = ""


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, expected: str, actual: str) -> EvalResult:
        ...
