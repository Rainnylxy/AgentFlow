"""Thinking Engine 核心抽象。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ThinkContext:
    """思考上下文：Agent 所需的所有运行时信息。"""
    user_input: str
    system_prompt: str
    messages: list
    tools: list[dict]
    llm_client: object
    memory: object
    max_iterations: int = 10
    feedback: list[str] = field(default_factory=list)

    def add_feedback(self, suggestions: list[str]) -> None:
        self.feedback.extend(suggestions)


@dataclass
class ThinkResult:
    """思考结果。"""
    output: str
    tool_calls: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    reflection_notes: list = field(default_factory=list)
    mode_used: str = "unknown"


class ThinkingStrategy(ABC):
    """思考策略的抽象基类。所有模式实现此接口。"""

    @abstractmethod
    async def run(self, context: ThinkContext) -> ThinkResult:
        ...
