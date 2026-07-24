"""Thinking Engine — 多模式 Agent 思考系统。

支持 ReAct / Plan-Execute / CoT / Reflection 四种模式，
具备自适应路由能力。
"""

from enum import Enum
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper
from agentflow.runtime.thinking.adaptive import AdaptiveRouter
from agentflow.runtime.thinking.routing import RoutingStrategy


class ThinkingMode(str, Enum):
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    COT = "cot"
    ADAPTIVE = "adaptive"
    ROUTING = "routing"

    def with_reflection(self, depth: int = 3) -> "ThinkingMode":
        """链式调用：给当前模式包裹反思层。"""
        object.__setattr__(self, '_reflection_depth', depth)
        return self


class ThinkingEngine:
    """管理多个思考策略，根据模式选择或自适应路由。

    用法:
        engine = ThinkingEngine(mode=ThinkingMode.ADAPTIVE, toolkit=my_toolkit)
        result = await engine.run(context)
    """

    def __init__(self, mode: ThinkingMode = ThinkingMode.ADAPTIVE, toolkit=None, registry=None, experts=None):
        self.mode = mode
        self.toolkit = toolkit
        self.registry = registry
        self.experts = experts or {}
        self._reflection_depth = getattr(mode, '_reflection_depth', 0)

    def _build_strategy(self, base: ThinkingStrategy) -> ThinkingStrategy:
        if self._reflection_depth > 0:
            return ReflectionWrapper(base, max_reflections=self._reflection_depth)
        return base

    def resolve_strategy(self, user_input: str, tools: list) -> ThinkingStrategy:
        if self.mode == ThinkingMode.ROUTING:
            if self.registry is None:
                raise ValueError("ThinkingMode.ROUTING requires a registry.")
            return RoutingStrategy(
                registry=self.registry,
                experts=self.experts,
                toolkit=self.toolkit,
            )

        if self.mode == ThinkingMode.ADAPTIVE:
            return AdaptiveRouter().route(user_input, tools)

        mapping = {
            ThinkingMode.REACT: ReActStrategy(toolkit=self.toolkit),
            ThinkingMode.PLAN_EXECUTE: PlanExecuteStrategy(toolkit=self.toolkit),
            ThinkingMode.COT: CoTStrategy(toolkit=self.toolkit),
        }
        base = mapping.get(self.mode, ReActStrategy(toolkit=self.toolkit))
        return self._build_strategy(base)

    async def run(self, context: ThinkContext) -> ThinkResult:
        strategy = self.resolve_strategy(context.user_input, context.tools)
        result = await strategy.run(context)
        return result


__all__ = [
    "ThinkingMode", "ThinkingEngine",
    "ThinkingStrategy", "ThinkContext", "ThinkResult",
    "ReActStrategy", "PlanExecuteStrategy", "CoTStrategy",
    "ReflectionWrapper", "AdaptiveRouter",
    "RoutingStrategy",
]
