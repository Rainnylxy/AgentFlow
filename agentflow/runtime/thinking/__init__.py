"""Thinking Engine — 多模式 Agent 思考系统。

支持 ReAct / Plan-Execute / CoT / Reflection 四种模式，
具备自适应路由能力。
"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy

__all__ = [
    "ThinkingStrategy", "ThinkContext", "ThinkResult",
    "ReActStrategy", "PlanExecuteStrategy", "CoTStrategy",
]
