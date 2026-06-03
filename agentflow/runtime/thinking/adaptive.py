"""自适应路由：根据任务信号自动选择最优思考模式。"""

from agentflow.runtime.thinking.base import ThinkingStrategy
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper


class AdaptiveRouter:
    """根据任务信号自动选择最优思考模式。

    信号检测基于关键词匹配（v1），后续可升级为 LLM 路由。
    """

    SIGNALS = {
        "multi_step": [
            "first", "then", "after", "step", "接下来", "然后", "之后",
            "1.", "2.", "3.",
        ],
        "deep_reasoning": [
            "why", "prove", "calculate", "analyze", "explain",
            "证明", "推导", "计算", "分析",
        ],
        "safe_critical": [
            "delete", "deploy", "drop", "charge",
            "删除", "部署", "提交代码", "commit",
        ],
    }

    def _detect(self, user_input: str) -> set[str]:
        text = user_input.lower()
        signals = set()
        for signal_type, keywords in self.SIGNALS.items():
            if any(kw in text for kw in keywords):
                signals.add(signal_type)
        return signals

    def route(self, user_input: str, tools: list) -> ThinkingStrategy:
        signals = self._detect(user_input)

        # 高风险 + 多步 → 规划 + 深度反思
        if "safe_critical" in signals:
            return ReflectionWrapper(PlanExecuteStrategy(), max_reflections=3)

        # 单纯高风险 → ReAct + 反思
        if "multi_step" in signals:
            return PlanExecuteStrategy()

        # 推理 → 链式思考
        if "deep_reasoning" in signals:
            return CoTStrategy()

        # 默认 → 快速 ReAct
        return ReActStrategy()
