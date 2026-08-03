"""自适应路由：根据任务信号自动选择最优思考模式。

提供两种路由实现：
  AdaptiveRouter — 关键词匹配（默认，无 LLM 依赖）
  LLMAdaptiveRouter — LLM 驱动分类（需注入 llm_client）
"""

import json
import logging

from agentflow.runtime.thinking.base import ThinkingStrategy
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """Classify this user request into exactly one category.

Categories:
- multi_step: Requires multiple sequential steps or tools
- deep_reasoning: Requires analysis, calculation, proof, or explanation
- safe_critical: Involves deletion, deployment, financial operations, or code commits
- default: Simple Q&A, single tool call, or conversational

Available tools: {tools_summary}

Return ONLY valid JSON: {{"signal": "<category>", "confidence": <0.0-1.0>}}"""


class AdaptiveRouter:
    """关键词匹配路由（v1），无 LLM 依赖。"""

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
        return _signal_to_strategy(self._detect(user_input))


class LLMAdaptiveRouter:
    """LLM 驱动分类路由（v2），利用 LLM 判断任务类型。

    用法:
        router = LLMAdaptiveRouter(llm_client)
        strategy = await router.route(user_input, tools)
    """

    def __init__(self, llm_client):
        self._llm = llm_client

    async def route(self, user_input: str, tools: list) -> ThinkingStrategy:
        # 构建工具摘要
        tools_summary = "none"
        if tools:
            names = []
            for t in tools:
                name = t.get("function", {}).get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                desc = t.get("function", {}).get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
                if name:
                    names.append(f"{name}: {desc[:60]}" if desc else name)
            if names:
                tools_summary = "; ".join(names[:10])

        prompt = _CLASSIFICATION_PROMPT.format(tools_summary=tools_summary)

        try:
            response = await self._llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ], max_tokens=100)
            result = json.loads(response.content)
            signal = result.get("signal", "default")
            confidence = result.get("confidence", 0.0)
            logger.debug(
                "LLM routing: signal=%s confidence=%.2f input=%.60s",
                signal, confidence, user_input,
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("LLM routing failed, fallback to keyword: %s", e)
            # LLM 调用失败 → 回退到关键词匹配
            return AdaptiveRouter().route(user_input, tools)

        return _signal_to_strategy({signal})


def _signal_to_strategy(signals: set[str]) -> ThinkingStrategy:
    """将信号集合映射到思考策略。"""
    if "safe_critical" in signals:
        return ReflectionWrapper(PlanExecuteStrategy(), max_reflections=3)
    if "multi_step" in signals:
        return PlanExecuteStrategy()
    if "deep_reasoning" in signals:
        return CoTStrategy()
    return ReActStrategy()
