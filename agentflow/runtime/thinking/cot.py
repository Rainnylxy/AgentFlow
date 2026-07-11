"""Chain-of-Thought 模式：深度推理 → 最终答案，支持工具调用。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class CoTStrategy(ThinkingStrategy):
    """Chain-of-Thought (CoT) 策略。

    Phase 1 — Think：深度推理（可调用工具）
    Phase 2 — Answer：基于推理给出最终答案（可调用工具）

    每阶段内部通过基类 _execute_tool_loop() 执行工具循环。
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []
        all_tool_calls = []
        tools_param = context.tools if context.tools else None

        # Phase 1: Deep thinking
        think_messages = [{"role": "system", "content": context.system_prompt}]
        for ref_msg in context.reference_messages:
            think_messages.append(dict(ref_msg))
        think_messages.append({"role": "user", "content": (
            f"Question: {context.user_input}\n\n"
            "Think through this step by step. Consider all angles, "
            "break down the problem, and reason carefully before arriving at a conclusion."
        )})
        think_text, think_tool_calls = await self._execute_tool_loop(
            context, think_messages, tools_param
        )
        steps.append({"phase": "think", "output": think_text})
        all_tool_calls.extend(think_tool_calls)

        # Phase 2: Final answer
        answer_messages = [{"role": "system", "content": context.system_prompt}]
        for ref_msg in context.reference_messages:
            answer_messages.append(dict(ref_msg))
        answer_messages.extend([
            {"role": "user", "content": think_text},
            {"role": "user", "content": (
                "Based on your reasoning above, give the final answer. "
                "Call any necessary tools to finalize."
            )},
        ])
        answer_text, answer_tool_calls = await self._execute_tool_loop(
            context, answer_messages, tools_param
        )
        steps.append({"phase": "answer", "output": answer_text})
        all_tool_calls.extend(answer_tool_calls)

        return ThinkResult(
            output=answer_text,
            tool_calls=all_tool_calls,
            steps=steps,
            mode_used="cot",
        )
