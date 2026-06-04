"""Chain-of-Thought 模式：深度推理 → 最终答案，支持工具调用。"""

from __future__ import annotations

import json
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class CoTStrategy(ThinkingStrategy):
    """Chain-of-Thought (CoT) 策略。

    Phase 1 — Think：深度推理（可调用工具）
    Phase 2 — Answer：基于推理给出最终答案（可调用工具）

    每阶段内部有 ReAct 式 tool 执行循环。
    """

    def __init__(self, toolkit=None):
        self.toolkit = toolkit

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []
        all_tool_calls = []
        tools_param = context.tools if context.tools else None

        # Phase 1: Deep thinking
        think_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Question: {context.user_input}\n\n"
                "Think through this step by step. Consider all angles, "
                "break down the problem, and reason carefully before arriving at a conclusion."
            )},
        ]
        think_text, think_tool_calls = await self._execute_with_tools(
            context, think_messages, tools_param
        )
        steps.append({"phase": "think", "output": think_text})
        all_tool_calls.extend(think_tool_calls)

        # Phase 2: Final answer
        answer_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": think_text},
            {"role": "user", "content": (
                "Based on your reasoning above, give the final answer. "
                "Call any necessary tools to finalize."
            )},
        ]
        answer_text, answer_tool_calls = await self._execute_with_tools(
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

    async def _execute_with_tools(
        self, context: ThinkContext, messages: list, tools_param: list | None
    ) -> tuple[str, list[dict]]:
        """带 tool 执行循环的单次对话。

        Returns:
            (final_text, all_tool_calls_made)
        """
        tool_calls_made = []

        for _ in range(context.max_iterations):
            response = await context.llm_client.chat(messages, tools=tools_param)

            if response.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                })

                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])

                    if self.toolkit:
                        result = self.toolkit.execute(func_name, func_args)
                        tool_output = result.output if result.success else result.error
                    else:
                        tool_output = f"[No toolkit] Called {func_name}({func_args})"

                    tool_calls_made.append({
                        "tool": func_name,
                        "input": func_args,
                        "output": tool_output,
                    })

                    messages.append({
                        "role": "tool",
                        "content": tool_output,
                        "tool_call_id": tc.get("id", ""),
                    })
            else:
                return response.content, tool_calls_made

        return messages[-1].get("content", ""), tool_calls_made
