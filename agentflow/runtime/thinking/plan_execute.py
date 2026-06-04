"""Plan-Execute 模式：先制定计划，再逐步执行，支持工具调用。"""

from __future__ import annotations

import json
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class PlanExecuteStrategy(ThinkingStrategy):
    """Plan-Execute 策略。

    Phase 1 — Plan：生成结构化步骤列表
    Phase 2 — Execute：逐步骤执行（LLM 可在此期间调用工具）
    Phase 3 — Finalize：汇总结果（LLM 可在此期间调用工具）

    每阶段内部有 ReAct 式 tool 执行循环：
    如果 LLM 回复中有 tool_calls，执行工具并把结果喂回 LLM，
    直到 LLM 给出纯文本回复（无 tool_calls），该阶段结束。
    """

    def __init__(self, toolkit=None):
        self.toolkit = toolkit

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []
        all_tool_calls = []
        tools_param = context.tools if context.tools else None

        # Phase 1: Plan
        plan_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                "Break this down into clear steps. Output as a numbered plan."
            )},
        ]
        plan_text, plan_tool_calls = await self._execute_with_tools(
            context, plan_messages, tools_param
        )
        steps.append({"phase": "plan", "output": plan_text})
        all_tool_calls.extend(plan_tool_calls)

        # Phase 2: Execute
        execute_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Plan:\n{plan_text}\n\n"
                "Execute the plan step by step. For each step, describe what you did "
                "and the result. You can call tools if needed."
            )},
        ]
        execute_text, execute_tool_calls = await self._execute_with_tools(
            context, execute_messages, tools_param
        )
        steps.append({"phase": "execute", "output": execute_text})
        all_tool_calls.extend(execute_tool_calls)

        # Phase 3: Finalize
        finalize_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Execution results:\n{execute_text}\n\n"
                "Finalize the result. Call any necessary tools to save or export "
                "the final output."
            )},
        ]
        final_text, final_tool_calls = await self._execute_with_tools(
            context, finalize_messages, tools_param
        )
        steps.append({"phase": "finalize", "output": final_text})
        all_tool_calls.extend(final_tool_calls)

        return ThinkResult(
            output=final_text,
            tool_calls=all_tool_calls,
            steps=steps,
            mode_used="plan_execute",
        )

    async def _execute_with_tools(
        self, context: ThinkContext, messages: list, tools_param: list | None
    ) -> tuple[str, list[dict]]:
        """带 tool 执行循环的单次对话。

        循环执行：LLM 回复 → 如果有 tool_calls → 执行工具 → 结果喂回 → 继续
        直到 LLM 给出无 tool_calls 的纯文本回复，或达到 max_iterations。

        Returns:
            (final_text, all_tool_calls_made)
        """
        tool_calls_made = []

        for _ in range(context.max_iterations):
            response = await context.llm_client.chat(messages, tools=tools_param)

            if response.tool_calls:
                # 记录 assistant 的 tool_calls
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
                # 无 tool_calls → 该阶段结束
                return response.content, tool_calls_made

        # max_iterations 耗尽
        return messages[-1].get("content", ""), tool_calls_made
