"""Plan-Execute 模式：先制定计划，再逐步执行，支持工具调用。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class PlanExecuteStrategy(ThinkingStrategy):
    """Plan-Execute 策略。

    Phase 1 — Plan：生成结构化步骤列表
    Phase 2 — Execute：逐步骤执行（LLM 可在此期间调用工具）
    Phase 3 — Finalize：汇总结果（LLM 可在此期间调用工具）

    每阶段内部通过基类 _execute_tool_loop() 执行工具循环。
    """

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
        plan_text, plan_tool_calls = await self._execute_tool_loop(
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
        execute_text, execute_tool_calls = await self._execute_tool_loop(
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
        final_text, final_tool_calls = await self._execute_tool_loop(
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
