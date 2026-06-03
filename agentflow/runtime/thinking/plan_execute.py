"""Plan-Execute 模式：先制定计划，再逐步执行。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class PlanExecuteStrategy(ThinkingStrategy):
    """Plan-Execute 策略。

    Phase 1 — Plan：生成结构化步骤列表
    Phase 2 — Execute：逐步骤执行
    Phase 3 — Finalize：汇总所有步骤结果
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []

        # Phase 1: Plan
        plan_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                "Break this down into clear steps. Output as a numbered plan."
            )},
        ]
        plan_response = await context.llm_client.chat(plan_messages)
        plan_text = plan_response.content
        steps.append({"phase": "plan", "output": plan_text})

        # Phase 2: Execute
        execute_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Plan:\n{plan_text}\n\n"
                "Execute the plan step by step. For each step, describe what you did and the result."
            )},
        ]
        execute_response = await context.llm_client.chat(execute_messages)
        steps.append({"phase": "execute", "output": execute_response.content})

        # Phase 3: Finalize
        finalize_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Execution results:\n{execute_response.content}\n\n"
                "Summarize the final answer concisely."
            )},
        ]
        final_response = await context.llm_client.chat(finalize_messages)
        steps.append({"phase": "finalize", "output": final_response.content})

        return ThinkResult(
            output=final_response.content,
            steps=steps,
            mode_used="plan_execute",
        )
