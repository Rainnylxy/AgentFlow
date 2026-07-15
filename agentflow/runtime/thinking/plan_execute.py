"""Plan-Execute 模式：先制定计划，再逐步执行，每步独立 query，支持工具调用。

KV Cache 优化：跨步骤的不变内容（system_prompt + task + plan）放在消息前缀，
仅每步的 prev_results 和 step 指令放在 user 消息尾部，最大化前缀缓存命中。
"""

import re

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class PlanExecuteStrategy(ThinkingStrategy):
    """Plan-Execute 策略。

    Phase 1 — Plan：生成结构化步骤列表
    Phase 2 — Execute：每步独立 query，上一步结果喂给下一步
    Phase 3 — Finalize：汇总所有步骤结果

    每阶段内部通过基类 _execute_tool_loop() 执行工具循环。
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []
        all_tool_calls = []
        tools_param = context.tools if context.tools else None

        # Phase 1: Plan
        plan_messages = [{"role": "system", "content": context.system_prompt}]
        for ref_msg in context.reference_messages:
            plan_messages.append(dict(ref_msg))
        plan_messages.append({"role": "user", "content": (
            f"Task: {context.user_input}\n\n"
            "Break this down into clear, numbered steps. "
            "Output one step per line, format: 'N. <description>'"
        )})
        plan_text, plan_tool_calls = await self._execute_tool_loop(
            context, plan_messages, tools_param
        )
        steps.append({"phase": "plan", "output": plan_text})
        all_tool_calls.extend(plan_tool_calls)

        # Phase 2: Execute — each step as an independent query
        # 将 task + plan 冻结为 system 消息，放在前缀区以命中 KV Cache
        plan_steps = self._parse_steps(plan_text)
        step_results = []

        for i, step_desc in enumerate(plan_steps):
            prev_context = ""
            if step_results:
                prev_context = "Previous steps results:\n" + "\n".join(
                    f"Step {j+1}: {r}" for j, r in enumerate(step_results)
                ) + "\n\n"

            # 前缀区：system_prompt → frozen(task+plan) → references → 全部命中缓存
            step_messages = [
                {"role": "system", "content": context.system_prompt},
                {"role": "system", "content": (
                    f"Overall task: {context.user_input}\n\n"
                    f"Full plan:\n{plan_text}"
                )},
            ]
            for ref_msg in context.reference_messages:
                step_messages.append(dict(ref_msg))
            # 变量区：仅此一条随步骤变化
            step_messages.append({"role": "user", "content": (
                f"{prev_context}"
                f"Now execute Step {i+1}/{len(plan_steps)}: {step_desc}\n\n"
                "Execute this specific step. Call tools if needed. "
                "Report what you did and the result."
            )})
            step_text, step_tool_calls = await self._execute_tool_loop(
                context, step_messages, tools_param
            )
            step_results.append(step_text)
            steps.append({
                "phase": "execute", "step": i + 1,
                "description": step_desc, "output": step_text,
            })
            all_tool_calls.extend(step_tool_calls)

        # Phase 3: Finalize — 同样将 task 冻结到前缀区
        execute_summary = "\n".join(
            f"Step {j+1}: {r}" for j, r in enumerate(step_results)
        )
        finalize_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "system", "content": f"Task: {context.user_input}"},
        ]
        for ref_msg in context.reference_messages:
            finalize_messages.append(dict(ref_msg))
        finalize_messages.append({"role": "user", "content": (
            f"Execution results:\n{execute_summary}\n\n"
            "Synthesize the final result from the step results above."
        )})
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

    @staticmethod
    def _parse_steps(plan_text: str) -> list[str]:
        """Parse a numbered plan into individual step descriptions."""
        lines = plan_text.strip().split("\n")
        parsed = []
        pattern = re.compile(r"^\d+[\.\)]\s+")
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                step = pattern.sub("", line.strip()).strip()
                if step:
                    parsed.append(step)
        return parsed if parsed else [plan_text.strip()]
