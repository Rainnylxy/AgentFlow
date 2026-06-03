"""Chain-of-Thought 模式：深度推理 → 最终答案。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class CoTStrategy(ThinkingStrategy):
    """Chain-of-Thought (CoT) 策略。

    Phase 1 — Think：深度推理，不调工具
    Phase 2 — Answer：基于推理给出最终答案
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []

        # Phase 1: Deep thinking
        think_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Question: {context.user_input}\n\n"
                "Think through this step by step. Consider all angles, "
                "break down the problem, and reason carefully before arriving at a conclusion."
            )},
        ]
        think_response = await context.llm_client.chat(think_messages)
        steps.append({"phase": "think", "output": think_response.content})

        # Phase 2: Final answer
        answer_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": think_response.content},
            {"role": "user", "content": "Based on your reasoning above, give the final answer."},
        ]
        answer_response = await context.llm_client.chat(answer_messages)
        steps.append({"phase": "answer", "output": answer_response.content})

        return ThinkResult(
            output=answer_response.content,
            steps=steps,
            mode_used="cot",
        )
