"""Reflection 反思装饰器：在任何策略外层包裹自我审视循环。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class ReflectionWrapper(ThinkingStrategy):
    """在任何策略外层包裹反思循环。

    每轮执行后，用 LLM 做自我审视：
    - 通过（PASS）→ 返回结果
    - 失败（FAIL）→ 注入反馈，重试
    """

    def __init__(self, inner: ThinkingStrategy, max_reflections: int = 3):
        self.inner = inner
        self.max_reflections = max_reflections

    async def run(self, context: ThinkContext) -> ThinkResult:
        all_notes = []

        for i in range(self.max_reflections):
            result = await self.inner.run(context)

            # Self-check
            review_messages = [
                {"role": "system", "content": "You are a quality reviewer. Review the response below."},
                {"role": "user", "content": (
                    f"Original task: {context.user_input}\n\n"
                    f"Agent response: {result.output}\n\n"
                    "Review the response. Answer ONLY with:\n"
                    "PASS if the response is correct and complete.\n"
                    "FAIL: <reason> if there is an issue that needs correction."
                )},
            ]
            review_response = await context.llm_client.chat(review_messages, max_tokens=context.max_output_tokens)
            review_text = review_response.content.strip()
            all_notes.append(review_text)

            if review_text.startswith("PASS"):
                result.reflection_notes = all_notes
                return result

            # FAIL — inject feedback for retry
            context.add_feedback([review_text])

        result.reflection_notes = all_notes
        return result
