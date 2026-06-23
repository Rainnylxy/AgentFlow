"""ReAct 模式：Thought → Action → Observation 循环。"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class ReActStrategy(ThinkingStrategy):
    """ReAct (Reasoning + Acting) 策略。

    执行循环：
    1. Thought: LLM 生成推理（可能包含工具调用）
    2. Action: 执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案，或达到 max_iterations
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        messages = [{"role": "system", "content": context.system_prompt}]

        # 注入已有消息历史
        for msg in context.messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            messages.append(msg_dict)

        tools_param = context.tools if context.tools else None

        # 使用基类的共享工具执行循环
        final_text, tool_calls_made = await self._execute_tool_loop(
            context, messages, tools_param
        )

        # 如果最后一条消息是 tool 角色 → LLM 被截断，未给出最终答案
        if messages and messages[-1].get("role") == "tool":
            return ThinkResult(
                output="Agent reached maximum iterations without a final answer.",
                tool_calls=tool_calls_made,
                steps=[{"type": "truncated", "iterations": context.max_iterations}],
                mode_used="react",
            )

        return ThinkResult(
            output=final_text,
            tool_calls=tool_calls_made,
            steps=[{"type": "final", "output": final_text}],
            mode_used="react",
        )
