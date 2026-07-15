"""ReAct 模式：Thought → Action → Observation 循环。

KV Cache 优化：system（ReAct 引导 + 用户 prompt）+ references 构成稳定前缀，
工作记忆中的已有消息追加在尾部。tool_loop 内消息增长方向与缓存方向一致。
"""

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


# ReAct 模式的思考引导——注入到 system_prompt 前面
REACT_SYSTEM_PROMPT = """## 推理与行动 (ReAct) 模式

你需要按照以下模式思考和行动：

1. **Thought（推理）**: 分析当前状态，判断是否需要更多信息。
2. **Action（行动）**: 如果需要信息，调用相应的工具。工具返回结果后继续推理。
3. **Observation（观察）**: 根据工具返回的结果更新你的理解。
4. 重复 Thought → Action → Observation 循环。
5. 当信息足够时，直接给出 **最终答案**，不要再调用工具。

重要规则：
- 每次只调用需要的工具，不要一次调用多个，除非它们互不依赖。
- 工具调用失败时，尝试其他方法，不要重复相同的失败调用。
- 确认信息充足后立即给出答案，不要无效循环。
"""


class ReActStrategy(ThinkingStrategy):
    """ReAct (Reasoning + Acting) 策略。

    自动注入 ReAct 思考引导到 system prompt。
    执行循环：
    1. Thought: LLM 生成推理（可能包含工具调用）
    2. Action: 执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案，或达到 max_iterations

    KV Cache：ReAct 引导 + system_prompt + references 构成前缀缓存区，
    tool_loop 内的消息追加方向自然形成缓存命中。
    """

    async def run(self, context: ThinkContext) -> ThinkResult:
        # 前缀区：ReAct 引导 + 用户 system_prompt → references → 全部命中缓存
        system_content = REACT_SYSTEM_PROMPT + "\n\n" + context.system_prompt
        messages = [{"role": "system", "content": system_content}]

        # Reference 参考卡（pinned，永不裁剪，构成缓存前缀）
        for ref_msg in context.reference_messages:
            messages.append(dict(ref_msg))

        # 变量区：工作记忆中的已有消息历史（追加在缓存前缀之后）
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
