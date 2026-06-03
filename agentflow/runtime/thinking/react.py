"""ReAct 模式：Thought → Action → Observation 循环。"""

import json
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class ReActStrategy(ThinkingStrategy):
    """ReAct (Reasoning + Acting) 策略。

    执行循环：
    1. Thought: LLM 生成推理（可能包含工具调用）
    2. Action: 执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案，或达到 max_iterations
    """

    def __init__(self, toolkit=None):
        self.toolkit = toolkit

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

        steps = []
        tool_calls_made = []

        for i in range(context.max_iterations):
            tools_param = context.tools if context.tools else None
            response = await context.llm_client.chat(messages, tools=tools_param)

            if response.tool_calls:
                # 记录 assistant 的 tool_calls 到消息历史
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                })

                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])

                    # 通过 toolkit 执行（如果有的话）
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

                steps.append({
                    "iteration": i,
                    "type": "tool_call",
                    "calls": [tc["function"]["name"] for tc in response.tool_calls],
                })
            else:
                # 最终回答
                messages.append({"role": "assistant", "content": response.content})
                steps.append({
                    "iteration": i,
                    "type": "final",
                    "output": response.content,
                })
                return ThinkResult(
                    output=response.content,
                    tool_calls=tool_calls_made,
                    steps=steps,
                    mode_used="react",
                )

        return ThinkResult(
            output="Agent reached maximum iterations without a final answer.",
            tool_calls=tool_calls_made,
            steps=steps,
            mode_used="react",
        )
