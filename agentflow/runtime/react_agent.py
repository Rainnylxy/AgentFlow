"""ReAct Agent：Thought → Action → Observation 循环

⚠️ DEPRECATED: 此模块已废弃，请使用 AgentBuilder + ThinkingMode.REACT 替代。
旧用法:
    agent = ReActAgent(name=..., llm_client=..., system_prompt=..., ...)
新用法:
    agent = (AgentBuilder(name)
        .with_llm(llm_client)
        .with_prompt(system_prompt)
        .with_thinking(ThinkingMode.REACT)
        .build())
"""

import json
import warnings
from agentflow.runtime.agent import BaseAgent, AgentResult
from agentflow.runtime.memory import Message


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) Agent — 已废弃。

    ⚠️ DEPRECATED: Use AgentBuilder + ThinkingMode.REACT instead.

    执行循环：
    1. Thought: LLM 生成推理（通常需要工具调用）
    2. Action: 执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案，或达到 max_iterations
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ReActAgent is deprecated. Use AgentBuilder + ThinkingMode.REACT instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    async def run(self, user_input: str, stream=None, agent_trace=None) -> AgentResult:
        self.memory.short_term.clear()
        self.memory.short_term.add(Message(role="user", content=user_input))

        steps = []
        tool_calls_made = []

        for i in range(self.max_iterations):
            # 构建消息列表（包括 tool_call_id）
            messages = [{"role": "system", "content": self.system_prompt}]
            for m in self.memory.short_term.get_context_window():
                msg_dict = {"role": m.role, "content": m.content}
                if m.tool_call_id:
                    msg_dict["tool_call_id"] = m.tool_call_id
                if m.tool_calls:
                    msg_dict["tool_calls"] = m.tool_calls
                messages.append(msg_dict)

            # 构建可用工具列表
            tool_list = self.tool_registry.list_tools()
            tools = None
            if tool_list:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    }
                    for t in tool_list
                ]

            response = await self.llm_client.chat(messages, tools=tools, max_tokens=self.max_output_tokens)

            # 如果有 tool_call，执行工具
            if response.tool_calls:
                # 先将 assistant 的 tool_calls 消息加入历史（API 要求：tool 消息前必须有 assistant 的 tool_calls）
                self.memory.short_term.add(Message(
                    role="assistant",
                    content="",
                    tool_calls=response.tool_calls,
                ))

                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])
                    result = await self.tool_registry.execute(func_name, func_args)

                    tool_calls_made.append({
                        "tool": func_name,
                        "input": func_args,
                        "output": result.output,
                    })

                    # 将工具结果注入对话历史（必须带 tool_call_id）
                    self.memory.short_term.add(Message(
                        role="tool",
                        content=result.output or "",
                        tool_call_id=tc.get("id", ""),
                    ))

                steps.append({
                    "iteration": i,
                    "type": "tool_call",
                    "calls": [tc["function"]["name"] for tc in response.tool_calls],
                })
            else:
                # 最终回答
                self.memory.short_term.add(Message(
                    role="assistant", content=response.content
                ))
                steps.append({
                    "iteration": i,
                    "type": "final",
                    "output": response.content,
                })
                return AgentResult(
                    output=response.content,
                    tool_calls=tool_calls_made,
                    steps=steps,
                )

        return AgentResult(
            output="Agent reached maximum iterations without a final answer.",
            tool_calls=tool_calls_made,
            steps=steps,
        )
