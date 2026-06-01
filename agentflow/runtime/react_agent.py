"""ReAct Agent：Thought → Action → Observation 循环"""

import json
from agentflow.runtime.agent import BaseAgent, AgentResult
from agentflow.runtime.memory import Message


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) Agent。

    执行循环：
    1. Thought: LLM 生成推理（通常需要工具调用）
    2. Action: 执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案，或达到 max_iterations
    """

    async def run(self, user_input: str) -> AgentResult:
        self.memory.short_term.clear()
        self.memory.short_term.add(Message(role="user", content=user_input))

        steps = []
        tool_calls_made = []

        for i in range(self.max_iterations):
            # 构建消息列表
            messages = [{"role": "system", "content": self.system_prompt}]
            messages += [
                {"role": m.role, "content": m.content}
                for m in self.memory.short_term.get_context_window()
            ]

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

            response = await self.llm_client.chat(messages, tools=tools)

            # 如果有 tool_call，执行工具
            if response.tool_calls:
                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])
                    result = self.tool_registry.execute(func_name, func_args)

                    tool_calls_made.append({
                        "tool": func_name,
                        "input": func_args,
                        "output": result.output,
                    })

                    # 将工具结果注入对话历史
                    self.memory.short_term.add(Message(
                        role="tool",
                        content=f"[{func_name}]: {result.output}",
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
