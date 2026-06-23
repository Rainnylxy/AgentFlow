"""Thinking Engine 核心抽象。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ThinkContext:
    """思考上下文：Agent 所需的所有运行时信息。"""
    user_input: str
    system_prompt: str
    messages: list
    tools: list[dict]
    llm_client: object
    memory: object
    max_iterations: int = 10
    feedback: list[str] = field(default_factory=list)

    def add_feedback(self, suggestions: list[str]) -> None:
        self.feedback.extend(suggestions)


@dataclass
class ThinkResult:
    """思考结果。"""
    output: str
    tool_calls: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    reflection_notes: list = field(default_factory=list)
    mode_used: str = "unknown"


class ThinkingStrategy(ABC):
    """思考策略的抽象基类。所有模式实现此接口。

    提供共享的 _execute_tool_loop() 方法——所有思考策略的
    工具执行循环逻辑完全一致，子类只需定义何时进入循环、
    如何构建消息上下文、以及循环退出后的处理。
    """

    def __init__(self, toolkit=None):
        self.toolkit = toolkit

    @abstractmethod
    async def run(self, context: ThinkContext) -> ThinkResult:
        ...

    async def _execute_tool_loop(
        self, context: ThinkContext, messages: list, tools_param: list | None
    ) -> "tuple[str, list[dict]]":
        """共享的工具执行循环：LLM 回复 → 有工具调用则执行 → 结果喂回 → 继续。

        所有思考策略（ReAct / CoT / PlanExecute）的核心循环逻辑：
        - 调用 LLM
        - 如果有 tool_calls：执行工具，将 assistant 消息和 tool 结果追加到 messages
        - 如果没有 tool_calls：返回最终文本
        - 达到 max_iterations 时截断

        Args:
            context: 思考上下文（含 llm_client、max_iterations）
            messages: 当前对话消息列表（原地修改）
            tools_param: OpenAI 格式的工具定义列表

        Returns:
            (final_text_response, list_of_tool_calls_made)
            每个 tool_call 格式为 {"tool": name, "input": dict, "output": str}
        """
        tool_calls_made = []

        for _ in range(context.max_iterations):
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

                    # 通过 toolkit 执行（如果有）
                    if self.toolkit:
                        result = await self.toolkit.execute(func_name, func_args)
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
            else:
                # 无 tool_calls → 最终响应，追加到消息历史
                messages.append({"role": "assistant", "content": response.content})
                return response.content, tool_calls_made

        # max_iterations 耗尽时的兜底返回（最后一条是 tool 消息）
        return messages[-1].get("content", ""), tool_calls_made
