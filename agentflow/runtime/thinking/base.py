"""Thinking Engine 核心抽象。"""

from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from agentflow.runtime.hooks import StreamEvent
from agentflow.trace.tracer import AgentTrace, AgentTurn, ToolCallRecord


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
    stream: Optional[Callable[[StreamEvent], Awaitable[None]]] = None
    skill_tool_map: dict[str, str] = field(default_factory=dict)  # use_skill_xxx → skill_name
    agent_trace: Optional[AgentTrace] = None  # 思考引擎逐轮填充的追踪数据
    reference_messages: list[dict] = field(default_factory=list)  # Reference 参考卡消息，永不裁剪，策略需注入在 system prompt 之后
    _skills_map: dict = field(default_factory=dict)  # skill_name → Skill（内部用）

    def add_feedback(self, suggestions: list[str]) -> None:
        self.feedback.extend(suggestions)

    async def emit(self, event: StreamEvent) -> None:
        """发送流式事件。如果设置了 stream 回调则调用。"""
        if self.stream:
            await self.stream(event)


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

    async def _handle_skill_activation(
        self, skill_name: str, context: ThinkContext, messages: list
    ) -> str:
        """拦截 skill 激活调用——懒加载 Skill body 并注入到下一轮。"""
        # 从 memory 或 builder 传下的 skills 中查找
        # Skill 对象通过 context 传入
        skill = getattr(context, '_skills_map', {}).get(skill_name)
        if skill is None:
            return f"Skill '{skill_name}' not found."

        if skill._loaded:
            return f"Skill '{skill_name}' is already active."

        await skill.ensure_loaded()

        # 注入 Skill 内容到 system prompt 区域（不插在 tool_calls 和 tool 结果之间）
        skill_prompt = f"[Activated Skill: {skill.name}]\n{skill.prompt}"
        messages.insert(1, {
            "role": "system",
            "content": skill_prompt,
        })

        # 如果 skill 有 steps，注入步骤约束
        if skill.steps:
            steps_desc = "\n".join(
                f"{i+1}. {s.name}: {s.description} "
                f"(allowed tools: {s.allowed_tools or 'none'})"
                for i, s in enumerate(skill.steps)
            )
            messages.insert(2, {
                "role": "system",
                "content": f"[Skill Steps]\n{steps_desc}",
            })

        return f"Skill '{skill_name}' activated. Content loaded and injected into context."

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
        trace = context.agent_trace

        for i in range(context.max_iterations):
            t_turn_start = time.monotonic()

            # 每轮开始：创建 turn 并捕获完整 messages 快照（LLM 调用前的上下文）
            if trace:
                turn = AgentTurn(
                    turn=i + 1,
                    messages_snapshot=copy.deepcopy(messages),
                )
                trace.turns.append(turn)
                trace.total_turns = len(trace.turns)

            response = await context.llm_client.chat(messages, tools=tools_param)

            if response.tool_calls:
                # 回填本轮思考内容和 LLM 响应元信息
                if trace:
                    trace.turns[-1].thinking = response.content or f"Decided to call: {[tc['function']['name'] for tc in response.tool_calls]}"
                    trace.turns[-1].finish_reason = response.finish_reason
                    trace.turns[-1].tokens = dict(response.usage) if response.usage else {}
                # 流式：发送思考事件
                await context.emit(StreamEvent(
                    type="thinking",
                    content=response.content or f"Calling tools: {[tc['function']['name'] for tc in response.tool_calls]}",
                    data={"iteration": i, "tool_calls": [tc["function"]["name"] for tc in response.tool_calls]},
                ))

                # 记录 assistant 的 tool_calls 到消息历史
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                })

                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])

                    # 流式：发送工具调用事件
                    await context.emit(StreamEvent(
                        type="tool_call",
                        content=f"Calling {func_name}...",
                        data={"tool": func_name, "input": func_args},
                    ))

                    # 拦截 skill 激活调用
                    t_tool_start = time.monotonic()
                    if func_name in context.skill_tool_map:
                        skill_name = context.skill_tool_map[func_name]
                        tool_output = await self._handle_skill_activation(
                            skill_name, context, messages
                        )
                        tool_success = True
                    elif self.toolkit:
                        result = await self.toolkit.execute(func_name, func_args)
                        tool_output = result.output if result.success else result.error
                        tool_success = result.success
                    else:
                        tool_output = f"[No toolkit] Called {func_name}({func_args})"
                        tool_success = True

                    t_tool_dur = int((time.monotonic() - t_tool_start) * 1000)

                    # 记录工具调用 trace
                    if trace and trace.turns:
                        trace.turns[-1].tool_calls.append(ToolCallRecord(
                            tool=func_name,
                            input=func_args,
                            output=tool_output[:500],
                            success=tool_success,
                            duration_ms=t_tool_dur,
                        ))
                        trace.total_tool_calls += 1

                    # 流式：发送工具结果事件
                    await context.emit(StreamEvent(
                        type="tool_result",
                        content=tool_output[:200],
                        data={"tool": func_name, "output": tool_output, "success": True},
                    ))

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
                # 无 tool_calls → 最终响应
                messages.append({"role": "assistant", "content": response.content})
                # 回填最终回答 + LLM 响应元信息到当前 turn
                if trace:
                    trace.turns[-1].final_answer = response.content
                    trace.turns[-1].finish_reason = response.finish_reason
                    trace.turns[-1].tokens = dict(response.usage) if response.usage else {}
                # 填写汇总
                if trace:
                    # 聚合所有 turn 的 token
                    agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    for t in trace.turns:
                        for k in agg:
                            agg[k] += t.tokens.get(k, 0)
                    trace.total_tokens = agg
                    trace.success = True
                # 流式：发送最终答案事件
                await context.emit(StreamEvent(
                    type="final",
                    content=response.content[:200],
                    data={"iterations": i + 1},
                ))
                return response.content, tool_calls_made

        # max_iterations 耗尽时的兜底返回
        if trace:
            trace.success = False
            trace.error = "Reached max iterations without final answer"
        await context.emit(StreamEvent(
            type="error",
            content="Agent reached maximum iterations without a final answer.",
        ))
        return messages[-1].get("content", ""), tool_calls_made
