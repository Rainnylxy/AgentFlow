"""AgentBuilder — Agent 构建的统一入口。

用法:
    agent = (AgentBuilder("my-agent")
        .with_llm(llm_client)
        .with_tools(my_tool)
        .with_memory(MemoryProfile.standard())
        .with_prompt(PromptTemplate.preset("customer_support"))
        .with_thinking(ThinkingMode.ADAPTIVE)
        .build())
"""

from __future__ import annotations

import os
from pathlib import Path

from agentflow.runtime.agent import BaseAgent, AgentResult
from agentflow.runtime.thinking import ThinkingEngine, ThinkingMode, ThinkContext
from agentflow.runtime.toolkit import ToolKit, Tool
from agentflow.runtime.tool_registry import ToolType
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile
from agentflow.runtime.memory.reference import Reference
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.prompt.section import Section
from agentflow.runtime.skill import Skill, SkillLoader


class SkillSection(Section):
    """将 Skill 的 prompt 包装为 Prompt Section。"""
    name = "skill"
    order = 15  # 在 role_card (10) 之后，style_guide (20) 之前

    def __init__(self, skill: Skill):
        super().__init__()
        self.name = f"skill:{skill.name}"
        self._skill = skill

    def render(self, context: dict) -> str:
        return self._skill.to_system_prompt()


class AgentBuilder:
    """Agent 构建器——统一入口。

    链式 API，每个 with_* 方法返回 self。
    build() 组装所有子系统并返回可用的 Agent。

    用法:
        agent = (AgentBuilder("my-agent")
            .with_llm(llm)
            .with_skill("code-review")    # 从 skills/ 目录加载
            .with_tools(save_file)
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._llm_client = None
        self._toolkit = ToolKit()
        self._memory_profile = MemoryProfile.standard()
        self._prompt_template = None
        self._thinking_mode = ThinkingMode.ADAPTIVE
        self._max_iterations = 10
        self._max_output_tokens = None
        self._max_input_tokens = None
        self._system_prompt_str = None
        self._skills_dir = Path("skills")  # 默认 skills/ 目录
        self._skill_names: list[str] = []
        self._reference = Reference()  # 参考卡，跨 run 持久，永不裁剪

    def with_llm(self, llm_client) -> "AgentBuilder":
        self._llm_client = llm_client
        return self

    def with_tools(self, *tools_or_kit) -> "AgentBuilder":
        """接收 @tool 函数、Tool 实例或 ToolKit 实例。"""
        for item in tools_or_kit:
            if isinstance(item, ToolKit):
                for t in item.list():
                    self._toolkit.add(t)
            elif isinstance(item, Tool):
                self._toolkit.add(item)
            else:
                # @tool 装饰的函数
                self._toolkit.add(item)
        return self

    def with_memory(self, profile: MemoryProfile) -> "AgentBuilder":
        self._memory_profile = profile
        return self

    def with_prompt(self, prompt) -> "AgentBuilder":
        """接收 PromptTemplate 或纯字符串（向后兼容）。"""
        if isinstance(prompt, str):
            self._system_prompt_str = prompt
        elif isinstance(prompt, PromptTemplate):
            self._prompt_template = prompt
        return self

    def with_thinking(self, mode: ThinkingMode) -> "AgentBuilder":
        self._thinking_mode = mode
        return self

    def with_skill(self, skill_name: str) -> "AgentBuilder":
        """加载一个 Skill（从 skills/ 目录下的 .md 文件）。

        Skill 的 prompt 会注入到 system prompt 中，
        Skill 声明的 tools 需要在 with_tools() 中注册过。
        """
        self._skill_names.append(skill_name)
        return self

    def with_skills_dir(self, directory: "str | Path") -> "AgentBuilder":
        """设置 Skill 文件目录，默认为 skills/。"""
        self._skills_dir = Path(directory)
        return self

    def with_reference(self, key: str, content: str) -> "AgentBuilder":
        """添加一条参考卡（Reference）。

        Reference 跨 agent.run() 持久化，永不参与滑动窗口裁剪。
        适用于角色设定、世界观、项目约定等长期上下文。

        Args:
            key: 参考条目标识（如 "characters", "style", "summary"）
            content: 参考内容文本
        """
        self._reference.set(key, content)
        return self

    def with_max_iterations(self, n: int) -> "AgentBuilder":
        self._max_iterations = n
        return self

    def with_max_output_tokens(self, n: int) -> "AgentBuilder":
        self._max_output_tokens = n
        return self

    def with_max_input_tokens(self, n: int) -> "AgentBuilder":
        """设置输入上下文窗口的 token 上限（WorkingMemory 截断阈值）。"""
        self._max_input_tokens = n
        return self

    def build_sync(self) -> BaseAgent:
        """同步版 build()，向后兼容。"""
        import asyncio
        return asyncio.run(self.build())

    async def build(self) -> BaseAgent:
        if self._llm_client is None:
            raise ValueError("with_llm() is required. Provide an LLM client.")

        # 加载 Skills（懒加载：只读 name/description/tools，不读 body，不调 LLM）
        loader = SkillLoader(skills_dir=self._skills_dir, llm_client=self._llm_client)
        skills: list[Skill] = []
        for skill_name in self._skill_names:
            skill = await loader.load_meta(skill_name)
            skills.append(skill)

        # 如果只有 skills、没有 prompt_template，创建一个容纳它们
        if not self._prompt_template and not self._system_prompt_str:
            self._prompt_template = PromptTemplate(self._name)

        # 将 Skills 注入 Prompt 模板
        for skill in skills:
            if self._prompt_template:
                self._prompt_template.add(SkillSection(skill))

        if self._max_input_tokens is not None:
            self._memory_profile.working.max_tokens = self._max_input_tokens

        memory = MemoryManager(profile=self._memory_profile)
        thinking_engine = ThinkingEngine(mode=self._thinking_mode, toolkit=self._toolkit)

        # 确定 system prompt
        if self._system_prompt_str:
            system_prompt = self._system_prompt_str
        elif self._prompt_template:
            system_prompt = self._prompt_template.render({
                "tools": self._toolkit.list(),
                "agent_name": self._name,
            })
        else:
            system_prompt = PromptTemplate.preset("default").render({
                "agent_name": self._name,
            })

        return _BuiltAgent(
            name=self._name,
            llm_client=self._llm_client,
            system_prompt=system_prompt,
            toolkit=self._toolkit,
            memory=memory,
            thinking_engine=thinking_engine,
            max_iterations=self._max_iterations,
            max_output_tokens=self._max_output_tokens,
            skills=skills,
            reference=self._reference,
        )


class _BuiltAgent(BaseAgent):
    """AgentBuilder 构建出的完整 Agent。

    内部委托给 ThinkingEngine 执行思考循环。
    Skill 激活采用真实懒加载——LLM 调用 activate_skill:xxx 工具时才加载。
    """

    def __init__(self, name, llm_client, system_prompt, toolkit, memory, thinking_engine, max_iterations, max_output_tokens=None, skills=None, reference=None):
        super().__init__(
            name=name,
            llm_client=llm_client,
            system_prompt=system_prompt,
            tool_registry=toolkit,
            memory_manager=memory,
            max_iterations=max_iterations,
            max_output_tokens=max_output_tokens,
        )
        self.thinking_engine = thinking_engine
        self.toolkit = toolkit
        self._skills: list = skills or []
        self.reference: Reference = reference or Reference()

    def _register_skill_activation_tools(self) -> dict[str, str]:
        """为每个未加载的 Skill 注册 use_skill_xxx 工具。

        返回 {tool_name: skill_name} 映射表，供拦截使用。
        """
        mapping = {}
        for skill in self._skills:
            if skill._loaded:
                continue
            tool_name = f"use_skill_{skill.name}"
            if self.toolkit.has(tool_name):
                mapping[tool_name] = skill.name
                continue
            self.toolkit.add(Tool(
                name=tool_name,
                description=f"激活「{skill.name}」能力：{skill.description}。当需要 {skill.name} 相关能力时调用。",
                tool_type=ToolType.LOCAL,
                func=None,
                parameters={"type": "object", "properties": {}, "required": []},
            ))
            mapping[tool_name] = skill.name
        return mapping

    def set_prompt(self, prompt: str) -> None:
        """修改 system prompt（build 之后也可调用）。

        直接覆盖已渲染的系统提示词。下次 agent.run() 生效。

        Args:
            prompt: 新的系统提示词字符串
        """
        self.system_prompt = prompt

    def set_reference(self, key: str, content: str) -> None:
        """设置一条参考卡。跨 agent.run() 持久化，永不裁剪。

        与 update_reference 等价，语义上更符合"首次设置"的场景。

        Args:
            key: 参考条目标识（如 "characters", "style"）
            content: 参考内容文本
        """
        self.reference.set(key, content)

    def update_reference(self, key: str, content: str) -> None:
        """更新一条参考卡。与 set_reference 等价，语义上强调"更新已有条目"。

        Args:
            key: 参考条目标识
            content: 新的参考内容
        """
        self.reference.set(key, content)

    def remove_reference(self, key: str) -> None:
        """删除一条参考卡。

        Args:
            key: 参考条目标识
        """
        self.reference.remove(key)

    def clear_references(self) -> None:
        """清空所有参考卡。"""
        self.reference.clear()

    def remember(self, key: str, content: str) -> None:
        """存入语义记忆（Semantic Memory）。

        agent.run() 的 pre_turn 阶段会自动根据 user_input 做关键词检索，
        匹配到的语义记忆会以 [Memory] system message 形式注入上下文。

        Args:
            key: 记忆标识（如 "user_name", "preference"）
            content: 记忆内容文本
        """
        self.memory.semantic.store(key, content)

    async def run(self, user_input: str, stream=None, agent_trace=None) -> AgentResult:
        """执行 Agent。

        stream: 可选的流式回调 async (StreamEvent) -> None。
        agent_trace: 可选的外部 AgentTrace（多 Agent 场景下编排器传入）。
                     不传则自动创建。
        """
        # 注册 Skill 激活工具
        skill_tool_map = self._register_skill_activation_tools()

        # 检索门：从语义记忆拉取相关事实
        retrieved = self.memory.pre_turn(user_input)

        # 记录到 trace
        if agent_trace:
            agent_trace.memory_retrieved = [
                {"subject": f.subject, "predicate": f.predicate,
                 "object": f.object, "confidence": f.confidence}
                for f in retrieved
            ]

        # 记忆事实注入工作记忆
        for fact in retrieved:
            from agentflow.runtime.memory.working import Message
            self.memory.working.add(Message(
                role="system",
                content=f"[Memory] {fact.subject} {fact.predicate} {fact.object}",
            ))

        # 用户消息加入工作记忆
        from agentflow.runtime.memory.working import Message
        self.memory.working.add(Message(role="user", content=user_input))

        # 构建 ThinkContext（含 AgentTrace 用于记录执行轨迹）
        # 如果编排器通过 context 传了 _agent_trace，优先用它（多 Agent 场景）
        tools_for_llm = self.toolkit.list_for_llm() if hasattr(self, 'toolkit') else []
        from agentflow.trace.tracer import AgentTrace
        # 如果编排器传了 agent_trace 就用它（多 Agent 场景），否则自己建（单 Agent 场景）
        if agent_trace is None:
            agent_trace = AgentTrace(agent_id=self.name)
        context = ThinkContext(
            user_input=user_input,
            system_prompt=self.system_prompt,
            messages=self.memory.working.get_context_window(),
            tools=tools_for_llm,
            llm_client=self.llm_client,
            memory=self.memory,
            max_iterations=self.max_iterations,
            max_output_tokens=self.max_output_tokens,
            stream=stream,
            skill_tool_map=skill_tool_map,
            agent_trace=agent_trace,
            reference_messages=self.reference.to_messages(),
        )
        # 将 Skill 对象注入 context，供拦截器查找
        context._skills_map = {s.name: s for s in self._skills}

        # 执行思考
        think_result = await self.thinking_engine.run(context)

        # 工作记忆记录 assistant 回复
        from agentflow.runtime.memory.working import Message
        self.memory.working.add(Message(role="assistant", content=think_result.output))

        # 记忆门 + 遗忘门
        await self.memory.post_turn()

        # 记录到 trace
        if agent_trace:
            agent_trace.memory_stored = [
                {"subject": f.subject, "predicate": f.predicate,
                 "object": f.object, "fact_type": f.fact_type,
                 "confidence": f.confidence}
                for f in self.memory._last_extracted
            ]
            agent_trace.memory_forgotten = self.memory._last_forgotten

        return AgentResult(
            output=think_result.output,
            tool_calls=think_result.tool_calls,
            steps=think_result.steps,
            agent_trace=agent_trace,
        )
