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
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile
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
        self._system_prompt_str = None
        self._skills_dir = Path("skills")  # 默认 skills/ 目录
        self._skill_names: list[str] = []

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

    def with_max_iterations(self, n: int) -> "AgentBuilder":
        self._max_iterations = n
        return self

    def build(self) -> BaseAgent:
        if self._llm_client is None:
            raise ValueError("with_llm() is required. Provide an LLM client.")

        # 加载 Skills
        loader = SkillLoader(skills_dir=self._skills_dir)
        skills: list[Skill] = []
        for skill_name in self._skill_names:
            skill = loader.load(skill_name)
            skills.append(skill)

        # 如果只有 skills、没有 prompt_template，创建一个容纳它们
        if not self._prompt_template and not self._system_prompt_str:
            self._prompt_template = PromptTemplate(self._name)

        # 将 Skills 注入 Prompt 模板
        for skill in skills:
            if self._prompt_template:
                self._prompt_template.add(SkillSection(skill))

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
        )


class _BuiltAgent(BaseAgent):
    """AgentBuilder 构建出的完整 Agent。

    内部委托给 ThinkingEngine 执行思考循环。
    """

    def __init__(self, name, llm_client, system_prompt, toolkit, memory, thinking_engine, max_iterations):
        super().__init__(
            name=name,
            llm_client=llm_client,
            system_prompt=system_prompt,
            tool_registry=toolkit,
            memory_manager=memory,
            max_iterations=max_iterations,
        )
        self.thinking_engine = thinking_engine
        self.toolkit = toolkit

    async def run(self, user_input: str) -> AgentResult:
        # 检索门：从语义记忆拉取相关事实
        retrieved = self.memory.pre_turn(user_input)

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

        # 构建 ThinkContext
        tools_for_llm = self.toolkit.list_for_llm() if hasattr(self, 'toolkit') else []
        context = ThinkContext(
            user_input=user_input,
            system_prompt=self.system_prompt,
            messages=self.memory.working.get_context_window(),
            tools=tools_for_llm,
            llm_client=self.llm_client,
            memory=self.memory,
            max_iterations=self.max_iterations,
        )

        # 执行思考
        think_result = await self.thinking_engine.run(context)

        # 工作记忆记录 assistant 回复
        from agentflow.runtime.memory.working import Message
        self.memory.working.add(Message(role="assistant", content=think_result.output))

        # 记忆门 + 遗忘门
        self.memory.post_turn()

        return AgentResult(
            output=think_result.output,
            tool_calls=think_result.tool_calls,
            steps=think_result.steps,
        )
