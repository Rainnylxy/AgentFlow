"""Memory Manager — 三层记忆系统的统一管理入口。

协调 Working / Episodic / Semantic 三层记忆，
提供自主记忆门、遗忘门、检索门、压缩门。

事实提取策略可通过 fact_extractor 参数注入，
默认使用 KeywordFactExtractor（关键词模式匹配）。

WorkingMemory 压缩策略：当消息被 max_turns 挤出窗口时，
post_turn 阶段自动调用 LLM 将溢出消息压缩为摘要，
注入到上下文窗口头部。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Awaitable
from agentflow.runtime.memory.working import WorkingMemory, Message
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact
from agentflow.runtime.memory.semantic import SemanticMemory
from agentflow.runtime.memory.token_counter import TokenCounter
from agentflow.runtime.memory.fact_extractor import (
    BaseFactExtractor,
    KeywordFactExtractor,
    NoOpFactExtractor,
)


@dataclass
class WorkingConfig:
    max_turns: int = 20
    max_tokens: int = 8000


@dataclass
class MemoryProfile:
    """三层记忆的配置预设。"""
    working: WorkingConfig = field(default_factory=WorkingConfig)
    episodic_max: int = 200
    semantic_enabled: bool = False
    semantic_embedder: Optional[str] = None
    auto_memorize: bool = True
    auto_forget: bool = True
    auto_retrieve: bool = True
    auto_compress: bool = True  # post_turn 时自动压缩溢出消息

    @classmethod
    def light(cls) -> "MemoryProfile":
        return cls(working=WorkingConfig(max_turns=10), episodic_max=0, auto_compress=False)

    @classmethod
    def standard(cls) -> "MemoryProfile":
        return cls(working=WorkingConfig(max_turns=20), episodic_max=200)

    @classmethod
    def deep(cls) -> "MemoryProfile":
        return cls(
            working=WorkingConfig(max_turns=40),
            episodic_max=500,
            semantic_enabled=True,
        )


class MemoryManager:
    """三层记忆管理器 — Agent 自主管理记忆生命周期。

    用法:
        mgr = MemoryManager(summarizer=llm_summarize)
        facts = mgr.pre_turn("user question")   # 检索门
        # ... run LLM turn using mgr.working ...
        await mgr.post_turn()                    # 记忆门 + 遗忘门 + 压缩门

    事实提取策略:
        mgr = MemoryManager(fact_extractor=NoOpFactExtractor())  # 不提取事实
        mgr = MemoryManager(fact_extractor=LLMFactExtractor(llm))  # LLM 驱动

    WorkingMemory 压缩:
        mgr = MemoryManager(summarizer=my_summarizer)  # 启用自动压缩
        # post_turn 时自动将溢出消息压缩为摘要

    Backward-compatible attributes:
        .short_term  -> alias for .working
        .long_term   -> alias for .semantic
    """

    def __init__(
        self,
        profile: Optional["MemoryProfile"] = None,
        verbose: bool = False,
        fact_extractor: Optional["BaseFactExtractor"] = None,
        summarizer: Optional[Callable[[list[Message]], Awaitable[str]]] = None,
        token_counter: Optional["TokenCounter"] = None,
    ):
        self.profile = profile or MemoryProfile.standard()
        self.verbose = verbose
        self.working = WorkingMemory(
            max_turns=self.profile.working.max_turns,
            max_tokens=self.profile.working.max_tokens,
            summarizer=summarizer,
            token_counter=token_counter,
        )
        self.episodic = EpisodicMemory(max_facts=self.profile.episodic_max)
        self.semantic = SemanticMemory(embedder=self.profile.semantic_embedder)
        self._turn_count = 0
        self._last_extracted: list[MemoryFact] = []
        self._last_forgotten: int = 0
        self._last_compressed: str | None = None

        # 事实提取策略：light profile 默认不提取，其余默认关键词
        if fact_extractor is not None:
            self._fact_extractor = fact_extractor
        elif self.profile.episodic_max == 0:
            self._fact_extractor = NoOpFactExtractor()
        else:
            self._fact_extractor = KeywordFactExtractor()

    # --- Backward-compatible aliases ---
    @property
    def short_term(self) -> WorkingMemory:
        """Backward-compatible alias for .working"""
        return self.working

    @property
    def long_term(self) -> SemanticMemory:
        """Backward-compatible alias for .semantic"""
        return self.semantic

    def pre_turn(self, user_input: Optional[str] = None) -> list[MemoryFact]:
        """检索门：每个 turn 之前，从语义记忆检索相关事实。"""
        self._turn_count += 1

        if not self.profile.auto_retrieve or user_input is None:
            return []

        results = self.semantic.search(user_input, top_k=5)
        facts = []
        for r in results:
            fact = MemoryFact(
                fact_type="entity",
                subject=f"kb:{r['key']}",
                predicate="contains",
                object=r["content"][:200],
                confidence=0.7,
                timestamp=datetime.now(),
                source_turn=self._turn_count,
                ttl=86400,
            )
            facts.append(fact)
        return facts

    async def post_turn(self) -> None:
        """记忆门 + 遗忘门 + 压缩门：每个 turn 之后触发（异步）。"""
        if self.profile.auto_memorize:
            await self._extract_facts()

        if self.profile.auto_forget:
            self._last_forgotten = self.episodic.forget_expired()

        if self.profile.auto_compress and self.working.needs_compression:
            self._last_compressed = await self.working.compress()

    async def _extract_facts(self) -> None:
        """从工作记忆中提取结构化事实，委托给注入的 FactExtractor 策略。

        异步设计：即使当前是同步关键词提取器，接口保持 async，
        以便后续无缝升级为 LLM 驱动提取。
        """
        messages = self.working.get_context_window()
        facts = await self._fact_extractor.extract(messages, self._turn_count)
        self._last_extracted = list(facts)
        for fact in facts:
            self.episodic.add(fact)
