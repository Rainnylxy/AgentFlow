# Memory 子系统详细设计

> 日期: 2026-06-01 | 状态: Draft | 关联: [[2026-06-01-agent-builder-design]]

## 一、目标

将当前简单的消息列表升级为三层记忆系统，Agent 能够自主决定记住什么、忘掉什么、何时检索，让记忆管理不再依赖开发者手动控制。

## 二、三层记忆模型

```
Layer 1: Working Memory  ─→  当前对话，完整消息
       │ 容量: 10-30 轮     │  生命周期: 单次 run()
       │ 存储: 内存列表      │
       ▼
Layer 2: Episodic Memory ─→  跨会话结构化事实
       │ 容量: 100-500 条   │  生命周期: 天/周
       │ 存储: SQLite/Pebble │
       ▼
Layer 3: Semantic Memory  ─→  长期知识，向量检索
       容量: 无上限          生命周期: 永久
       存储: Chroma/向量 DB
```

### 2.1 Layer 1: Working Memory（工作记忆）

**存什么**：当前对话的完整消息列表（`Message` 对象）。

**能力**：
- 滑动窗口管理（超出上限自动截断最旧消息）
- 支持消息类型标记（user / assistant / tool / system）
- 为每一轮标注 tool_call_id 关联

**不做**：不提取、不总结，保持原始完整性供 LLM 推理。

### 2.2 Layer 2: Episodic Memory（情节记忆）

**存什么**：从对话中提取的结构化事实。

```python
@dataclass
class MemoryFact:
    fact_type: Literal["entity", "decision", "event", "preference"]
    subject: str       # "user" | "agent" | "tool:weather"
    predicate: str     # "prefers" | "stated" | "executed"
    object: str        # "quick replies" | "refund policy"
    confidence: float  # 0.0 - 1.0
    timestamp: datetime
    source_turn: int   # 第几轮对话
    ttl: int           # 生存时间（秒）

    def is_expired(self) -> bool: ...
    def decay(self, factor: float) -> None: ...  # 降置信度
```

**提取流程**（记忆门）：

```
每个 turn 后自动触发：
  Working Memory 中的对话
    → 小 Prompt: "这条信息值得记住吗？如果是，提取为事实"
    → 结构化 MemoryFact 列表
    → 存入 Episodic 后端
```

**遗忘流程**（遗忘门）：

```
Agent 空闲时 / 每次 run() 启动时：
  遍历 Episodic
    → TTL 到期 → confidence -= 0.3
    → confidence < 0.3 → 删除
    → 容量超限 → 按 (confidence × 新鲜度) 排序 → 淘汰低分项
```

### 2.3 Layer 3: Semantic Memory（语义记忆）

**存什么**：需要长期保留的知识，以向量嵌入存储。

**检索流程**（检索门）：

```
每个 run() 启动时：
  Agent 自主生成检索 query
    → embedding(query)
    → Chroma 向量相似度搜索 (top_k=5)
    → 返回相关 MemoryFact
    → 注入 Working Memory（作为 system 消息）
```

**存储后端**：
- 默认：Chroma（本地文件，零配置）
- 可选：Pinecone / Weaviate / PostgreSQL pgvector

## 三、自主管理控制器

```python
class MemoryManager:
    def __init__(self, profile: MemoryProfile):
        self.working = WorkingMemory(profile.working)
        self.episodic = EpisodicMemory(profile.episodic)
        self.semantic = SemanticMemory(profile.semantic)
        self.auto_memorize = profile.auto_memorize
        self.auto_forget = profile.auto_forget
        self.auto_retrieve = profile.auto_retrieve

    async def pre_turn(self) -> list[MemoryFact]:
        """每轮之前：检索门"""
        ...

    async def post_turn(self, messages: list[Message]):
        """每轮之后：记忆门 + 遗忘门（周期性触发）"""
        ...

    def get_context_window(self) -> list[Message]:
        """获取当前 Working Memory 内容，供 LLM 使用"""
        ...
```

## 四、MemoryProfile 预设

```python
class MemoryProfile:
    @classmethod
    def light(cls):    # 聊天机器人：只工作记忆
        return cls(working=WorkingConfig(max_turns=10),
                   episodic=None, semantic=None)

    @classmethod
    def standard(cls): # 客服 Agent：工作 + 情节
        return cls(working=WorkingConfig(max_turns=20),
                   episodic=EpisodicConfig(max_facts=200, ttl_hours=168),
                   semantic=None)

    @classmethod
    def deep(cls):     # 研究 Agent：三层全开
        return cls(working=WorkingConfig(max_turns=40),
                   episodic=EpisodicConfig(max_facts=500, ttl_hours=720),
                   semantic=SemanticConfig(embedder="text-embedding-3-small", top_k=5))
```

## 五、关键细节

### 5.1 记忆门 Prompt

提取事实的小 Prompt 由系统内置，用户可覆盖。它是轻量级的（不需要完整 LLM 推理），未来可考虑用本地小模型（如 onnx-runtime）离线执行以节省成本。

### 5.2 遗忘策略

不出错地忘记：优先淘汰"低置信度 + 旧时间戳"的条目。用户也可标记某条为 `protected` 永久保留。

### 5.3 与评测联动

结构化记忆使 Faithfulness 检测更精确——可以直接交叉比对 Agent 声称的内容与 Episodic 中存储的工具输出事实。

## 六、与现有代码的关系

| 现有模块 | 处理 |
|----------|------|
| `memory.py` (MemoryManager) | 完全重写，同名兼容 |
| `Message` dataclass | 保留，作为 Working Memory 基本单元 |
| Episodic / Semantic | 全新模块 |

## 七、待定内容

- 多 Agent 间的共享记忆
- 记忆的可解释性面板（Dashboard 中展示记忆图）
- 基于用户反馈的记忆强化学习
