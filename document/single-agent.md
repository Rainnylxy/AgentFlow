# 单 Agent 运行时

AgentFlow 单 Agent 运行时提供了从构建到执行的完整生命周期管理。核心设计理念是**一切可插拔**：LLM、工具、记忆、思考策略、Skill 均可独立替换。

## 架构

```
user_input
    │
    ▼
┌─────────────────────────────────────────────┐
│               AgentBuilder                   │
│  .with_llm()  .with_tools()  .with_memory() │
│  .with_thinking()  .with_skill()            │
│  .with_prompt()  .with_reference()          │
└─────────────────┬───────────────────────────┘
                  │ build()
                  ▼
┌─────────────────────────────────────────────┐
│              _BuiltAgent                     │
│                                              │
│  run(user_input)                             │
│    ├─ pre_turn: 语义记忆检索                  │
│    ├─ 构建 ThinkContext                       │
│    ├─ ThinkingEngine.run()                   │
│    │    ├─ resolve_strategy()                │
│    │    │    ├─ ReAct / CoT / PlanExecute    │
│    │    │    └─ AdaptiveRouter               │
│    │    └─ strategy.run()                    │
│    │         └─ tool_loop (LLM ⇄ Tool)       │
│    └─ post_turn: 记忆门 + 遗忘门 + 压缩门     │
└─────────────────────────────────────────────┘
```

## 核心组件

### AgentBuilder — 链式构建入口

统一的 Agent 构建 API，所有子系统通过 `with_*` 方法注入。

```python
from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.llm_client import OpenAIClient
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.memory.manager import MemoryProfile

llm = OpenAIClient(api_key="...", model="gpt-4o")

agent = await (
    AgentBuilder("my-agent")
    .with_llm(llm)
    .with_tools(search, calculator)
    .with_memory(MemoryProfile.standard())
    .with_thinking(ThinkingMode.ADAPTIVE)
    .with_skill("code-review")
    .with_reference("project_rules", "Always use type hints.")
    .with_max_iterations(15)
    .build()
)

result = await agent.run("帮我审查 src/utils.py")
print(result.output)
```

**核心 `with_*` 方法：**

| 方法                           | 用途                                              |
| ------------------------------ | ------------------------------------------------- |
| `with_llm(client)`             | 注入 LLM 客户端（必填）                           |
| `with_tools(*tools)`           | 注册工具：`@tool` 函数、`Tool` 实例或 `ToolKit`   |
| `with_memory(profile)`         | 选择记忆配置：`light()` / `standard()` / `deep()` |
| `with_thinking(mode)`          | 思考模式：REACT / COT / PLAN_EXECUTE / ADAPTIVE   |
| `with_skill(name)`             | 从 `skills/` 目录加载 Skill                       |
| `with_prompt(template)`        | 注入 PromptTemplate 或纯字符串                    |
| `with_reference(key, content)` | 添加永久上下文（不参与滑动窗口裁剪）              |
| `with_max_iterations(n)`       | 最大思考轮次（默认 10）                           |
| `with_max_input_tokens(n)`     | 输入上下文窗口上限                                |

### ThinkingEngine — 多模式思考系统

管理 5 种思考策略，支持显式指定或自适应路由。

```python
from agentflow.runtime.thinking import ThinkingEngine, ThinkingMode

engine = ThinkingEngine(mode=ThinkingMode.ADAPTIVE, toolkit=toolkit)
result = await engine.run(context)
```

**五种策略：**

| 策略             | 模式                                 | 适用场景             |
| ---------------- | ------------------------------------ | -------------------- |
| **ReAct**        | 思考 → 行动 → 观察 循环              | 工具调用密集任务     |
| **CoT**          | 深度思考 → 最终答案（两阶段）        | 推理、分析类任务     |
| **Plan-Execute** | 制定计划 → 逐步执行 → 综合           | 复杂多步任务         |
| **Adaptive**     | 基于关键词信号自动路由               | 不确定用哪种策略时   |
| **Reflection**   | 包装任意策略，执行后 LLM 评审 + 修正 | 需要高质量输出的场景 |

**AdaptiveRouter 路由逻辑**（`agentflow/runtime/thinking/adaptive.py`）：

- 检测到 `multi_step` 信号 → Plan-Execute
- 检测到 `deep_reasoning` 信号 → CoT
- 检测到 `safe_critical` 信号 → ReAct（可审计）
- 默认 → ReAct

**Reflection 包装器**用法：

```python
mode = ThinkingMode.REACT.with_reflection(depth=3)
engine = ThinkingEngine(mode=mode)
# 每个 ReAct 循环后，LLM 评审输出，不通过则重新执行，最多 3 次
```

### Memory — 三层记忆系统

```
┌──────────────────────────────────────────┐
│              MemoryManager                │
│                                           │
│  pre_turn(user_input)                     │
│    ├─ 检索门: Semantic 关键词匹配          │
│    └─ 注入 [Memory] 消息到 Working         │
│                                           │
│  post_turn()                              │
│    ├─ 记忆门: 提取新事实 → Episodic        │
│    ├─ 遗忘门: 淘汰过期事实                  │
│    └─ 压缩门: 溢出消息 LLM 摘要压缩         │
└──────────────────────────────────────────┘
```

| 层级              | 组件         | 容量                   | 生命周期   | 用途                   |
| ----------------- | ------------ | ---------------------- | ---------- | ---------------------- |
| **L1 Working**    | 滑动窗口消息 | 20 turns / 8000 tokens | 单次 run() | 当前对话上下文         |
| **L2 Episodic**   | 时间线事实   | 200 条，TTL 过期       | 跨 run()   | 本次会话的关键信息     |
| **L3 Semantic**   | 键值存储     | 无上限                 | 永久       | 跨会话的用户偏好、知识 |
| **ReferenceCard** | 固定文本     | 无限制                 | 永久       | 角色设定、项目约定     |

**配置预设：**

```python
MemoryProfile.light()     # 轻量：10 turns，无情景记忆
MemoryProfile.standard()  # 标准：20 turns，200 条情景记忆
MemoryProfile.deep()      # 深度：40 turns，500 条情景 + 语义记忆
```

**Reference 永久上下文：**

```python
builder.with_reference("characters", "主角：小明，25 岁程序员")
builder.with_reference("style", "回复使用中文，保持幽默风格")
```

Reference 的内容永不参与滑动窗口裁剪，每次 `run()` 都会注入到上下文头部。

### ToolKit — 工具注册与执行

`@tool` 装饰器自动将 Python 函数转为 OpenAI function-calling 格式的 JSON Schema。

```python
from agentflow.runtime.toolkit import tool, ToolKit

@tool
def search(query: str) -> str:
    """Search the knowledge base for relevant information."""
    return kb.search(query)

@tool(name="calc", description="Evaluate a math expression")
def calculate(expression: str) -> str:
    return str(eval(expression))

toolkit = ToolKit()
toolkit.add(search)
toolkit.add(calculate)

# 获取 OpenAI 格式的工具列表
schemas = toolkit.list_for_llm()
# 执行工具
result = await toolkit.execute("search", {"query": "Python"})
```

**类型推导**：`@tool` 从函数签名自动推导 JSON Schema，支持 `int`、`str`、`float`、`bool`、`list[X]`、`Optional[X]` 和 Pydantic 模型。

**三源统一**（`ToolRegistry`）：LOCAL 函数 / MCP 服务 / REST API 统一注册和执行接口。

### Skill — 可复用能力模块

Skill 是 Markdown 文件定义的可复用 Agent 能力，支持懒加载。

```markdown
<!-- skills/code-review.md -->

---

name: code-review
description: Review code for bugs, style, and security issues
tools: [read_file, grep]
---

# Code Review Skill

## 流程

1. 阅读目标文件
2. 检查潜在 bug
3. 检查安全漏洞
4. 输出 review 报告
```

```python
builder.with_skill("code-review")
```

**懒加载机制**：

- `build()` 阶段只读 frontmatter（name + description + tools），不读 body，不调 LLM
- 运行时 Agent 调用 `use_skill_code_review` 工具时，才加载 body 并提取结构化 Step
- Step 提取由 `StepExtractor` 调用 LLM 完成

### LLM Client — 统一的 LLM 调用接口

```python
from agentflow.runtime.llm_client import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    timeout=120.0,
)
response = await client.chat([
    {"role": "user", "content": "Hello"}
])
# LLMResponse(content=..., tool_calls=[...], reasoning_content=..., usage={...})
```

**指数退避重试**：`base_delay * 2^attempt`，自动重试 429/5xx、网络错误、超时。

**reasoning_content**：自动捕获 DeepSeek R1 / OpenAI o1 的思考链。

**代理支持**：通过 `proxy` 参数或 `AGENTFLOW_PROXY` 环境变量设置 HTTP 代理。

### Prompt 模板 — 可组合的 System Prompt

```python
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.prompt.section import RoleCard, SafetyRules, ToolManual, FormatGuide

template = PromptTemplate("my-agent")
template.add(RoleCard(name="小助手", role="客服", tone="友善专业"))
template.add(SafetyRules(rules=["不泄露用户信息", "退款 > ¥500 需审批"]))
template.add(ToolManual())
template.add(FormatGuide(format="markdown"))

system_prompt = template.render({"agent_name": "agent_01"})
```

**内置 Section**：`RoleCard`、`SafetyRules`、`ToolManual`、`FormatGuide`、`TimeContext`，每个 Section 有独立 order 属性，render 时自动排序。

内置预设：`PromptTemplate.preset("customer_support")`、`PromptTemplate.preset("coding_assistant")`。

## 使用示例

### 最小可运行 Agent

```python
from agentflow.runtime.llm_client import OpenAIClient
from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.thinking import ThinkingMode

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 晴天, 25°C"

async def main():
    llm = OpenAIClient(
        api_key="sk-xxx",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    agent = await (
        AgentBuilder("assistant")
        .with_llm(llm)
        .with_tools(get_weather)
        .with_thinking(ThinkingMode.REACT)
        .build()
    )
    result = await agent.run("北京今天天气怎么样？")
    print(result.output)

import asyncio
asyncio.run(main())
```

### 带记忆的 Agent

```python
agent = await (
    AgentBuilder("memory-bot")
    .with_llm(llm)
    .with_memory(MemoryProfile.deep())
    .build()
)

# 第一次对话
await agent.run("我叫小明，喜欢 Python")
# 第二次对话——记忆生效
result = await agent.run("我喜欢什么编程语言？")
# → "你喜欢 Python"
```

## 最佳实践

1. **开发阶段用 Adaptive 模式**，快速原型；生产环境根据任务特征选择固定策略
2. **用 Reference 存角色设定和项目约定**，不要塞进 system prompt，方便跨 run 复用
3. **Tool 的描述要写清楚**，LLM 靠 description 决定是否调用
4. **Memory 按需选择**：简单对话用 `light()`，需要跨轮记忆用 `standard()`，需要永久知识用 `deep()`
