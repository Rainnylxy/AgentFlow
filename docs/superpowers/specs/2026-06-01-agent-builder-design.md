# Agent Builder 优化设计（总体架构）

> 日期: 2026-06-01 | 状态: Draft | 关联: [[2026-06-01-toolkit-design]] [[2026-06-01-memory-design]] [[2026-06-01-prompt-design]] [[2026-06-01-thinking-design]]

## 一、背景

AgentFlow 当前的单 Agent 构建体验存在以下问题：

1. **样板代码多** — 构建一个 Agent 需要手动创建 LLMClient、ToolRegistry、MemoryManager、手写 system_prompt、选择 ReActAgent，5-6 步组装，缺少统一入口
2. **Tool 定义繁琐** — 需要手写 JSON Schema、name、description，与 Python 函数签名重复
3. **记忆系统简单** — 只有一个 short_term 列表，不支持跨会话、结构化、长期记忆
4. **Prompt 硬编码** — system_prompt 是纯字符串，无法复用、模板化
5. **思考模式单一** — 只有 ReAct 模式，不支持复杂任务的 Plan-Execute 或高风险任务的反思修正

## 二、目标

以 **AgentBuilder（Builder 模式）** 作为统一入口，内部四个子系统各自独立演化：

```python
agent = (AgentBuilder("support-agent")
    .with_tools(ToolKit.from_module("my_tools"))
    .with_memory(MemoryProfile.standard())
    .with_prompt(PromptTemplate.preset("customer_support"))
    .with_thinking(ThinkingMode.ADAPTIVE)
    .build())
```

## 三、架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   AgentBuilder (Facade)                      │
│  .with_tools()  .with_memory()  .with_prompt()               │
│  .with_thinking()  .build()  .from_yaml()                    │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
  ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼──────┐
  │ ToolKit │   │ Memory  │   │ Prompt  │   │ Thinking  │
  │ 子系统  │   │ 子系统  │   │ 子系统  │   │ Engine    │
  └─────────┘   └─────────┘   └─────────┘   └──────────┘
```

四个子系统独立构造、独立测试、独立替换，Builder 只做协调和依赖注入。

## 四、四个子系统简介

### 4.1 ToolKit 子系统

**做什么**：让工具定义从"手写样板代码"变成"装饰器一键注册"，同时统一本地函数、MCP Server、REST API 三种工具源。

**核心能力**：
- `@tool` 装饰器：从函数签名 + docstring 自动推导 Tool schema，Pydantic 做参数校验
- 三源统一：本地函数、MCP、REST 三种工具对 Agent 透明
- 自动 Schema 生成：将注册的工具转为 OpenAI function-calling 格式

详见：[[2026-06-01-toolkit-design]]

### 4.2 Memory 子系统

**做什么**：将简单的消息列表升级为三层记忆系统（工作/情节/语义），Agent 自主决定记忆、遗忘、检索。

**核心能力**：
- 三层模型：Working（当前对话）+ Episodic（跨会话结构化事实）+ Semantic（长期向量检索）
- 结构化存储：不存原始文本，提取为 `MemoryFact`（主体-谓词-客体-置信度）
- 自主管理：记忆门（记住什么）、遗忘门（淘汰什么）、检索门（需要什么）全程自动

详见：[[2026-06-01-memory-design]]

### 4.3 Prompt 模板子系统

**做什么**：将手写 system_prompt 字符串变成模块化拼装，不同 Agent 复用不同模块。

**核心能力**：
- Section 模块化：角色卡、工具册、规则集、示例集各自独立
- Jinja2 模板渲染：变量注入 + 条件逻辑
- 内置模板库 + 自定义注册
- 场景预设：`PromptTemplate.preset("customer_support")` 一键生成

详见：[[2026-06-01-prompt-design]]

### 4.4 Thinking 引擎

**做什么**：让 Agent 支持多种思考模式（ReAct / Plan-Execute / CoT），具备自适应选择和反思自修正能力。

**核心能力**：
- 四种策略：ReAct、Plan-Execute、CoT（链式思考）、Reflection（反思装饰器）
- 自适应路由：根据任务信号自动选择最优策略
- 反思自修正：事实核查 + 完备性检查 + 策略调整

详见：[[2026-06-01-thinking-design]]

## 五、文件布局

```
agentflow/runtime/
  agent.py              # BaseAgent + AgentResult（保留, 微调）
  builder.py             # NEW: AgentBuilder
  toolkit.py             # NEW: ToolKit + @tool 装饰器
  tool_registry.py       # 扩展: MCP/REST 执行支持
  memory/
    __init__.py          # NEW: MemoryProfile + MemoryManager(重写)
    working.py           # 工作记忆
    episodic.py          # 情节记忆
    semantic.py          # 语义记忆
    manager.py           # 自主管理控制器
  prompt/
    __init__.py          # NEW: PromptTemplate + PromptRenderer
    section.py           # Section 基类
    templates/           # 内置模板库
      builtin/           # 通用模块（角色卡、工具册、安全规则）
      domains/           # 领域模块（客服、编程、研究）
  thinking/
    __init__.py          # NEW: ThinkingEngine + ThinkingMode
    base.py              # ThinkingStrategy 抽象
    react.py             # ReActStrategy
    plan_execute.py      # PlanExecuteStrategy
    cot.py               # CoTStrategy
    reflection.py        # ReflectionWrapper
    adaptive.py          # AdaptiveRouter
```

## 六、与现有代码的关系

| 现有模块 | 处理方式 |
|----------|----------|
| `tool_registry.py` | 保留并扩展，增加 MCP/REST 执行器 |
| `memory.py` | 重写为 memory/ 子包，保持同名接口兼容 |
| `react_agent.py` | 逻辑迁移到 thinking/react.py |
| `agent.py` (BaseAgent) | 保留，简化为容器 + 委托 |
| `llm_client.py` | 不变，仍作为 LLM 调用抽象层 |

## 七、设计原则

1. **组件即插拔** — 每个子系统可独立替换，换记忆策略不影响工具系统
2. **渐进复杂度** — 简单场景走默认，复杂场景可深度定制
3. **与评测联动** — Thinking 的反思记录成为 D7 Adaptability 数据源；Memory 的结构化事实辅助 D4 Faithfulness 检测
4. **向后兼容** — `BaseAgent(system_prompt="...")` 继续可用，内部自动升级
