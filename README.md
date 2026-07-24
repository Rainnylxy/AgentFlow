# AgentFlow

> 生产级多 Agent 编排与评测框架 —— Agent 开发的 Vercel

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Go](https://img.shields.io/badge/go-1.22+-00ADD8.svg)](https://go.dev/)

**AgentFlow** 是一个开源的多 Agent 编排与评测框架，提供从单 Agent 构建、多 Agent 编排到标准化评测的完整工具链。

## 为什么需要 AgentFlow？

LangGraph、CrewAI、AutoGen 等框架让搭建 Agent 原型变得极其容易，但从原型到生产之间存在巨大鸿沟：

- **没有观测性** — Agent 内部推理链路不透明，出了错不知道哪一步挂了
- **没有容错机制** — 一个 Agent 超时，整个工作流级联崩溃
- **没有标准化评测** — 评测方式碎片化，无法量化 Agent 质量
- **没有 A/B 对比** — 改了一句 prompt，Agent 变好了还是变差了？没有数据支撑

AgentFlow 提供完全开源、可自部署的解决方案。

## 核心特性

### Workflow DSL

声明式定义多 Agent 拓扑（DAG），支持顺序、条件分支、并行、循环、子图嵌套。YAML 序列化 + Mermaid 可视化。

### Orchestration Engine

Python `DAGExecutor` — 拓扑分层 + 层内异步并发，集成熔断器、指数退避重试、超时控制、降级逻辑。Go 高性能引擎（gRPC）开发中。

### Agent Runtime

| 子系统             | 能力                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| **AgentBuilder**   | 链式 API，一行组装 LLM + 工具 + 记忆 + 思考策略 + Skill                    |
| **ThinkingEngine** | 5 种思考策略：ReAct / CoT / Plan-Execute / Adaptive / Reflection           |
| **Memory**         | 三层记忆（Working / Episodic / Semantic）+ ReferenceCard 永久上下文        |
| **ToolKit**        | `@tool` 装饰器自动生成 JSON Schema，支持 MCP / REST / Local 三源统一       |
| **Skill**          | Markdown 定义的可复用能力模块，懒加载 + LLM 步骤提取                       |
| **Prompt**         | 可组合的 Section 系统（RoleCard / SafetyRules / ToolManual / FormatGuide） |
| **MessageBus**     | 意图驱动的 Agent 间消息传递（delegate / handoff / task_complete）          |
| **Hooks**          | 生命周期钩子，Workflow / Group / Node / Tool 四级可插拔扩展                |

### Trace Store

统一 Trace 模型，覆盖单 Agent 执行轨迹到多 Agent 工作流全链路。支持 A/B Diff 对比、瓶颈分析、Token 统计。

### Eval Engine

10 维评测矩阵，覆盖任务完成、推理质量、安全边界、效率成本四大类。EvalSuite 批量运行 + `diagnose()` 低分归因 + `compare()` A/B 回归检测。

### Benchmark Suite

标准化 Agent 评测集：Tool-Use / Multi-Hop QA / Long-Context。

### CLI

`agentflow new / dev / eval / trace` 四命令开发工作流。

## 技术架构

```
┌──────────────────────────────────────────────────┐
│              AgentFlow CLI (Python)               │
│        agentflow new / dev / eval / trace         │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│         Orchestration Layer                       │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  DAGExecutor     │  │  Go Engine (开发中)    │  │
│  │  (Python, 主力)   │  │  gRPC + goroutine    │  │
│  └──────────────────┘  └──────────────────────┘  │
│  Retry / Fallback / Timeout / Loop               │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│           Agent Runtime (Python)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │Thinking  │ │Memory    │ │ToolKit / Skill   │  │
│  │Engine    │ │(3-Layer) │ │Prompt / Message  │  │
│  └──────────┘ └──────────┘ └──────────────────┘  │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│          Evaluation Engine (Python)               │
│  10-Dimensional Agent Quality Assessment          │
│  Task | Reasoning | Safety | Reliability          │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│          Trace Store (Python)                     │
│  WorkflowTrace / AgentTrace / Diff                │
└──────────────────────────────────────────────────┘
```

## 技术栈

| 层级       | 技术                                            |
| ---------- | ----------------------------------------------- |
| Agent 框架 | LangGraph + LangChain                           |
| 编排引擎   | Python asyncio（主力）/ Go + gRPC（开发中）     |
| LLM 客户端 | OpenAI-compatible API（支持 DeepSeek、Qwen 等） |
| 向量相似度 | sentence-transformers                           |
| 序列化     | YAML + Protocol Buffers                         |
| CLI        | Typer + Rich                                    |
| 测试       | pytest                                          |

## 为什么编排引擎有 Go 版本？

Python DAGExecutor 覆盖了当前所有编排需求，Go 版本面向更高性能要求的场景：

1. **高性能** — goroutine 天然适合 DAG 拓扑排序 + 并发执行
2. **可靠性** — 静态类型 + 编译检查，减少生产环境意外
3. **部署简单** — 单个二进制分发，不依赖 Python 环境

两者通过 gRPC 通信，Go 引擎调用 Python Agent Runtime 执行具体节点。

## 快速开始

```bash
# 克隆项目
git clone https://github.com/Rainnylxy/AgentFlow.git
cd AgentFlow

# 安装依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 创建新项目
agentflow new my-agent-app

# 启动开发服务器（运行工作流）
agentflow dev

# 运行评测
agentflow eval

# 查看 Trace
agentflow trace
```

## 文档

| 文档                                        | 内容                                                                  |
| ------------------------------------------- | --------------------------------------------------------------------- |
| [单 Agent 运行时](document/single-agent.md) | AgentBuilder → ThinkingEngine → Memory → ToolKit → Skill → LLM Client |
| [多 Agent 编排](document/multi-agent.md)    | Workflow DSL → DAGExecutor → MessageBus → Hooks → Trace 模型          |
| [评测体系](document/eval-system.md)         | 10 维评测矩阵 → Evaluator 详解 → EvalSuite → Benchmark Suite          |
| [设计文档](docs/superpowers/specs/)         | 各子系统的架构设计文档                                                |

## 许可证

[MIT](LICENSE)
