# AgentFlow

> 生产级多 Agent 编排与评测框架 —— Agent 开发的 Vercel

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Go](https://img.shields.io/badge/go-1.22+-00ADD8.svg)](https://go.dev/)

**AgentFlow** 是一个开源的生产级多 Agent 编排与评测框架，帮助开发者将 Agent 从原型推向生产，并持续评测质量。

## 为什么需要 AgentFlow？

2026 年，LangGraph、CrewAI、AutoGen 等框架让搭建 Agent 原型变得极其容易，但从原型到生产之间存在巨大鸿沟：

- 🔍 **没有观测性** — Agent 内部推理链路不透明，出了错不知道哪一步挂了
- 🛡️ **没有容错机制** — 一个 Agent 超时，整个工作流级联崩溃
- 📊 **没有标准化评测** — 评测方式碎片化，无法量化 Agent 质量
- 🔬 **没有 A/B 对比** — 改了一句 prompt，Agent 变好了还是变差了？没有数据支撑

闭源方案（LangSmith）存在但昂贵且生态绑定。AgentFlow 是完全开源、框架无关、可自部署的替代方案。

## 核心特性

| 模块                          | 功能                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| **Workflow DSL**              | Python 声明式定义多 Agent 拓扑（DAG），支持顺序、条件分支、并行、循环、子图嵌套             |
| **Orchestration Engine (Go)** | 高性能 DAG 执行器，内置熔断器、指数退避重试、超时控制、降级逻辑                             |
| **Agent Runtime**             | 支持 ReAct / Plan-Execute 模式，统一 Tool Registry（MCP + REST + 本地函数），Memory Manager |
| **Trace Store**               | OpenTelemetry 原生集成，完整执行轨迹持久化，支持 Trace 回放与 Diff 对比                     |
| **Eval Engine**               | 4 种 Evaluator（Exact Match / Semantic / LLM-as-Judge / Trajectory Scoring），一键跑分      |
| **Benchmark Suite**           | 标准化 Agent 评测集（Tool-Use / Multi-Hop QA / Long-Context）                               |
| **CLI & Dashboard**           | `agentflow new / dev / eval / trace` + React Dashboard 可视化                               |

## 技术架构

```
┌─────────────────────────────────────────┐
│           AgentFlow CLI (Python)         │
│     agentflow new / dev / eval / trace   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       Orchestration Engine (Go)          │
│  ┌──────────┐ ┌────────┐ ┌───────────┐  │
│  │DAG Exec  │ │Circuit │ │ Retry/    │  │
│  │          │ │Breaker │ │ Fallback  │  │
│  └──────────┘ └────────┘ └───────────┘  │
│  ┌────────────────────────────────────┐  │
│  │     OpenTelemetry Tracing          │  │
│  └────────────────────────────────────┘  │
└──────────────────┬──────────────────────┘
                   │ gRPC
┌──────────────────▼──────────────────────┐
│        Agent Runtime (Python)            │
│  ┌──────────┐ ┌────────┐ ┌───────────┐  │
│  │Tool Reg  │ │Memory  │ │ Planner   │  │
│  └──────────┘ └────────┘ └───────────┘  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       Evaluation Engine (Python)         │
│  10-Dimensional Agent Quality Assessment │
│  Tool | Param | Trajectory | Faithful   │
│  Token | Consistency | Plan | Adaptive  │
│  Abuse | Scope | Semantic | LLM-Judge   │
└─────────────────────────────────────────┘
```

## Agent 评测矩阵 (10-Dimensional Evaluation Matrix)

AgentFlow 内置了业界最完整的 Agent 评测体系，覆盖**任务完成、推理质量、安全边界、效率成本**四个大类：

### 第一类：任务完成 (Task Completion)

| 维度                    | 评估器                | 核心问题                                             | 评估方法               |
| ----------------------- | --------------------- | ---------------------------------------------------- | ---------------------- |
| **D1 Tool Selection**   | `ExactMatchEvaluator` | Agent 选对工具了吗？                                 | 工具名精确匹配         |
| **D2 Tool Parameter**   | `ToolParamEvaluator`  | 工具参数对吗？`{"city":"北京"}` vs `{"city":"背景"}` | JSON Schema + 语义匹配 |
| **D3 Answer Semantics** | `SemanticEvaluator`   | 回答的语义正确吗？                                   | 向量相似度             |
| **D4 Answer Quality**   | `LLMJudgeEvaluator`   | 强模型怎么看这份回答？                               | LLM 打分 + Rubric      |

### 第二类：推理质量 (Reasoning Quality)

| 维度                | 评估器                  | 核心问题                           | 评估方法                    |
| ------------------- | ----------------------- | ---------------------------------- | --------------------------- |
| **D5 Trajectory**   | `TrajectoryEvaluator`   | 推理路径高效吗？有没有重复调工具？ | 步骤冗余度 + 是否有 Thought |
| **D6 Plan Quality** | `PlanQualityEvaluator`  | 复杂任务的分步计划合理吗？         | 步数效率 + 工具覆盖度       |
| **D7 Adaptability** | `AdaptabilityEvaluator` | 工具失败时能切换策略吗？           | 策略多样性 + 重试冗余度     |
| **D8 Consistency**  | `ConsistencyEvaluator`  | 同样输入跑 3 次，答案一致吗？      | 多次运行方差分析            |

### 第三类：安全与边界 (Safety & Boundaries)

| 维度                    | 评估器                    | 核心问题                            | 评估方法                         |
| ----------------------- | ------------------------- | ----------------------------------- | -------------------------------- |
| **D9 Tool Abuse**       | `ToolAbuseEvaluator`      | 调了禁止的工具吗？传了 SQL 注入吗？ | 14 种危险模式正则 + 禁止工具列表 |
| **D10 Scope Adherence** | `ScopeAdherenceEvaluator` | Agent 越权了吗？角色边界守住了吗？  | 工具权限校验 + 越权行为检测      |

### 第四类：可信度与效率 (Reliability & Efficiency)

| 维度                 | 评估器                     | 核心问题                 | 评估方法                           |
| -------------------- | -------------------------- | ------------------------ | ---------------------------------- |
| **Faithfulness**     | `FaithfulnessEvaluator`    | Agent 编造工具结果了吗？ | 交叉比对 tool output vs agent 声称 |
| **Token Efficiency** | `TokenEfficiencyEvaluator` | 花了多少 Token？值不值？ | Token 消耗 / baseline 比值         |

### 评测矩阵总览

```
                    ┌─────────────────────────────┐
                    │     AgentFlow 评测矩阵       │
                    │    10 Dimensions            │
                    └─────────────┬───────────────┘
            ┌──────────┬─────────┼─────────┬──────────┐
            ▼          ▼         ▼         ▼          ▼
        Task Comp   Reasoning  Safety    Reliability  Quality
        D1 工具选择  D5 轨迹    D9 工具滥用  Faithfulness  D4 LLM Judge
        D2 参数准确  D6 计划    D10 越权    Token Eff.
        D3 语义相似  D7 自适应
                    D8 一致性
```

## 技术栈

| 层级         | 技术                                  |
| ------------ | ------------------------------------- |
| Agent 框架   | LangGraph                             |
| 编排引擎     | Go + gRPC + OpenTelemetry             |
| Agent 运行时 | Python (FastAPI, LangChain)           |
| 序列化       | Protocol Buffers                      |
| Trace 存储   | Pebble (Go)                           |
| CLI          | Typer + Rich                          |
| Dashboard    | React + Recharts                      |
| 观测性       | OpenTelemetry (Jaeger / Grafana 兼容) |

## 为什么编排引擎用 Go？

1. **高性能** — goroutine 天然适合 DAG 拓扑排序 + 并发执行
2. **可靠性** — 静态类型 + 编译检查，减少生产环境意外
3. **部署简单** — 单个二进制分发，不依赖 Python 环境

编排引擎负责"什么时候执行哪个节点、失败了怎么办"；Agent Runtime 负责"怎么跟 LLM 交互、怎么调用工具"。两者通过 gRPC 通信，可独立部署和扩展。

## 快速开始

> 🚧 项目正在开发中，预计 2026 年 7 月发布 v0.1.0

```bash
# 安装 AgentFlow
pip install agentflow

# 创建新项目
agentflow new my-agent-app

# 启动开发服务器
agentflow dev

# 运行评测
agentflow eval

# 查看 Trace
agentflow trace
```

## 项目路线图

| 周次           | 里程碑                                                      |
| -------------- | ----------------------------------------------------------- |
| W1 (6/1-6/7)   | Go 环境搭建 + 项目骨架 + Workflow DSL 设计                  |
| W2 (6/8-6/14)  | DAG Executor (Go) + 熔断/重试/超时 + Agent Runtime (Python) |
| W3 (6/15-6/21) | Tool Registry + Memory Manager + Eval Engine                |
| W4 (6/22-6/28) | Trace Store + Benchmark Suite + CLI + 集成测试              |
| W5 (6/29-7/5)  | Dashboard + 文档 + Demo 视频 + 博客                         |

## 文档

完整文档请访问 [AgentFlow Docs](https://agentflow.dev)（即将上线）

## 许可证

[MIT](LICENSE)
