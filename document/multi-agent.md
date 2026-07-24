# 多 Agent 编排

AgentFlow 的编排系统支持将多个 Agent 组织为 DAG 工作流，并发执行、条件分支、消息通信、全链路追踪。

## 架构

```
┌─────────────────────────────────────────┐
│            Workflow DSL                  │
│  Workflow → [Node, Node, ...]           │
│  Edge(condition) → 控制流                │
│  YAML 序列化 / Mermaid 可视化            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           DAGExecutor                    │
│                                          │
│  execute(workflow, agent_fn, tool_fn)    │
│    ├─ parallel_groups()  拓扑分层        │
│    ├─ 逐层 asyncio.gather() 并发         │
│    ├─ 边条件评估 → 决定下游是否触发       │
│    ├─ Node.loop → 循环执行              │
│    ├─ Fallback → SKIP / DEFAULT / RAISE │
│    └─ Subgraph → 递归嵌套               │
│                                          │
│  集成: MessageBus / Hooks / Trace        │
└─────────────────────────────────────────┘
```

## 核心组件

### Workflow DSL — 声明式 DAG 定义

四种节点类型覆盖所有编排场景：

```python
from agentflow.dsl.types import (
    Workflow, Node, NodeKind, Edge,
    AgentConfig, ToolConfig, HumanConfig, LoopConfig,
    FallbackPolicy,
)

workflow = Workflow(
    name="customer_support",
    nodes=[
        Node(id="classify", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o", prompt="分类用户意图",
                               tools=["sentiment"], thinking="react")),
        Node(id="faq", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o-mini", prompt="从 FAQ 库回答",
                               tools=["search_faq"])),
        Node(id="escalate", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o", prompt="升级给人工客服",
                               tools=["create_ticket"])),
        Node(id="approve", kind=NodeKind.HUMAN,
             human=HumanConfig(prompt="确认升级请求", timeout_sec=3600)),
    ],
    edges=[
        Edge(from_node="classify", to_node="faq",
             condition="score > 0.5"),
        Edge(from_node="classify", to_node="escalate",
             condition="score <= 0.5"),
        Edge(from_node="escalate", to_node="approve"),
    ],
)
```

**四种节点类型：**

| Kind       | 配置类          | 用途                                             |
| ---------- | --------------- | ------------------------------------------------ |
| `AGENT`    | `AgentConfig`   | LLM 驱动：模型 + prompt + 工具 + 思考策略 + 记忆 |
| `TOOL`     | `ToolConfig`    | 纯函数调用，零 Token 开销                        |
| `HUMAN`    | `HumanConfig`   | 人工确认（Human-in-the-Loop），支持超时回退      |
| `SUBGRAPH` | `subgraph` 引用 | 嵌套另一个 Workflow，递归执行                    |

**控制流在边上：**

```python
Edge(from_node="step1", to_node="step2",
     condition="score > 0.8")           # 引用上游节点的输出字段
Edge(from_node="step1", to_node="step2",
     condition="step1.score > 0.8")      # 引用任意节点的输出字段
```

`condition` 为空 → 无条件流转；非空 → 正则解析比较表达式（支持 `>`, `<`, `>=`, `<=`, `==`, `!=`）。可引用上游节点字段或指定 `node_id.field`。

**循环配置：**

```python
Node(id="improve", kind=NodeKind.AGENT,
     loop=LoopConfig(max_iterations=5, condition="output.score > 0.9"))
```

**降级策略：**

```python
Node(id="step", fallback=FallbackPolicy.SKIP)          # 失败跳过
Node(id="step", fallback=FallbackPolicy.DEFAULT_VALUE,
     default_value="fallback answer")                    # 返回默认值
Node(id="step", fallback=FallbackPolicy.FALLBACK_NODE,
     fallback_node_id="fallback_step")                   # 跳转到降级节点
```

**YAML 序列化：**

```yaml
# workflow.yaml
name: customer_support
nodes:
  - id: classify
    kind: agent
    agent:
      model: gpt-4o
      prompt: 分类用户意图
      thinking: react
  - id: faq
    kind: agent
    agent:
      model: gpt-4o-mini
      prompt: 从 FAQ 库回答
      tools: [search_faq]
edges:
  - from: classify
    to: faq
    condition: "score > 0.5"
  - from: classify
    to: escalate
    condition: "score <= 0.5"
```

### DAGExecutor — 异步编排引擎

```python
from agentflow.runtime.orchestrator import DAGExecutor

executor = DAGExecutor(default_timeout_ms=120_000, workflows={"sub_wf": sub_workflow})

results, trace = await executor.execute(
    workflow,
    agent_fn=my_agent.run,      # async (node_id, inputs, stream_cb) -> str
    tool_fn=toolkit.execute,    # async (tool_name, inputs) -> str
    human_fn=console.ask,       # async (prompt, inputs) -> str
    hooks=MyHooks(),
)
```

**调度策略**：

1. `parallel_groups()` 拓扑排序 → 分层（同一层节点无依赖关系）
2. 每层内部 `asyncio.gather()` 并发执行
3. 每层执行前评估 Edge condition → 决定下游节点是否触发
4. `Node.loop` 非空时循环执行直到条件满足
5. 失败时按 `FallbackPolicy` 降级
6. SUBGRAPH 节点递归调用 `execute()` 嵌套执行

**Subgraph 支持**：

```python
executor = DAGExecutor(workflows={
    "sub_analysis": Workflow(name="sub_analysis", nodes=[...], edges=[...])
})
# 主 Workflow 中引用
Node(id="deep_analysis", kind=NodeKind.SUBGRAPH, subgraph="sub_analysis")
```

### MessageBus — Agent 间消息传递

显式的 Agent-to-Agent 通信，与共享记忆互补。

```python
from agentflow.runtime.message_bus import MessageBus, AgentMessage, Intent

bus = MessageBus()

# Agent A 委派任务给 Agent B
bus.send(AgentMessage(
    from_agent="planner",
    to_agent="worker",
    intent=Intent.DELEGATE,
    payload={"task": "查询库存", "params": {"sku": "A001"}},
))

# Agent B 完成任务并返回结果
bus.send(AgentMessage(
    from_agent="worker",
    to_agent="planner",
    intent=Intent.TASK_COMPLETE,
    payload={"result": {"stock": 42}},
))

# Agent B 收到自己未读的消息
msgs = bus.receive("worker")

# 广播
bus.broadcast(from_agent="orchestrator", intent="info",
              payload={"status": "phase_2_starting"})
```

**六种消息意图：**

| Intent               | 含义                         |
| -------------------- | ---------------------------- |
| `DELEGATE`           | 委派任务给其他 Agent         |
| `TASK_COMPLETE`      | 任务完成，携带结果           |
| `NEED_CLARIFICATION` | 请求澄清                     |
| `HANDOFF`            | 将整个会话转交给另一个 Agent |
| `ERROR`              | 执行出错通知                 |
| `INFO`               | 一般性通知                   |

### Hooks — 生命周期扩展

四级可插拔钩子，用户继承 `ExecutionHooks` 按需重写。

```python
from agentflow.runtime.hooks import ExecutionHooks, HookContext

class MyHooks(ExecutionHooks):
    async def on_workflow_start(self, workflow, ctx: HookContext):
        ctx.shared["start_time"] = time.time()

    async def on_node_start(self, node, ctx: HookContext):
        print(f"[{node.id}] 开始执行")

    async def on_tool_call(self, tool_name, inputs, ctx: HookContext):
        if tool_name == "delete_db":
            raise PermissionError("不允许调用危险工具")

    async def on_stream(self, event, ctx: HookContext):
        print(f"[{event.node_id}] {event.type}: {event.content[:50]}")

    async def on_workflow_end(self, workflow, trace, ctx: HookContext):
        elapsed = time.time() - ctx.shared["start_time"]
        print(f"总耗时: {elapsed:.2f}s")
```

**完整钩子列表：**
`on_workflow_start` → `on_group_start` → `on_node_start` → `on_tool_call` / `on_tool_result` / `on_stream` → `on_node_end` → `on_group_end` → `on_workflow_end`

`HookContext.shared` 是贯穿整个生命周期的共享字典，可在任意钩子中读写。

### Trace 模型 — 全链路追踪

统一 Trace 模型覆盖从单 Agent 推理到多 Agent 编排的完整链路。

```
WorkflowTrace
├── dag_groups: [[node_id, ...], ...]    并行分组
├── node_traces: {node_id: AgentTrace}   每节点的 Agent 轨迹
├── message_flow: [MessageRecord]        消息时间线
└── summary: WorkflowSummary             汇总
```

**AgentTrace**（单节点内）：

```python
@dataclass
class AgentTrace:
    agent_id: str
    turns: list[AgentTurn]               # 每轮思考-行动-观察
    memory_retrieved: list[dict]         # pre_turn 检索到的记忆
    memory_stored: list[dict]            # post_turn 提取的事实
    total_turns: int
    total_tool_calls: int
    total_tokens: dict                   # {input, output}
    total_duration_ms: int
    success: bool
```

**AgentTurn**（单轮）：

```python
@dataclass
class AgentTurn:
    turn: int
    thinking: str                        # 思考内容
    tool_calls: list[ToolCallRecord]     # 工具调用
    final_answer: str
    finish_reason: str                   # stop / tool_calls / length
    reasoning: str                       # DeepSeek R1 / OpenAI o1 CoT
    tokens: dict                         # 本轮 Token 消耗
    messages_snapshot: list[dict]        # 本轮 LLM 输入快照
    tools_snapshot: list[dict]           # 本轮可用工具快照
```

**A/B Diff 对比**：

```python
diff = trace_v1.diff(trace_v2)
# → {
#     "trace_old": ..., "trace_new": ...,
#     "changes": [{"node": "researcher", "change": "duration",
#                   "old_ms": 1200, "new_ms": 850}, ...],
#     "summary_diff": {"duration_delta_ms": -350, "failed_delta": 0},
#   }
```

## 使用示例

### 多 Agent 工作流完整示例

```python
from agentflow.dsl.types import Workflow, Node, NodeKind, Edge, AgentConfig
from agentflow.runtime.orchestrator import DAGExecutor

# 定义 Workflow
wf = Workflow(
    name="research_report",
    nodes=[
        Node(id="researcher", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o", prompt="搜索并收集信息",
                               tools=["web_search", "fetch_page"],
                               thinking="plan_execute")),
        Node(id="writer", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o", prompt="撰写报告",
                               thinking="cot")),
        Node(id="reviewer", kind=NodeKind.AGENT,
             agent=AgentConfig(model="gpt-4o-mini", prompt="审校报告",
                               tools=["check_facts"])),
    ],
    edges=[
        Edge(from_node="researcher", to_node="writer"),
        Edge(from_node="writer", to_node="reviewer"),
    ],
)

# 执行
executor = DAGExecutor()
results, trace = await executor.execute(
    wf,
    agent_fn=build_and_run_agent,
)

# 分析
for node_id, agent_trace in trace.node_traces.items():
    print(f"{node_id}: {agent_trace.total_turns} turns, "
          f"{agent_trace.total_tool_calls} tool calls, "
          f"{agent_trace.total_duration_ms}ms")

# 定位瓶颈
print(f"关键路径: {trace.summary.critical_path}")
print(f"瓶颈节点: {trace.summary.bottleneck}")
```

### 带条件的 Workflow

```python
nodes = [
    Node(id="analyze", kind=NodeKind.AGENT,
         agent=AgentConfig(model="gpt-4o", prompt="分析输入内容")),
    Node(id="simple_reply", kind=NodeKind.AGENT,
         agent=AgentConfig(model="gpt-4o-mini", prompt="简单回复")),
    Node(id="deep_research", kind=NodeKind.AGENT,
         agent=AgentConfig(model="gpt-4o", prompt="深度调研",
                           tools=["web_search"])),
]
edges = [
    Edge("analyze", "simple_reply", condition="score > 0.5"),
    Edge("analyze", "deep_research", condition="score <= 0.5"),
]
```

## 动态路由（Dynamic Routing）

静态 DAG 要求所有 Agent 节点和边在 Workflow 定义时就固定好。动态路由则允许**运行时**根据任务语义自动选择最合适的专家 Agent，并支持专家之间的 handoff 转交。

### 核心组件

#### AgentRegistry — 能力注册与语义匹配

每个专家 Agent 注册时声明自己的能力卡：

```python
from agentflow.runtime.agent_registry import AgentCapability, AgentRegistry

registry = AgentRegistry()
registry.register(AgentCapability(
    agent_id="refund_expert",
    description="处理退款、账单、支付纠纷",
    tools=["lookup_refund", "process_refund"],
    examples=["如何退款？", "我的退款在哪里？"],
    priority=0,
))
registry.register(AgentCapability(
    agent_id="shipping_expert",
    description="处理发货、物流追踪、配送问题",
    tools=["track_order", "update_address"],
    examples=["我的包裹在哪？", "修改收货地址"],
    priority=0,
))
registry.register(AgentCapability(
    agent_id="general_cs",
    description="通用客服、问候、简单咨询",
    tools=[],
    examples=["你好", "你们几点下班？"],
))

# 语义匹配：Jaccard 相似度 + priority 加权
candidates = registry.match("我想退款我的订单", top_k=3)
# → [(refund_expert, 0.85), (general_cs, 0.20), (shipping_expert, 0.15)]
```

**匹配策略**：v1 使用 Jaccard 关键词相似度，priority 每点 +1% 加权。升级路径预留 embedding 向量检索接口。

#### RoutingStrategy — 路由编排策略

新增 `ThinkingMode.ROUTING`，与 ReAct / CoT / PlanExecute 同级，核心状态机：

```
ANALYZE → ROUTE → EXECUTE → CHECK_HANDOFF
   ^                            |
   |        (handoff)           |
   +--- back to ANALYZE --------+
   |        (no handoff)        |
   +--- DONE -------------------+
```

- **ANALYZE**：`registry.match()` 语义匹配，返回 top-3 候选
- **ROUTE**：LLM 从候选中选择最优专家（带理由）
- **EXECUTE**：直接调用 `expert.run()`，保留完整的记忆/工具/思考链
- **CHECK_HANDOFF**：检测专家是否发出 handoff 信号，是则重新路由

#### Handoff 协议 — Agent 间任务转交

专家在无法完成任务时，通过标准化的文本块发出转交信号：

```
---HANDOFF---
reason: 跨境支付不在我的范围内
suggest: 处理国际汇款和跨境支付的代理
context: 用户需要向英国账户汇款 £500，账户已验证
---END---
```

```python
from agentflow.runtime.handoff import parse_handoff_block

handoff = parse_handoff_block(expert_output)
if handoff:
    print(f"Reason: {handoff.reason}")
    print(f"Suggested: {handoff.suggested_agent}")
    print(f"Partial result: {handoff.partial_result}")
```

### 用法：Builder 一行开启

```python
from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.thinking import ThinkingMode

router = (
    AgentBuilder("support_router")
    .with_llm(llm_client)
    .with_registry(registry)
    .with_experts({
        "refund_expert": refund_agent,
        "shipping_expert": shipping_agent,
        "general_cs": cs_agent,
    })
    .with_thinking(ThinkingMode.ROUTING)
    .with_max_iterations(10)
    .build_sync()
)

result = await router.run("我想退款我的订单 #12345")
# → 自动路由到 refund_expert，输出退款结果
```

### 与静态 DAG 的关系

静态 DAG 和动态路由**互补**，不是互斥：

| 场景                             | 推荐方式                |
| -------------------------------- | ----------------------- |
| 流程固定（订单处理、审批流水线） | 静态 DAG                |
| 任务多样、需要语义匹配           | 动态路由                |
| 混合场景                         | DAG 中嵌入 ROUTING 节点 |

动态路由的 Router 本身是一个 AGENT 节点，可以放进 DAG 的任意位置。它使用 `ThinkingMode.ROUTING` 思考策略，与 ReAct / CoT 一样通过 `AgentBuilder` 构建。

## 最佳实践

1. **先画 DAG 再写代码**：用 `agentflow/dsl/visualizer.py` 生成 Mermaid 图，确认拓扑正确
2. **条件分支用具体的关键词**：`"simple" in output` 比 `output == "simple"` 更容错
3. **Human 节点必须设超时**：避免 Workflow 因等待人工输入永久挂起
4. **AGENT 节点用不同模型**：计算密集用 gpt-4o-mini，推理密集用 gpt-4o，控制成本
5. **编排器传 trace 给 Agent**：多 Agent 场景下编排器创建 `WorkflowTrace`，每个 Agent 拿到注入的 `AgentTrace`，确保追踪完整
6. **Capability 描述要具体**：`description` 和 `examples` 的质量直接决定路由准确度，用自然语言写清楚"能做什么、不能做什么"
7. **Handoff 上限设为 3 次**：防止无限转交，Router 在 `max_handoffs` 次后自动终止并返回部分结果
