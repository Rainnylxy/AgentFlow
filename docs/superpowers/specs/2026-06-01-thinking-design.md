# Thinking 引擎详细设计

> 日期: 2026-06-01 | 状态: Draft | 关联: [[2026-06-01-agent-builder-design]]

## 一、目标

让 Agent 支持多种思考模式——ReAct 快速工具调用、Plan-Execute 复杂规划、CoT 深度推理、Reflection 自我修正。Agent 可以自适应选择最优模式，也可以在运行中自我反省和纠错。

## 二、核心抽象

### 2.1 统一接口

```python
class ThinkingStrategy(ABC):
    """思考策略的抽象基类。所有模式实现此接口。"""

    @abstractmethod
    async def run(self, context: ThinkContext) -> ThinkResult:
        """执行思考并返回结果。"""
        ...

@dataclass
class ThinkContext:
    """思考上下文：Agent 所需的所有运行时信息。"""
    user_input: str
    system_prompt: str
    messages: list[Message]         # Working Memory
    tools: list[ToolSchema]         # 可用工具
    llm_client: LLMClient
    memory: MemoryManager
    max_iterations: int = 10

@dataclass
class ThinkResult:
    """思考结果。"""
    output: str
    tool_calls: list[dict]
    steps: list[dict]               # 每步的详细记录
    reflection_notes: list[str]     # 反思记录（如果有的话）
    mode_used: str                  # 实际使用的模式
```

### 2.2 ThinkingEngine——策略的编排者

```python
class ThinkingEngine:
    """管理多个策略，根据模式选择或自适应路由。"""

    def __init__(self, mode: ThinkingMode, llm_client: LLMClient):
        self.mode = mode
        self.llm_client = llm_client

    async def run(self, context: ThinkContext) -> ThinkResult:
        strategy = self._resolve_strategy(context)
        return await strategy.run(context)

    def _resolve_strategy(self, context: ThinkContext) -> ThinkingStrategy:
        if self.mode == ThinkingMode.ADAPTIVE:
            return AdaptiveRouter().route(context.user_input, context.tools)
        return self.mode.to_strategy(self.llm_client)
```

## 三、四种策略

### 3.1 ReActStrategy

```
循环：
  Thought: LLM 生成推理
  Action: 如果需工具 → 执行工具调用
  Observation: 工具结果反馈给 LLM
  Final: LLM 给出最终答案 → 退出
```

- **适用**：简单工具调用（查天气、查 KB、计算）
- **超时**：默认 10 轮，可配置
- **来自**：现有 `ReActAgent` 逻辑迁移

### 3.2 PlanExecuteStrategy

```
Phase 1 — Plan：
  "我需要完成的任务是 X，这需要以下步骤：1. ... 2. ... 3. ..."
  生成结构化步骤列表（每步包含：目标、所需工具、预期输出）

Phase 2 — Execute（逐步骤）：
  for each step:
    执行该步（可以是 ReAct 子循环）
    检查该步输出是否符合预期
    不符合 → 调整计划 / 重试

Phase 3 — Finalize：
  汇总所有步骤结果 → 生成最终答案
```

- **适用**：复杂多步任务（部署、数据分析、研究写作）
- **关键**：Plan 阶段生成的步骤必须明确、可执行、可验证

### 3.3 CoTStrategy（链式思考）

```
Phase 1 — Think（深度推理，不调工具）：
  "让我逐步分析这个问题...
   已知条件：...
   推理步骤：...
   中间结论：...（可能有多次自问自答）"

Phase 2 — Answer：
  基于推理输出最终答案
```

- **适用**：数学题、逻辑推理、需要严密论证的场景
- **关键**：Think 阶段不使用工具，纯靠 LLM 推理能力

### 3.4 ReflectionWrapper（反思装饰器）

```python
class ReflectionWrapper(ThinkingStrategy):
    """在任何策略外面包裹反思循环。"""

    def __init__(self, inner: ThinkingStrategy, max_reflections: int = 3):
        self.inner = inner
        self.max_reflections = max_reflections

    async def run(self, context: ThinkContext) -> ThinkResult:
        for i in range(self.max_reflections):
            result = await self.inner.run(context)

            # 三条检查
            fact_check = await self._check_facts(result, context)
            completeness = await self._check_completeness(result, context)
            strategy_ok = await self._check_strategy(result, context)

            notes = [fact_check, completeness, strategy_ok]
            result.reflection_notes.extend(notes)

            if all(n.passed for n in notes):
                return result  # 自我满意

            # 有问题 → 注入反馈，重试
            context.add_feedback([n.suggestion for n in notes if not n.passed])

        return result  # 反思深度用尽
```

**三条自我检查**：

| 检查项   | 问题                         | 方法                                        |
| -------- | ---------------------------- | ------------------------------------------- |
| 事实核查 | "工具结果和我声称的一致吗？" | 交叉比对 tool_calls 输出 vs agent 最终答案  |
| 完备性   | "用户的问题我都回答了吗？"   | 对比 user_input 关键点 vs 答案覆盖          |
| 策略     | "当前方法对吗？要换模式吗？" | 检查工具调用是否循环重复、是否该升级为 Plan |

## 四、自适应路由

```python
class AdaptiveRouter:
    COMPLEXITY_SIGNALS = {
        "multi_step":     ["first", "then", "step", "接下来", "然后", "之后"],
        "deep_reasoning": ["why", "prove", "calculate", "证明", "推导", "计算"],
        "safe_critical":  ["delete", "deploy", "charge", "删除", "部署", "扣款"],
    }

    def route(self, user_input: str, tools: list) -> ThinkingStrategy:
        signals = self._detect(user_input)

        # 高风险 + 多步 → 规划 + 深度反思
        if "safe_critical" in signals and "multi_step" in signals:
            return ReflectionWrapper(PlanExecuteStrategy(), max_reflections=3)

        # 高风险 → 即使是简单任务也要反思
        if "safe_critical" in signals:
            return ReflectionWrapper(ReActStrategy(), max_reflections=2)

        # 多步 → 先规划
        if "multi_step" in signals:
            return PlanExecuteStrategy()

        # 深度推理 → 链式思考
        if "deep_reasoning" in signals:
            return CoTStrategy()

        # 默认 → 快速 ReAct
        return ReActStrategy()
```

**注意**：自适应路由是第一版实现，后续可升级为让 LLM 自身做路由决策（更准确但更慢）。

## 五、使用方式

### 5.1 固定模式

```python
# ReAct
agent = AgentBuilder("...").with_thinking(ThinkingMode.REACT).build()

# Plan-Execute
agent = AgentBuilder("...").with_thinking(ThinkingMode.PLAN_EXECUTE).build()

# Plan-Execute + 3 层反思
agent = AgentBuilder("...").with_thinking(
    ThinkingMode.PLAN_EXECUTE.with_reflection(depth=3)
).build()
```

### 5.2 自适应（默认推荐）

```python
agent = AgentBuilder("...").with_thinking(ThinkingMode.ADAPTIVE).build()
# 不指定 .with_thinking() 时默认就是 ADAPTIVE
```

### 5.3 组合自定义

```python
agent = AgentBuilder("trader").with_thinking(
    ThinkingMode.REACT
    .with_reflection(depth=2)
    .with_safety_gate()  # 高风险操作前 LLM 必须显式确认
).build()
```

## 六、关键细节

### 6.1 反思数据与评测联动

反思过程中的 `reflection_notes` 全部记录在 Trace 中，作为：

- **D7 Adaptability**：Agent 是否换了策略、是否自我修正成功
- **D8 Consistency**：多次跑同一任务，反思是否能收敛到一致结果

### 6.2 模式切换的平滑性

ReflectionWrapper 在决定"换策略"时，不是从头开始——已执行的工具调用结果保留，新策略从当前状态继续。

### 6.3 成本控制

反思和自适应路由本身消耗 Token。通过 `max_reflections` 控制反思深度，通过信号量匹配（而非 LLM 调用）降低路由成本。

## 七、与现有代码的关系

| 现有模块                      | 处理                              |
| ----------------------------- | --------------------------------- |
| `react_agent.py` (ReActAgent) | 逻辑迁移到 `ReActStrategy`        |
| `agent.py` (BaseAgent.run)    | 简化为委托 `ThinkingEngine.run()` |
| `BaseAgent`                   | 保留，作为 Builder 的输出类型     |

## 八、待定内容

- 基于强化学习的策略选择优化
- 多 Agent 间的思考模式协调
- 用户可干预的思考过程（中途暂停、引导方向）
