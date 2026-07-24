# 评测体系

AgentFlow 内置 10 维评测矩阵，覆盖任务完成、推理质量、安全边界、效率成本四大类。EvalSuite 提供批量运行、Trace 关联、低分归因和 A/B 回归对比。

## 架构

```
┌─────────────────────────────────────────┐
│              EvalSuite                    │
│  run(agent_fn) → SuiteReport             │
│  diagnose(report) → 低分归因              │
│  compare(old, new) → A/B 回归检测         │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    ▼             ▼             ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Task    │ │Reasoning │ │Safety    │ │Reliability   │
│D1-D4   │ │D5-D8     │ │D9-D10    │ │Faithfulness  │
│        │ │          │ │          │ │Token Eff.    │
└────────┘ └──────────┘ └──────────┘ └──────────────┘
```

## 10 维评测矩阵

### 第一类：任务完成 (Task Completion)

| 维度                    | 评估器                | 核心问题               | 评估方法               |
| ----------------------- | --------------------- | ---------------------- | ---------------------- |
| **D1 Tool Selection**   | `ExactMatchEvaluator` | Agent 选对工具了吗？   | 工具名精确匹配         |
| **D2 Tool Parameter**   | `ToolParamEvaluator`  | 工具参数对吗？         | JSON Schema + 语义匹配 |
| **D3 Answer Semantics** | `SemanticEvaluator`   | 回答的语义正确吗？     | 向量余弦相似度         |
| **D4 Answer Quality**   | `LLMJudgeEvaluator`   | 强模型怎么看这份回答？ | LLM 打分 + Rubric      |

### 第二类：推理质量 (Reasoning Quality)

| 维度                | 评估器                  | 核心问题                           | 评估方法                    |
| ------------------- | ----------------------- | ---------------------------------- | --------------------------- |
| **D5 Trajectory**   | `TrajectoryEvaluator`   | 推理路径高效吗？有没有重复调工具？ | 步骤冗余度 + Thought 存在性 |
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

## 核心组件

### EvalSuite — 评测套件运行器

```python
from agentflow.eval.suite import EvalSuite, EvalCase
from agentflow.eval.exact_match import ExactMatchEvaluator
from agentflow.eval.semantic import SemanticEvaluator

suite = EvalSuite(
    name="weather_agent_v1",
    cases=[
        EvalCase(id="basic", input="北京天气",
                 expected="晴", evaluator=ExactMatchEvaluator()),
        EvalCase(id="semantic", input="上海天气怎么样",
                 expected="上海今天晴天，25°C",
                 evaluator=SemanticEvaluator()),
    ],
)

report = await suite.run(lambda inp: agent.run(inp))
print(f"通过率: {report.pass_rate:.1%} ({report.passed}/{report.total})")
```

### 各 Evaluator 详解

**D1 ExactMatchEvaluator** — 精确匹配，支持 JSON 比较和空白归一化：

```python
from agentflow.eval.exact_match import ExactMatchEvaluator
evaluator = ExactMatchEvaluator()
result = await evaluator.evaluate("北京", "北京")  # score=1.0
```

**D2 ToolParamEvaluator** — 逐字段比较工具参数，数字容差、字符串语义包含：

```python
from agentflow.eval.tool_param import ToolParamEvaluator
evaluator = ToolParamEvaluator()
result = await evaluator.evaluate(
    '{"city": "北京", "days": 3}',
    '{"city": "背景", "days": 3}',  # typo: 背景 → 北京
)
# score ~0.8 (city 不匹配，days 匹配)
```

**D3 SemanticEvaluator** — 语义向量相似度：

```python
from agentflow.eval.semantic import SemanticEvaluator
evaluator = SemanticEvaluator()
result = await evaluator.evaluate(
    "今天北京天气晴朗，适合出行",
    "北京今日晴好，宜外出",
)
# score ~0.9 (语义相近)
```

**D4 LLMJudgeEvaluator** — 用强模型按 Rubric 打分：

```python
from agentflow.eval.llm_judge import LLMJudgeEvaluator
evaluator = LLMJudgeEvaluator(
    llm_client=gpt4_client,
    rubric="评估回答的准确性、完整性和可读性，每项 1-5 分",
)
result = await evaluator.evaluate(expected_answer, actual_answer)
```

**D5 TrajectoryEvaluator** — 评估推理路径效率：

```python
from agentflow.eval.trajectory import TrajectoryEvaluator
evaluator = TrajectoryEvaluator()
result = await evaluator.evaluate(golden_trajectory, actual_trajectory)
# 检查：是否有 Thought、步骤是否去重、最终答案是否存在
```

**D6 PlanQualityEvaluator** — 评估分步计划质量：

```python
from agentflow.eval.plan_quality import PlanQualityEvaluator
evaluator = PlanQualityEvaluator()
result = await evaluator.evaluate(expected_plan, actual_plan)
# 检查：步数效率、工具覆盖度、最终答案
```

**D7 AdaptabilityEvaluator** — 评估失败时的策略切换能力：

```python
from agentflow.eval.adaptability import AdaptabilityEvaluator
evaluator = AdaptabilityEvaluator()
result = await evaluator.evaluate(expected_trajectory, actual_trajectory)
# 检查：策略多样性、是否避免无效重试、是否降级
```

**D8 ConsistencyEvaluator** — 多次运行的方差分析：

```python
from agentflow.eval.consistency import ConsistencyEvaluator
evaluator = ConsistencyEvaluator()
result = await evaluator.evaluate(reference_run, multi_run_results)
# 检查：工具集、输出长度、步数的方差
```

**D9 ToolAbuseEvaluator** — 14 种危险模式检测：

```python
from agentflow.eval.tool_abuse import ToolAbuseEvaluator
evaluator = ToolAbuseEvaluator(forbidden_tools=["delete_db", "exec"])
result = await evaluator.evaluate("", agent_output_with_tool_calls)
# 检测：SQL 注入、路径遍历、XSS、权限提升、命令注入等
```

**D10 ScopeAdherenceEvaluator** — 越权检测：

```python
from agentflow.eval.scope_adherence import ScopeAdherenceEvaluator
evaluator = ScopeAdherenceEvaluator(
    role_definition="客服 Agent，只能查询和查看",
    allowed_tools=["search", "read"],
    forbidden_actions=["delete", "exec"],
)
result = await evaluator.evaluate("", agent_output)
# 检查：是否调用了未授权的工具、是否执行了越界操作
```

**FaithfulnessEvaluator** — 幻觉检测：

```python
from agentflow.eval.faithfulness import FaithfulnessEvaluator
evaluator = FaithfulnessEvaluator()
result = await evaluator.evaluate(tool_outputs, agent_claim)
# 检查：数字是否一致、引号内容是否真实、是否有编造陈述
```

**TokenEfficiencyEvaluator** — Token 效率：

```python
from agentflow.eval.token_efficiency import TokenEfficiencyEvaluator
evaluator = TokenEfficiencyEvaluator(baseline_tokens=2000)
result = await evaluator.evaluate("", actual_result)
# score = min(1.0, baseline / actual_tokens)
```

### diagnose() — 低分归因

找出得分低于阈值的 case，按分数升序排列，关联 trace_id 定位问题。

```python
low_cases = suite.diagnose(report, min_score=0.5)
for case in low_cases:
    print(f"{case['case_id']}: score={case['score']:.2f}, "
          f"reason={case['reason']}, trace={case['trace_id']}")
```

输出示例：

```
tool_param_03: score=0.32, reason=参数 city 不匹配: 期望'上海' 实际'伤害',
               trace=eval:weather_v1:tool_param_03
```

### compare() — A/B 回归检测

对比两次评测的 SuiteReport，量化改进和回归。

```python
diff = suite.compare(report_v1, report_v2)
# → {
#     "total": 20, "improved": 3, "regressed": 1, "unchanged": 16,
#     "old_pass_rate": 0.75, "new_pass_rate": 0.85,
#   }
```

## Benchmark Suite

内置三套标准化评测集，可直接运行。

```python
from agentflow.benchmark.tool_use import ToolUseBenchmark
from agentflow.benchmark.multi_hop_qa import MultiHopQABenchmark
from agentflow.benchmark.long_context import LongContextBenchmark

# 工具使用基准：评估 Agent 选择合适的工具和参数的能力
bench = ToolUseBenchmark()
cases = bench.get_cases()           # → [ToolUseCase, ...]
report = bench.run_benchmark(agent_fn)  # sync

# 多跳推理基准：评估 Agent 串联多步信息的能力
bench = MultiHopQABenchmark()
cases = bench.get_cases()           # → [MultiHopCase, ...]

# 长上下文基准：评估 Agent 处理大量上下文的能力
bench = LongContextBenchmark()
cases = bench.get_cases()           # → [LongContextCase, ...]
```

## 使用示例

### 完整评测流程

```python
from agentflow.eval.suite import EvalSuite, EvalCase
from agentflow.eval.exact_match import ExactMatchEvaluator
from agentflow.eval.tool_param import ToolParamEvaluator
from agentflow.eval.semantic import SemanticEvaluator
from agentflow.eval.llm_judge import LLMJudgeEvaluator
from agentflow.eval.trajectory import TrajectoryEvaluator

# 定义评测套件
suite = EvalSuite(name="customer_agent_v1", cases=[
    EvalCase(id="greeting", input="你好", expected="你好！有什么可以帮您？",
             evaluator=SemanticEvaluator()),
    EvalCase(id="tool_select", input="查询订单 12345",
             expected="query_order", evaluator=ExactMatchEvaluator()),
    EvalCase(id="tool_param", input="退款订单 12345，金额 50",
             expected='{"order_id":"12345","amount":50}',
             evaluator=ToolParamEvaluator()),
    EvalCase(id="quality", input="我的订单为什么还没到？",
             expected="查询物流信息并给出预计到达时间",
             evaluator=LLMJudgeEvaluator(llm_client=judge_llm,
                 rubric="评估客服回复的共情能力、信息准确性、解决效率")),
])

# 运行评测
async def run_agent(input_text: str) -> str:
    result = await agent.run(input_text)
    return result.output

report = await suite.run(run_agent)
print(f"Pass rate: {report.pass_rate:.1%}")

# 定位低分 case
for case in suite.diagnose(report, min_score=0.6):
    print(f"  [{case['case_id']}] score={case['score']:.2f}: {case['reason']}")
```

### A/B 对比

```python
# 修改 prompt 后重新评测
report_v1 = await suite_v1.run(run_agent)
agent.set_prompt(new_prompt)
report_v2 = await suite_v2.run(run_agent)

diff = suite_v1.compare(report_v1, report_v2)
print(f"改进: {diff['improved']}, 回退: {diff['regressed']}, "
      f"通过率: {diff['old_pass_rate']:.1%} → {diff['new_pass_rate']:.1%}")
```

## 最佳实践

1. **先跑 Benchmark 建立基线**：在改 prompt 之前跑一遍 Benchmark Suite，知道当前水平
2. **评测维度组合使用**：单靠 ExactMatch 不够，配合 Semantic 和 LLM Judge 覆盖准确性和语义
3. **A/B 对比每次改 prompt 都做**：compare() 告诉你改 prompt 到底是改进还是回退
4. **低分 case 用 diagnose() 定位**：关联 trace_id 查看具体哪一步出了问题
5. **LLM Judge 选更强的模型**：除非特殊情况，评测模型应该比被测模型更强
