# Trace Prompt Snapshot — 设计文档

日期: 2026-07-13 | 状态: ✅ 已实现

## 目标

在 AgentFlow trace 系统中，为每次 query 的每一轮 LLM 调用记录完整 prompt（messages 快照），
使得可以精确回放 Agent 在每一轮看到了什么上下文。

## 方案

选择方案 A（完整快照）：在 `AgentTurn` 新增 `messages_snapshot` 字段，
在 `_execute_tool_loop()` 中每次调用 LLM 前 `copy.deepcopy(messages)` 存入。

### 改动范围（2 个文件）

**1. `agentflow/trace/tracer.py`**

- `AgentTurn` 新增字段 `messages_snapshot: list[dict]`
- `WorkflowTrace.to_dict()` 的 turns 导出中包含 `messages_snapshot`

**2. `agentflow/runtime/thinking/base.py`**

- 新增 `import copy`
- 重构 `_execute_tool_loop()` 循环体：每轮迭代先创建 `AgentTurn` + 捕获快照，再调用 LLM，最后回填结果
- 旧逻辑在收到 tool_calls 后才创建 turn，导致 final_answer 和 tool_call 被合并到同一个 turn；新逻辑每轮 LLM 调用独立成 turn

### 副作用

- `AgentTrace.total_turns` 语义变化：旧逻辑将 tool_call + final_answer 合并为 1 个 turn，新逻辑每轮 LLM 调用独立成 turn，`total_turns` 会增加（更准确）
- 相关测试断言已更新

## 验证

- 所有 218 个测试通过
- 单 Agent（ReAct/CoT/PlanExecute）每轮 turn 都有完整 messages_snapshot
- 多 Agent 编排通过传入 `agent_trace` 共用同一套逻辑
- `to_dict()` 完整导出，不截断
