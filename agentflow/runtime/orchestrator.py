"""纯 Python 异步 DAG 编排引擎 — v0.1。

用法:
    executor = DAGExecutor()
    results, trace = await executor.execute(
        workflow,
        agent_fn=my_agent.run,     # async callable for AGENT nodes
        tool_fn=toolkit.execute,   # async callable for TOOL nodes
        human_fn=console.ask,      # async callable for HUMAN nodes
    )

调度策略:
    - parallel_groups() 将 DAG 分层，层内 asyncio.gather() 并发
    - 边上的 condition 在每层执行前评估，决定下游节点是否触发
    - Node.loop 非空时，节点循环执行直到条件满足
    - FallbackPolicy: SKIP / DEFAULT_VALUE / FALLBACK_NODE / RAISE
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional, Union

from agentflow.dsl.types import Workflow, Node, NodeKind, FallbackPolicy
from agentflow.runtime.message_bus import MessageBus, AgentMessage
from agentflow.runtime.hooks import ExecutionHooks, HookContext, StreamEvent, StreamCallback
from agentflow.trace.tracer import WorkflowTrace, AgentTrace, AgentTurn, ToolCallRecord, MessageRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果 & 踪迹
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """单个节点的执行结果。"""
    node_id: str
    output: Any = None
    success: bool = True
    error: str = ""
    attempts: int = 1
    duration_ms: int = 0
    skipped_by_condition: bool = False


@dataclass
class ExecutionTrace:
    """一次 DAG 执行的轨迹。"""
    workflow_name: str
    total_duration_ms: int = 0
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    groups: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DAG 执行器
# ---------------------------------------------------------------------------

AgentFn = Callable[[str, dict, Optional[StreamCallback]], Awaitable[str]]
ToolFn = Callable[[str, dict], Awaitable[str]]
HumanFn = Callable[[str, dict], Awaitable[str]]


class DAGExecutor:
    """纯 Python 异步 DAG 编排引擎。"""

    def __init__(
        self,
        default_timeout_ms: int = 120_000,
        workflows: dict[str, Workflow] | None = None,
    ):
        self.default_timeout_ms = default_timeout_ms
        self._workflows: dict[str, Workflow] = workflows or {}  # subgraph 注册表

    def register_workflow(self, wf: Workflow) -> None:
        """注册一个可被 SUBGRAPH 节点引用的 Workflow。"""
        self._workflows[wf.name] = wf

    async def execute(
        self,
        workflow: Workflow,
        agent_fn: AgentFn | None = None,
        tool_fn: ToolFn | None = None,
        human_fn: HumanFn | None = None,
        message_bus: MessageBus | None = None,
        hooks: ExecutionHooks | None = None,
        stream: StreamCallback | None = None,
    ) -> "tuple[dict[str, NodeResult], WorkflowTrace]":
        """执行整个 Workflow DAG。

        hooks: 生命周期钩子，在节点/工具/组的关键节点回调。
        stream: 流式回调，agent_fn 内部的 emit 事件会传递到这里。
        """
        from agentflow.dsl.graph import parallel_groups

        hooks = hooks or ExecutionHooks()
        hctx = HookContext(workflow_name=workflow.name)

        # 包装 stream：同时路由到 hooks.on_stream 和外部回调
        async def _stream_wrapper(event: StreamEvent) -> None:
            await hooks.on_stream(event, hctx)
            if stream:
                await stream(event)

        # 统一使用 _stream_wrapper，agent_fn 收到的 emit 就是它
        _emit: StreamCallback | None = _stream_wrapper if (stream or type(hooks) != ExecutionHooks) else None

        await hooks.on_workflow_start(workflow, hctx)

        trace = WorkflowTrace.start(workflow_name=workflow.name)

        t0 = time.monotonic()
        groups = parallel_groups(workflow)
        trace.dag_groups = groups
        all_results: dict[str, NodeResult] = {}
        bus = message_bus or MessageBus()

        for group in groups:
            if not group:
                continue

            hctx.group = group
            await hooks.on_group_start(group, hctx)

            # 评估边条件——当前层哪些节点应跳过
            active = []
            for nid in group:
                if self._should_skip(nid, workflow, all_results):
                    all_results[nid] = NodeResult(
                        node_id=nid, success=True,
                        skipped_by_condition=True,
                    )
                else:
                    active.append(nid)

            if active:
                group_results = await asyncio.gather(*[
                    self._execute_node(
                        workflow, nid, all_results,
                        agent_fn, tool_fn, human_fn, bus, hooks, _emit, hctx, trace,
                    )
                    for nid in active
                ])
                for nr in group_results:
                    all_results[nr.node_id] = nr

            await hooks.on_group_end(group, hctx)

        # 导出消息流
        for msg in bus.all_messages():
            trace.message_flow.append(MessageRecord(
                timestamp=msg.timestamp,
                from_agent=msg.from_agent,
                to_agent=msg.to_agent,
                intent=msg.intent,
                payload=msg.payload,
            ))

        # 填 Trace 数据
        for nid, nr in all_results.items():
            at = trace.node_traces.get(nid)
            if at is None:
                at = AgentTrace(agent_id=nid)
                trace.node_traces[nid] = at
            at.success = nr.success
            at.error = nr.error
            if at.total_duration_ms == 0:
                at.total_duration_ms = nr.duration_ms

            if nr.skipped_by_condition:
                trace.summary.nodes_skipped += 1
            elif nr.success:
                trace.summary.nodes_executed += 1
            else:
                trace.summary.nodes_failed += 1

        trace.finish()
        await hooks.on_workflow_end(workflow, trace, hctx)
        return all_results, trace

    # ------------------------------------------------------------------
    # 边条件评估
    # ------------------------------------------------------------------

    def _should_skip(
        self, node_id: str, workflow: Workflow,
        results: dict[str, NodeResult],
    ) -> bool:
        """评估指向 node_id 的所有入边的条件。

        如果任意入边有条件且条件为 False → 跳过该节点。
        无条件边 = 无条件通过。
        """
        incoming = [e for e in workflow.edges if e.to_node == node_id]
        if not incoming:
            return False  # 入口节点

        for edge in incoming:
            if edge.condition:
                upstream = results.get(edge.from_node)
                if not upstream or not upstream.success:
                    return True
                if not self._eval_edge_condition(edge.condition, upstream, results):
                    return True
        return False

    def _eval_edge_condition(
        self, expr: str, upstream: NodeResult,
        all_results: dict[str, NodeResult],
    ) -> bool:
        """评估边上的条件表达式。

        支持:
          - 'True' / 'False'
          - 'node_id.score > 0.5'  ← 引用任意节点输出
          - 'score > 0.5'          ← 引用上游节点（省略 node_id）
        """
        import re
        expr = expr.strip()

        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False

        # 'node_id.field > value' 格式
        m = re.match(r"(\w+)\.(\w+)\s*(>|<|>=|<=|==|!=)\s*([\d.]+)", expr)
        if m:
            nid, field, op, val = m.groups()
            target = all_results.get(nid)
            if target and target.output:
                return self._compare(getattr(target, field, target.output), op, float(val))
            return False

        # 'field > value' 格式（默认引用上游节点）
        m = re.match(r"(\w+)\s*(>|<|>=|<=|==|!=)\s*([\d.]+)", expr)
        if m:
            field, op, val = m.groups()
            if upstream.output:
                return self._compare(getattr(upstream, field, upstream.output), op, float(val))
            return False

        return False

    @staticmethod
    def _compare(actual, op: str, expected) -> bool:
        if op == ">":  return actual > expected
        if op == "<":  return actual < expected
        if op == ">=": return actual >= expected
        if op == "<=": return actual <= expected
        if op == "==": return actual == expected
        if op == "!=": return actual != expected
        return False

    # ------------------------------------------------------------------
    # 单节点执行
    # ------------------------------------------------------------------

    async def _execute_node(
        self,
        workflow: Workflow,
        node_id: str,
        results_so_far: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
        bus: MessageBus,
        hooks: ExecutionHooks,
        stream: StreamCallback | None,
        hctx: HookContext,
        workflow_trace: WorkflowTrace,
    ) -> NodeResult:
        node = self._get_node(workflow, node_id)
        if node is None:
            return NodeResult(node_id=node_id, success=False, error=f"Node not found")

        # 确保 WorkflowTrace 中有该节点的 AgentTrace
        if node_id not in workflow_trace.node_traces:
            workflow_trace.node_traces[node_id] = AgentTrace(agent_id=node_id)

        hctx.node_id = node_id
        hctx.node_kind = node.kind.value
        await hooks.on_node_start(node, hctx)

        # 收集发给该节点的未读消息
        incoming = bus.receive(node_id)

        node_agent_trace = workflow_trace.node_traces.get(node_id)

        # 循环节点 —— wrapper
        if node.loop:
            result = await self._execute_with_loop(
                node, workflow, results_so_far,
                agent_fn, tool_fn, human_fn, bus, incoming, hooks, stream, hctx,
                workflow_trace, node_agent_trace,
            )
        else:
            result = await self._execute_once(
                node, workflow, results_so_far, agent_fn, tool_fn, human_fn, bus, incoming,
                hooks, stream, hctx, node_agent_trace,
            )

        await hooks.on_node_end(node, result, hctx)
        return result

    async def _execute_once(
        self,
        node: Node,
        workflow: Workflow,
        results_so_far: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
        bus: MessageBus,
        incoming: list[AgentMessage],
        hooks: ExecutionHooks,
        stream: StreamCallback | None,
        hctx: HookContext,
        node_agent_trace: AgentTrace | None = None,
    ) -> NodeResult:
        """单次执行一个节点（不含循环）。"""
        timeout_ms = node.timeout_ms or self.default_timeout_ms
        max_retries = max(0, node.retry_max)
        last_error = ""

        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                ctx = self._build_context(node, workflow, results_so_far, incoming, bus, node_agent_trace)
                # 注入 hook 共享上下文
                ctx["_hook_shared"] = hctx.shared
                output = await asyncio.wait_for(
                    self._dispatch(node, ctx, agent_fn, tool_fn, human_fn, hooks, stream, hctx),
                    timeout=timeout_ms / 1000.0,
                )
                return NodeResult(
                    node_id=node.id,
                    output=output,
                    attempts=attempt + 1,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_ms}ms"
                logger.warning("Node '%s' timed out (attempt %d/%d)", node.id, attempt + 1, max_retries + 1)
            except Exception as e:
                last_error = str(e)
                logger.warning("Node '%s' failed (attempt %d/%d): %s", node.id, attempt + 1, max_retries + 1, e)

            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

        return self._apply_fallback(node, last_error)

    async def _dispatch(
        self,
        node: Node,
        ctx: dict,
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
        hooks: ExecutionHooks,
        stream: StreamCallback | None,
        hctx: HookContext,
    ) -> str:
        """根据 NodeKind 分发到正确的执行函数。

        ctx 包含: previous_outputs, incoming_messages, message_bus, _hook_shared
        """
        if node.kind == NodeKind.AGENT:
            if agent_fn is None:
                raise RuntimeError(f"AGENT node '{node.id}': agent_fn not provided")
            try:
                return await agent_fn(node.id, ctx, stream)
            except TypeError:
                # 向后兼容：agent_fn 如果不接受第三个参数，回退到两参数版本
                return await agent_fn(node.id, ctx)  # type: ignore[call-arg]

        elif node.kind == NodeKind.TOOL:
            if tool_fn is None:
                raise RuntimeError(f"TOOL node '{node.id}': tool_fn not provided")
            if node.tool is None:
                raise RuntimeError(f"TOOL node '{node.id}': tool config missing")
            await hooks.on_tool_call(node.tool.name, node.tool.inputs, hctx)
            result = await tool_fn(node.tool.name, node.tool.inputs)
            await hooks.on_tool_result(node.tool.name, result, hctx)
            return result

        elif node.kind == NodeKind.HUMAN:
            return await self._execute_human(node, ctx, human_fn)

        elif node.kind == NodeKind.SUBGRAPH:
            return await self._execute_subgraph(node, ctx, agent_fn, tool_fn, human_fn, hooks, stream, hctx)

        raise RuntimeError(f"Unknown node kind: {node.kind}")

    async def _execute_subgraph(
        self,
        node: Node,
        ctx: dict,
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
        hooks: ExecutionHooks,
        stream: StreamCallback | None,
        hctx: HookContext,
    ) -> str:
        """递归执行子 Workflow。

        在注册表中查找子 Workflow，以当前 context 作为输入递归执行。
        子 Workflow 的入口节点拿到父 context；子 Workflow 结束后，
        最终节点的输出作为本节点的返回值。
        """
        sub_name = node.subgraph
        if not sub_name:
            raise RuntimeError(f"SUBGRAPH node '{node.id}': subgraph name not set")

        child_wf = self._workflows.get(sub_name)
        if child_wf is None:
            raise RuntimeError(
                f"SUBGRAPH node '{node.id}': unknown workflow '{sub_name}'. "
                f"Available: {list(self._workflows.keys())}"
            )

        # 从 ctx 提取消息总线，父子共享
        bus = ctx.get("message_bus")
        parent_outputs = ctx.get("previous_outputs", {})

        # 为子 Workflow 创建包装的 agent_fn——将父 context 注入子入口
        async def child_agent_fn(nid: str, child_ctx: dict) -> str:
            # 子节点同时看到父 context 和自己的 child_ctx
            merged = {**ctx, **child_ctx}
            merged["previous_outputs"] = {
                **parent_outputs,
                **child_ctx.get("previous_outputs", {}),
            }
            if agent_fn is not None:
                return await agent_fn(nid, merged)
            return f"sub:{nid}"

        logger.info("Entering subgraph '%s' from node '%s'", sub_name, node.id)
        child_results, _ = await self.execute(
            child_wf,
            agent_fn=child_agent_fn,
            tool_fn=tool_fn,
            human_fn=human_fn,
            message_bus=bus,
            hooks=hooks,
            stream=stream,
        )

        # 返回子 Workflow 最后一个节点的输出
        last_nid = child_wf.nodes[-1].id
        last_result = child_results.get(last_nid)
        return last_result.output if last_result else f"subgraph '{sub_name}' completed"
        return last_result.output if last_result else f"subgraph '{sub_name}' completed"

    async def _execute_human(
        self,
        node: Node,
        ctx: dict,
        human_fn: HumanFn | None,
    ) -> str:
        """执行人工确认节点——等待输入，超时则用默认值。

        如果提供了 human_fn，调用它（通常是 console input 或 Web 回调）。
        如果没提供，直接返回 default_response。
        超时和 fallback 逻辑由 human_fn 内部处理，
        编排器只负责在 human_fn 不可用时降级。
        """
        human_conf = node.human
        if human_conf is None:
            raise RuntimeError(f"HUMAN node '{node.id}': human config missing")

        prompt = human_conf.prompt or f"Human approval required for '{node.id}'"
        timeout = human_conf.timeout_sec

        if human_fn is not None:
            try:
                return await human_fn(node.id, {
                    "prompt": prompt,
                    "timeout_sec": timeout,
                    **ctx,
                })
            except asyncio.TimeoutError:
                logger.warning("Human node '%s' timed out after %ds, using default", node.id, timeout)
                return human_conf.default_response or "(timeout, no response)"

        # 没有 human_fn → 直接返回默认值（自动审批模式）
        logger.info("Human node '%s': no human_fn, auto-approving", node.id)
        return human_conf.default_response or "(auto-approved)"

    async def _execute_with_loop(
        self,
        node: Node, workflow: Workflow,
        results_so_far: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
        bus: MessageBus,
        incoming: list[AgentMessage],
        hooks: ExecutionHooks,
        stream: StreamCallback | None,
        hctx: HookContext,
        workflow_trace: WorkflowTrace,
        node_agent_trace: AgentTrace | None = None,
    ) -> NodeResult:
        """循环执行节点直到条件满足或达到上限。"""
        assert node.loop is not None
        max_loop = node.loop.max_iterations
        outputs = []
        last_error = ""

        for i in range(max_loop):
            result = await self._execute_once(
                node, workflow, results_so_far, agent_fn, tool_fn, human_fn, bus, incoming,
                hooks, stream, hctx, node_agent_trace,
            )
            outputs.append(result.output if result.success else {"error": result.error})

            if result.success and node.loop.condition:
                if self._eval_loop_condition(node.loop.condition, result):
                    return NodeResult(node_id=node.id, output=outputs, attempts=i + 1)
            elif result.success and node.loop.break_on_tool:
                # 如果指定了 break_on_tool，检查 output
                if isinstance(result.output, dict) and result.output.get("tool") == node.loop.break_on_tool:
                    return NodeResult(node_id=node.id, output=outputs, attempts=i + 1)
            elif not result.success:
                last_error = result.error
                return NodeResult(node_id=node.id, success=False, error=last_error)

        return NodeResult(node_id=node.id, output=outputs, error="loop exhausted")

    @staticmethod
    def _eval_loop_condition(expr: str, result: NodeResult) -> bool:
        """评估循环退出条件。"""
        import re
        expr = expr.strip()
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False
        m = re.match(r"(\w+)\s*(>|<|>=|<=|==|!=)\s*([\d.]+)", expr)
        if m and result.output:
            field, op, val = m.groups()
            actual = result.output.get(field) if isinstance(result.output, dict) else getattr(result, field, result.output)
            return DAGExecutor._compare(actual, op, float(val))
        return False

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _apply_fallback(self, node: Node, error: str) -> NodeResult:
        if node.fallback == FallbackPolicy.SKIP:
            return NodeResult(node_id=node.id, output=None, success=True, error=error)
        elif node.fallback == FallbackPolicy.DEFAULT_VALUE:
            return NodeResult(node_id=node.id, output=node.default_value, success=True)
        elif node.fallback == FallbackPolicy.FALLBACK_NODE:
            return NodeResult(node_id=node.id, output={"fallback_to": node.fallback_node_id}, success=True)
        else:
            return NodeResult(node_id=node.id, success=False, error=error)

    # ------------------------------------------------------------------
    # 记忆作用域 → context 构建
    # ------------------------------------------------------------------

    def _build_context(
        self,
        node: Node,
        workflow: Workflow,
        results_so_far: dict[str, NodeResult],
        incoming: list[AgentMessage],
        bus: MessageBus,
        node_agent_trace: AgentTrace | None = None,
    ) -> dict:
        """根据节点 memory_scope 构建 context。

        workflow  — 看全局，所有前驱输出 + 消息 + bus
        inherit  — 只看直接上游节点的输出（默认）
        none     — 不看任何记忆，只有 incoming_messages + bus
        """
        scope = node.agent.memory_scope if node.agent else "inherit"

        base = {
            "incoming_messages": incoming,
            "message_bus": bus,
            "_agent_trace": node_agent_trace,
        }

        if scope == "none":
            base["previous_outputs"] = {}
            return base

        if scope == "workflow":
            # 全局：所有已完成的节点输出都可见
            base["previous_outputs"] = results_so_far
            return base

        # scope == "inherit"（默认）：只看直接上游
        upstream_ids = self._get_upstream_ids(node.id, workflow)
        base["previous_outputs"] = {
            uid: results_so_far[uid]
            for uid in upstream_ids
            if uid in results_so_far
        }
        return base

    def _get_upstream_ids(self, node_id: str, workflow: Workflow) -> set[str]:
        """返回 node_id 的所有直接上游节点 id。"""
        return {e.from_node for e in workflow.edges if e.to_node == node_id}

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_node(self, workflow: Workflow, node_id: str) -> Optional[Node]:
        for n in workflow.nodes:
            if n.id == node_id:
                return n
        return None
