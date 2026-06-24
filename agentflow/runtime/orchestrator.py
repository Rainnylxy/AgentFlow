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
from typing import Any, Callable, Awaitable, Optional

from agentflow.dsl.types import Workflow, Node, NodeKind, FallbackPolicy

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

AgentFn = Callable[[str, dict], Awaitable[str]]
ToolFn = Callable[[str, dict], Awaitable[str]]
HumanFn = Callable[[str, dict], Awaitable[str]]


class DAGExecutor:
    """纯 Python 异步 DAG 编排引擎。"""

    def __init__(self, default_timeout_ms: int = 120_000):
        self.default_timeout_ms = default_timeout_ms

    async def execute(
        self,
        workflow: Workflow,
        agent_fn: AgentFn | None = None,
        tool_fn: ToolFn | None = None,
        human_fn: HumanFn | None = None,
    ) -> "tuple[dict[str, NodeResult], ExecutionTrace]":
        """执行整个 Workflow DAG。

        每层执行前评估边条件，跳过不满足条件的下游节点。
        """
        from agentflow.dsl.graph import parallel_groups

        t0 = time.monotonic()
        groups = parallel_groups(workflow)
        all_results: dict[str, NodeResult] = {}

        for group in groups:
            if not group:
                continue

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
                        agent_fn, tool_fn, human_fn,
                    )
                    for nid in active
                ])
                for nr in group_results:
                    all_results[nr.node_id] = nr

        t1 = time.monotonic()
        trace = ExecutionTrace(
            workflow_name=workflow.name,
            total_duration_ms=int((t1 - t0) * 1000),
            node_results=all_results,
            groups=groups,
        )
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
    ) -> NodeResult:
        node = self._get_node(workflow, node_id)
        if node is None:
            return NodeResult(node_id=node_id, success=False, error=f"Node not found")

        # 循环节点 —— wrapper
        if node.loop:
            return await self._execute_with_loop(
                node, workflow, results_so_far,
                agent_fn, tool_fn, human_fn,
            )

        return await self._execute_once(
            node, results_so_far, agent_fn, tool_fn, human_fn,
        )

    async def _execute_once(
        self,
        node: Node,
        results_so_far: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
    ) -> NodeResult:
        """单次执行一个节点（不含循环）。"""
        timeout_ms = node.timeout_ms or self.default_timeout_ms
        max_retries = max(0, node.retry_max)
        last_error = ""

        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                output = await asyncio.wait_for(
                    self._dispatch(node, results_so_far, agent_fn, tool_fn, human_fn),
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
        ctx: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
    ) -> str:
        """根据 NodeKind 分发到正确的执行函数。"""
        if node.kind == NodeKind.AGENT:
            if agent_fn is None:
                raise RuntimeError(f"AGENT node '{node.id}': agent_fn not provided")
            return await agent_fn(node.id, {"previous_outputs": ctx})

        elif node.kind == NodeKind.TOOL:
            if tool_fn is None:
                raise RuntimeError(f"TOOL node '{node.id}': tool_fn not provided")
            if node.tool is None:
                raise RuntimeError(f"TOOL node '{node.id}': tool config missing")
            return await tool_fn(node.tool.name, node.tool.inputs)

        elif node.kind == NodeKind.HUMAN:
            if human_fn is None:
                raise RuntimeError(f"HUMAN node '{node.id}': human_fn not provided")
            return await human_fn(node.id, {"prompt": node.human.prompt if node.human else ""})

        elif node.kind == NodeKind.SUBGRAPH:
            # Subgraph 由 agent_fn 处理（内部递归执行子 Workflow）
            if agent_fn is None:
                raise RuntimeError(f"SUBGRAPH node '{node.id}': agent_fn not provided")
            return await agent_fn(node.id, {"previous_outputs": ctx, "subgraph": node.subgraph})

        raise RuntimeError(f"Unknown node kind: {node.kind}")

    async def _execute_with_loop(
        self,
        node: Node, workflow: Workflow,
        results_so_far: dict[str, NodeResult],
        agent_fn: AgentFn | None,
        tool_fn: ToolFn | None,
        human_fn: HumanFn | None,
    ) -> NodeResult:
        """循环执行节点直到条件满足或达到上限。"""
        assert node.loop is not None
        max_loop = node.loop.max_iterations
        outputs = []
        last_error = ""

        for i in range(max_loop):
            result = await self._execute_once(
                node, results_so_far, agent_fn, tool_fn, human_fn,
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
    # 辅助
    # ------------------------------------------------------------------

    def _get_node(self, workflow: Workflow, node_id: str) -> Optional[Node]:
        for n in workflow.nodes:
            if n.id == node_id:
                return n
        return None
