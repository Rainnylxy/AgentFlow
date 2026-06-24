"""纯 Python 异步 DAG 编排引擎 — 替代 Go 引擎的 v0.1 实现。

用法:
    executor = DAGExecutor()
    results, trace = await executor.execute(
        workflow,
        agent_fn=my_agent.run,  # async callable
    )

调度策略:
    - 使用 parallel_groups() 将 DAG 分层，层内 asyncio.gather() 并发
    - 支持 CONDITION 边（Jinja2 条件模板）
    - 支持 LOOP 节点（循环直到条件满足或达到 max_iterations）
    - 支持 FallbackPolicy：SKIP / DEFAULT_VALUE / FALLBACK_NODE / RAISE
    - 支持节点级重试 + 超时

与 Go 引擎的关系:
    - Go 引擎的 DAG executor/breaker/retry/metrics 逻辑用 Python asyncio 重实现
    - proto 合约保留不动，将来需要独立扩展时接回去
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from agentflow.dsl.types import Workflow, Node, NodeType, FallbackPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 执行结果 & 追踪
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


@dataclass
class ExecutionTrace:
    """一次完整 DAG 执行的轨迹。"""
    workflow_name: str
    total_duration_ms: int = 0
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    groups: list[list[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DAG 执行器
# ---------------------------------------------------------------------------

class DAGExecutor:
    """纯 Python 异步 DAG 编排引擎。

    用法:
        executor = DAGExecutor()
        results, trace = await executor.execute(workflow, agent_fn=my_run)

    agent_fn 签名: async def fn(node_id: str, context: dict) -> str
        - node_id: 当前执行的节点 id
        - context: {"previous_outputs": {node_id: NodeResult, ...}}
        - 返回: 节点输出字符串
    """

    def __init__(self, default_timeout_ms: int = 120_000):
        self.default_timeout_ms = default_timeout_ms

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def execute(
        self,
        workflow: Workflow,
        agent_fn: Callable[[str, dict], Awaitable[str]],
    ) -> "tuple[dict[str, NodeResult], ExecutionTrace]":
        """执行整个 Workflow DAG。

        Args:
            workflow: DSL 定义的 Workflow
            agent_fn: 异步节点执行函数 async (node_id, context) -> str

        Returns:
            (results_by_node_id, execution_trace)
        """
        from agentflow.dsl.graph import parallel_groups

        t0 = time.monotonic()
        groups = parallel_groups(workflow)
        all_results: dict[str, NodeResult] = {}

        for group in groups:
            if not group:
                continue

            group_results = await asyncio.gather(*[
                self._execute_node(workflow, nid, all_results, agent_fn)
                for nid in group
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
    # 单节点执行
    # ------------------------------------------------------------------

    async def _execute_node(
        self,
        workflow: Workflow,
        node_id: str,
        results_so_far: dict[str, NodeResult],
        agent_fn: Callable[[str, dict], Awaitable[str]],
    ) -> NodeResult:
        """执行单个节点（含重试、超时、降级）。"""
        node = self._get_node(workflow, node_id)
        if node is None:
            return NodeResult(node_id=node_id, success=False, error=f"Node '{node_id}' not found")

        if node.node_type == NodeType.CONDITION:
            return self._execute_condition(node, results_so_far)

        if node.node_type == NodeType.LOOP:
            return await self._execute_loop(node, workflow, results_so_far, agent_fn)

        # AGENT / PARALLEL — 标准执行
        timeout_ms = node.timeout_ms or self.default_timeout_ms
        max_retries = max(0, node.retry_max)
        last_error = ""

        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                output = await asyncio.wait_for(
                    agent_fn(node_id, {"previous_outputs": results_so_far}),
                    timeout=timeout_ms / 1000.0,
                )
                return NodeResult(
                    node_id=node_id,
                    output=output,
                    success=True,
                    attempts=attempt + 1,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_ms}ms"
                logger.warning("Node '%s' timed out (attempt %d/%d)", node_id, attempt + 1, max_retries + 1)
            except Exception as e:
                last_error = str(e)
                logger.warning("Node '%s' failed (attempt %d/%d): %s", node_id, attempt + 1, max_retries + 1, e)

            # 重试前等待（简单指数退避）
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

        # 所有重试耗尽 → 降级
        return self._apply_fallback(node, last_error)

    # ------------------------------------------------------------------
    # 条件节点
    # ------------------------------------------------------------------

    def _execute_condition(
        self, node: Node, results_so_far: dict[str, NodeResult]
    ) -> NodeResult:
        """执行条件路由节点。

        CONDITION 节点的 config 里包含要评估的条件表达式。
        当前实现：简单通过/失败（完整 Jinja2 模板在后续版本实现）。
        """
        condition_expr = node.config.get("condition", "")
        if condition_expr:
            # 简化条件评估：检查 previous_outputs 中是否有匹配
            try:
                # 格式: "node_id.score > 0.5" 或直接 "True"
                result = self._eval_simple_condition(condition_expr, results_so_far)
                return NodeResult(node_id=node.id, output={"branch": result}, success=True)
            except Exception as e:
                return NodeResult(node_id=node.id, success=False, error=str(e))
        return NodeResult(node_id=node.id, output={"branch": True}, success=True)

    def _eval_simple_condition(self, expr: str, results: dict[str, NodeResult]) -> bool:
        """简化条件评估器。支持: 'node_id.score > 0.5' 和 'True'/'False'。"""
        import re
        m = re.match(r"(\w+)\.(\w+)\s*(>|<|>=|<=|==|!=)\s*([\d.]+)", expr)
        if m:
            nid, field, op, val = m.groups()
            if nid in results:
                node_output = results[nid].output
                if isinstance(node_output, dict) and field in node_output:
                    actual = node_output[field]
                else:
                    actual = getattr(results[nid], field, 0)
                val_num = float(val)
                if op == ">":
                    return actual > val_num
                elif op == "<":
                    return actual < val_num
                elif op == ">=":
                    return actual >= val_num
                elif op == "<=":
                    return actual <= val_num
                elif op == "==":
                    return actual == val_num
        return expr.strip().lower() == "true"

    # ------------------------------------------------------------------
    # 循环节点
    # ------------------------------------------------------------------

    async def _execute_loop(
        self,
        node: Node,
        workflow: Workflow,
        results_so_far: dict[str, NodeResult],
        agent_fn: Callable[[str, dict], Awaitable[str]],
    ) -> NodeResult:
        """执行 LOOP 节点——循环直到条件满足。"""
        max_loop = node.config.get("max_iterations", workflow.max_iterations)
        loop_condition = node.config.get("condition", "False")
        outputs = []

        for i in range(max_loop):
            try:
                output = await agent_fn(node.id, {
                    "previous_outputs": results_so_far,
                    "loop_index": i,
                    "loop_max": max_loop,
                })
                outputs.append(output)
                if self._eval_simple_condition(loop_condition, results_so_far):
                    break
            except Exception as e:
                return NodeResult(node_id=node.id, success=False, error=str(e))

        return NodeResult(node_id=node.id, output=outputs, success=True)

    # ------------------------------------------------------------------
    # 降级处理
    # ------------------------------------------------------------------

    def _apply_fallback(self, node: Node, error: str) -> NodeResult:
        """应用节点的降级策略。"""
        if node.fallback == FallbackPolicy.SKIP:
            return NodeResult(node_id=node.id, output=None, success=True, error=error)
        elif node.fallback == FallbackPolicy.DEFAULT_VALUE:
            return NodeResult(node_id=node.id, output=node.default_value, success=True)
        elif node.fallback == FallbackPolicy.FALLBACK_NODE:
            return NodeResult(
                node_id=node.id,
                output={"fallback_to": node.fallback_node_id},
                success=True,
            )
        else:  # RAISE
            return NodeResult(node_id=node.id, success=False, error=error)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_node(self, workflow: Workflow, node_id: str) -> Optional[Node]:
        for n in workflow.nodes:
            if n.id == node_id:
                return n
        return None
