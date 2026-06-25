"""DAG 编排器测试 — 验证纯 Python 异步引擎的 DAG 执行逻辑。"""

from __future__ import annotations

import asyncio
import pytest
from agentflow.dsl.types import Node, Edge, Workflow, NodeKind, FallbackPolicy
from agentflow.runtime.orchestrator import DAGExecutor, NodeResult


# ---------------------------------------------------------------------------
# agent_fn 模拟
# ---------------------------------------------------------------------------

def make_agent_fn(responses: dict[str, str] | None = None, delay: float = 0):
    """创建一个可控的异步 agent 执行函数。

    responses: node_id → 返回内容映射
    delay: 模拟延迟（秒）
    """
    responses = responses or {}

    async def fn(node_id: str, context: dict) -> str:
        if delay:
            await asyncio.sleep(delay)
        return responses.get(node_id, f"default output from {node_id}")

    return fn


def make_failing_agent_fn(fail_on: str, error_msg: str = "boom"):
    """创建一个对特定节点抛异常的 agent 函数。"""
    async def fn(node_id: str, context: dict) -> str:
        if node_id == fail_on:
            raise RuntimeError(error_msg)
        return f"ok from {node_id}"
    return fn


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestDAGExecutorLinear:
    """线性 DAG: A → B → C"""

    @pytest.fixture
    def linear_workflow(self):
        return Workflow(
            name="linear",
            nodes=[
                Node(id="a", kind=NodeKind.AGENT),
                Node(id="b", kind=NodeKind.AGENT),
                Node(id="c", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="a", to_node="b"),
                Edge(from_node="b", to_node="c"),
            ],
        )

    async def test_linear_execution_order(self, linear_workflow):
        """线性 DAG 按依赖顺序执行，每个节点拿到前驱结果。"""
        execution_order = []

        async def tracking_fn(node_id, context):
            execution_order.append(node_id)
            prev = context.get("previous_outputs", {})
            return f"result from {node_id}, saw previous: {list(prev.keys())}"

        executor = DAGExecutor()
        results, trace = await executor.execute(linear_workflow, tracking_fn)

        assert execution_order == ["a", "b", "c"]
        assert results["a"].success
        assert results["b"].success
        assert results["c"].success
        assert trace.summary.total_duration_ms >= 0

    async def test_linear_with_mock_responses(self, linear_workflow):
        """线性执行各节点返回正确的 mock 输出。"""
        responses = {"a": "A-output", "b": "B-output", "c": "C-output"}
        executor = DAGExecutor()
        results, _ = await executor.execute(linear_workflow, make_agent_fn(responses))

        assert results["a"].output == "A-output"
        assert results["b"].output == "B-output"
        assert results["c"].output == "C-output"


class TestDAGExecutorDiamond:
    """菱形 DAG: A → B, A → C, B → D, C → D (B,C 可并行)"""

    @pytest.fixture
    def diamond_workflow(self):
        return Workflow(
            name="diamond",
            nodes=[
                Node(id="entry", kind=NodeKind.AGENT),
                Node(id="left", kind=NodeKind.AGENT),
                Node(id="right", kind=NodeKind.AGENT),
                Node(id="end", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="entry", to_node="left"),
                Edge(from_node="entry", to_node="right"),
                Edge(from_node="left", to_node="end"),
                Edge(from_node="right", to_node="end"),
            ],
        )

    async def test_parallel_execution_groups(self, diamond_workflow):
        """菱形 DAG 的并行分组正确：[[entry], [left,right], [end]]"""
        execution_order = []

        async def tracking_fn(node_id, context):
            await asyncio.sleep(0.01)  # 小延迟让并行更明显
            execution_order.append(node_id)
            return f"output-{node_id}"

        executor = DAGExecutor()
        results, trace = await executor.execute(diamond_workflow, tracking_fn)

        assert execution_order[0] == "entry"
        assert execution_order[-1] == "end"
        assert set(execution_order[1:3]) == {"left", "right"}
        assert trace.dag_groups == [["entry"], ["left", "right"], ["end"]]

    async def test_diamond_collects_all_outputs(self, diamond_workflow):
        """所有节点输出都被正确收集。"""
        executor = DAGExecutor()
        results, _ = await executor.execute(
            diamond_workflow,
            make_agent_fn({"entry": "start", "left": "L", "right": "R", "end": "done"}),
        )
        assert len(results) == 4
        assert results["entry"].output == "start"
        assert results["left"].output == "L"
        assert results["right"].output == "R"
        assert results["end"].output == "done"


class TestDAGExecutorResilience:
    """容错：重试、超时、降级"""

    async def test_retry_on_failure(self):
        """节点失败 → 重试 → 最终成功。"""
        call_count = {"b": 0}

        async def flaky_fn(node_id, context):
            if node_id == "b":
                call_count["b"] += 1
                if call_count["b"] < 3:
                    raise RuntimeError("not yet")
            return f"ok-{node_id}"

        wf = Workflow(
            name="retry-test",
            nodes=[
                Node(id="a", kind=NodeKind.AGENT),
                Node(id="b", kind=NodeKind.AGENT, retry_max=3),
            ],
            edges=[Edge(from_node="a", to_node="b")],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, flaky_fn)

        assert results["b"].success
        assert results["b"].attempts == 3
        assert call_count["b"] == 3

    async def test_timeout_triggers_fallback_skip(self):
        """超时节点触发 SKIP 降级。"""
        async def slow_fn(node_id, context):
            if node_id == "slow":
                await asyncio.sleep(10)
            return "ok"

        wf = Workflow(
            name="timeout-test",
            nodes=[
                Node(id="slow", kind=NodeKind.AGENT,
                     timeout_ms=50, fallback=FallbackPolicy.SKIP),
                Node(id="fast", kind=NodeKind.AGENT),
            ],
            edges=[Edge(from_node="slow", to_node="fast")],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, slow_fn)

        assert results["slow"].success  # SKIP → success=True
        assert results["slow"].output is None
        assert "Timeout" in results["slow"].error

    async def test_fallback_default_value(self):
        """失败节点返回预设默认值。"""
        wf = Workflow(
            name="default-test",
            nodes=[
                Node(id="x", kind=NodeKind.AGENT,
                     fallback=FallbackPolicy.DEFAULT_VALUE, default_value="fallback-output"),
            ],
            edges=[],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, make_failing_agent_fn("x"))

        assert results["x"].success
        assert results["x"].output == "fallback-output"

    async def test_fallback_raise(self):
        """RAISE 策略：失败直接标记为不成功。"""
        wf = Workflow(
            name="raise-test",
            nodes=[Node(id="x", kind=NodeKind.AGENT, fallback=FallbackPolicy.RAISE)],
            edges=[],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, make_failing_agent_fn("x"))

        assert not results["x"].success
        assert "boom" in results["x"].error


class TestDAGExecutorCondition:
    """条件在边上——边条件决定下游节点是否执行"""

    async def test_edge_condition_skips_downstream(self):
        """Edge condition=False → 下游节点跳过。"""
        wf = Workflow(
            name="edge-condition-test",
            nodes=[
                Node(id="step", kind=NodeKind.AGENT),
                Node(id="a", kind=NodeKind.AGENT),
                Node(id="b", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="step", to_node="a"),
                Edge(from_node="step", to_node="b", condition="False"),  # 永不通
            ],
        )

        async def fn(node_id, context):
            return f"output-{node_id}"

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=fn)

        assert results["step"].success
        assert results["a"].success
        assert results["b"].skipped_by_condition


class TestDAGExecutorSingleNode:
    """单节点 / 无依赖"""

    async def test_single_node(self):
        wf = Workflow(
            name="solo",
            nodes=[Node(id="only", kind=NodeKind.AGENT)],
            edges=[],
        )
        executor = DAGExecutor()
        results, trace = await executor.execute(wf, make_agent_fn({"only": "solo-output"}))

        assert len(results) == 1
        assert results["only"].output == "solo-output"
        assert trace.summary.total_duration_ms >= 0

    async def test_node_not_found(self):
        """Workflow 引用了不存在的节点 id（由 validator 保证，此处测兜底）。"""
        wf = Workflow(
            name="missing-node",
            nodes=[Node(id="a", kind=NodeKind.AGENT)],
            edges=[],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, make_agent_fn({}))

        # 返回的 a 来自 workflow.nodes，所以能正常执行
        assert results["a"].success


class TestDAGExecutorIntegration:
    """编排器与真实 AsyncMock 集成"""

    async def test_with_real_agent_fn_pattern(self):
        """模拟真实 AgentBuilder 产出的 agent.run() 作为 agent_fn。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_agent = AsyncMock()

        async def mock_run(user_input: str):
            return MagicMock(output=f"processed: {user_input}")

        mock_agent.run.side_effect = mock_run

        # 真实的 agent_fn 包装
        async def agent_fn(node_id: str, context: dict) -> str:
            result = await mock_agent.run(f"execute {node_id}")
            return result.output

        wf = Workflow(
            name="integration-test",
            nodes=[
                Node(id="step1", kind=NodeKind.AGENT),
                Node(id="step2", kind=NodeKind.AGENT),
            ],
            edges=[Edge(from_node="step1", to_node="step2")],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn)

        assert results["step1"].success
        assert results["step2"].success
        assert mock_agent.run.call_count == 2


class TestMemoryScope:
    """记忆作用域：workflow / inherit / none"""

    @pytest.fixture
    def diamond_wf(self):
        from agentflow.dsl.types import AgentConfig
        return Workflow(
            name="scope-test",
            nodes=[
                Node(id="entry", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="inherit")),
                Node(id="left", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="inherit")),
                Node(id="right", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="inherit")),
                Node(id="end", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="workflow")),
            ],
            edges=[
                Edge(from_node="entry", to_node="left"),
                Edge(from_node="entry", to_node="right"),
                Edge(from_node="left", to_node="end"),
                Edge(from_node="right", to_node="end"),
            ],
        )

    async def test_inherit_only_sees_upstream(self, diamond_wf):
        """inherit 作用域：只看直接上游，不看到旁路。"""

        async def agent_fn(node_id, ctx):
            prev = ctx.get("previous_outputs", {})
            seen = list(prev.keys())
            return f"{node_id} saw: {seen}"

        executor = DAGExecutor()
        results, _ = await executor.execute(diamond_wf, agent_fn=agent_fn)

        # entry 没有上游
        assert "[]" in results["entry"].output or "saw:" in results["entry"].output

        # left / right 只看到 entry（直接上游），看不到彼此
        left_out = results["left"].output
        assert "entry" in left_out
        assert "right" not in left_out  # 旁路不可见

        right_out = results["right"].output
        assert "entry" in right_out
        assert "left" not in right_out   # 旁路不可见

    async def test_workflow_sees_all(self, diamond_wf):
        """workflow 作用域：看到所有已完成节点的输出。"""

        async def agent_fn(node_id, ctx):
            prev = ctx.get("previous_outputs", {})
            seen = sorted(prev.keys())
            return f"{node_id} saw: {seen}"

        executor = DAGExecutor()
        results, _ = await executor.execute(diamond_wf, agent_fn=agent_fn)

        # end 是 workflow 作用域，应看到 entry + left + right
        assert "entry" in results["end"].output
        assert "left" in results["end"].output
        assert "right" in results["end"].output

    async def test_none_scope_sees_nothing(self):
        """none 作用域：不传任何 previous_outputs。"""
        from agentflow.dsl.types import AgentConfig

        wf = Workflow(
            name="none-test",
            nodes=[
                Node(id="a", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="inherit")),
                Node(id="b", kind=NodeKind.AGENT,
                     agent=AgentConfig(memory_scope="none")),
            ],
            edges=[Edge(from_node="a", to_node="b")],
        )

        async def agent_fn(node_id, ctx):
            prev = ctx.get("previous_outputs", {})
            return f"{node_id} saw {len(prev)} outputs"

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn)

        assert "saw 0 outputs" in results["b"].output


class TestSubgraphExecution:
    """SUBGRAPH 节点 — 递归执行子 Workflow"""

    async def test_subgraph_runs_child_workflow(self):
        """SUBGRAPH 节点递归执行注册的子 Workflow。"""
        child = Workflow(
            name="child_wf",
            nodes=[
                Node(id="c1", kind=NodeKind.AGENT),
                Node(id="c2", kind=NodeKind.AGENT),
            ],
            edges=[Edge(from_node="c1", to_node="c2")],
        )

        parent = Workflow(
            name="parent_wf",
            nodes=[
                Node(id="start", kind=NodeKind.AGENT),
                Node(id="sub", kind=NodeKind.SUBGRAPH, subgraph="child_wf"),
                Node(id="finish", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="start", to_node="sub"),
                Edge(from_node="sub", to_node="finish"),
            ],
        )

        async def agent_fn(node_id, ctx):
            prev = ctx.get("previous_outputs", {})
            return f"agent:{node_id} prev:{sorted(prev.keys())}"

        executor = DAGExecutor()
        executor.register_workflow(child)

        results, trace = await executor.execute(parent, agent_fn=agent_fn)

        # 父节点正常执行
        assert results["start"].success
        assert results["sub"].success
        assert results["finish"].success

        # sub 的输出来自子 workflow 的最终节点
        assert "agent:c2" in results["sub"].output

    async def test_subgraph_receives_parent_context(self):
        """子 Workflow 的入口节点能看到父 context。"""
        child = Workflow(
            name="inherit_ctx",
            nodes=[Node(id="c_start", kind=NodeKind.AGENT)],
            edges=[],
        )

        parent = Workflow(
            name="parent_ctx",
            nodes=[
                Node(id="p1", kind=NodeKind.AGENT),
                Node(id="sub", kind=NodeKind.SUBGRAPH, subgraph="inherit_ctx"),
            ],
            edges=[Edge(from_node="p1", to_node="sub")],
        )

        async def agent_fn(node_id, ctx):
            prev = ctx.get("previous_outputs", {})
            return f"node:{node_id} parent_visible:{'p1' in prev}"

        executor = DAGExecutor()
        executor.register_workflow(child)
        results, _ = await executor.execute(parent, agent_fn=agent_fn)

        # 子节点应该能看到父 context 中的 p1
        assert "parent_visible:True" in results["sub"].output

    async def test_subgraph_not_found_raises(self):
        """引用未注册的 subgraph → 报错。"""
        wf = Workflow(
            name="bad_parent",
            nodes=[Node(id="sub", kind=NodeKind.SUBGRAPH, subgraph="no_such_wf")],
            edges=[],
        )

        executor = DAGExecutor()
        async def fn(nid, ctx): return "ok"
        results, _ = await executor.execute(wf, agent_fn=fn)

        assert not results["sub"].success
        assert "unknown workflow" in results["sub"].error.lower()


class TestHumanExecution:
    """HUMAN 节点 — 人工确认 + 超时 fallback"""

    async def test_human_with_callback(self):
        """提供 human_fn → 正常获取人工输入。"""
        from agentflow.dsl.types import HumanConfig

        async def mock_human(node_id, ctx):
            return "approved by human"

        wf = Workflow(
            name="human-test",
            nodes=[Node(
                id="approval",
                kind=NodeKind.HUMAN,
                human=HumanConfig(prompt="Approve?", timeout_sec=10),
            )],
            edges=[],
        )

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, human_fn=mock_human)

        assert results["approval"].success
        assert results["approval"].output == "approved by human"

    async def test_human_no_callback_uses_default(self):
        """没有 human_fn → 自动返回默认值（自动审批）。"""
        from agentflow.dsl.types import HumanConfig

        wf = Workflow(
            name="auto-approve",
            nodes=[Node(
                id="approval",
                kind=NodeKind.HUMAN,
                human=HumanConfig(
                    prompt="Approve?",
                    default_response="auto-approved: proceed",
                ),
            )],
            edges=[],
        )

        executor = DAGExecutor()
        # 不传 human_fn
        results, _ = await executor.execute(wf)

        assert results["approval"].success
        assert "approved" in results["approval"].output.lower()

    async def test_human_timeout_fallback(self):
        """human_fn 超时 → 返回 default_response。"""
        from agentflow.dsl.types import HumanConfig

        async def slow_human(node_id, ctx):
            # human_fn 自己处理超时，或者编排器 catch TimeoutError
            raise asyncio.TimeoutError("human did not respond")

        wf = Workflow(
            name="timeout-human",
            nodes=[Node(
                id="approval",
                kind=NodeKind.HUMAN,
                human=HumanConfig(
                    prompt="Hurry up!",
                    timeout_sec=1,
                    default_response="timeout: auto-rejected",
                ),
            )],
            edges=[],
        )

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, human_fn=slow_human)

        assert results["approval"].success
        assert "timeout" in results["approval"].output.lower()
