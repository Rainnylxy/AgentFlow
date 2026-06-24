"""Agent 间消息传递测试。"""

import asyncio
import pytest
from agentflow.runtime.message_bus import MessageBus, AgentMessage, Intent
from agentflow.runtime.orchestrator import DAGExecutor
from agentflow.dsl.types import Node, Edge, Workflow, NodeKind


class TestAgentMessage:
    def test_create_message(self):
        msg = AgentMessage(
            from_agent="planner",
            to_agent="worker",
            intent="delegate",
            payload={"task": "查库存"},
        )
        assert msg.from_agent == "planner"
        assert msg.intent == "delegate"
        assert msg.payload["task"] == "查库存"
        assert msg.id  # auto-generated

    def test_message_to_dict(self):
        msg = AgentMessage(from_agent="a", to_agent="b", intent="info")
        d = msg.to_dict()
        assert d["from"] == "a"
        assert d["to"] == "b"


class TestMessageBus:
    def test_send_and_receive(self):
        bus = MessageBus()
        bus.send(AgentMessage(from_agent="a", to_agent="b", intent="delegate"))

        msgs = bus.receive("b")
        assert len(msgs) == 1
        assert msgs[0].from_agent == "a"

        # 已读不再返回
        msgs2 = bus.receive("b")
        assert len(msgs2) == 0

    def test_broadcast(self):
        bus = MessageBus()
        bus.broadcast("planner", "info", {"status": "done"})

        for agent_id in ["worker_a", "worker_b", "worker_c"]:
            msgs = bus.receive(agent_id)
            assert len(msgs) == 1
            assert msgs[0].to_agent == "broadcast"

    def test_targeted_vs_broadcast(self):
        bus = MessageBus()
        bus.send(AgentMessage(from_agent="a", to_agent="b", intent="info"))
        bus.broadcast("a", "info", {})

        # b 收到定向 + 广播
        assert len(bus.receive("b")) == 2
        # c 只收到广播
        assert len(bus.receive("c")) == 1

    def test_peek_does_not_mark_read(self):
        bus = MessageBus()
        bus.send(AgentMessage(from_agent="a", to_agent="b"))
        assert len(bus.peek("b")) == 1
        assert len(bus.peek("b")) == 1  # still there
        assert len(bus.receive("b")) == 1
        assert len(bus.peek("b")) == 0  # now read

    def test_clear(self):
        bus = MessageBus()
        bus.send(AgentMessage(from_agent="a", to_agent="b"))
        bus.clear()
        assert len(bus.receive("b")) == 0


class TestOrchestratorWithMessages:
    """编排器 + 消息总线 集成测试"""

    async def test_agent_sends_message_to_downstream(self):
        """上游 Agent 发消息 → 下游 Agent 收到。"""
        bus = MessageBus()

        async def agent_fn(node_id, ctx):
            msgs = ctx.get("incoming_messages", [])
            msg_bus = ctx.get("message_bus")

            if node_id == "planner":
                # planner 发送委派任务
                msg_bus.send(AgentMessage(
                    from_agent="planner",
                    to_agent="worker",
                    intent="delegate",
                    payload={"task": "查库存"},
                ))
                return "task delegated"

            elif node_id == "worker":
                # worker 收到 planner 的消息
                if msgs:
                    task = msgs[0].payload.get("task")
                    return f"executing: {task}"
                return "no task received"

            return f"default-{node_id}"

        wf = Workflow(
            name="delegation",
            nodes=[
                Node(id="planner", kind=NodeKind.AGENT),
                Node(id="worker", kind=NodeKind.AGENT),
            ],
            edges=[Edge(from_node="planner", to_node="worker")],
        )

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn, message_bus=bus)

        assert results["planner"].output == "task delegated"
        assert "executing: 查库存" in results["worker"].output

    async def test_broadcast_notifies_all_downstream(self):
        """广播消息 → 所有下游 Agent 都收到。"""
        bus = MessageBus()

        async def agent_fn(node_id, ctx):
            msg_bus = ctx.get("message_bus")

            if node_id == "coordinator":
                msg_bus.broadcast("coordinator", "info", {"status": "go"})
                return "broadcasted"

            elif node_id in ("left", "right"):
                msgs = ctx.get("incoming_messages", [])
                received = "yes" if msgs else "no"
                return f"{node_id}: received={received}"

            return f"ok-{node_id}"

        wf = Workflow(
            name="broadcast",
            nodes=[
                Node(id="coordinator", kind=NodeKind.AGENT),
                Node(id="left", kind=NodeKind.AGENT),
                Node(id="right", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="coordinator", to_node="left"),
                Edge(from_node="coordinator", to_node="right"),
            ],
        )

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn, message_bus=bus)

        assert "received=yes" in results["left"].output
        assert "received=yes" in results["right"].output

    async def test_message_across_multiple_levels(self):
        """多层 DAG：消息逐层传递。"""
        bus = MessageBus()

        async def agent_fn(node_id, ctx):
            msg_bus = ctx.get("message_bus")
            msgs = ctx.get("incoming_messages", [])
            prev_msgs = [m.payload.get("trace", "") for m in msgs]

            # 把自己的 trace 发给下一层
            my_trace = f"{node_id}"
            msg_bus.broadcast(node_id, "info", {"trace": my_trace})

            if prev_msgs:
                return f"{node_id} saw: {prev_msgs}"
            return f"{node_id} (first)"

        wf = Workflow(
            name="levels",
            nodes=[
                Node(id="l1", kind=NodeKind.AGENT),
                Node(id="l2", kind=NodeKind.AGENT),
                Node(id="l3", kind=NodeKind.AGENT),
            ],
            edges=[
                Edge(from_node="l1", to_node="l2"),
                Edge(from_node="l2", to_node="l3"),
            ],
        )

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn, message_bus=bus)

        assert "(first)" in results["l1"].output
        assert "l1" in results["l2"].output   # l2 saw l1's trace
        assert "l2" in results["l3"].output   # l3 saw l2's trace

    async def test_message_without_bus_still_works(self):
        """不传 message_bus 时编排器自动创建一个，不影响执行。"""
        async def agent_fn(node_id, ctx):
            bus = ctx.get("message_bus")
            # 不发送任何消息，只验证 bus 存在
            assert bus is not None
            return f"ok-{node_id}"

        wf = Workflow(
            name="no-bus",
            nodes=[Node(id="a", kind=NodeKind.AGENT)],
            edges=[],
        )
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn)
        assert results["a"].success
