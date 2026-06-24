"""Hook 系统 + 流式回调测试。"""

import asyncio
import pytest
from agentflow.runtime.hooks import (
    ExecutionHooks, HookContext, StreamEvent,
    EnvironmentHook, LoggingHook, PermissionHook,
)
from agentflow.runtime.orchestrator import DAGExecutor
from agentflow.dsl.types import Node, Edge, Workflow, NodeKind, ToolConfig


class TestExecutionHooks:
    async def test_hooks_called_in_order(self):
        """验证 hook 按 Workflow → Group → Node → ... → Node → Group → Workflow 顺序调用。"""
        calls = []

        class TraceHook(ExecutionHooks):
            async def on_workflow_start(self, wf, ctx):
                calls.append("w_start")
            async def on_group_start(self, g, ctx):
                calls.append(f"g_start:{g}")
            async def on_node_start(self, node, ctx):
                calls.append(f"n_start:{node.id}")
            async def on_node_end(self, node, res, ctx):
                calls.append(f"n_end:{node.id}")
            async def on_group_end(self, g, ctx):
                calls.append(f"g_end:{g}")
            async def on_workflow_end(self, wf, trace, ctx):
                calls.append("w_end")

        wf = Workflow(
            name="hook-test",
            nodes=[Node(id="a", kind=NodeKind.AGENT), Node(id="b", kind=NodeKind.AGENT)],
            edges=[Edge(from_node="a", to_node="b")],
        )

        async def agent_fn(node_id, ctx, emit=None):
            return f"ok-{node_id}"

        executor = DAGExecutor()
        await executor.execute(wf, agent_fn=agent_fn, hooks=TraceHook())

        assert calls == [
            "w_start",
            "g_start:['a']",
            "n_start:a", "n_end:a",
            "g_end:['a']",
            "g_start:['b']",
            "n_start:b", "n_end:b",
            "g_end:['b']",
            "w_end",
        ]

    async def test_environment_hook_injects_context(self):
        """EnvironmentHook 在 ctx.shared 中注入环境信息。"""
        captured_env = {}

        class CaptureHook(ExecutionHooks):
            async def on_node_start(self, node, ctx):
                if "env" in ctx.shared:
                    captured_env["env"] = ctx.shared["env"]

        wf = Workflow(name="env", nodes=[Node(id="a")], edges=[])

        async def agent_fn(node_id, ctx, emit=None):
            shared = ctx.get("_hook_shared", {})
            return str(shared.get("env", "no-env"))

        executor = DAGExecutor()
        results, _ = await executor.execute(
            wf, agent_fn=agent_fn,
            hooks=EnvironmentHook(),
        )

        assert "os" in results["a"].output

    async def test_permission_hook_blocks_tool(self):
        """PermissionHook 拦截禁止的工具调用。"""
        wf = Workflow(
            name="perm-test",
            nodes=[Node(id="t1", kind=NodeKind.TOOL, tool=ToolConfig(name="dangerous_op"))],
            edges=[],
        )

        async def tool_fn(name, inputs):
            return "executed"

        executor = DAGExecutor()
        results, _ = await executor.execute(
            wf, tool_fn=tool_fn,
            hooks=PermissionHook(blocked_tools={"dangerous_op"}),
        )

        assert not results["t1"].success
        assert "blocked" in results["t1"].error.lower()

    async def test_permission_hook_allowlist(self):
        """白名单模式：只允许列出的工具。"""
        wf = Workflow(
            name="allow-test",
            nodes=[Node(id="t1", kind=NodeKind.TOOL, tool=ToolConfig(name="safe_op"))],
            edges=[],
        )

        async def tool_fn(name, inputs):
            return "executed safely"

        executor = DAGExecutor()
        results, _ = await executor.execute(
            wf, tool_fn=tool_fn,
            hooks=PermissionHook(allowed_tools={"safe_op"}),
        )

        assert results["t1"].success

    async def test_hook_shared_context_persists(self):
        """HookContext.shared 跨节点保持。"""
        class SharedHook(ExecutionHooks):
            async def on_node_start(self, node, ctx):
                ctx.shared.setdefault("call_count", 0)
                ctx.shared["call_count"] += 1

        wf = Workflow(
            name="shared",
            nodes=[Node(id="a"), Node(id="b"), Node(id="c")],
            edges=[Edge(from_node="a", to_node="b"), Edge(from_node="b", to_node="c")],
        )

        async def agent_fn(node_id, ctx, emit=None):
            count = ctx.get("_hook_shared", {}).get("call_count", 0)
            return f"call:{count}"

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn, hooks=SharedHook())

        # 每个节点递增，最后一个应该是 3
        assert "call:3" in results["c"].output


class TestStreaming:
    async def test_agent_receives_emit(self):
        """agent_fn 收到 emit 参数，可以发送流式事件。"""
        events = []

        async def stream_handler(event: StreamEvent):
            events.append(event)

        async def agent_fn(node_id, ctx, emit):
            if emit:
                await emit(StreamEvent(type="thinking", node_id=node_id, content="thinking..."))
            return f"done-{node_id}"

        wf = Workflow(name="stream", nodes=[Node(id="a")], edges=[])

        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn, stream=stream_handler)

        assert results["a"].success
        assert len(events) == 1
        assert events[0].type == "thinking"
        assert "thinking" in events[0].content

    async def test_stream_with_hooks_on_stream(self):
        """Hook 的 on_stream 也能收到流式事件。"""
        streamed = []

        class StreamHook(ExecutionHooks):
            async def on_stream(self, event, ctx):
                streamed.append(event)

        async def agent_fn(node_id, ctx, emit):
            if emit:
                await emit(StreamEvent(type="progress", content="50%"))
            return "ok"

        wf = Workflow(name="hook-stream", nodes=[Node(id="a")], edges=[])

        executor = DAGExecutor()
        await executor.execute(wf, agent_fn=agent_fn, hooks=StreamHook())

        assert len(streamed) == 1
        assert streamed[0].type == "progress"

    async def test_emit_none_is_safe(self):
        """不传 stream 回调时，emit 是 None，agent_fn 应安全处理。"""
        async def agent_fn(node_id, ctx, emit):
            # emit is None when no stream callback given
            if emit:
                await emit(StreamEvent(type="test"))
            return "no-stream"

        wf = Workflow(name="safe", nodes=[Node(id="a")], edges=[])
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=agent_fn)
        assert results["a"].output == "no-stream"

    async def test_old_style_agent_fn_still_works(self):
        """不接受 emit 参数的老 agent_fn 完全兼容。"""
        async def old_agent(node_id, ctx):
            return f"old-{node_id}"

        wf = Workflow(name="old", nodes=[Node(id="a")], edges=[])
        executor = DAGExecutor()
        results, _ = await executor.execute(wf, agent_fn=old_agent)
        assert results["a"].success
        assert "old-a" in results["a"].output


class TestLoggingHook:
    async def test_logging_hook_output(self):
        """LoggingHook 在关键点输出日志。"""
        import logging
        import io
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("agentflow.hooks.test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        hook = LoggingHook(logger=logger)

        wf = Workflow(name="log", nodes=[Node(id="a")], edges=[])

        async def agent_fn(node_id, ctx, emit=None):
            return "test"

        executor = DAGExecutor()
        await executor.execute(wf, agent_fn=agent_fn, hooks=hook)

        output = stream.getvalue()
        assert "started" in output
        assert "Node 'a'" in output
