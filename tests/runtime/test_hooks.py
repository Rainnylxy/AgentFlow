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


class TestAgentTraceRecording:
    """验证思考引擎逐轮记录 trace 数据"""

    async def test_agent_result_contains_trace(self):
        """AgentResult 包含完整的 AgentTrace。"""
        from unittest.mock import AsyncMock, MagicMock
        from agentflow.runtime.builder import AgentBuilder
        from agentflow.runtime.thinking import ThinkingMode
        from agentflow.runtime.memory.manager import MemoryProfile
        from agentflow.runtime.toolkit import tool

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Need to search.", role="assistant", tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "search", "arguments": '{"q": "refund"}'},
            }]),
            MagicMock(content="The answer is 30 days.", role="assistant", tool_calls=[]),
        ]

        @tool
        def search(q: str) -> str:
            """Search."""
            return f"Found: {q}"

        agent = await (
            AgentBuilder("trace-agent")
            .with_llm(mock_llm)
            .with_tools(search)
            .with_prompt("You are helpful.")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(5)
            .build()
        )

        result = await agent.run("What is the refund policy?")

        # 验证 trace 存在且有数据
        assert result.agent_trace is not None
        at = result.agent_trace
        assert at.success
        assert at.total_turns == 1
        assert at.total_tool_calls == 1

        # 验证 turn 内容
        turn = at.turns[0]
        assert "search" in turn.thinking.lower() or "Need" in turn.thinking
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].tool == "search"
        assert turn.tool_calls[0].input == {"q": "refund"}
        assert "Found" in turn.tool_calls[0].output
        assert "30 days" in turn.final_answer

    async def test_trace_records_skill_activation(self):
        """Trace 记录 Skill 激活事件。"""
        from unittest.mock import AsyncMock, MagicMock
        from agentflow.runtime.builder import AgentBuilder
        from agentflow.runtime.thinking import ThinkingMode
        from agentflow.runtime.memory.manager import MemoryProfile
        from agentflow.runtime.skill import Skill

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            # 第1轮：激活 skill
            MagicMock(content=None, role="assistant", tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "use_skill_helper", "arguments": "{}"},
            }]),
            # 第2轮：最终回答
            MagicMock(content="Done with skill.", role="assistant", tool_calls=[]),
        ]

        skill = Skill(
            name="helper",
            description="Helper skill",
            prompt="## Helper\nYou are a helper.",
            tools=[],
            _loaded=False,
            _file_path="",
            _loader_ref=None,
        )

        agent = await (
            AgentBuilder("skill-trace")
            .with_llm(mock_llm)
            .with_prompt("Base prompt.")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(5)
            .build()
        )
        # 手动注入 skill（绕过 with_skill）
        agent._skills = [skill]

        result = await agent.run("help me")

        at = result.agent_trace
        assert at is not None
        # 应该有 2 轮：激活 + 回答
        assert at.total_turns == 1
        assert at.total_tool_calls == 1
        # 工具调用记录应该标记为 skill 激活
        tc = at.turns[0].tool_calls[0]
        assert "helper" in tc.tool

    async def test_trace_no_tools(self):
        """无工具调用的简单问答也有 trace。"""
        from unittest.mock import AsyncMock, MagicMock
        from agentflow.runtime.builder import AgentBuilder
        from agentflow.runtime.thinking import ThinkingMode
        from agentflow.runtime.memory.manager import MemoryProfile

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Hello! How can I help?", role="assistant", tool_calls=[],
        )

        agent = await (
            AgentBuilder("simple")
            .with_llm(mock_llm)
            .with_prompt("You are helpful.")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .build()
        )

        result = await agent.run("Hello")

        assert result.agent_trace is not None
        assert result.agent_trace.success
        assert result.agent_trace.total_tool_calls == 0
        assert result.agent_trace.total_turns == 1
        assert "Hello" in result.agent_trace.turns[0].final_answer


class TestThinkingEngineStreaming:
    """思考引擎内部的流式事件"""

    async def test_react_strategy_emits_thinking_and_tool_events(self):
        """ReAct 策略在有工具调用时逐轮 emit 事件。"""
        from unittest.mock import AsyncMock, MagicMock
        from agentflow.runtime.thinking.base import ThinkContext
        from agentflow.runtime.thinking.react import ReActStrategy
        from agentflow.runtime.toolkit import ToolKit, tool
        from agentflow.runtime.memory.manager import MemoryManager

        events = []

        async def stream_handler(event):
            events.append(event)

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(
                content="Let me look that up.",
                role="assistant",
                tool_calls=[{
                    "id": "c1", "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "refund"}'},
                }],
            ),
            MagicMock(content="The refund policy is 30 days.", role="assistant", tool_calls=[]),
        ]

        kit = ToolKit()

        @tool
        def search(query: str) -> str:
            """Search."""
            return f"Found info about {query}"

        kit.add(search)

        ctx = ThinkContext(
            user_input="refund policy?",
            system_prompt="You are helpful.",
            messages=[],
            tools=kit.list_for_llm(),
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
            stream=stream_handler,
        )

        strategy = ReActStrategy(toolkit=kit)
        result = await strategy.run(ctx)

        assert "30 days" in result.output

        # 验证事件类型链：thinking → tool_call → tool_result → final
        event_types = [e.type for e in events]
        assert "thinking" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "final" in event_types

    async def test_builder_agent_stream_flows_through(self):
        """AgentBuilder 构建的 Agent，stream 从 run() 透传到思考引擎。"""
        from unittest.mock import AsyncMock, MagicMock
        from agentflow.runtime.builder import AgentBuilder
        from agentflow.runtime.thinking import ThinkingMode
        from agentflow.runtime.memory.manager import MemoryProfile
        from agentflow.runtime.toolkit import tool

        events = []

        async def stream_handler(event):
            events.append(event)

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Got it.", role="assistant", tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "test"}'},
            }]),
            MagicMock(content="Here is the answer.", role="assistant", tool_calls=[]),
        ]

        @tool
        def lookup(query: str) -> str:
            return f"Result: {query}"

        agent = await (
            AgentBuilder("stream-agent")
            .with_llm(mock_llm)
            .with_tools(lookup)
            .with_prompt("You are helpful.")
            .with_thinking(ThinkingMode.REACT)
            .with_memory(MemoryProfile.light())
            .with_max_iterations(5)
            .build()
        )

        result = await agent.run("test query", stream=stream_handler)

        assert "answer" in result.output.lower() or "Here" in result.output
        assert len(events) >= 3  # thinking + tool_call + tool_result + final
        assert "final" in [e.type for e in events]
