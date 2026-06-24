"""执行生命周期 Hook — AgentFlow 的扩展原语。

用户实现 ExecutionHooks 子类，注入到 DAGExecutor.execute()，
在 Workflow / Group / Node / Tool 的关键节点插入自己的逻辑。

这不是框架实现"审批/环境感知/日志"，而是框架提供钩子，
用户用 Agent 来填钩子。

用法:
    class MyHooks(ExecutionHooks):
        async def on_node_start(self, ctx):
            # 注入 git status 到 context
            ctx["env"] = get_environment_info()

        async def on_tool_call(self, tool_name, inputs):
            # 审批：检查是否允许调用
            if tool_name == "delete_db":
                raise PermissionError("不允许删除数据库")

    executor.execute(workflow, agent_fn=my_agent, hooks=MyHooks())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from agentflow.dsl.types import Workflow, Node


# ---------------------------------------------------------------------------
# Hook 上下文
# ---------------------------------------------------------------------------

@dataclass
class HookContext:
    """传递给每个 hook 的上下文信息。"""
    workflow_name: str = ""
    node_id: str = ""
    node_kind: str = ""
    group: list[str] = field(default_factory=list)
    # 用户可在 hook 中读写此字典，在后续 hook 和 agent_fn 中可见
    shared: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 流式事件
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """流式输出的单条事件。"""
    type: str           # "thinking" | "tool_call" | "tool_result" | "node_output" | "progress"
    node_id: str = ""
    content: str = ""
    data: dict = field(default_factory=dict)


StreamCallback = Callable[[StreamEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Hook 基类
# ---------------------------------------------------------------------------

class ExecutionHooks:
    """执行生命周期钩子基类。

    所有方法默认空实现。用户按需重写。
    方法按调用顺序排列。
    """

    # -- Workflow 级 --
    async def on_workflow_start(self, workflow: Workflow, ctx: HookContext) -> None:
        """Workflow 开始执行前。"""

    async def on_workflow_end(self, workflow: Workflow, trace: Any, ctx: HookContext) -> None:
        """Workflow 执行完成后。"""

    # -- Group 级（每层并行组） --
    async def on_group_start(self, group: list[str], ctx: HookContext) -> None:
        """每层 DAG 组开始前。"""

    async def on_group_end(self, group: list[str], ctx: HookContext) -> None:
        """每层 DAG 组完成后。"""

    # -- Node 级 --
    async def on_node_start(self, node: Node, ctx: HookContext) -> None:
        """节点执行前。可在此注入上下文到 ctx.shared。"""

    async def on_node_end(self, node: Node, result: Any, ctx: HookContext) -> None:
        """节点执行完成后。result 为 NodeResult。"""

    # -- Tool 级 --
    async def on_tool_call(self, tool_name: str, inputs: dict, ctx: HookContext) -> None:
        """工具调用前。可在此做权限检查、参数校验。"""

    async def on_tool_result(self, tool_name: str, result: Any, ctx: HookContext) -> None:
        """工具调用完成后。"""

    # -- 流式 --
    async def on_stream(self, event: StreamEvent, ctx: HookContext) -> None:
        """收到流式事件。默认忽略，子类可实现 WebSocket 推送等。"""


# ---------------------------------------------------------------------------
# 常见 Hook 实现
# ---------------------------------------------------------------------------

class EnvironmentHook(ExecutionHooks):
    """环境感知 Hook：在节点执行前注入系统环境信息。"""

    async def on_node_start(self, node: Node, ctx: HookContext) -> None:
        import os
        import platform
        from datetime import datetime

        ctx.shared["env"] = {
            "os": platform.system(),
            "cwd": os.getcwd(),
            "date": datetime.now().isoformat(),
            "python_version": platform.python_version(),
        }


class LoggingHook(ExecutionHooks):
    """日志 Hook：在关键生命周期点打印日志。"""

    def __init__(self, logger=None):
        import logging
        self._log = logger or logging.getLogger("agentflow.hooks")

    async def on_workflow_start(self, workflow, ctx):
        self._log.info("Workflow '%s' started (%d nodes)", workflow.name, len(workflow.nodes))

    async def on_group_start(self, group, ctx):
        self._log.debug("Group: %s", group)

    async def on_node_start(self, node, ctx):
        self._log.info("Node '%s' (%s) starting", node.id, node.kind.value)

    async def on_node_end(self, node, result, ctx):
        status = "OK" if result.success else "FAIL"
        self._log.info("Node '%s' → %s (%dms)", node.id, status, result.duration_ms)

    async def on_workflow_end(self, workflow, trace, ctx):
        self._log.info("Workflow '%s' done (%dms)", workflow.name, trace.total_duration_ms)


class PermissionHook(ExecutionHooks):
    """权限 Hook：在工具调用前检查权限，拒绝危险调用。"""

    def __init__(self, allowed_tools: set[str] | None = None,
                 blocked_tools: set[str] | None = None):
        self.allowed = allowed_tools
        self.blocked = blocked_tools or set()

    async def on_tool_call(self, tool_name, inputs, ctx):
        if tool_name in self.blocked:
            raise PermissionError(f"Tool '{tool_name}' is blocked")
        if self.allowed is not None and tool_name not in self.allowed:
            raise PermissionError(f"Tool '{tool_name}' is not in allowed list")
