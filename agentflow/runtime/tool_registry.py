"""Tool Registry：统一管理 MCP Server / REST API / 本地函数三种工具"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agentflow.runtime.security.engine import PolicyEngine
    from agentflow.runtime.security.policy import CallContext

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    MCP = "mcp"
    REST = "rest"
    LOCAL = "local"


# JSON Schema type → (Python canonical type, display name)
_SCHEMA_TYPE_MAP: dict[str, tuple] = {
    "string": (str, "string"),
    "integer": (int, "integer"),
    "number": (float, "number"),
    "boolean": (bool, "boolean"),
    "array": (list, "array"),
    "object": (dict, "object"),
}


def _check_type(value, schema_type: str, field_name: str) -> None:
    """校验 value 的类型是否匹配 JSON Schema type，不匹配抛出 TypeError。"""
    if schema_type not in _SCHEMA_TYPE_MAP:
        return  # 未知类型跳过
    expected_type, display_name = _SCHEMA_TYPE_MAP[schema_type]

    if expected_type is float:
        # number 接受 int 和 float
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Parameter '{field_name}': expected {display_name}, "
                f"got {type(value).__name__} ({value!r})"
            )
        return

    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{field_name}': expected {display_name}, "
            f"got {type(value).__name__} ({value!r})"
        )


def _validate_by_schema(inputs: dict, parameters: dict) -> dict:
    """按 JSON Schema 校验输入参数，校验失败抛出 ValueError / TypeError。"""
    props = parameters.get("properties", {})
    required_fields = parameters.get("required", [])
    validated = dict(inputs)

    # 1. 必填字段检查
    for field in required_fields:
        if field not in validated:
            raise ValueError(
                f"Missing required parameter: '{field}'. "
                f"Required: {required_fields}"
            )

    # 2. 每个字段类型检查
    for field, value in list(validated.items()):
        field_schema = props.get(field)
        if field_schema is None:
            continue  # 未知字段放行，LLM 可能加多余字段
        schema_type = field_schema.get("type")
        if schema_type:
            _check_type(value, schema_type, field)

    # 3. 填充默认值
    for field, field_schema in props.items():
        if field not in validated and "default" in field_schema:
            validated[field] = field_schema["default"]

    return validated


@dataclass
class Tool:
    name: str
    description: str
    tool_type: ToolType
    func: Optional[Callable] = None
    endpoint: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    params_model: Optional[Any] = None  # Pydantic BaseModel subclass for validation

    def validate_params(self, inputs: dict) -> dict:
        """校验参数：Pydantic Model 优先，否则按 JSON Schema 校验。"""
        if self.params_model is not None:
            validated = self.params_model(**inputs)
            return validated.model_dump()
        if self.parameters:
            return _validate_by_schema(inputs, self.parameters)
        return inputs


@dataclass
class ToolResult:
    success: bool
    output: Optional[str] = None
    error: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._policy_engine: Optional[PolicyEngine] = None

    def attach_policy_engine(self, engine: PolicyEngine) -> None:
        """Attach a PolicyEngine for pre-execution checks and audit logging.

        Once attached, every ``execute()`` call that provides a *context*
        will be checked against registered SecurityPolicies.
        """
        self._policy_engine = engine

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    async def execute(
        self, name: str, inputs: dict,
        context: Optional[CallContext] = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        # --- Security check (pre-execution) ---
        if self._policy_engine is not None and context is not None:
            policy_result = self._policy_engine.check(name, context, inputs)
            if policy_result.verdict == "deny":
                return ToolResult(success=False, error=f"Security: {policy_result.reason}")
            if policy_result.verdict == "pending_approval":
                return ToolResult(
                    success=False,
                    error=f"Security: approval required (id={policy_result.approval_id})",
                )
            if policy_result.sanitized_params is not None:
                inputs = policy_result.sanitized_params

        try:
            # Pydantic 校验
            validated_inputs = tool.validate_params(inputs)

            tool_result: ToolResult
            if tool.tool_type == ToolType.LOCAL and tool.func:
                output = tool.func(**validated_inputs)
                tool_result = ToolResult(success=True, output=str(output))
            elif tool.tool_type == ToolType.REST and tool.endpoint:
                output = await self._execute_rest(tool, validated_inputs)
                tool_result = ToolResult(success=True, output=output)
            elif tool.tool_type == ToolType.MCP:
                output = await self._execute_mcp(tool, validated_inputs)
                tool_result = ToolResult(success=True, output=output)
            else:
                tool_result = ToolResult(
                    success=False, error=f"Unsupported tool type: {tool.tool_type}"
                )
        except Exception as e:
            tool_result = ToolResult(success=False, error=str(e))

        # --- Audit (post-execution) ---
        if self._policy_engine is not None and context is not None:
            self._policy_engine.audit(
                name, context, inputs,
                success=tool_result.success,
                output=tool_result.output or "",
                error=tool_result.error,
            )

        return tool_result

    async def _execute_rest(self, tool: Tool, inputs: dict) -> str:
        """预留 REST 调用实现。"""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(tool.endpoint, json=inputs)
            resp.raise_for_status()
            return resp.text

    async def _execute_mcp(self, tool: Tool, inputs: dict) -> str:
        """通过 MCPServerManager 调用 MCP 工具。

        tool.endpoint 格式: "<server_name>:<tool_name>"
        如 "github:search_repositories"
        """
        if not tool.endpoint or ":" not in tool.endpoint:
            raise ValueError(
                f"MCP tool '{tool.name}' requires endpoint in format "
                f"'<server_name>:<tool_name>', got '{tool.endpoint}'"
            )
        server_name, mcp_tool_name = tool.endpoint.split(":", 1)
        manager = MCPServerManager.get_instance()
        return await manager.call_tool(server_name, mcp_tool_name, inputs)

    def register_mcp_tools(self, server_name: str) -> list[Tool]:
        """从 .mcp.json 配置注册指定 MCP server 的所有工具。"""
        manager = MCPServerManager.get_instance()
        tools_list = asyncio.run(manager.list_tools(server_name))
        registered = []
        for td in tools_list:
            mcp_tool = Tool(
                name=td.get("name", ""),
                description=td.get("description", ""),
                tool_type=ToolType.MCP,
                endpoint=f"{server_name}:{td.get('name', '')}",
                parameters=td.get("inputSchema", {}),
            )
            self.register(mcp_tool)
            registered.append(mcp_tool)
        return registered

    def shutdown_mcp(self) -> None:
        """关闭所有 MCP server 连接。"""
        try:
            manager = MCPServerManager.get_instance()
            asyncio.run(manager.shutdown())
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MCP Server Manager — stdio + JSON-RPC 2.0
# ---------------------------------------------------------------------------

_MCP_REQUEST_TIMEOUT = 30.0


class MCPServerManager:
    """管理 MCP stdio 子进程，提供 JSON-RPC 2.0 通信。

    单例模式，从项目根目录的 .mcp.json 读取服务器配置。

    用法:
        manager = MCPServerManager.get_instance()
        tools = await manager.list_tools("github")
        result = await manager.call_tool("github", "search_repositories", {"query": "AgentFlow"})
        await manager.shutdown()
    """

    _instance: Optional["MCPServerManager"] = None

    @classmethod
    def get_instance(cls) -> "MCPServerManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._servers: dict[str, dict] = {}       # server_name → {process, session_id, next_id}
        self._config: dict[str, dict] = {}         # server_name → {command, args, env}
        self._load_config()

    def _load_config(self) -> None:
        """从 .mcp.json 加载 MCP server 配置。"""
        config_path = Path(".mcp.json")
        if not config_path.exists():
            logger.debug("No .mcp.json found, MCP servers not configured")
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse .mcp.json: %s", e)
            return
        for name, cfg in data.get("mcpServers", {}).items():
            self._config[name] = {
                "command": cfg.get("command", ""),
                "args": cfg.get("args", []),
                "env": cfg.get("env", {}),
            }

    async def _ensure_started(self, server_name: str) -> None:
        """按需启动 MCP server 并完成初始化握手。"""
        if server_name in self._servers:
            return

        cfg = self._config.get(server_name)
        if not cfg:
            raise ValueError(
                f"MCP server '{server_name}' not found in .mcp.json. "
                f"Available: {list(self._config.keys())}"
            )

        # 解析环境变量
        env = os.environ.copy()
        for k, v in cfg.get("env", {}).items():
            if v.startswith("${") and v.endswith("}"):
                env_var = v[2:-1]
                env[k] = os.getenv(env_var, "")
            else:
                env[k] = v

        # 启动子进程
        cmd = [cfg["command"]] + list(cfg.get("args", []))
        logger.info("Starting MCP server '%s': %s", server_name, " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # 初始化握手
        session_id = ""
        try:
            init_response = await self._send_request(
                process, "initialize",
                {"protocolVersion": "2024-11-05",
                 "capabilities": {}, "clientInfo": {"name": "agentflow", "version": "0.1.0"}},
                timeout=_MCP_REQUEST_TIMEOUT,
            )
            session_id = (
                init_response.get("result", {}).get("protocolVersion", "")
                or init_response.get("result", {}).get("serverInfo", {}).get("name", "")
            )
            # 发送 initialized 通知
            self._send_notification(process, "notifications/initialized", {})
        except Exception:
            process.kill()
            raise

        self._servers[server_name] = {
            "process": process,
            "session_id": session_id,
            "next_id": 1,
        }

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取 MCP server 的工具列表。"""
        await self._ensure_started(server_name)
        srv = self._servers[server_name]
        response = await self._send_request(
            srv["process"], "tools/list", {},
            timeout=_MCP_REQUEST_TIMEOUT,
        )
        tools = response.get("result", {}).get("tools", [])
        return tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict,
    ) -> str:
        """调用 MCP server 上的指定工具。"""
        await self._ensure_started(server_name)
        srv = self._servers[server_name]
        response = await self._send_request(
            srv["process"], "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=_MCP_REQUEST_TIMEOUT,
        )
        result = response.get("result", {})
        content = result.get("content", [])
        # 提取文本内容
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
        return json.dumps(result, ensure_ascii=False)

    async def shutdown(self) -> None:
        """关闭所有 MCP server 子进程。"""
        for name, srv in list(self._servers.items()):
            process = srv["process"]
            logger.info("Shutting down MCP server '%s'", name)
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            except Exception:
                pass
        self._servers.clear()
        MCPServerManager._instance = None

    # -- JSON-RPC 2.0 helpers --

    @staticmethod
    async def _send_request(
        process: asyncio.subprocess.Process,
        method: str,
        params: dict,
        timeout: float = 30.0,
    ) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        request_bytes = (json.dumps(request) + "\n").encode("utf-8")
        process.stdin.write(request_bytes)
        await process.stdin.drain()

        try:
            line = await asyncio.wait_for(
                process.stdout.readline(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MCP request '{method}' timed out after {timeout}s"
            )

        if not line:
            raise ConnectionError(
                f"MCP server closed stdout during '{method}'"
            )
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON-RPC response for '{method}': {e}"
            )

    @staticmethod
    def _send_notification(
        process: asyncio.subprocess.Process,
        method: str,
        params: dict,
    ) -> None:
        """发送 JSON-RPC 通知（无 id，不需响应）。"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        request_bytes = (json.dumps(notification) + "\n").encode("utf-8")
        process.stdin.write(request_bytes)
