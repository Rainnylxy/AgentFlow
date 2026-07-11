"""Tool Registry：统一管理 MCP Server / REST API / 本地函数三种工具"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ToolType(str, Enum):
    MCP = "mcp"
    REST = "rest"
    LOCAL = "local"


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
        """用 Pydantic 模型校验参数，校验失败抛出 ValidationError。"""
        if self.params_model is not None:
            validated = self.params_model(**inputs)
            return validated.model_dump()
        return inputs


@dataclass
class ToolResult:
    success: bool
    output: Optional[str] = None
    error: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

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

    async def execute(self, name: str, inputs: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        try:
            # Pydantic 校验
            validated_inputs = tool.validate_params(inputs)

            if tool.tool_type == ToolType.LOCAL and tool.func:
                output = tool.func(**validated_inputs)
                return ToolResult(success=True, output=str(output))
            elif tool.tool_type == ToolType.REST and tool.endpoint:
                output = await self._execute_rest(tool, validated_inputs)
                return ToolResult(success=True, output=output)
            elif tool.tool_type == ToolType.MCP:
                output = await self._execute_mcp(tool, validated_inputs)
                return ToolResult(success=True, output=output)
            return ToolResult(success=False, error=f"Unsupported tool type: {tool.tool_type}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _execute_rest(self, tool: Tool, inputs: dict) -> str:
        """预留 REST 调用实现。"""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(tool.endpoint, json=inputs)
            resp.raise_for_status()
            return resp.text

    async def _execute_mcp(self, tool: Tool, inputs: dict) -> str:
        """预留 MCP 调用实现。"""
        return f"[MCP] {tool.endpoint}: {inputs}"
