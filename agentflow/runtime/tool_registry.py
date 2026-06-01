"""Tool Registry：统一管理 MCP Server / REST API / 本地函数三种工具"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


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

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, name: str, inputs: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        try:
            if tool.tool_type == ToolType.LOCAL and tool.func:
                output = tool.func(**inputs)
                return ToolResult(success=True, output=str(output))
            return ToolResult(success=False, error=f"Unsupported tool type: {tool.tool_type}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
