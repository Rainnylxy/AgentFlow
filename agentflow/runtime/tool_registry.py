"""Tool Registry：统一管理 MCP Server / REST API / 本地函数三种工具"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


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
