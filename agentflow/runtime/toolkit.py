"""ToolKit: @tool 装饰器 + 统一工具集合，支持本地/MCP/REST 三源统一。"""

import inspect
from typing import Callable, Optional, Any, List, get_type_hints

from agentflow.runtime.tool_registry import Tool, ToolType, ToolResult, ToolRegistry

# Re-export for convenience
__all__ = ["tool", "ToolKit", "Tool", "ToolType", "ToolResult"]


def _type_to_json_schema(py_type) -> dict:
    """将 Python 类型注解转为 JSON Schema 基本类型。"""
    mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        dict: "object",
        list: "array",
        type(None): "null",
    }
    type_str = mapping.get(py_type, "string")
    return {"type": type_str}


def _function_to_parameters(func: Callable) -> dict:
    """从函数签名 + 类型注解推导 JSON Schema parameters。"""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        py_type = hints.get(name, str)
        prop = _type_to_json_schema(py_type)
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        properties[name] = prop

    return {"type": "object", "properties": properties, "required": required}


def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    params: Optional[Any] = None,
) -> Tool:
    """将 Python 函数一键转换为 AgentFlow Tool。

    用法:
        @tool
        def lookup(query: str) -> str:
            \"\"\"Search the knowledge base.\"\"\"
            ...

        @tool(name="custom_name", params=PydanticModel)
        def my_func(...) -> ...:
            ...
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        raw_desc = (func.__doc__ or "").strip()
        tool_desc = description or (raw_desc.split("\n")[0] if raw_desc else "")
        tool_params = {}

        if params is not None:
            # Pydantic model -> JSON Schema
            try:
                tool_params = params.model_json_schema()
            except AttributeError:
                # fallback for Pydantic v1
                tool_params = params.schema()
        else:
            tool_params = _function_to_parameters(func)

        t = Tool(
            name=tool_name,
            description=tool_desc,
            tool_type=ToolType.LOCAL,
            func=func,
            parameters=tool_params,
            params_model=params,
        )
        return t

    if _func is not None:
        return decorator(_func)
    return decorator


class ToolKit:
    """统一的工具集合，支持本地/MCP/REST 三源统一注册。"""

    def __init__(self):
        self._registry = ToolRegistry()

    def add(self, tool_: Tool) -> "ToolKit":
        self._registry.register(tool_)
        return self

    def list(self) -> List[Tool]:
        return self._registry.list_tools()

    def execute(self, name: str, inputs: dict) -> ToolResult:
        return self._registry.execute(name, inputs)

    def list_for_llm(self) -> List[dict]:
        """生成 OpenAI function-calling 格式的工具列表。"""
        schemas = []
        for t in self._registry.list_tools():
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return schemas
