import pytest
from agentflow.runtime.tool_registry import ToolRegistry, Tool, ToolType, ToolResult


class TestTool:
    def test_local_tool(self):
        t = Tool(name="add", description="Add numbers", tool_type=ToolType.LOCAL,
                 func=lambda a, b: a + b,
                 parameters={"properties": {"a": {"type": "int"}, "b": {"type": "int"}}})
        assert t.tool_type == ToolType.LOCAL

    def test_rest_tool(self):
        t = Tool(name="weather", description="Get weather", tool_type=ToolType.REST,
                 endpoint="https://api.weather.com/v1")
        assert t.endpoint == "https://api.weather.com/v1"

    def test_tool_with_pydantic_validation(self):
        from pydantic import BaseModel, Field

        class AddParams(BaseModel):
            a: int = Field(description="First number")
            b: int = Field(description="Second number")

        t = Tool(name="add", description="Add numbers", tool_type=ToolType.LOCAL,
                 func=lambda a, b: a + b,
                 params_model=AddParams)

        validated = t.validate_params({"a": 1, "b": 2})
        assert validated == {"a": 1, "b": 2}

        with pytest.raises(Exception):
            t.validate_params({"a": "not_a_number", "b": 2})


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register(Tool(name="t1", description="d1", tool_type=ToolType.LOCAL, func=lambda x: x))
        reg.register(Tool(name="t2", description="d2", tool_type=ToolType.LOCAL, func=lambda x: x))
        assert len(reg.list_tools()) == 2

    def test_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(Tool(name="x", description="d", tool_type=ToolType.LOCAL, func=lambda x: x))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(Tool(name="x", description="d", tool_type=ToolType.LOCAL, func=lambda x: x))

    async def test_execute_local(self):
        reg = ToolRegistry()
        reg.register(Tool(name="greet", description="Greet", tool_type=ToolType.LOCAL,
                          func=lambda name: f"Hello, {name}!"))
        result = await reg.execute("greet", {"name": "World"})
        assert result.success is True
        assert result.output == "Hello, World!"

    async def test_execute_not_found(self):
        reg = ToolRegistry()
        result = await reg.execute("ghost", {})
        assert result.success is False
        assert "not found" in result.error

    async def test_execute_with_pydantic_validation(self):
        from pydantic import BaseModel, Field

        class AddParams(BaseModel):
            a: int = Field(description="First number")
            b: int = Field(description="Second number")

        reg = ToolRegistry()
        reg.register(Tool(name="add", description="Add numbers", tool_type=ToolType.LOCAL,
                          func=lambda a, b: a + b,
                          params_model=AddParams))
        result = await reg.execute("add", {"a": 1, "b": 2})
        assert result.success is True
        assert result.output == "3"

        result = await reg.execute("add", {"a": "not_a_number", "b": 2})
        assert result.success is False
