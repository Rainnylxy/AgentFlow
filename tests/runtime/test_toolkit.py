import pytest
from agentflow.runtime.toolkit import tool, ToolKit
from agentflow.runtime.tool_registry import Tool, ToolType


class TestToolDecorator:
    def test_decorator_basic(self):
        """装饰器从函数签名自动推导 name / description / parameters。"""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers together."
        assert add.tool_type == ToolType.LOCAL
        assert add.parameters["type"] == "object"
        assert "a" in add.parameters["properties"]
        assert "b" in add.parameters["properties"]

    def test_decorator_executes(self):
        """装饰后的工具仍可被调用执行。"""
        from agentflow.runtime.tool_registry import ToolRegistry

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        reg = ToolRegistry()
        reg.register(greet)
        result = reg.execute("greet", {"name": "World"})
        assert result.success
        assert "Hello, World!" == result.output

    def test_decorator_override_name(self):
        """可以手动覆盖工具名。"""

        @tool(name="my_add")
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        assert add.name == "my_add"

    def test_decorator_with_pydantic(self):
        """Pydantic 参数模型精确控制 schema 并提供运行时校验。"""
        from pydantic import BaseModel, Field

        class WeatherParams(BaseModel):
            city: str = Field(description="City name")
            unit: str = Field(default="celsius")

        @tool(params=WeatherParams, description="Get current weather")
        def get_weather(city: str, unit: str = "celsius") -> str:
            return f"{city}: 22°{unit}"

        assert get_weather.params_model is not None
        # 校验通过
        validated = get_weather.validate_params({"city": "Beijing"})
        assert validated["unit"] == "celsius"
        # 校验失败
        with pytest.raises(Exception):
            get_weather.validate_params({"wrong_key": "x"})


class TestToolKit:
    def test_toolkit_add_and_list(self):
        """ToolKit 容器：注册和列出工具。"""

        @tool
        def func_a(x: int) -> int:
            """A function."""
            return x * 2

        @tool
        def func_b(s: str) -> str:
            """B function."""
            return s.upper()

        kit = ToolKit()
        kit.add(func_a)
        kit.add(func_b)

        tools = kit.list()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"func_a", "func_b"}

    def test_toolkit_list_for_llm(self):
        """生成 OpenAI function-calling 格式的工具列表。"""

        @tool
        def lookup(query: str) -> str:
            """Search the knowledge base."""
            return f"Result for {query}"

        kit = ToolKit()
        kit.add(lookup)
        schemas = kit.list_for_llm()

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "lookup"
        assert schema["function"]["description"] == "Search the knowledge base."

    def test_toolkit_execute(self):
        """ToolKit 内部委托给 ToolRegistry 执行。"""

        @tool
        def echo(text: str) -> str:
            """Echo back."""
            return text

        kit = ToolKit()
        kit.add(echo)
        result = kit.execute("echo", {"text": "hello"})
        assert result.success
        assert result.output == "hello"

    def test_toolkit_execute_not_found(self):
        """执行不存在的工具返回错误。"""
        kit = ToolKit()
        result = kit.execute("nobody", {})
        assert not result.success
        assert "not found" in result.error
