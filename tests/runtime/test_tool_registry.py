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


class TestSchemaValidation:
    """JSON Schema 自动校验 —— 无 Pydantic Model 时的防线。"""

    def _make_tool(self, func, **kwargs):
        return Tool(
            name=func.__name__,
            description=func.__doc__ or "",
            tool_type=ToolType.LOCAL,
            func=func,
            **kwargs,
        )

    def test_required_field_missing(self):
        """缺少必填参数"""
        t = self._make_tool(
            func=lambda name: f"Hello, {name}!",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        with pytest.raises(ValueError, match="Missing required parameter: 'name'"):
            t.validate_params({})

    def test_type_mismatch_string_got_int(self):
        """LLM 传了整数给字符串参数"""
        t = self._make_tool(
            func=lambda city: f"Weather: {city}",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        with pytest.raises(TypeError, match="expected string, got int"):
            t.validate_params({"city": 123})

    def test_type_mismatch_integer_got_string(self):
        """LLM 传了字符串给整数参数"""
        t = self._make_tool(
            func=lambda limit: limit,
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": ["limit"],
            },
        )
        with pytest.raises(TypeError, match="expected integer, got str"):
            t.validate_params({"limit": "fifty"})

    def test_number_accepts_int(self):
        """number 类型接受 int 和 float"""
        t = self._make_tool(
            func=lambda price: price,
            parameters={
                "type": "object",
                "properties": {"price": {"type": "number"}},
                "required": ["price"],
            },
        )
        assert t.validate_params({"price": 99}) == {"price": 99}
        assert t.validate_params({"price": 99.9}) == {"price": 99.9}

    def test_boolean_rejects_string(self):
        """bool 参数拒绝字符串"""
        t = self._make_tool(
            func=lambda force: force,
            parameters={
                "type": "object",
                "properties": {"force": {"type": "boolean"}},
            },
        )
        with pytest.raises(TypeError, match="expected boolean, got str"):
            t.validate_params({"force": "true"})

    def test_default_value_injected(self):
        """schema 中有 default 值时自动填充"""
        t = self._make_tool(
            func=lambda query, limit=10: f"{query}:{limit}",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        )
        result = t.validate_params({"query": "test"})
        assert result["query"] == "test"
        assert result["limit"] == 10

    def test_unknown_field_passthrough(self):
        """LLM 多传的字段放行，不报错"""
        t = self._make_tool(
            func=lambda name: name,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        result = t.validate_params({"name": "test", "extra": "ignored"})
        assert result["name"] == "test"
        assert "extra" in result

    def test_hallucinated_field_name_does_not_crash(self):
        """LLM 拼错参数名 —— 不会抛异常（因为校验不到该字段），
        但 extra 字段会传给函数，由 Python 函数签名兜底。"""
        t = self._make_tool(
            func=lambda name: name,
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        # tyop_field 没在 schema 中，不会报错；但 name 缺失会报错
        with pytest.raises(ValueError, match="Missing required parameter"):
            t.validate_params({"naem": "test"})

    async def test_end_to_end_schema_validation(self):
        """端到端：registry.execute 正确拦截 LLM 传的错误类型"""
        reg = ToolRegistry()
        reg.register(Tool(
            name="search",
            description="Search",
            tool_type=ToolType.LOCAL,
            func=lambda query, limit=10: f"Found {limit} results for {query}",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ))

        # 正确参数
        result = await reg.execute("search", {"query": "Python"})
        assert result.success is True
        assert "Python" in result.output

        # LLM 把 limit 传成了字符串
        result = await reg.execute("search", {"query": "Python", "limit": "twenty"})
        assert result.success is False
        assert "expected integer" in result.error
