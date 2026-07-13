# Agent Builder 优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 AgentBuilder 门面 + 四个独立子系统（ToolKit / Memory / Prompt / Thinking）重构 Agent 构建流程，消灭样板代码，让构建一个生产级 Agent 只需 5 行代码。

**Architecture:** 组件化架构。每个子系统有自己的配置、工厂和接口，AgentBuilder 作为门面协调各组件并完成依赖注入。四个子系统按顺序独立实施：ToolKit → Memory → Prompt → Thinking → 最终整合 Builder。

**Tech Stack:** Python 3.10+, Pydantic, Jinja2, pytest + pytest-asyncio, Chroma (optional)

**Specs:**

- [[2026-06-01-agent-builder-design]]
- [[2026-06-01-toolkit-design]]
- [[2026-06-01-memory-design]]
- [[2026-06-01-prompt-design]]
- [[2026-06-01-thinking-design]]

---

## 文件结构全景

```
Create:
  agentflow/runtime/toolkit.py
  agentflow/runtime/memory/__init__.py
  agentflow/runtime/memory/working.py
  agentflow/runtime/memory/episodic.py
  agentflow/runtime/memory/semantic.py
  agentflow/runtime/memory/manager.py
  agentflow/runtime/prompt/__init__.py
  agentflow/runtime/prompt/section.py
  agentflow/runtime/thinking/__init__.py
  agentflow/runtime/thinking/base.py
  agentflow/runtime/thinking/react.py
  agentflow/runtime/thinking/plan_execute.py
  agentflow/runtime/thinking/cot.py
  agentflow/runtime/thinking/reflection.py
  agentflow/runtime/thinking/adaptive.py
  agentflow/runtime/builder.py
  tests/runtime/test_toolkit.py
  tests/runtime/test_memory_working.py
  tests/runtime/test_memory_episodic.py
  tests/runtime/test_memory_semantic.py
  tests/runtime/test_memory_manager.py
  tests/runtime/test_prompt.py
  tests/runtime/test_thinking_react.py
  tests/runtime/test_thinking_plan_execute.py
  tests/runtime/test_thinking_cot.py
  tests/runtime/test_thinking_reflection.py
  tests/runtime/test_thinking_adaptive.py
  tests/runtime/test_builder.py

Modify:
  agentflow/runtime/tool_registry.py — 扩展 MCP/REST 执行支持
  agentflow/runtime/agent.py — BaseAgent 简化，委托给 ThinkingEngine
  agentflow/runtime/memory.py — 保持向后兼容，重导出到 memory/ 子包
  agentflow/runtime/__init__.py — 导出新模块

Remove (logic migrated, not deleted):
  agentflow/runtime/react_agent.py — 逻辑迁移到 thinking/react.py
```

---

## Phase 1: ToolKit 子系统

### Task 1.1: Pydantic 参数模型 + Tool 扩展

**Files:**

- Modify: `agentflow/runtime/tool_registry.py`
- Test: `tests/runtime/test_tool_registry.py` (existing, add cases)

首先扩展 Tool dataclass 和 ToolRegistry，为 `@tool` 装饰器打好基础。

- [ ] **Step 1: 扩展 Tool 支持 Pydantic params 和 MCP/REST 类型**

在 `agentflow/runtime/tool_registry.py` 的 `ToolType` 枚举新增 MCP（已存在则跳过），在 `Tool` dataclass 新增 `params_model` 字段：

```python
# agentflow/runtime/tool_registry.py (修改部分)

from typing import Optional, Any

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
```

- [ ] **Step 2: 扩展 ToolRegistry.execute 支持 REST 和 MCP（预留接口）**

```python
# agentflow/runtime/tool_registry.py — ToolRegistry.execute 方法修改

class ToolRegistry:
    # ... existing code ...

    def execute(self, name: str, inputs: dict) -> ToolResult:
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
                # 预留：同步包装异步 HTTP 调用
                import asyncio
                output = asyncio.run(self._execute_rest(tool, validated_inputs))
                return ToolResult(success=True, output=output)
            elif tool.tool_type == ToolType.MCP:
                # 预留：MCP 调用
                import asyncio
                output = asyncio.run(self._execute_mcp(tool, validated_inputs))
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
```

- [ ] **Step 3: 运行现有测试确保不破坏兼容性**

```bash
cd /d/Codes_lxy/VibeCoding/AgentFlow && python -m pytest tests/runtime/test_tool_registry.py -v
```

Expected: 4 tests PASS (existing tests continue to work)

- [ ] **Step 4: 提交**

```bash
git add agentflow/runtime/tool_registry.py
git commit -m "feat(tool): add Pydantic param validation and MCP/REST execute support"
```

---

### Task 1.2: `@tool` 装饰器

**Files:**

- Create: `agentflow/runtime/toolkit.py`
- Test: `tests/runtime/test_toolkit.py`

- [ ] **Step 1: 写测试（TDD）**

```python
# tests/runtime/test_toolkit.py

import pytest
from agentflow.runtime.toolkit import tool, ToolKit
from agentflow.runtime.tool_registry import ToolType, ToolResult


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

    def test_decorator_keeps_function_callable(self):
        """装饰器不破坏原始函数的直接调用能力。"""

        @tool
        def multiply(x: int, y: int) -> int:
            return x * y

        # 可以直接调用
        assert multiply._func(x=3, y=4) == 12


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
```

- [ ] **Step 2: 运行测试，确认全部 FAIL**

```bash
python -m pytest tests/runtime/test_toolkit.py -v
```

Expected: ALL FAIL (ToolKit 和 @tool 尚未实现)

- [ ] **Step 3: 实现 `@tool` 装饰器 + ToolKit**

```python
# agentflow/runtime/toolkit.py

import inspect
from typing import Callable, Optional, Any, get_type_hints
from agentflow.runtime.tool_registry import Tool, ToolType, ToolResult, ToolRegistry


def _type_to_json_schema(py_type) -> dict:
    """将 Python 类型注解转为 JSON Schema 基本类型。"""
    mapping = {int: "integer", float: "number", str: "string", bool: "boolean", dict: "object", list: "array"}
    type_str = mapping.get(py_type, "string")
    return {"type": type_str}


def _function_to_parameters(func: Callable) -> dict:
    """从函数签名 + 类型注解推导 JSON Schema parameters。"""
    hints = get_type_hints(func)
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
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
        tool_params = {}

        if params is not None:
            # Pydantic model -> JSON Schema
            tool_params = params.model_json_schema()
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
        # 保留原始函数的引用，方便直接调用
        t._func = func
        return t

    if _func is not None:
        return decorator(_func)
    return decorator


class ToolKit:
    """统一的工具集合，支持本地/MCP/REST 三源统一注册。"""

    def __init__(self):
        self._registry = ToolRegistry()

    def add(self, tool: Tool) -> "ToolKit":
        self._registry.register(tool)
        return self

    def list(self) -> list[Tool]:
        return self._registry.list_tools()

    def execute(self, name: str, inputs: dict) -> ToolResult:
        return self._registry.execute(name, inputs)

    def list_for_llm(self) -> list[dict]:
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
```

- [ ] **Step 4: 运行测试，确认全部 PASS**

```bash
python -m pytest tests/runtime/test_toolkit.py -v
```

Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add agentflow/runtime/toolkit.py tests/runtime/test_toolkit.py
git commit -m "feat(toolkit): add @tool decorator and ToolKit container"
```

---

## Phase 2: Memory 子系统

### Task 2.1: Working Memory（工作记忆）

**Files:**

- Create: `agentflow/runtime/memory/__init__.py` (package init, empty for now)
- Create: `agentflow/runtime/memory/working.py`
- Test: `tests/runtime/test_memory_working.py`

- [ ] **Step 1: 写测试**

```python
# tests/runtime/test_memory_working.py

from agentflow.runtime.memory.working import WorkingMemory, Message


class TestWorkingMemory:
    def test_add_and_retrieve(self):
        wm = WorkingMemory(max_turns=20)
        wm.add(Message(role="user", content="Hello"))
        wm.add(Message(role="assistant", content="Hi!"))
        msgs = wm.get_context_window()
        assert len(msgs) == 2

    def test_sliding_window_by_turns(self):
        wm = WorkingMemory(max_turns=3)
        for i in range(6):
            wm.add(Message(role="user", content=f"msg-{i}"))
        msgs = wm.get_context_window()
        assert len(msgs) == 3
        assert msgs[0].content == "msg-3"

    def test_token_limit(self):
        wm = WorkingMemory(max_turns=100, max_tokens=30)
        wm.add(Message(role="user", content="x" * 500))  # ~125 tokens
        msgs = wm.get_context_window()
        total_chars = sum(len(m.content) for m in msgs)
        assert total_chars <= 30 * 4 + 50

    def test_clear(self):
        wm = WorkingMemory(max_turns=10)
        wm.add(Message(role="user", content="test"))
        wm.clear()
        assert len(wm.get_context_window()) == 0

    def test_role_filter(self):
        wm = WorkingMemory(max_turns=10)
        wm.add(Message(role="system", content="sys"))
        wm.add(Message(role="user", content="hello"))
        wm.add(Message(role="tool", content="result"))
        user_only = wm.get_context_window(roles={"user"})
        assert len(user_only) == 1
        assert user_only[0].role == "user"
```

- [ ] **Step 2: 运行测试，确认 FAIL**

- [ ] **Step 3: 实现 WorkingMemory**

```python
# agentflow/runtime/memory/working.py

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str = ""
    tool_calls: list = field(default_factory=list)


class WorkingMemory:
    """Layer 1: 工作记忆 — 当前对话的完整消息窗口。

    支持滑动窗口（按轮数截断）和 token 限制（按字符估算截断）。
    """

    def __init__(self, max_turns: int = 20, max_tokens: int = 8000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        while len(self._messages) > self.max_turns:
            self._messages.pop(0)

    def clear(self) -> None:
        self._messages.clear()

    def get_context_window(self, roles: set[str] | None = None) -> list[Message]:
        """获取对话窗口，按 token 限制从后向前截取。

        Args:
            roles: 可选的角色过滤集合，如 {"user", "assistant"}
        """
        result = []
        total_chars = 0
        char_limit = self.max_tokens * 4  # 粗略 1 token ≈ 4 chars

        for msg in reversed(self._messages):
            if roles and msg.role not in roles:
                result.insert(0, msg)
                continue
            total_chars += len(msg.content)
            if total_chars > char_limit:
                break
            result.insert(0, msg)
        return result

    def __len__(self):
        return len(self._messages)
```

- [ ] **Step 4: 运行测试 PASS**

- [ ] **Step 5: 提交**

---

### Task 2.2: Episodic Memory（情节记忆）

**Files:**

- Create: `agentflow/runtime/memory/episodic.py`
- Test: `tests/runtime/test_memory_episodic.py`

- [ ] **Step 1: 写测试**

```python
# tests/runtime/test_memory_episodic.py

import time
from datetime import datetime, timedelta
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact


class TestMemoryFact:
    def test_creation(self):
        fact = MemoryFact(
            fact_type="preference",
            subject="user",
            predicate="prefers",
            object="quick replies",
            confidence=0.9,
            timestamp=datetime.now(),
            source_turn=3,
            ttl=86400,
        )
        assert fact.fact_type == "preference"
        assert not fact.is_expired()

    def test_expiry(self):
        past = datetime.now() - timedelta(hours=25)
        fact = MemoryFact(
            fact_type="event", subject="user", predicate="did", object="login",
            confidence=0.8, timestamp=past, source_turn=1, ttl=3600,  # 1h TTL
        )
        assert fact.is_expired()

    def test_decay(self):
        fact = MemoryFact(
            fact_type="entity", subject="tool:weather", predicate="returned", object="22°C",
            confidence=0.8, timestamp=datetime.now(), source_turn=1, ttl=86400,
        )
        fact.decay(0.5)
        assert fact.confidence == 0.4


class TestEpisodicMemory:
    def test_store_and_retrieve_by_subject(self):
        mem = EpisodicMemory(max_facts=100)
        f1 = MemoryFact("event", "user", "asked", "refund", 0.9, datetime.now(), 1, 86400)
        f2 = MemoryFact("event", "agent", "responded", "policy", 0.9, datetime.now(), 1, 86400)
        mem.add(f1)
        mem.add(f2)
        user_facts = mem.get_by_subject("user")
        assert len(user_facts) == 1
        assert user_facts[0].object == "refund"

    def test_capacity_eviction(self):
        """容量超限时淘汰低置信度 + 旧时间戳的条目。"""
        mem = EpisodicMemory(max_facts=3)
        now = datetime.now()

        # 高置信度，新
        mem.add(MemoryFact("event", "u", "did", "a", 0.9, now, 1, 86400))
        # 中置信度
        mem.add(MemoryFact("event", "u", "did", "b", 0.5, now, 2, 86400))
        # 高置信度但旧
        old = now - timedelta(days=30)
        mem.add(MemoryFact("event", "u", "did", "c", 0.9, old, 3, 86400))

        # 新加一条触发淘汰
        mem.add(MemoryFact("event", "u", "did", "d", 0.7, now, 4, 86400))

        assert mem.count() == 3  # 淘汰了 1 条
        # 被淘汰的应该是 b（置信度最低），或 c（虽然高置信但太旧）
        subjects = [f.object for f in mem.get_all()]
        assert "a" in subjects  # 最高置信度 + 最新，不应被淘汰

    def test_forget_expired(self):
        """遗忘门：清除所有过期事实。"""
        mem = EpisodicMemory(max_facts=100)
        now = datetime.now()
        past = now - timedelta(hours=25)

        mem.add(MemoryFact("event", "u", "did", "fresh", 0.9, now, 1, ttl=86400))
        mem.add(MemoryFact("event", "u", "did", "stale", 0.9, past, 2, ttl=3600))

        removed = mem.forget_expired()
        assert removed >= 1
        assert mem.count() == 1
```

- [ ] **Step 2: 运行测试 FAIL**

- [ ] **Step 3: 实现 MemoryFact + EpisodicMemory**

```python
# agentflow/runtime/memory/episodic.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional


@dataclass
class MemoryFact:
    """结构化记忆事实：主体-谓词-客体 三元组。"""

    fact_type: str  # "entity" | "decision" | "event" | "preference"
    subject: str
    predicate: str
    object: str
    confidence: float  # 0.0 - 1.0
    timestamp: datetime
    source_turn: int
    ttl: int  # 生存时间（秒）

    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)

    def decay(self, factor: float) -> None:
        """衰减置信度，factor 为衰减因子（0~1）。"""
        self.confidence = max(0.0, self.confidence - factor)


class EpisodicMemory:
    """Layer 2: 情节记忆 — 跨会话的结构化事实存储。

    基于 SQLite 或内存字典的简单实现。
    内置自动淘汰和过期清理机制。
    """

    def __init__(self, max_facts: int = 200, backend: str = "memory"):
        self.max_facts = max_facts
        self._facts: list[MemoryFact] = []

    def add(self, fact: MemoryFact) -> None:
        self._facts.append(fact)
        self._evict_if_needed()

    def get_all(self) -> list[MemoryFact]:
        return list(self._facts)

    def get_by_subject(self, subject: str) -> list[MemoryFact]:
        return [f for f in self._facts if f.subject == subject]

    def get_by_type(self, fact_type: str) -> list[MemoryFact]:
        return [f for f in self._facts if f.fact_type == fact_type]

    def count(self) -> int:
        return len(self._facts)

    def forget_expired(self) -> int:
        """遗忘门：删除所有已过期的事实，返回删除数量。"""
        before = len(self._facts)
        self._facts = [f for f in self._facts if not f.is_expired()]
        return before - len(self._facts)

    def _evict_if_needed(self) -> None:
        """容量超限时淘汰低质量条目。"""
        if len(self._facts) <= self.max_facts:
            return
        # 按 (confidence × 新鲜度) 排序，淘汰最低分
        now = datetime.now()
        scored = sorted(
            self._facts,
            key=lambda f: f.confidence / max(1.0, (now - f.timestamp).total_seconds() / 86400),
        )
        self._facts = scored[-(self.max_facts):]
```

- [ ] **Step 4: 运行测试 PASS**

- [ ] **Step 5: 提交**

---

### Task 2.3: Semantic Memory（语义记忆）+ MemoryProfile + MemoryManager

**Files:**

- Create: `agentflow/runtime/memory/semantic.py`
- Create: `agentflow/runtime/memory/manager.py`
- Update: `agentflow/runtime/memory/__init__.py`
- Test: `tests/runtime/test_memory_semantic.py`, `tests/runtime/test_memory_manager.py`

- [ ] **Step 1: 写 SemanticMemory 测试**

```python
# tests/runtime/test_memory_semantic.py

import pytest
from agentflow.runtime.memory.semantic import SemanticMemory


class TestSemanticMemory:
    def test_store_and_search_basic(self):
        """基本的关键词检索（无向量嵌入时的 fallback）。"""
        mem = SemanticMemory()
        mem.store("key1", "AgentFlow is a Go+Python framework")
        mem.store("key2", "LangChain is a Python LLM framework")

        results = mem.search("Go")
        assert len(results) == 1
        assert "AgentFlow" in str(results[0])

    def test_get_missing(self):
        mem = SemanticMemory()
        assert mem.get("nonexistent") is None

    def test_top_k_limit(self):
        mem = SemanticMemory()
        for i in range(10):
            mem.store(f"key{i}", f"Document number {i} containing keyword")
        results = mem.search("keyword", top_k=3)
        assert len(results) <= 3
```

- [ ] **Step 2: 写 MemoryManager 集成测试**

```python
# tests/runtime/test_memory_manager.py

from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile, MemoryConfig
from agentflow.runtime.memory.working import Message


class TestMemoryProfile:
    def test_light_profile(self):
        p = MemoryProfile.light()
        assert p.working.max_turns == 10
        assert p.episodic_max == 0

    def test_standard_profile(self):
        p = MemoryProfile.standard()
        assert p.working.max_turns == 20
        assert p.episodic_max == 200

    def test_deep_profile(self):
        p = MemoryProfile.deep()
        assert p.working.max_turns == 40
        assert p.episodic_max == 500
        assert p.semantic_enabled is True


class TestMemoryManager:
    def test_pre_turn_retrieves_from_semantic(self):
        """检索门：run 开始时从 Semantic 拉取相关事实。"""
        mgr = MemoryManager(verbose=True)
        mgr.semantic.store("kb_1", "Refund policy: 30 days unconditional")
        mgr.semantic.store("kb_2", "Contact support: email support@example.com")
        mgr.semantic.store("kb_3", "Weather is sunny")

        facts = mgr.pre_turn("I want a refund")
        assert len(facts) > 0
        assert any("refund" in str(f).lower() for f in facts)

    def test_post_turn_extracts_facts(self):
        """记忆门：每轮后自动提取结构化事实。"""
        mgr = MemoryManager(verbose=True)
        mgr.working.add(Message(role="user", content="I live in Beijing"))
        mgr.working.add(Message(role="assistant", content="Got it, Beijing it is."))

        mgr.post_turn()
        # 应该提取出 user 住在 Beijing 的事实
        facts = mgr.episodic.get_all()
        assert any("Beijing" in str(f) for f in facts)

    def test_full_cycle(self):
        """完整的记忆生命周期：搜索 → 加入工作记忆 → 提取 → 遗忘。"""
        mgr = MemoryManager()
        mgr.pre_turn("What's the weather?")
        mgr.working.add(Message(role="user", content="What's the weather?"))
        mgr.working.add(Message(role="assistant", content="It's 22°C in Beijing."))
        mgr.post_turn()

        # 情节记忆中应有提取的事实
        assert mgr.episodic.count() > 0
```

- [ ] **Step 3: 运行测试 FAIL**

- [ ] **Step 4: 实现 SemanticMemory**

```python
# agentflow/runtime/memory/semantic.py

from collections import OrderedDict
from typing import Optional


class SemanticMemory:
    """Layer 3: 语义记忆 — 长期知识库。

    默认使用关键词匹配（零依赖），可选升级为 Chroma 向量检索。
    """

    def __init__(self, embedder: Optional[str] = None, top_k_default: int = 5):
        self.embedder = embedder
        self.top_k_default = top_k_default
        self._store: OrderedDict[str, dict] = OrderedDict()

    def store(self, key: str, content: str, metadata: dict | None = None) -> None:
        self._store[key] = {"content": content, "metadata": metadata or {}}

    def get(self, key: str) -> Optional[dict]:
        return self._store.get(key)

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """关键词检索（fallback 实现）。"""
        k = top_k or self.top_k_default
        query_words = query.lower().split()
        scored = []

        for key, entry in self._store.items():
            content = entry["content"].lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored.append((score, {"key": key, **entry}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]
```

- [ ] **Step 5: 实现 MemoryProfile + MemoryManager**

```python
# agentflow/runtime/memory/manager.py

from dataclasses import dataclass, field
from agentflow.runtime.memory.working import WorkingMemory, Message
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact
from agentflow.runtime.memory.semantic import SemanticMemory


@dataclass
class WorkingConfig:
    max_turns: int = 20
    max_tokens: int = 8000


@dataclass
class MemoryProfile:
    working: WorkingConfig = field(default_factory=WorkingConfig)
    episodic_max: int = 200
    semantic_enabled: bool = False
    semantic_embedder: str | None = None
    auto_memorize: bool = True
    auto_forget: bool = True
    auto_retrieve: bool = True

    @classmethod
    def light(cls) -> "MemoryProfile":
        return cls(working=WorkingConfig(max_turns=10), episodic_max=0)

    @classmethod
    def standard(cls) -> "MemoryProfile":
        return cls(working=WorkingConfig(max_turns=20), episodic_max=200)

    @classmethod
    def deep(cls) -> "MemoryProfile":
        return cls(
            working=WorkingConfig(max_turns=40),
            episodic_max=500,
            semantic_enabled=True,
        )


class MemoryManager:
    """三层记忆管理器 — Agent 自主管理记忆生命周期。"""

    def __init__(self, profile: MemoryProfile | None = None, verbose: bool = False):
        self.profile = profile or MemoryProfile.standard()
        self.verbose = verbose
        self.working = WorkingMemory(
            max_turns=self.profile.working.max_turns,
            max_tokens=self.profile.working.max_tokens,
        )
        self.episodic = EpisodicMemory(max_facts=self.profile.episodic_max)
        self.semantic = SemanticMemory(embedder=self.profile.semantic_embedder)
        self._turn_count = 0

    def pre_turn(self, user_input: str | None = None) -> list[MemoryFact]:
        """检索门：每个 turn 之前，从语义记忆检索相关事实。"""
        self._turn_count += 1

        if not self.profile.auto_retrieve or user_input is None:
            return []

        results = self.semantic.search(user_input, top_k=5)
        facts = []
        for r in results:
            facts.append(MemoryFact(
                fact_type="entity",
                subject=f"kb:{r['key']}",
                predicate="contains",
                object=r["content"][:200],
                confidence=0.7,
                timestamp=__import__('datetime').datetime.now(),
                source_turn=self._turn_count,
                ttl=86400,
            ))
        return facts

    def post_turn(self) -> None:
        """记忆门 + 遗忘门：每个 turn 之后触发。"""
        if self.profile.auto_memorize:
            self._extract_facts()

        if self.profile.auto_forget:
            self.episodic.forget_expired()

    def _extract_facts(self) -> None:
        """从工作记忆中提取结构化事实（简化版：关键词触发）。"""
        from datetime import datetime
        for msg in self.working.get_context_window():
            content_lower = msg.content.lower()
            # 简单的模式匹配提取（第一版，不调用 LLM）
            if "live in" in content_lower or "住在" in content_lower or "from" in content_lower:
                fact = MemoryFact(
                    fact_type="preference",
                    subject=msg.role,
                    predicate="location",
                    object=msg.content[:100],
                    confidence=0.6,
                    timestamp=datetime.now(),
                    source_turn=self._turn_count,
                    ttl=86400 * 7,  # 7 day TTL
                )
                self.episodic.add(fact)
```

```python
# agentflow/runtime/memory/__init__.py

from agentflow.runtime.memory.working import WorkingMemory, Message
from agentflow.runtime.memory.episodic import EpisodicMemory, MemoryFact
from agentflow.runtime.memory.semantic import SemanticMemory
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile, WorkingConfig

__all__ = [
    "WorkingMemory", "Message",
    "EpisodicMemory", "MemoryFact",
    "SemanticMemory",
    "MemoryManager", "MemoryProfile", "WorkingConfig",
]
```

- [ ] **Step 6: 运行测试 PASS**

- [ ] **Step 7: 确保旧测试仍通过（向后兼容）**

```bash
python -m pytest tests/runtime/test_memory.py -v
```

旧 `test_memory.py` 引用的是 `agentflow.runtime.memory.MemoryManager`，需要更新旧模块做重导出。

- [ ] **Step 8: 更新旧 memory.py 做重导出**

```python
# agentflow/runtime/memory.py — 修改为从 memory/ 子包重导出

from agentflow.runtime.memory.working import Message, WorkingMemory as ShortTermMemory
from agentflow.runtime.memory.semantic import SemanticMemory as LongTermMemory
from agentflow.runtime.memory.manager import MemoryManager

__all__ = ["Message", "ShortTermMemory", "LongTermMemory", "MemoryManager"]
```

- [ ] **Step 9: 运行全部 memory 相关测试**

```bash
python -m pytest tests/runtime/test_memory.py tests/runtime/test_memory_working.py tests/runtime/test_memory_episodic.py tests/runtime/test_memory_semantic.py tests/runtime/test_memory_manager.py -v
```

Expected: ALL PASS

- [ ] **Step 10: 提交**

---

## Phase 3: Prompt 模板子系统

### Task 3.1: Section + PromptTemplate

**Files:**

- Create: `agentflow/runtime/prompt/__init__.py`
- Create: `agentflow/runtime/prompt/section.py`
- Test: `tests/runtime/test_prompt.py`

- [ ] **Step 1: 写测试**

```python
# tests/runtime/test_prompt.py

import pytest
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.prompt.section import Section, RoleCard, SafetyRules, ToolManual


class TestSection:
    def test_role_card_render(self):
        s = RoleCard(name="Alice", role="客服", tone="友善专业")
        result = s.render({})
        assert "Alice" in result
        assert "客服" in result

    def test_safety_rules_render(self):
        s = SafetyRules(rules=["规则1", "规则2"])
        result = s.render({})
        assert "规则1" in result
        assert "规则2" in result

    def test_tool_manual_empty(self):
        s = ToolManual()
        result = s.render({"tools": []})
        assert "工具" in result.lower() or "tool" in result.lower()


class TestPromptTemplate:
    def test_add_section_and_render(self):
        template = PromptTemplate("test")
        template.add(RoleCard(name="Bob", role="助手", tone="友好"))
        template.add(SafetyRules(rules=["不泄露隐私"]))

        result = template.render()
        assert "Bob" in result
        assert "不泄露隐私" in result
        # RoleCard order=10 < SafetyRules order=20，所以角色在前面
        assert result.index("Bob") < result.index("不泄露隐私")

    def test_remove_section(self):
        template = PromptTemplate("test")
        template.add(RoleCard(name="Bob", role="助手", tone="友好"))
        template.add(SafetyRules(rules=["规则"]))
        template.remove("safety_rules")
        result = template.render()
        assert "规则" not in result

    def test_override_params(self):
        template = PromptTemplate("test")
        template.add(RoleCard(name="Bob", role="助手", tone="友好"))
        template.override("role_card", tone="严肃")
        result = template.render()
        assert "严肃" in result

    def test_preset_customer_support(self):
        template = PromptTemplate.preset("customer_support")
        result = template.render()
        assert "客服" in result or "support" in result.lower()
        assert "工具" in result.lower() or "tool" in result.lower()

    def test_context_injection(self):
        template = PromptTemplate("test")
        template.add(RoleCard(name="{{ agent_name }}", role="助手", tone="友好"))
        result = template.render({"agent_name": "MyAgent"})
        assert "MyAgent" in result

    def test_from_string_fallback(self):
        """从纯字符串创建模板的兼容模式。"""
        template = PromptTemplate.from_string("You are a helpful assistant.")
        result = template.render()
        assert "helpful assistant" in result
```

- [ ] **Step 2: 运行测试 FAIL**

- [ ] **Step 3: 实现 Section 基类 + 内置模块**

```python
# agentflow/runtime/prompt/section.py

from abc import ABC, abstractmethod
from string import Template


class Section(ABC):
    """Prompt 模块基类。"""
    name: str = "base"
    order: int = 50  # 排序权重，越小越靠前

    @abstractmethod
    def render(self, context: dict) -> str:
        ...


class RoleCard(Section):
    name = "role_card"
    order = 10

    def __init__(self, name: str, role: str, tone: str = "professional", audience: str = ""):
        self._name = name
        self._role = role
        self._tone = tone
        self._audience = audience

    def render(self, context: dict) -> str:
        template = Template(
            "## Role\n"
            "You are $name, a $role.\n"
            "Tone: $tone.\n"
            "$audience_line"
        )
        audience_line = f"Audience: {self._audience}." if self._audience else ""
        # Support Jinja2-style variable injection from context
        name = context.get("name", self._name)
        role = context.get("role", self._role)
        tone = context.get("tone", self._tone)
        return template.safe_substitute(
            name=name, role=role, tone=tone, audience_line=audience_line
        )


class SafetyRules(Section):
    name = "safety_rules"
    order = 20

    def __init__(self, rules: list[str] | None = None):
        self.rules = rules or []

    def render(self, context: dict) -> str:
        all_rules = context.get("rules", self.rules)
        if not all_rules:
            return ""
        lines = ["## Safety Rules"]
        for i, rule in enumerate(all_rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)


class ToolManual(Section):
    name = "tool_manual"
    order = 40

    def render(self, context: dict) -> str:
        tools = context.get("tools", [])
        if not tools:
            return "## Tools\nNo tools available."

        lines = ["## Available Tools"]
        for t in tools:
            name = t.name if hasattr(t, 'name') else t.get('name', 'unknown')
            desc = t.description if hasattr(t, 'description') else t.get('description', '')
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)


class FormatGuide(Section):
    name = "format_guide"
    order = 30

    def __init__(self, format: str = "markdown"):
        self.format = format

    def render(self, context: dict) -> str:
        fmt = context.get("format", self.format)
        return f"## Output Format\nRespond in {fmt} format."


class TimeContext(Section):
    name = "time_context"
    order = 60

    def render(self, context: dict) -> str:
        from datetime import datetime
        now = context.get("current_time", datetime.now().isoformat())
        return f"## Context\nCurrent time: {now}"
```

- [ ] **Step 4: 实现 PromptTemplate**

```python
# agentflow/runtime/prompt/__init__.py

from string import Template
from agentflow.runtime.prompt.section import Section, RoleCard, SafetyRules, ToolManual, FormatGuide


class PromptTemplate:
    """将多个 Section 组装为完整的 System Prompt。"""

    def __init__(self, name: str):
        self.name = name
        self._sections: list[Section] = []

    def add(self, section: Section) -> "PromptTemplate":
        self._sections.append(section)
        return self

    def remove(self, name: str) -> "PromptTemplate":
        self._sections = [s for s in self._sections if s.name != name]
        return self

    def override(self, name: str, **params) -> "PromptTemplate":
        for s in self._sections:
            if s.name == name:
                for k, v in params.items():
                    if hasattr(s, f"_{k}"):
                        setattr(s, f"_{k}", v)
                    elif hasattr(s, k):
                        setattr(s, k, v)
                break
        return self

    def render(self, context: dict | None = None) -> str:
        ctx = context or {}
        # 自动注入默认上下文
        if "current_time" not in ctx:
            from datetime import datetime
            ctx["current_time"] = datetime.now().isoformat()

        # 按 order 排序
        sorted_sections = sorted(self._sections, key=lambda s: s.order)
        parts = []
        for s in sorted_sections:
            rendered = s.render(ctx)
            if rendered.strip():
                parts.append(rendered)

        return "\n\n".join(parts)

    @classmethod
    def preset(cls, name: str) -> "PromptTemplate":
        """按场景预设生成 PromptTemplate。"""
        if name == "customer_support":
            template = cls("customer_support")
            template.add(RoleCard(name="小助手", role="客服代表", tone="友善专业"))
            template.add(SafetyRules(rules=[
                "不泄露用户个人信息",
                "退款超过 ¥500 需主管审批",
                "无法解决的问题及时转人工",
            ]))
            template.add(ToolManual())
            template.add(FormatGuide(format="markdown"))
            return template

        if name == "coding_assistant":
            template = cls("coding_assistant")
            template.add(RoleCard(name="CodeBot", role="编程助手", tone="简洁技术"))
            template.add(SafetyRules(rules=[
                "不执行危险命令（rm -rf, DROP TABLE 等）",
                "代码修改前先展示 diff",
                "鼓励写测试",
            ]))
            template.add(ToolManual())
            return template

        # 默认
        template = cls(name)
        template.add(RoleCard(name="Assistant", role="AI 助手", tone="helpful"))
        return template

    @classmethod
    def from_string(cls, prompt: str) -> "PromptTemplate":
        """从纯字符串创建（向后兼容）。"""
        template = cls("inline")
        template.add(_StringSection(prompt))
        return template


class _StringSection(Section):
    """将纯字符串包装为 Section 的适配器。"""
    name = "inline_prompt"
    order = 0

    def __init__(self, content: str):
        self.content = content

    def render(self, context: dict) -> str:
        return Template(self.content).safe_substitute(**context)
```

- [ ] **Step 5: 运行测试 PASS**

- [ ] **Step 6: 提交**

---

## Phase 4: Thinking 引擎

### Task 4.1: ThinkingStrategy 抽象 + ReActStrategy

**Files:**

- Create: `agentflow/runtime/thinking/__init__.py`
- Create: `agentflow/runtime/thinking/base.py`
- Create: `agentflow/runtime/thinking/react.py`
- Test: `tests/runtime/test_thinking_react.py`

- [ ] **Step 1: 写测试**

```python
# tests/runtime/test_thinking_react.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.toolkit import ToolKit, tool
from agentflow.runtime.memory.manager import MemoryManager


class TestReActStrategy:
    def test_simple_answer_no_tools(self):
        """无工具调用的简单问答。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="The answer is 42.", role="assistant", tool_calls=[],
        )

        ctx = ThinkContext(
            user_input="What is the meaning of life?",
            system_prompt="You are helpful.",
            messages=[],
            tools=[],
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=10,
        )

        strategy = ReActStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "42" in result.output
        assert result.mode_used == "react"
        assert len(result.steps) == 1

    def test_uses_tool_then_answers(self):
        """Agent 先调用工具，再给出最终答案。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(
                content=None, role="assistant",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }],
            ),
            MagicMock(content="2 + 3 = 5", role="assistant", tool_calls=[]),
        ]

        @tool
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        kit = ToolKit()
        kit.add(add)

        ctx = ThinkContext(
            user_input="What is 2+3?",
            system_prompt="You do math.",
            messages=[],
            tools=kit.list_for_llm(),
            llm_client=mock_llm,
            memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = ReActStrategy(toolkit=kit)
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "5" in result.output
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "add"

    def test_stops_at_max_iterations(self):
        """达到最大迭代次数时强制停止。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content=None, role="assistant",
            tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "ping"}'},
            }],
        )

        @tool
        def echo(text: str) -> str:
            return text

        kit = ToolKit()
        kit.add(echo)

        ctx = ThinkContext(
            user_input="ping", system_prompt="You loop.",
            messages=[], tools=kit.list_for_llm(),
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=2,
        )

        strategy = ReActStrategy(toolkit=kit)
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert "maximum iterations" in result.output.lower()
```

- [ ] **Step 2: 运行测试 FAIL**

- [ ] **Step 3: 实现 base.py + react.py**

```python
# agentflow/runtime/thinking/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThinkContext:
    user_input: str
    system_prompt: str
    messages: list
    tools: list[dict]
    llm_client: object
    memory: object
    max_iterations: int = 10
    feedback: list[str] = field(default_factory=list)

    def add_feedback(self, suggestions: list[str]) -> None:
        self.feedback.extend(suggestions)


@dataclass
class ThinkResult:
    output: str
    tool_calls: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    reflection_notes: list = field(default_factory=list)
    mode_used: str = "unknown"


class ThinkingStrategy(ABC):
    @abstractmethod
    async def run(self, context: ThinkContext) -> ThinkResult:
        ...
```

```python
# agentflow/runtime/thinking/react.py

import json
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class ReActStrategy(ThinkingStrategy):
    """ReAct 模式：Thought → Action → Observation 循环。"""

    def __init__(self, toolkit=None):
        self.toolkit = toolkit

    async def run(self, context: ThinkContext) -> ThinkResult:
        messages = [{"role": "system", "content": context.system_prompt}]
        for msg in context.messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            messages.append(msg_dict)

        steps = []
        tool_calls_made = []

        for i in range(context.max_iterations):
            response = await context.llm_client.chat(messages, tools=context.tools or None)

            if response.tool_calls:
                # 添加 assistant tool_calls 到消息
                messages.append({
                    "role": "assistant", "content": response.content or "",
                    "tool_calls": response.tool_calls,
                })

                for tc in response.tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"])

                    if self.toolkit:
                        result = self.toolkit.execute(func_name, func_args)
                        tool_output = result.output or result.error
                    else:
                        tool_output = f"[No toolkit] Called {func_name}({func_args})"

                    tool_calls_made.append({
                        "tool": func_name,
                        "input": func_args,
                        "output": tool_output,
                    })

                    messages.append({
                        "role": "tool",
                        "content": tool_output,
                        "tool_call_id": tc.get("id", ""),
                    })

                steps.append({
                    "iteration": i,
                    "type": "tool_call",
                    "calls": [tc["function"]["name"] for tc in response.tool_calls],
                })
            else:
                messages.append({"role": "assistant", "content": response.content})
                steps.append({
                    "iteration": i,
                    "type": "final",
                    "output": response.content,
                })
                return ThinkResult(
                    output=response.content,
                    tool_calls=tool_calls_made,
                    steps=steps,
                    mode_used="react",
                )

        return ThinkResult(
            output="Agent reached maximum iterations without a final answer.",
            tool_calls=tool_calls_made,
            steps=steps,
            mode_used="react",
        )
```

- [ ] **Step 4: 运行测试 PASS**

- [ ] **Step 5: 提交**

---

### Task 4.2: PlanExecuteStrategy + CoTStrategy

**Files:**

- Create: `agentflow/runtime/thinking/plan_execute.py`
- Create: `agentflow/runtime/thinking/cot.py`
- Test: `tests/runtime/test_thinking_plan_execute.py`, `tests/runtime/test_thinking_cot.py`

- [ ] **Step 1: 写 PlanExecute 测试**

```python
# tests/runtime/test_thinking_plan_execute.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.memory.manager import MemoryManager


class TestPlanExecuteStrategy:
    def test_generates_plan_then_executes(self):
        """首先生成计划，然后逐步执行。"""
        mock_llm = AsyncMock()
        # 第一次调用：生成计划
        # 第二次调用：执行步骤
        # 第三次调用：汇总
        mock_llm.chat.side_effect = [
            MagicMock(content="PLAN:\n1. Search for refund policy\n2. Apply to user case\n3. Summarize", role="assistant", tool_calls=[]),
            MagicMock(content="EXECUTE: Found refund policy - 30 days unconditional.", role="assistant", tool_calls=[]),
            MagicMock(content="FINAL: You are eligible for a refund within 30 days.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="Can I get a refund?",
            system_prompt="You handle refunds.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=10,
        )

        strategy = PlanExecuteStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "plan_execute"
        assert "refund" in result.output.lower()
        assert len(result.steps) >= 2  # at least plan step + execute step
```

- [ ] **Step 2: 写 CoT 测试**

```python
# tests/runtime/test_thinking_cot.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.memory.manager import MemoryManager


class TestCoTStrategy:
    def test_think_then_answer(self):
        """先深度推理，再给出答案。"""
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content="Let me think step by step...\nGiven: 10 apples, eat 3, buy 5.\n10 - 3 + 5 = 12.", role="assistant", tool_calls=[]),
            MagicMock(content="FINAL ANSWER: You have 12 apples.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="I have 10 apples, eat 3, buy 5. How many left?",
            system_prompt="You solve math problems.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        strategy = CoTStrategy()
        import asyncio
        result = asyncio.run(strategy.run(ctx))

        assert result.mode_used == "cot"
        assert "12" in result.output
```

- [ ] **Step 3: 实现 PlanExecuteStrategy**

```python
# agentflow/runtime/thinking/plan_execute.py

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class PlanExecuteStrategy(ThinkingStrategy):
    """Plan-Execute 模式：先制定计划，再逐步执行。"""

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []

        # Phase 1: Plan
        plan_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                "Break this down into clear steps. Output as a numbered plan."
                "Each step should describe what needs to be done."
            )},
        ]
        plan_response = await context.llm_client.chat(plan_messages)
        plan_text = plan_response.content
        steps.append({"phase": "plan", "output": plan_text})

        # Phase 2: Execute (simplified — single execution pass)
        execute_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Plan:\n{plan_text}\n\n"
                "Execute the plan step by step. For each step, describe what you did and the result."
            )},
        ]
        execute_response = await context.llm_client.chat(execute_messages)
        steps.append({"phase": "execute", "output": execute_response.content})

        # Phase 3: Finalize
        finalize_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Task: {context.user_input}\n\n"
                f"Execution results:\n{execute_response.content}\n\n"
                "Summarize the final answer concisely."
            )},
        ]
        final_response = await context.llm_client.chat(finalize_messages)
        steps.append({"phase": "finalize", "output": final_response.content})

        return ThinkResult(
            output=final_response.content,
            steps=steps,
            mode_used="plan_execute",
        )
```

- [ ] **Step 4: 实现 CoTStrategy**

```python
# agentflow/runtime/thinking/cot.py

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class CoTStrategy(ThinkingStrategy):
    """Chain-of-Thought 模式：深度推理 → 最终答案。"""

    async def run(self, context: ThinkContext) -> ThinkResult:
        steps = []

        # Phase 1: Deep thinking
        think_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": (
                f"Question: {context.user_input}\n\n"
                "Think through this step by step. Consider all angles, break down the problem, "
                "and reason carefully before arriving at a conclusion."
            )},
        ]
        think_response = await context.llm_client.chat(think_messages)
        steps.append({"phase": "think", "output": think_response.content})

        # Phase 2: Final answer
        answer_messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": think_response.content},
            {"role": "user", "content": "Based on your reasoning above, give the final answer."},
        ]
        answer_response = await context.llm_client.chat(answer_messages)
        steps.append({"phase": "answer", "output": answer_response.content})

        return ThinkResult(
            output=answer_response.content,
            steps=steps,
            mode_used="cot",
        )
```

- [ ] **Step 5: 运行测试 PASS**

- [ ] **Step 6: 提交**

---

### Task 4.3: ReflectionWrapper + AdaptiveRouter + ThinkingEngine

**Files:**

- Create: `agentflow/runtime/thinking/reflection.py`
- Create: `agentflow/runtime/thinking/adaptive.py`
- Update: `agentflow/runtime/thinking/__init__.py`
- Test: `tests/runtime/test_thinking_reflection.py`, `tests/runtime/test_thinking_adaptive.py`

- [ ] **Step 1: 写 ReflectionWrapper 测试**

```python
# tests/runtime/test_thinking_reflection.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.thinking.base import ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper
from agentflow.runtime.memory.manager import MemoryManager


class TestReflectionWrapper:
    def test_passes_when_strategy_succeeds(self):
        """当 inner strategy 成功时，反思通过并返回结果。"""
        mock_llm = AsyncMock()
        # ReAct 成功返回
        mock_llm.chat.side_effect = [
            MagicMock(content="The answer is 42.", role="assistant", tool_calls=[]),
            # Reflection check: "any issues?" → no issues
            MagicMock(content="No issues found. The answer is correct.", role="assistant", tool_calls=[]),
        ]

        ctx = ThinkContext(
            user_input="What is the answer?",
            system_prompt="You are helpful.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        wrapped = ReflectionWrapper(ReActStrategy(), max_reflections=2)
        import asyncio
        result = asyncio.run(wrapped.run(ctx))

        assert "42" in result.output
        assert len(result.reflection_notes) > 0

    def test_retries_on_failure(self):
        """当反思发现问题时，重试。"""
        call_count = [0]
        mock_llm = AsyncMock()

        async def chat_side_effect(messages, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt: gives a wrong answer
                return MagicMock(content="The answer is 100.", role="assistant", tool_calls=[])
            elif call_count[0] == 2:
                # Reflection: finds issue
                return MagicMock(content="ISSUE FOUND: The answer seems incorrect. Should recalculate.", role="assistant", tool_calls=[])
            elif call_count[0] == 3:
                # Second attempt: correct answer
                return MagicMock(content="The correct answer is 42.", role="assistant", tool_calls=[])
            else:
                # Second reflection: ok
                return MagicMock(content="No issues found.", role="assistant", tool_calls=[])

        mock_llm.chat.side_effect = chat_side_effect

        ctx = ThinkContext(
            user_input="What is 6*7?",
            system_prompt="You do math.",
            messages=[], tools=[],
            llm_client=mock_llm, memory=MemoryManager(),
            max_iterations=5,
        )

        wrapped = ReflectionWrapper(ReActStrategy(), max_reflections=2)
        import asyncio
        result = asyncio.run(wrapped.run(ctx))

        # Should eventually get the right answer
        assert len(result.reflection_notes) >= 1
```

- [ ] **Step 2: 写 AdaptiveRouter 测试**

```python
# tests/runtime/test_thinking_adaptive.py

import pytest
from agentflow.runtime.thinking.adaptive import AdaptiveRouter
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper


class TestAdaptiveRouter:
    def test_routes_simple_to_react(self):
        router = AdaptiveRouter()
        strategy = router.route("What is the weather in Beijing?", [])
        assert isinstance(strategy, ReActStrategy)

    def test_routes_multi_step_to_plan_execute(self):
        router = AdaptiveRouter()
        strategy = router.route("First check the weather, then book a hotel, then send me a confirmation", [])
        assert isinstance(strategy, PlanExecuteStrategy)

    def test_routes_reasoning_to_cot(self):
        router = AdaptiveRouter()
        strategy = router.route("Prove that the sum of angles in a triangle is 180 degrees", [])
        assert isinstance(strategy, CoTStrategy)

    def test_routes_safe_critical_to_reflection(self):
        router = AdaptiveRouter()
        strategy = router.route("Delete the production database and redeploy", [])
        assert isinstance(strategy, ReflectionWrapper)

    def test_default_to_react(self):
        router = AdaptiveRouter()
        strategy = router.route("Hello!", [])
        assert isinstance(strategy, ReActStrategy)
```

- [ ] **Step 3: 实现 ReflectionWrapper**

```python
# agentflow/runtime/thinking/reflection.py

from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class ReflectionWrapper(ThinkingStrategy):
    """在任何策略外层包裹反思循环。"""

    def __init__(self, inner: ThinkingStrategy, max_reflections: int = 3):
        self.inner = inner
        self.max_reflections = max_reflections

    async def run(self, context: ThinkContext) -> ThinkResult:
        all_notes = []

        for i in range(self.max_reflections):
            result = await self.inner.run(context)

            # Self-check: ask LLM to review the result
            review_messages = [
                {"role": "system", "content": "You are a quality reviewer."},
                {"role": "user", "content": (
                    f"Original task: {context.user_input}\n\n"
                    f"Agent response: {result.output}\n\n"
                    "Review the response. Answer ONLY with:\n"
                    "PASS if the response is correct and complete.\n"
                    "FAIL: <reason> if there is an issue that needs correction."
                )},
            ]
            review_response = await context.llm_client.chat(review_messages)
            review_text = review_response.content.strip()
            all_notes.append(review_text)

            if review_text.startswith("PASS"):
                result.reflection_notes = all_notes
                return result

            # FAIL — inject feedback for retry
            context.add_feedback([review_text])

        result.reflection_notes = all_notes
        return result
```

- [ ] **Step 4: 实现 AdaptiveRouter**

```python
# agentflow/runtime/thinking/adaptive.py

from agentflow.runtime.thinking.base import ThinkingStrategy
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper


class AdaptiveRouter:
    """根据任务信号自动选择最优思考模式。"""

    SIGNALS = {
        "multi_step": [
            "first", "then", "after", "step", "接下来", "然后", "之后",
            "1.", "2.", "3.",
        ],
        "deep_reasoning": [
            "why", "prove", "calculate", "analyze", "explain",
            "证明", "推导", "计算", "分析",
        ],
        "safe_critical": [
            "delete", "deploy", "drop", "charge", "扣款",
            "删除", "部署", "提交代码", "commit",
        ],
    }

    def _detect(self, user_input: str) -> set[str]:
        text = user_input.lower()
        signals = set()
        for signal_type, keywords in self.SIGNALS.items():
            if any(kw in text for kw in keywords):
                signals.add(signal_type)
        return signals

    def route(self, user_input: str, tools: list) -> ThinkingStrategy:
        signals = self._detect(user_input)

        if "safe_critical" in signals:
            return ReflectionWrapper(PlanExecuteStrategy(), max_reflections=3)

        if "multi_step" in signals:
            return PlanExecuteStrategy()

        if "deep_reasoning" in signals:
            return CoTStrategy()

        return ReActStrategy()
```

- [ ] **Step 5: 实现 ThinkingEngine + ThinkingMode**

```python
# agentflow/runtime/thinking/__init__.py

from enum import Enum
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper
from agentflow.runtime.thinking.adaptive import AdaptiveRouter


class ThinkingMode(str, Enum):
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    COT = "cot"
    ADAPTIVE = "adaptive"

    def with_reflection(self, depth: int = 3) -> "ThinkingMode":
        """链式调用：给当前模式包裹反思层。"""
        self._reflection_depth = depth
        return self


class ThinkingEngine:
    """管理多个思考策略，根据模式选择或自适应路由。"""

    def __init__(self, mode: ThinkingMode = ThinkingMode.ADAPTIVE, toolkit=None):
        self.mode = mode
        self.toolkit = toolkit
        self._reflection_depth = getattr(mode, '_reflection_depth', 0)

    def _build_strategy(self, base: ThinkingStrategy) -> ThinkingStrategy:
        if self._reflection_depth > 0:
            return ReflectionWrapper(base, max_reflections=self._reflection_depth)
        return base

    def resolve_strategy(self, user_input: str, tools: list) -> ThinkingStrategy:
        if self.mode == ThinkingMode.ADAPTIVE:
            return AdaptiveRouter().route(user_input, tools)

        mapping = {
            ThinkingMode.REACT: ReActStrategy(toolkit=self.toolkit),
            ThinkingMode.PLAN_EXECUTE: PlanExecuteStrategy(),
            ThinkingMode.COT: CoTStrategy(),
        }
        base = mapping.get(self.mode, ReActStrategy(toolkit=self.toolkit))
        return self._build_strategy(base)

    async def run(self, context: ThinkContext) -> ThinkResult:
        strategy = self.resolve_strategy(context.user_input, context.tools)
        result = await strategy.run(context)
        return result
```

- [ ] **Step 6: 运行全部 thinking 测试 PASS**

- [ ] **Step 7: 提交**

---

## Phase 5: AgentBuilder 门面 + 全链路整合

### Task 5.1: AgentBuilder

**Files:**

- Create: `agentflow/runtime/builder.py`
- Modify: `agentflow/runtime/agent.py` — BaseAgent 简化为容器
- Test: `tests/runtime/test_builder.py`

- [ ] **Step 1: 写 Builder 测试**

```python
# tests/runtime/test_builder.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.memory.manager import MemoryProfile
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.thinking import ThinkingMode


class TestAgentBuilder:
    def test_minimal_build(self):
        """最简 Builder：仅名称 + mock LLM。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Hello!", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("minimal")
            .with_llm(mock_llm)
            .build())

        assert agent.name == "minimal"
        import asyncio
        result = asyncio.run(agent.run("Hi"))
        assert "Hello" in result.output

    def test_with_tools(self):
        """Builder 集成 ToolKit。"""

        @tool
        def echo(text: str) -> str:
            """Echo text back."""
            return text

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content=None, tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "hello world"}'},
            }]),
            MagicMock(content="You said: hello world", tool_calls=[]),
        ]

        agent = (AgentBuilder("tool-agent")
            .with_llm(mock_llm)
            .with_tools(echo)
            .build())

        import asyncio
        result = asyncio.run(agent.run("Echo hello world"))
        assert len(result.tool_calls) == 1

    def test_with_memory_profile(self):
        """Builder 集成 MemoryProfile。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Got it.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("mem-agent")
            .with_llm(mock_llm)
            .with_memory(MemoryProfile.light())
            .build())

        assert agent.memory.profile.working.max_turns == 10

    def test_with_prompt_template(self):
        """Builder 集成 PromptTemplate。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="I am a support agent.", role="assistant", tool_calls=[],
        )

        template = PromptTemplate.preset("customer_support")
        agent = (AgentBuilder("prompt-agent")
            .with_llm(mock_llm)
            .with_prompt(template)
            .build())

        import asyncio
        result = asyncio.run(agent.run("Help!"))
        assert result.output is not None

    def test_with_thinking_mode(self):
        """Builder 集成 ThinkingMode。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Answer.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("thinker")
            .with_llm(mock_llm)
            .with_thinking(ThinkingMode.COT)
            .build())

        assert agent.thinking_engine.mode == ThinkingMode.COT

    def test_full_build(self):
        """全组件 Builder 集成测试。"""
        @tool
        def lookup(query: str) -> str:
            """Search for info."""
            return f"Result for {query}"

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            MagicMock(content=None, tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "lookup", "arguments": '{"query": "refund"}'},
            }]),
            MagicMock(content="Based on our policy, you can refund within 30 days.", tool_calls=[]),
        ]

        agent = (AgentBuilder("full-agent")
            .with_llm(mock_llm)
            .with_tools(lookup)
            .with_memory(MemoryProfile.standard())
            .with_prompt(PromptTemplate.preset("customer_support"))
            .with_thinking(ThinkingMode.REACT)
            .with_max_iterations(3)
            .build())

        import asyncio
        result = asyncio.run(agent.run("I want a refund"))
        assert "refund" in result.output.lower()

    def test_builder_from_string_prompt(self):
        """向后兼容：纯字符串 System Prompt。"""
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = MagicMock(
            content="Yes.", role="assistant", tool_calls=[],
        )

        agent = (AgentBuilder("compat")
            .with_llm(mock_llm)
            .with_prompt("You are helpful.")
            .build())

        import asyncio
        result = asyncio.run(agent.run("Hello"))
        assert result.output is not None
```

- [ ] **Step 2: 运行测试 FAIL**

- [ ] **Step 3: 实现 AgentBuilder**

```python
# agentflow/runtime/builder.py

from agentflow.runtime.agent import BaseAgent, AgentResult
from agentflow.runtime.thinking import ThinkingEngine, ThinkingMode, ThinkContext
from agentflow.runtime.toolkit import ToolKit, Tool
from agentflow.runtime.memory.manager import MemoryManager, MemoryProfile
from agentflow.runtime.prompt import PromptTemplate


class AgentBuilder:
    """Agent 构建器——统一入口。

    用法:
        agent = (AgentBuilder("my-agent")
            .with_llm(llm_client)
            .with_tools(my_tool)
            .with_memory(MemoryProfile.standard())
            .with_prompt(PromptTemplate.preset("customer_support"))
            .with_thinking(ThinkingMode.ADAPTIVE)
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._llm_client = None
        self._toolkit = ToolKit()
        self._memory_profile = MemoryProfile.standard()
        self._prompt_template = PromptTemplate.preset("default")
        self._thinking_mode = ThinkingMode.ADAPTIVE
        self._max_iterations = 10
        self._system_prompt_str = None  # 向后兼容：纯字符串 prompt

    def with_llm(self, llm_client) -> "AgentBuilder":
        self._llm_client = llm_client
        return self

    def with_tools(self, *tools_or_kit) -> "AgentBuilder":
        """接收 @tool 装饰的函数、Tool 实例 或 ToolKit 实例。"""
        for item in tools_or_kit:
            if isinstance(item, ToolKit):
                for t in item.list():
                    self._toolkit.add(t)
            elif isinstance(item, Tool):
                self._toolkit.add(item)
            else:
                # 假定是 @tool 装饰的函数
                self._toolkit.add(item)
        return self

    def with_memory(self, profile: MemoryProfile) -> "AgentBuilder":
        self._memory_profile = profile
        return self

    def with_prompt(self, prompt) -> "AgentBuilder":
        """接收 PromptTemplate 或纯字符串。"""
        if isinstance(prompt, str):
            self._system_prompt_str = prompt
        elif isinstance(prompt, PromptTemplate):
            self._prompt_template = prompt
        return self

    def with_thinking(self, mode: ThinkingMode) -> "AgentBuilder":
        self._thinking_mode = mode
        return self

    def with_max_iterations(self, n: int) -> "AgentBuilder":
        self._max_iterations = n
        return self

    def build(self) -> BaseAgent:
        if self._llm_client is None:
            raise ValueError("with_llm() is required. Provide an LLM client.")

        memory = MemoryManager(profile=self._memory_profile)
        thinking_engine = ThinkingEngine(mode=self._thinking_mode, toolkit=self._toolkit)

        # 确定 system prompt
        if self._system_prompt_str:
            system_prompt = self._system_prompt_str
        else:
            system_prompt = self._prompt_template.render({
                "tools": self._toolkit.list(),
                "agent_name": self._name,
            })

        return _BuiltAgent(
            name=self._name,
            llm_client=self._llm_client,
            system_prompt=system_prompt,
            toolkit=self._toolkit,
            memory=memory,
            thinking_engine=thinking_engine,
            max_iterations=self._max_iterations,
        )


class _BuiltAgent(BaseAgent):
    """AgentBuilder 构建出的完整 Agent。"""

    def __init__(self, name, llm_client, system_prompt, toolkit, memory, thinking_engine, max_iterations):
        super().__init__(name, llm_client, system_prompt, toolkit, memory, max_iterations)
        self.thinking_engine = thinking_engine

    async def run(self, user_input: str) -> AgentResult:
        # 检索门
        retrieved = self.memory.pre_turn(user_input)

        # 记忆结果注入工作记忆
        for fact in retrieved:
            from agentflow.runtime.memory.working import Message
            self.memory.working.add(Message(
                role="system",
                content=f"[Memory] {fact.subject} {fact.predicate} {fact.object}",
            ))

        self.memory.working.add(
            __import__('agentflow.runtime.memory.working', fromlist=['Message']).Message(
                role="user", content=user_input
            )
        )

        # 构建 ThinkContext
        tools_for_llm = self.tool_registry.list_for_llm() if hasattr(self.tool_registry, 'list_for_llm') else []
        context = ThinkContext(
            user_input=user_input,
            system_prompt=self.system_prompt,
            messages=self.memory.working.get_context_window(),
            tools=tools_for_llm,
            llm_client=self.llm_client,
            memory=self.memory,
            max_iterations=self.max_iterations,
        )

        # 执行思考
        think_result = await self.thinking_engine.run(context)

        # 记忆门 + 遗忘门
        from agentflow.runtime.memory.working import Message
        self.memory.working.add(Message(role="assistant", content=think_result.output))
        self.memory.post_turn()

        return AgentResult(
            output=think_result.output,
            tool_calls=think_result.tool_calls,
            steps=think_result.steps,
        )
```

- [ ] **Step 4: 更新 BaseAgent 以兼容新接口**

```python
# agentflow/runtime/agent.py — 修改 __init__ 以接受 ToolKit

from agentflow.runtime.toolkit import ToolKit
from agentflow.runtime.memory.manager import MemoryManager

class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm_client,
        system_prompt: str,
        tool_registry,   # 兼容 ToolRegistry 和 ToolKit
        memory_manager,  # 兼容旧的 MemoryManager 和新的
        max_iterations: int = 10,
    ):
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.memory = memory_manager
        self.max_iterations = max_iterations
```

- [ ] **Step 5: 运行全部测试确保一切通过**

```bash
python -m pytest tests/runtime/ -v
```

- [ ] **Step 6: 运行 demo 确保端到端不被破坏**

```bash
python examples/demo_e2e.py
```

- [ ] **Step 7: 最终提交**

```bash
git add agentflow/runtime/builder.py tests/runtime/test_builder.py agentflow/runtime/agent.py
git commit -m "feat(builder): add AgentBuilder facade with full subsystem integration"
```

---

## 风险与注意事项

1. **向后兼容**：旧 `MemoryManager` 和 `ReActAgent` 作为重导出保留，不破坏现有测试和 demo
2. **渐进可用**：每个 Phase 完成后可独立使用，不强制等全部完成
3. **旧 ReactAgent**：Phase 4 完成后，`react_agent.py` 标记为 deprecated，重导出到 `thinking/react.py`
4. **测试隔离**：所有新测试使用 mock LLM，不依赖外部 API

## 实施顺序依赖

```
Phase 1 (ToolKit) ──┐
                     ├──→ Phase 4 (Thinking) ──→ Phase 5 (Builder)
Phase 2 (Memory) ────┤
                     │
Phase 3 (Prompt) ───┘
```

Phase 1-3 相互独立，可以任意顺序或并行实施。Phase 4 依赖 Phase 1 和 2。Phase 5 依赖全部四个子系统。
