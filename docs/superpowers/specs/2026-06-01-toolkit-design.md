# ToolKit 子系统详细设计

> 日期: 2026-06-01 | 状态: Draft | 关联: [[2026-06-01-agent-builder-design]]

## 一、目标

将工具定义的体验从"手写 JSON Schema + 手动注册"升级为"装饰器一键注册"，同时统一本地函数、MCP Server、REST API 三种工具源，让 Agent 对工具来源完全透明。

## 二、核心组件

### 2.1 `@tool` 装饰器

把任意 Python 函数一键转换为 Agent 可用的 Tool。

**自动推导规则**：

| 字段          | 来源                                        |
| ------------- | ------------------------------------------- |
| `name`        | 函数名（可用 `@tool(name="...")` 覆盖）     |
| `description` | 函数 docstring 首行                         |
| `parameters`  | 类型注解 → JSON Schema（Pydantic 负责转换） |
| `return`      | 返回值自动包装为 `ToolResult`               |

**三种用法**：

```python
# L1: 零配置（简单工具）
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# L2: Pydantic 精确控制（复杂参数）
class WeatherParams(BaseModel):
    city: str = Field(description="City name")
    unit: str = Field(default="celsius")

@tool(params=WeatherParams, description="Get current weather")
def get_weather(city: str, unit: str = "celsius") -> dict: ...

# L3: 手动覆盖全部字段（迁移旧代码）
@tool(name="legacy_lookup", description="...",
      params={"type": "object", "properties": {...}})
def old_function(*args, **kwargs): ...
```

### 2.2 ToolKit 容器

```python
class ToolKit:
    """统一的工具集合，组合注册、校验、执行、Schema 导出。"""

    def add(self, source: Tool | ToolKit | MCPTool | RESTTool) -> "ToolKit": ...
    def from_module(module_path: str) -> "ToolKit": ...  # 自动发现 @tool
    def list_for_llm(self) -> list[dict]: ...              # OpenAI function-calling 格式
    def validate(self, name: str, params: dict) -> ValidationResult: ...
    def execute(self, name: str, params: dict) -> ToolResult: ...
    def dry_run(self, name: str, params: dict) -> ToolResult: ...
```

### 2.3 三源统一

三种工具类型实现同一个 `ToolLike` 接口：

```python
class ToolLike(Protocol):
    name: str
    description: str
    parameters: dict

    async def execute(self, params: dict) -> ToolResult: ...
    def validate(self, params: dict) -> ValidationResult: ...

# 本地函数 → LocalTool
tool = LocalTool(func=lookup_kb, params=LookupParams)

# MCP → MCPTool
tool = MCPTool(server="weather-server", tool="get_forecast", transport="stdio")

# REST → RESTTool
tool = RESTTool(method="POST", url="https://api.example.com/chat",
                auth_header="Bearer ${TOKEN}")
```

三者在 ToolKit 中混用，Agent 无感知。

## 三、关键细节

### 3.1 Pydantic 参数校验

`@tool` 装饰器在调用函数前自动通过 Pydantic 校验输入参数，类型不匹配时返回明确错误信息给 LLM，帮助 Agent 自我修正。

### 3.2 MCP 传输

MCPTool 支持两种传输方式：

- **stdio**：启动子进程，通过标准输入/输出通信
- **HTTP**：向 MCP Server 发送 HTTP 请求

连接管理和重连逻辑封装在 MCPTool 内部。

### 3.3 自动发现

```python
# 扫描整个 Python 模块，收集所有 @tool 装饰的函数
toolkit = ToolKit.from_module("my_project.tools")

# 等同于：
# from my_project.tools import lookup_kb, get_weather
# toolkit.add(lookup_kb)
# toolkit.add(get_weather)
```

### 3.4 错误处理

工具执行失败时不抛异常，返回 `ToolResult(success=False, error="原因")`，由 LLM 自行决定是否重试或切换策略。这个数据同时记录到 Trace，作为 Adaptability 评测数据。

## 四、与现有代码的关系

| 现有模块           | 处理                                     |
| ------------------ | ---------------------------------------- |
| `tool_registry.py` | 保留作为后端存储，ToolKit 委托给它       |
| `Tool` dataclass   | 保留，`@tool` 背后生成的就是 `Tool` 实例 |
| `ToolResult`       | 保留，不变                               |
| `ToolType` enum    | 扩展：增加 `MCP`、`REST`                 |
| MCP/REST 执行      | 新增 `mcp_client.py`、`rest_client.py`   |

## 五、待定内容（不在本期实现）

- 工具调用的流式返回（tool call streaming）
- 工具的版本管理与回滚
