# Prompt 模板子系统详细设计

> 日期: 2026-06-01 | 状态: Draft | 关联: [[2026-06-01-agent-builder-design]]

## 一、目标

将手写 system_prompt 字符串升级为模块化拼装系统，让不同 Agent 复用相同的角色定义、工具说明、安全规则等模块，通过参数定制而非复制粘贴。

## 二、核心概念

### 2.1 Section——Prompt 的最小复用单元

```python
class Section(ABC):
    """Prompt 模块基类。每个 Section 代表 Prompt 中一个独立段落。"""
    name: str           # 模块名，如 "support_role", "safety_rules"
    order: int = 50     # 排序权重（决定在最终 Prompt 中的位置）

    @abstractmethod
    def render(self, context: dict) -> str:
        """根据上下文渲染出本段 Prompt 文本。"""
        ...
```

### 2.2 PromptTemplate——Section 的容器

```python
class PromptTemplate:
    """将多个 Section 组装为完整的 System Prompt。"""

    def __init__(self, name: str):
        self.name = name
        self.sections: list[Section] = []

    def add(self, section: Section) -> "PromptTemplate": ...
    def add_after(self, after_name: str, section: Section) -> "PromptTemplate": ...
    def remove(self, name: str) -> "PromptTemplate": ...
    def override(self, name: str, **params) -> "PromptTemplate": ...

    def render(self, context: dict | None = None) -> str:
        """按 order 排序，逐个调用 section.render()，拼接为最终 Prompt。"""
        ...
```

## 三、内置 Section 分类

### 3.1 通用模块（builtin/）

| 模块 | 说明 | 关键参数 |
|------|------|----------|
| `role_card` | 角色定义 | `name`, `role`, `tone`, `audience` |
| `tool_manual` | 工具使用说明 | 自动从 ToolKit 注入工具列表 |
| `safety_rules` | 安全约束 | `rules: list[str]` |
| `format_guide` | 输出格式要求 | `format: "markdown" \| "json" \| "plain"` |
| `time_context` | 当前时间上下文 | 自动注入当前日期时间 |

### 3.2 领域模块（domains/）

| 模块 | 说明 |
|------|------|
| `customer_support` | 客服场景套装（角色 + 响应规范 + 升级规则） |
| `coding_assistant` | 编程助手套装（代码规范 + 安全约束 + 测试要求） |
| `research_analyst` | 研究分析套装（引证要求 + 批判思维 + 来源标注） |

## 四、使用方式

### 4.1 手动组装

```python
template = PromptTemplate("custom_agent")
template.add(RoleCard(name="Alice", role="客服", tone="专业友善"))
template.add(ToolManual())                    # 自动读取 ToolKit
template.add(SafetyRules(rules=["不泄露个人信息", "不承诺无法兑现的事项"]))
template.add(FormatGuide(format="markdown"))

system_prompt = template.render()
```

### 4.2 场景预设

```python
# 一键生成客服 Agent 的完整 Prompt
template = PromptTemplate.preset("customer_support")

# 覆盖部分参数
template.override("role_card", tone="活泼幽默")
template.override("safety_rules", max_refund=500)
```

### 4.3 自定义 Section

```python
@register_section("my_company_rules")
class CompanyRules(Section):
    name = "my_company_rules"
    order = 30
    prompt = """## 公司规则
    1. 工作时间：{{ work_hours }}
    2. 退款上限：¥{{ max_refund }}
    3. 严重问题转人工：{{ escalation_phone }}"""

# 注册后即可被任何 PromptTemplate 引用
template.add(CompanyRules(work_hours="9:00-18:00", max_refund=500, ...))
```

### 4.4 从 YAML 加载

```yaml
# prompts/support_agent.yaml
name: support_agent
sections:
  - type: role_card
    params:
      name: 小助手
      role: 客服代表
      tone: 友善专业
  - type: tool_manual
  - type: safety_rules
    params:
      rules:
        - 不泄露用户个人信息
        - 退款超过 ¥500 需主管审批
```

```python
template = PromptTemplate.from_yaml("prompts/support_agent.yaml")
```

## 五、模板引擎

使用 Jinja2 作为模板引擎，Section 内部的 `prompt` 字符串支持：

- 变量插值：`{{ variable }}`
- 条件逻辑：`{% if condition %}...{% endif %}`
- 循环：`{% for item in list %}...{% endfor %}`

ToolManual 中的工具列表、SafetyRules 中的规则列表天然适合用循环渲染。

## 六、关键细节

### 6.1 Section 排序

每个 Section 有 `order` 属性，最终 Prompt 按 order 升序排列。默认值 50。内置模块的默认排序：

```
order=10: role_card      (角色定义最先)
order=20: safety_rules   (安全规则紧随)
order=30: format_guide   (输出格式)
order=40: tool_manual    (工具说明)
order=50: 自定义模块     (默认)
order=60: time_context   (时间上下文最后)
```

### 6.2 自动上下文注入

`PromptTemplate.render()` 时自动注入：
- `tools`：从当前 Agent 的 ToolKit 读取
- `current_time`：当前日期时间
- `agent_name`：从 AgentBuilder 传入

开发者无需手动传递这些上下文变量。

### 6.3 渲染缓存

Section 的渲染结果可缓存。当 `render()` 的 context 参数不变时，返回缓存结果。`tool_manual` 类 Section 在 ToolKit 变化时自动失效缓存。

## 七、与现有代码的关系

| 现有模块 | 处理 |
|----------|------|
| `system_prompt: str` 参数 | 保留，内部自动包装为 `PromptTemplate.from_string()` |
| ReActAgent 中的 prompt 拼接 | 移除，统一由 PromptTemplate 管理 |

## 八、待定内容

- Prompt 渲染的可视化预览（Dashboard）
- Prompt 的 A/B 变体管理与效果对比
- 基于评测结果自动优化 Prompt 参数
