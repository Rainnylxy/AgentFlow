"""Prompt Template — 模块化 System Prompt 组装系统。"""

from typing import Dict, List, Optional

from agentflow.runtime.prompt.section import Section, RoleCard, SafetyRules, ToolManual, FormatGuide, TimeContext


class PromptTemplate:
    """将多个 Section 组装为完整的 System Prompt。"""

    def __init__(self, name: str):
        self.name = name
        self._sections: List[Section] = []

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
                    private_key = f"_{k}"
                    if hasattr(s, private_key):
                        setattr(s, private_key, v)
                    elif hasattr(s, k):
                        setattr(s, k, v)
                break
        return self

    def render(self, context: Optional[dict] = None) -> str:
        ctx = context or {}
        if "current_time" not in ctx:
            from datetime import datetime
            ctx["current_time"] = datetime.now().isoformat()

        sorted_sections = sorted(self._sections, key=lambda s: s.order)
        parts = []
        for s in sorted_sections:
            rendered = s.render(ctx)
            if rendered.strip():
                parts.append(rendered)

        return "\n\n".join(parts)

    @classmethod
    def preset(cls, name: str) -> "PromptTemplate":
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
                "不执行危险命令",
                "代码修改前先展示 diff",
                "鼓励写测试",
            ]))
            template.add(ToolManual())
            return template

        template = cls(name)
        template.add(RoleCard(name="Assistant", role="AI 助手", tone="helpful"))
        return template

    @classmethod
    def from_string(cls, prompt: str) -> "PromptTemplate":
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
        from string import Template
        return Template(self.content).safe_substitute(**context)
