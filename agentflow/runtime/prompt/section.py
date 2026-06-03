"""Section 模块库 — Prompt 模板的可复用段落。"""

from abc import ABC, abstractmethod
from typing import List, Optional
from string import Template


class Section(ABC):
    """Prompt 模块基类。每个 Section 代表 Prompt 中一个独立段落。"""
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
        name = context.get("name", self._name)
        role = context.get("role", self._role)
        tone = context.get("tone", self._tone)
        audience = context.get("audience", self._audience)

        lines = [
            "## Role",
            f"You are {name}, a {role}.",
            f"Tone: {tone}.",
        ]
        if audience:
            lines.append(f"Audience: {audience}.")
        return "\n".join(lines)


class SafetyRules(Section):
    name = "safety_rules"
    order = 20

    def __init__(self, rules: Optional[List[str]] = None):
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
