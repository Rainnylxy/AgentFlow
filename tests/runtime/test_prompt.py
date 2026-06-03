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
        assert "tool" in result.lower() or "工具" in result.lower()


class TestPromptTemplate:
    def test_add_section_and_render(self):
        template = PromptTemplate("test")
        template.add(RoleCard(name="Bob", role="助手", tone="友好"))
        template.add(SafetyRules(rules=["不泄露隐私"]))

        result = template.render()
        assert "Bob" in result
        assert "不泄露隐私" in result
        # RoleCard order=10 < SafetyRules order=20
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

    def test_from_string_fallback(self):
        template = PromptTemplate.from_string("You are a helpful assistant.")
        result = template.render()
        assert "helpful assistant" in result
