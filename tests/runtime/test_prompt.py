import pytest
import os
import tempfile

import pytest
from agentflow.runtime.prompt import PromptTemplate, PromptRegistry, PromptVersion, PromptDiff
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


# ==============================================================================
# PromptRegistry — version management
# ==============================================================================

class TestPromptRegistry:
    @pytest.fixture
    def reg(self):
        """Create a registry with a temporary base directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield PromptRegistry(base_dir=tmp)

    def test_save_and_get(self, reg):
        v = reg.save("support", "You are a helpful assistant.", {"author": "alice"})
        assert v.name == "support"
        assert len(v.version) == 12
        assert v.template == "You are a helpful assistant."

        retrieved = reg.get("support")
        assert retrieved is not None
        assert retrieved.version == v.version
        assert retrieved.metadata["author"] == "alice"

    def test_get_by_version(self, reg):
        v1 = reg.save("support", "Prompt v1")
        v2 = reg.save("support", "Prompt v2")

        assert reg.get("support").version == v2.version  # latest
        assert reg.get("support", v1.version).template == "Prompt v1"

    def test_get_nonexistent_name(self, reg):
        assert reg.get("nonexistent") is None

    def test_get_nonexistent_version(self, reg):
        reg.save("support", "test prompt")
        assert reg.get("support", "deadbeef1234") is None

    def test_content_hash_deduplication(self, reg):
        """Same content should return the same version (no duplicate files)."""
        v1 = reg.save("support", "You are helpful.")
        v2 = reg.save("support", "You are helpful.")

        assert v1.version == v2.version
        # Should only create one version file
        versions = reg.list_versions("support")
        assert len(versions) == 1

    def test_list_versions(self, reg):
        reg.save("support", "V1")
        reg.save("support", "V2")
        reg.save("support", "V3")

        versions = reg.list_versions("support")
        assert len(versions) == 3
        # Newest first
        assert versions[0].template == "V3"

    def test_list_versions_respects_limit(self, reg):
        for i in range(10):
            reg.save("support", f"V{i}")
        assert len(reg.list_versions("support", limit=3)) == 3

    def test_list_versions_empty(self, reg):
        assert reg.list_versions("unknown") == []

    def test_delete_version(self, reg):
        v1 = reg.save("support", "V1")
        v2 = reg.save("support", "V2")
        assert reg.delete("support", v1.version) is True
        assert reg.get("support", v1.version) is None
        assert reg.get("support").version == v2.version  # v2 still current

    def test_delete_current_promotes_next(self, reg):
        v1 = reg.save("support", "V1")
        v2 = reg.save("support", "V2")
        reg.delete("support", v2.version)
        assert reg.get("support").version == v1.version

    def test_delete_nonexistent(self, reg):
        assert reg.delete("support", "nope") is False

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def test_diff_two_versions(self, reg):
        v1 = reg.save("support", "Line 1\nLine 2\nLine 3")
        v2 = reg.save("support", "Line 1\nLine 2 updated\nLine 3")

        diff = reg.diff("support", v1.version, v2.version)
        assert diff is not None
        assert diff.old_version == v1.version
        assert diff.new_version == v2.version
        assert diff.added_lines > 0
        assert diff.removed_lines > 0
        assert diff.unified_diff
        assert "Line 2 updated" in diff.unified_diff
        assert not diff.is_identical

    def test_diff_identical_versions(self, reg):
        v1 = reg.save("support", "Same content")
        v2 = reg.save("support", "Same content")

        diff = reg.diff("support", v1.version, v2.version)
        assert diff is not None
        assert diff.is_identical

    def test_diff_nonexistent_version(self, reg):
        v1 = reg.save("support", "Content")
        assert reg.diff("support", v1.version, "nope") is None
        assert reg.diff("support", "nope", v1.version) is None

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def test_rollback_sets_current(self, reg):
        v1 = reg.save("support", "Original prompt")
        v2 = reg.save("support", "Updated prompt")

        result = reg.rollback("support", v1.version)
        assert result is not None
        assert result.template == "Original prompt"
        assert reg.get("support").template == "Original prompt"

    def test_rollback_preserves_history(self, reg):
        v1 = reg.save("support", "Original")
        reg.save("support", "Updated")

        reg.rollback("support", v1.version)
        # v2 should still exist
        versions = reg.list_versions("support")
        assert len(versions) == 2

    def test_rollback_nonexistent_version(self, reg):
        assert reg.rollback("support", "nope") is None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def test_metadata_stored(self, reg):
        v = reg.save("support", "Prompt", {"author": "alice", "tags": ["v1", "prod"]})
        retrieved = reg.get("support")
        assert retrieved.metadata["author"] == "alice"
        assert "prod" in retrieved.metadata["tags"]
