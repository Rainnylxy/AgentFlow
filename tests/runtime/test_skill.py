import os
import tempfile
from agentflow.runtime.skill import Skill, SkillLoader


class TestSkillLoader:
    def test_load_from_file(self):
        """从 .md 文件加载 Skill。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "---\n"
                "name: test-skill\n"
                "description: A test skill\n"
                "tools: [tool_a, tool_b]\n"
                "---\n"
                "\n"
                "## Test Skill\n"
                "This is the prompt body.\n"
            )
            path = f.name

        try:
            loader = SkillLoader()
            skill = loader.load(path)

            assert skill.name == "test-skill"
            assert skill.description == "A test skill"
            assert skill.tools == ["tool_a", "tool_b"]
            assert "## Test Skill" in skill.prompt
            assert "prompt body" in skill.prompt
        finally:
            os.unlink(path)

    def test_load_by_name_from_dir(self):
        """从 skills_dir 按名加载。"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = os.path.join(tmpdir, "test-skill.md")
            with open(skill_path, "w", encoding="utf-8") as f:
                f.write(
                    "---\n"
                    "name: test-skill\n"
                    "description: Test\n"
                    "---\n"
                    "\n"
                    "Skill body.\n"
                )

            loader = SkillLoader(skills_dir=tmpdir)
            skill = loader.load("test-skill")

            assert skill.name == "test-skill"
            assert "Skill body" in skill.prompt

    def test_load_all(self):
        """加载目录下所有 .md 文件。"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["skill-a", "skill-b"]:
                skill_path = os.path.join(tmpdir, f"{name}.md")
                with open(skill_path, "w", encoding="utf-8") as f:
                    f.write(f"---\nname: {name}\n---\n\nBody of {name}.\n")

            loader = SkillLoader()
            skills = loader.load_all(tmpdir)

            assert len(skills) == 2
            names = {s.name for s in skills}
            assert names == {"skill-a", "skill-b"}

    def test_skill_to_system_prompt(self):
        """Skill.to_system_prompt() 返回 prompt 文本。"""
        skill = Skill(
            name="test",
            description="test",
            prompt="## Test\nContent here.",
            tools=[],
        )
        result = skill.to_system_prompt()
        assert "## Test" in result
        assert "Content here" in result

    def test_missing_file_raises(self):
        """加载不存在的文件抛出 FileNotFoundError。"""
        loader = SkillLoader()
        try:
            loader.load("nonexistent-file.md")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass
