"""Skill 系统：可复用的 Agent 能力模块。

Skill 文件格式（Markdown + YAML frontmatter）:

    ---
    name: my-skill
    description: 一句话描述
    tools: [tool_a, tool_b]
    ---

    # Skill 正文（Markdown 格式的 prompt 内容）
    ...

用法:
    loader = SkillLoader()
    skill = loader.load("skills/my-skill.md")
    agent = AgentBuilder("agent").with_skill("my-skill").build()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    """一个可复用的 Agent 能力模块。"""

    name: str
    description: str
    prompt: str  # 正文（markdown 部分）
    tools: list[str] = field(default_factory=list)  # 需要的工具名列表

    def to_system_prompt(self) -> str:
        """生成可用于 System Prompt 的文本。"""
        return self.prompt


class SkillLoader:
    """从 Markdown 文件加载 Skill。"""

    def __init__(self, skills_dir: str | Path | None = None):
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._cache: dict[str, Skill] = {}

    def load(self, path_or_name: str) -> Skill:
        """加载单个 Skill 文件。

        Args:
            path_or_name: Skill 文件路径，或 skill 名（自动在 skills_dir 下查找）

        Returns:
            Skill 对象
        """
        # 如果已缓存，直接返回
        if path_or_name in self._cache:
            return self._cache[path_or_name]

        # 解析路径
        path = Path(path_or_name)
        if not path.exists() or not path.suffix:
            # 作为 skill 名，在 skills_dir 下查找
            if self._skills_dir is None:
                raise FileNotFoundError(
                    f"Skill '{path_or_name}' not found. No skills_dir configured."
                )
            path = self._skills_dir / f"{path_or_name}.md"

        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        return self._load_from_file(path)

    def load_all(self, skills_dir: str | Path) -> list[Skill]:
        """加载目录下的所有 .md Skill 文件。"""
        dir_path = Path(skills_dir)
        if not dir_path.is_dir():
            return []

        skills = []
        for md_file in sorted(dir_path.glob("*.md")):
            try:
                skill = self._load_from_file(md_file)
                skills.append(skill)
            except ValueError:
                continue  # 跳过非 skill 的 .md 文件

        return skills

    def _load_from_file(self, path: Path) -> Skill:
        """解析单个 Skill 文件。"""
        content = path.read_text(encoding="utf-8")

        # 解析 YAML frontmatter（--- 分隔）
        parts = content.split("---")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid skill file '{path}': missing YAML frontmatter (--- ... ---)"
            )

        # frontmatter 是第二部分，正文从第三部分开始
        if content.startswith("---"):
            yaml_text = parts[1].strip()
            body = "---".join(parts[2:]).strip()
        else:
            raise ValueError(
                f"Invalid skill file '{path}': must start with ---"
            )

        # 解析 YAML
        import yaml as _yaml
        try:
            meta = _yaml.safe_load(yaml_text)
        except Exception as e:
            raise ValueError(f"Invalid YAML frontmatter in '{path}': {e}")

        if meta is None:
            raise ValueError(f"Empty YAML frontmatter in '{path}'")

        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        tools = meta.get("tools", [])

        skill = Skill(
            name=name,
            description=description,
            prompt=body,
            tools=tools if isinstance(tools, list) else [],
        )

        self._cache[str(path)] = skill
        self._cache[name] = skill
        return skill
