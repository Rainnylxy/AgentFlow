"""Skill 系统：可复用的 Agent 能力模块。

懒加载设计：
    build() 时只读 frontmatter（name + description + tools）——不读 body，不调 LLM。
    prompt 和 steps 在第一次使用时按需加载。

用法:
    loader = SkillLoader()
    skill = await loader.load_meta("customer-support")   # 快，无 LLM
    await skill.ensure_loaded()                           # 按需加载 body + steps
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """从 Skill 自然语言描述中提取的结构化步骤。"""
    name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    completion_signal: str = "llm_judge"


@dataclass
class Skill:
    """一个可复用的 Agent 能力模块。

    懒加载：build() 时只填充 name/description/tools。
    prompt 和 steps 在第一次 ensure_loaded() 时才从文件读取。
    """

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)

    prompt: str = ""       # 正文（markdown 部分）——懒加载
    steps: list[Step] = field(default_factory=list)  # 结构化步骤——懒加载

    # -- 内部懒加载用 --
    _loaded: bool = field(default=False, repr=False)
    _file_path: str = field(default="", repr=False)
    _loader_ref: object = field(default=None, repr=False)

    def to_system_prompt(self) -> str:
        """生成可用于 System Prompt 的文本。"""
        if not self._loaded:
            return f"## {self.name}\n{self.description}"
        return self.prompt

    async def ensure_loaded(self) -> None:
        """按需加载——如果还没加载，从源文件读 body + 提取 steps。"""
        if self._loaded or not self._file_path:
            return
        if self._loader_ref is None:
            return
        full = await self._loader_ref._load_from_file(Path(self._file_path))
        self.prompt = full.prompt
        self.steps = full.steps
        self._loaded = True


# ---------------------------------------------------------------------------
# Step 提取（LLM 驱动，可选）
# ---------------------------------------------------------------------------

STEP_EXTRACTION_PROMPT = """You are analyzing a skill description written in natural language.
Extract structured steps from it.

Skill text:
{body}

Output a JSON array of steps. Each step has:
- name: short step name
- description: what this step does
- allowed_tools: list of tool names allowed in this step (empty = no tools allowed)
- completion_signal: "llm_judge" or "tool_called:<tool_name>"

Return ONLY valid JSON, no other text.
Example:
[
  {{"name": "分析结构", "description": "...", "allowed_tools": [], "completion_signal": "llm_judge"}},
  {{"name": "保存", "description": "...", "allowed_tools": ["save"], "completion_signal": "tool_called:save"}}
]"""


class StepExtractor:
    """使用 LLM 从 Skill 自然语言正文中提取结构化步骤。"""

    def __init__(self, llm_client):
        self._llm_client = llm_client

    async def extract(self, body: str) -> list[Step]:
        if not self._llm_client:
            return []
        prompt = STEP_EXTRACTION_PROMPT.format(body=body[:4000])
        try:
            response = await self._llm_client.chat([{"role": "user", "content": prompt}])
            data = json.loads(response.content)
            if not isinstance(data, list):
                return []
            return [
                Step(
                    name=item.get("name", f"step_{i}"),
                    description=item.get("description", ""),
                    allowed_tools=item.get("allowed_tools", []),
                    completion_signal=item.get("completion_signal", "llm_judge"),
                )
                for i, item in enumerate(data)
            ]
        except (json.JSONDecodeError, KeyError, Exception):
            return []


# ---------------------------------------------------------------------------
# Skill Loader
# ---------------------------------------------------------------------------

class SkillLoader:
    """从 Markdown 文件加载 Skill。

    支持懒加载：
        loader = SkillLoader()
        skill = await loader.load_meta("customer-support")  # 只读 frontmatter
        await skill.ensure_loaded()                          # 按需读 body

    支持 LLM Step 提取：
        loader = SkillLoader(llm_client=client)
        skill = await loader.load("customer-support")        # 完整加载 + 提取
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        llm_client=None,
    ):
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._llm_client = llm_client
        self._cache: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # 懒加载 — 方法 1：只读元数据（build() 用）
    # ------------------------------------------------------------------

    async def load_meta(self, path_or_name: str) -> Skill:
        """只加载元数据（name、description、tools）——不读 body，不调 LLM。

        这是 build() 时用的方法。Skill 的 prompt 和 steps 留空，
        等运行时按需调用 skill.ensure_loaded() 加载。
        """
        if path_or_name in self._cache:
            return self._cache[path_or_name]

        path = self._resolve_path(path_or_name)
        return self._load_meta_from_file(path)

    def _load_meta_from_file(self, path: Path) -> Skill:
        """只解析 frontmatter。"""
        content = path.read_text(encoding="utf-8")
        yaml_text, _ = self._split_frontmatter(content, path)

        import yaml as _yaml
        try:
            meta = _yaml.safe_load(yaml_text) or {}
        except Exception as e:
            raise ValueError(f"Invalid YAML frontmatter in '{path}': {e}")

        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        tools = meta.get("tools", [])

        skill = Skill(
            name=name,
            description=description,
            tools=tools if isinstance(tools, list) else [],
            prompt="",
            steps=[],
            _loaded=False,
            _file_path=str(path),
            _loader_ref=self,
        )

        self._cache[str(path)] = skill
        self._cache[name] = skill
        return skill

    # ------------------------------------------------------------------
    # 完整加载 — 方法 2（用到 Skill 内容时调用）
    # ------------------------------------------------------------------

    async def load(self, path_or_name: str) -> Skill:
        """完整加载 Skill（body + 可选 LLM Step 提取）。

        如果之前已通过 load_meta() 缓存，则返回缓存对象。
        完整加载会自动触发 ensure_loaded()。
        """
        # 如果已经在缓存中（可能是 load_meta 放的），直接用
        if path_or_name in self._cache:
            skill = self._cache[path_or_name]
        else:
            path = self._resolve_path(path_or_name)
            skill = self._load_meta_from_file(path)

        if not skill._loaded:
            full = await self._load_from_file(Path(skill._file_path) if skill._file_path else self._resolve_path(path_or_name))
            skill.prompt = full.prompt
            skill.steps = full.steps
            skill._loaded = True
        return skill

    async def load_all(self, skills_dir: str | Path) -> list[Skill]:
        """加载目录下所有 .md Skill 文件的完整内容。"""
        dir_path = Path(skills_dir)
        if not dir_path.is_dir():
            return []
        skills = []
        for md_file in sorted(dir_path.glob("*.md")):
            try:
                skill = await self.load(str(md_file))
                skills.append(skill)
            except ValueError:
                continue
        return skills

    def load_sync(self, path_or_name: str) -> Skill:
        """同步版加载（不进行 LLM Step 提取）。向后兼容。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(self.load(path_or_name), loop)
            return future.result()
        except RuntimeError:
            return asyncio.run(self.load(path_or_name))

    # ------------------------------------------------------------------
    # 文件解析（内部）
    # ------------------------------------------------------------------

    def _resolve_path(self, path_or_name: str) -> Path:
        path = Path(path_or_name)
        if path.exists() and path.suffix:
            return path
        if self._skills_dir is None:
            raise FileNotFoundError(f"Skill '{path_or_name}' not found. No skills_dir configured.")
        resolved = self._skills_dir / f"{path_or_name}.md"
        if not resolved.exists():
            raise FileNotFoundError(f"Skill file not found: {resolved}")
        return resolved

    @staticmethod
    def _split_frontmatter(content: str, path: Path) -> tuple[str, str]:
        """拆分 YAML frontmatter 和 body。返回 (yaml_text, body)。"""
        if not content.startswith("---"):
            raise ValueError(f"Invalid skill file '{path}': must start with ---")
        parts = content.split("---")
        if len(parts) < 2:
            raise ValueError(f"Invalid skill file '{path}': missing YAML frontmatter")
        yaml_text = parts[1].strip()
        body = "---".join(parts[2:]).strip()
        return yaml_text, body

    async def _load_from_file(self, path: Path) -> Skill:
        """解析完整 Skill 文件（含 LLM Step 提取）。"""
        content = path.read_text(encoding="utf-8")
        yaml_text, body = self._split_frontmatter(content, path)

        import yaml as _yaml
        try:
            meta = _yaml.safe_load(yaml_text) or {}
        except Exception as e:
            raise ValueError(f"Invalid YAML frontmatter in '{path}': {e}")

        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        tools = meta.get("tools", [])

        # LLM 驱动 Step 提取（如果有 LLM client）
        extractor = StepExtractor(self._llm_client)
        steps = await extractor.extract(body)

        return Skill(
            name=name,
            description=description,
            tools=tools if isinstance(tools, list) else [],
            prompt=body,
            steps=steps,
            _loaded=True,
            _file_path=str(path),
            _loader_ref=self,
        )
