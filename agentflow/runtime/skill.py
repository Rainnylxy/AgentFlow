"""Skill 系统：可复用的 Agent 能力模块。

Skill 文件格式（Markdown + YAML frontmatter）:

    ---
    name: my-skill
    description: 一句话描述
    tools: [tool_a, tool_b]
    ---

    # Skill 正文（Markdown 格式的 prompt 内容）

    ## 流程
    1. 分析用户需求
    2. 调用 tool_a 获取数据
    3. 汇总结果

    ## 约束
    - 第三步之前不要调用工具

用户用纯自然语言写流程和约束，SkillLoader.load() 在加载时通过 LLM
自动提取结构化 Step（用户无感知）。运行时便可对每一步做工具门禁等硬约束。

用法:
    loader = SkillLoader(llm_client=client)
    skill = await loader.load("skills/my-skill.md")
    # skill.steps 已被自动解析为结构化步骤
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
    """从 Skill 自然语言描述中提取的结构化步骤。

    由 StepExtractor 在加载时自动解析，用户不可见。
    运行时用于工具门禁、完成检测等硬约束。
    """
    name: str                     # 步骤名，如 "分析结构"
    description: str              # 步骤描述
    allowed_tools: list[str] = field(default_factory=list)  # 空 = 不许调工具
    completion_signal: str = "llm_judge"  # llm_judge | tool_called:name | output_schema:...


@dataclass
class Skill:
    """一个可复用的 Agent 能力模块。

    包含 prompt（自然语言正文）和结构化 steps（LLM 自动提取）。
    """

    name: str
    description: str
    prompt: str                          # 正文（markdown 部分）
    tools: list[str] = field(default_factory=list)       # 需要的工具名列表
    steps: list[Step] = field(default_factory=list)      # LLM 提取的结构化步骤

    def to_system_prompt(self) -> str:
        """生成可用于 System Prompt 的文本。"""
        return self.prompt


# ---------------------------------------------------------------------------
# Step Extraction（LLM 驱动）
# ---------------------------------------------------------------------------

STEP_EXTRACTION_PROMPT = """You are analyzing a skill description written in natural language.
Extract structured steps from it.

Skill text:
{body}

Output a JSON array of steps. Each step has:
- name: short step name (Chinese or English)
- description: what this step does
- allowed_tools: list of tool names allowed in this step (empty = no tools allowed)
- completion_signal: one of:
  - "llm_judge" (LLM decides when this step is done)
  - "tool_called:<tool_name>" (step is done when this tool is called)

Return ONLY valid JSON, no other text.
Example:
[
  {{"name": "分析结构", "description": "分析输入内容的结构", "allowed_tools": [], "completion_signal": "llm_judge"}},
  {{"name": "保存结果", "description": "调用保存工具", "allowed_tools": ["save"], "completion_signal": "tool_called:save"}}
]"""


class StepExtractor:
    """使用 LLM 从 Skill 自然语言正文中提取结构化步骤。

    加载时调用一次，结果缓存在 Skill.steps 中。
    """

    def __init__(self, llm_client):
        self._llm_client = llm_client

    async def extract(self, body: str) -> list[Step]:
        """从 Skill 正文文本中提取步骤。"""
        if not self._llm_client:
            return []

        prompt = STEP_EXTRACTION_PROMPT.format(body=body[:4000])  # 截断保护
        try:
            response = await self._llm_client.chat([
                {"role": "user", "content": prompt}
            ])
            data = json.loads(response.content)
            if not isinstance(data, list):
                return []

            steps = []
            for i, item in enumerate(data):
                step = Step(
                    name=item.get("name", f"step_{i}"),
                    description=item.get("description", ""),
                    allowed_tools=item.get("allowed_tools", []),
                    completion_signal=item.get("completion_signal", "llm_judge"),
                )
                steps.append(step)
            return steps
        except (json.JSONDecodeError, KeyError, Exception):
            return []


# ---------------------------------------------------------------------------
# Skill Loader
# ---------------------------------------------------------------------------

class SkillLoader:
    """从 Markdown 文件加载 Skill。

    支持可选的 LLM 驱动 Step 提取：
        loader = SkillLoader(llm_client=client)
        skill = await loader.load("skills/my-skill.md")
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        llm_client=None,
    ):
        self._skills_dir = Path(skills_dir) if skills_dir else None
        self._llm_client = llm_client
        self._cache: dict[str, Skill] = {}

    async def load(self, path_or_name: str) -> Skill:
        """加载单个 Skill 文件。

        Args:
            path_or_name: Skill 文件路径，或 skill 名（自动在 skills_dir 下查找）

        Returns:
            Skill 对象（含 LLM 提取的结构化 steps）
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

        return await self._load_from_file(path)

    async def load_all(self, skills_dir: str | Path) -> list[Skill]:
        """加载目录下的所有 .md Skill 文件。"""
        dir_path = Path(skills_dir)
        if not dir_path.is_dir():
            return []

        skills = []
        for md_file in sorted(dir_path.glob("*.md")):
            try:
                skill = await self._load_from_file(md_file)
                skills.append(skill)
            except ValueError:
                continue  # 跳过非 skill 的 .md 文件

        return skills

    def load_sync(self, path_or_name: str) -> Skill:
        """同步版加载（不进行 LLM Step 提取）。

        向后兼容：如果不想用 LLM 提取步骤，用此方法。
        """
        import asyncio
        # 检查是否在 event loop 中
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.load(path_or_name))
        else:
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self.load(path_or_name), loop
            )
            return future.result()

    async def _load_from_file(self, path: Path) -> Skill:
        """解析单个 Skill 文件。"""
        content = path.read_text(encoding="utf-8")

        # 解析 YAML frontmatter（--- 分隔）
        if not content.startswith("---"):
            raise ValueError(
                f"Invalid skill file '{path}': must start with ---"
            )

        parts = content.split("---")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid skill file '{path}': missing YAML frontmatter (--- ... ---)"
            )

        yaml_text = parts[1].strip()
        body = "---".join(parts[2:]).strip()

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

        # LLM 驱动 Step 提取（如果有 LLM client）
        extractor = StepExtractor(self._llm_client)
        steps = await extractor.extract(body)

        skill = Skill(
            name=name,
            description=description,
            prompt=body,
            tools=tools if isinstance(tools, list) else [],
            steps=steps,
        )

        self._cache[str(path)] = skill
        self._cache[name] = skill
        return skill
