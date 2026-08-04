"""Project configuration loader for agentflow.yaml.

Parses the project config file and exposes typed accessors for LLM settings,
skill directories, eval directories, and tool registrations.

Usage::

    config = ProjectConfig.load("./agentflow.yaml")
    print(config.llm.model)        # "gpt-4o"
    print(config.skills_dir)       # Path("skills/")
    print(config.tools.to_dict())  # {name: ToolSpec, ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# =============================================================================
# Sub-models
# =============================================================================


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""      # if empty, read from AGENTFLOW_API_KEY or OPENAI_API_KEY env
    max_retries: int = 3
    timeout: float = 120.0

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return os.getenv("AGENTFLOW_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    @property
    def resolved_base_url(self) -> str:
        env_url = os.getenv("AGENTFLOW_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
        return env_url or self.base_url


@dataclass
class ToolSpec:
    """A tool definition from agentflow.yaml tools section.

    Supports three kinds:
      - local: Python callable path (e.g. "my_project.tools:search")
      - mcp: MCP server reference (server_name:tool_name)
      - rest: HTTP endpoint URL
    """

    name: str
    description: str = ""
    kind: str = "local"          # local | mcp | rest
    path: str = ""               # module path, mcp endpoint, or URL
    params: dict = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Collection of tool specs from config."""

    tools: list[ToolSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    @classmethod
    def from_yaml(cls, data: list[dict] | None) -> ToolConfig:
        if not data:
            return cls()
        tools = []
        for item in data:
            tools.append(ToolSpec(
                name=item.get("name", ""),
                description=item.get("description", ""),
                kind=item.get("kind", "local"),
                path=item.get("path", ""),
                params=item.get("params", {}),
            ))
        return cls(tools=tools)


@dataclass
class RuntimeConfig:
    """Runtime tuning parameters."""

    max_iterations: int = 10
    max_output_tokens: int = 4096
    max_input_tokens: int = 8000
    default_timeout_ms: int = 120_000
    thinking_mode: str = "adaptive"  # react | cot | plan_execute | adaptive | routing
    memory_profile: str = "standard"  # light | standard | deep


# =============================================================================
# ProjectConfig
# =============================================================================


@dataclass
class ProjectConfig:
    """Parsed agentflow.yaml project configuration."""

    name: str = ""
    version: str = "0.1.0"

    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)

    skills_dir: Path = Path("skills")
    evals_dir: Path = Path("evals")
    workflows_dir: Path = Path("workflows")

    # Raw config dict for extension
    _raw: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        """Load and parse an agentflow.yaml file.

        Returns a default config if the file doesn't exist (for backward
        compatibility with projects that don't have agentflow.yaml yet).
        """
        path = Path(path)
        if not path.exists():
            return cls._default(path.parent)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data, project_root=path.parent)

    @classmethod
    def from_dict(cls, data: dict, project_root: Path | None = None) -> ProjectConfig:
        root = project_root or Path(".")

        # LLM
        llm_raw = data.get("llm", {})
        llm = LLMConfig(
            provider=llm_raw.get("provider", "openai"),
            model=llm_raw.get("model", "gpt-4o"),
            base_url=llm_raw.get("base_url", "https://api.openai.com/v1"),
            api_key=llm_raw.get("api_key", ""),
            max_retries=llm_raw.get("max_retries", 3),
            timeout=llm_raw.get("timeout", 120.0),
        )

        # Runtime
        rt_raw = data.get("runtime", {})
        runtime = RuntimeConfig(
            max_iterations=rt_raw.get("max_iterations", 10),
            max_output_tokens=rt_raw.get("max_output_tokens", 4096),
            max_input_tokens=rt_raw.get("max_input_tokens", 8000),
            default_timeout_ms=rt_raw.get("default_timeout_ms", 120_000),
            thinking_mode=rt_raw.get("thinking_mode", "adaptive"),
            memory_profile=rt_raw.get("memory_profile", "standard"),
        )

        # Tools
        tools = ToolConfig.from_yaml(data.get("tools", []))

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            llm=llm,
            runtime=runtime,
            tools=tools,
            skills_dir=root / data.get("skills_dir", "skills"),
            evals_dir=root / data.get("evals_dir", "evals"),
            workflows_dir=root / data.get("workflows_dir", "workflows"),
            _raw=data,
        )

    @classmethod
    def _default(cls, project_root: Path) -> ProjectConfig:
        """Return a sensible default config."""
        return cls(
            name=project_root.name,
            skills_dir=project_root / "skills",
            evals_dir=project_root / "evals",
            workflows_dir=project_root / "workflows",
        )

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def discover_skills(self) -> list[str]:
        """List skill names found in skills_dir (one per .md file)."""
        if not self.skills_dir.exists():
            return []
        return sorted(
            f.stem for f in self.skills_dir.glob("*.md")
            if f.is_file()
        )

    def discover_workflows(self) -> list[Path]:
        """List workflow YAML files in workflows_dir."""
        if not self.workflows_dir.exists():
            return []
        return sorted(self.workflows_dir.glob("*.yaml"))
