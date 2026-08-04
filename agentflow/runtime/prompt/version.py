"""Prompt version management — content-addressable storage, diff, rollback.

No external dependencies. Pure file-system storage under
``~/.agentflow/prompts/`` with content-hash versioning.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _content_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def _prompt_dir(name: str, base: Path | None = None) -> Path:
    base = base or Path.home() / ".agentflow" / "prompts"
    return base / name


@dataclass
class PromptVersion:
    name: str
    version: str               # content sha256[:12]
    template: str               # full prompt text
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromptVersion:
        return cls(
            name=data["name"],
            version=data["version"],
            template=data["template"],
            created_at=data.get("created_at", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PromptDiff:
    old_version: str
    new_version: str
    unified_diff: str
    added_lines: int
    removed_lines: int

    @property
    def is_identical(self) -> bool:
        return self.added_lines == 0 and self.removed_lines == 0


class PromptRegistry:
    """Content-addressable prompt version storage.

    Versions are stored as JSON files under
    ``~/.agentflow/prompts/{name}/{version}.json``.
    A `current` pointer file under each prompt name records the latest version.

    Same content → same version hash → no duplicate storage.

    Usage::

        reg = PromptRegistry()
        v1 = reg.save("support", "You are a helpful assistant.", {"author": "alice"})
        v2 = reg.save("support", "You are a helpful and concise assistant.")

        latest = reg.get("support")  # → v2
        old = reg.get("support", v1.version)  # → v1

        diff = reg.diff("support", v1.version, v2.version)
        print(diff.unified_diff)

        reg.rollback("support", v1.version)  # sets current back to v1
    """

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".agentflow" / "prompts"
        self._base = Path(base_dir)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        template: str,
        metadata: dict | None = None,
    ) -> PromptVersion:
        """Persist a new version. If the content is identical to an existing
        version, returns the existing one (no duplicate storage).
        """
        version_id = _content_hash(template)

        # Check if this version already exists
        existing = self.get(name, version_id)
        if existing is not None:
            return existing

        pv = PromptVersion(
            name=name,
            version=version_id,
            template=template,
            metadata=metadata or {},
        )

        dir_path = _prompt_dir(name, self._base)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write version file
        version_path = dir_path / f"{version_id}.json"
        version_path.write_text(
            json.dumps(pv.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Update current pointer
        self._set_current(name, version_id)

        return pv

    def get(self, name: str, version: str | None = None) -> PromptVersion | None:
        """Retrieve a prompt version. If *version* is None, returns the latest."""
        if version is None:
            version = self._get_current(name)
            if version is None:
                # Fall back to newest file by mtime
                versions = self.list_versions(name, limit=1)
                return versions[0] if versions else None

        path = _prompt_dir(name, self._base) / f"{version}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PromptVersion.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list_versions(self, name: str, limit: int = 20) -> list[PromptVersion]:
        """List versions for a prompt, newest first."""
        dir_path = _prompt_dir(name, self._base)
        if not dir_path.exists():
            return []

        results: list[PromptVersion] = []
        for f in sorted(dir_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name == "_current":
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(PromptVersion.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        return results[:limit]

    def delete(self, name: str, version: str) -> bool:
        """Delete a specific version. If it is the current version, the next
        newest version becomes current. Returns True if deleted."""
        if self._get_current(name) == version:
            versions = self.list_versions(name)
            remaining = [v for v in versions if v.version != version]
            self._set_current(name, remaining[0].version if remaining else None)

        path = _prompt_dir(name, self._base) / f"{version}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, name: str, v1: str, v2: str) -> PromptDiff | None:
        """Compute line-level unified diff between two versions."""
        old = self.get(name, v1)
        new = self.get(name, v2)
        if old is None or new is None:
            return None

        old_lines = old.template.splitlines(keepends=True)
        new_lines = new.template.splitlines(keepends=True)

        udiff = "".join(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{name}@{v1}",
            tofile=f"{name}@{v2}",
        ))

        # Count added/removed lines
        added = sum(1 for line in new_lines if line not in old_lines)
        removed = sum(1 for line in old_lines if line not in new_lines)

        return PromptDiff(
            old_version=v1,
            new_version=v2,
            unified_diff=udiff or "(no changes)",
            added_lines=added,
            removed_lines=removed,
        )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, name: str, version: str) -> PromptVersion | None:
        """Set an existing version as the current version.

        This does NOT delete intermediate versions — it only updates the
        current pointer. The full history is preserved.
        """
        existing = self.get(name, version)
        if existing is None:
            return None

        self._set_current(name, version)
        return existing

    # ------------------------------------------------------------------
    # Current pointer
    # ------------------------------------------------------------------

    def _current_path(self, name: str) -> Path:
        return _prompt_dir(name, self._base) / "_current"

    def _get_current(self, name: str) -> str | None:
        path = self._current_path(name)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _set_current(self, name: str, version: str | None) -> None:
        dir_path = _prompt_dir(name, self._base)
        dir_path.mkdir(parents=True, exist_ok=True)
        path = self._current_path(name)
        if version is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(version, encoding="utf-8")
