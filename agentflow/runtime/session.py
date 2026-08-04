"""Session Management — lifecycle, persistence, and isolation for agent conversations.

Sessions are stored as JSON files under ``~/.agentflow/sessions/``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SessionStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    CLOSED = "closed"


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    metadata: dict = field(default_factory=dict)
    # Arbitrary user data (e.g., conversation history references)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "metadata": self.metadata,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        return cls(
            id=d["id"],
            status=SessionStatus(d.get("status", "active")),
            created_at=d.get("created_at", 0.0),
            closed_at=d.get("closed_at", 0.0),
            metadata=d.get("metadata", {}),
            data=d.get("data", {}),
        )

    @property
    def is_active(self) -> bool:
        return self.status != SessionStatus.CLOSED

    def close(self) -> None:
        self.status = SessionStatus.CLOSED
        self.closed_at = time.time()

    def mark_idle(self) -> None:
        self.status = SessionStatus.IDLE


class SessionManager:
    """CRUD + lifecycle for agent sessions.

    Usage::

        mgr = SessionManager()
        session = mgr.create(metadata={"user": "alice"})
        mgr.save(session)  # persist

        # Later
        s = mgr.get(session.id)
        s.close()
        mgr.save(s)
    """

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".agentflow" / "sessions"
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        s = Session(
            id=session_id or uuid.uuid4().hex[:12],
            metadata=metadata or {},
        )
        return s

    def save(self, session: Session) -> None:
        path = self._path(session.id)
        path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            return None

    def list(self, status: SessionStatus | None = None, limit: int = 20) -> list[Session]:
        results: list[Session] = []
        for f in sorted(self._base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                s = Session.from_dict(json.loads(f.read_text(encoding="utf-8")))
                if status is None or s.status == status:
                    results.append(s)
            except (json.JSONDecodeError, KeyError):
                continue
        return results[:limit]

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def close_session(self, session_id: str) -> Session | None:
        s = self.get(session_id)
        if s is None:
            return None
        s.close()
        self.save(s)
        return s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._base / f"{session_id}.json"
