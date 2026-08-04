"""SQLite-backed persistence store for traces, sessions, memories, and cost records.

Replaces the file-based TraceStore, SessionManager JSON file store, and
ephemeral cost tracking with a single SQLite database.

Usage::

    store = AgentFlowStore(":memory:")  # or Path to .db file
    store.save_trace(trace)
    store.save_session(session)
    store.record_cost(entry)
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentflow.errors import AgentFlowError
from agentflow.trace.tracer import WorkflowTrace

# Default location: ~/.agentflow/store.db
_DEFAULT_DB = Path.home() / ".agentflow" / "store.db"


# =============================================================================
# Models
# =============================================================================


@dataclass
class StoredSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "active"  # active | idle | closed
    created_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    metadata_json: str = "{}"
    data_json: str = "{}"

    @property
    def metadata(self) -> dict:
        try:
            return json.loads(self.metadata_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def data(self) -> dict:
        try:
            return json.loads(self.data_json)
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class StoredCostEntry:
    id: int = 0
    category: str = ""          # agent | workflow | session
    name: str = ""              # agent_id, workflow_name, or session_id
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class StoredMemory:
    """Persisted semantic / episodic memory fact."""
    id: int = 0
    session_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    fact_type: str = "semantic"  # semantic | episodic
    confidence: float = 1.0
    expires_at: float = 0.0      # 0 = never
    created_at: float = field(default_factory=time.time)


# =============================================================================
# Store
# =============================================================================


class AgentFlowStore:
    """SQLite-backed persistence store for all AgentFlow data.

    Creates the database and tables on first use. Thread-safe via SQLite's
    own locking — this is not async. Heavy operations should be run in a
    thread pool.

    Tables:
        traces     — WorkflowTrace JSON blobs
        sessions   — active/idle/closed sessions
        cost_log   — per-call cost records
        memories   — semantic and episodic memory facts
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = _DEFAULT_DB
        if isinstance(db_path, str) and db_path != ":memory:":
            db_path = Path(db_path)
        if isinstance(db_path, Path):
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS traces (
                workflow_id TEXT PRIMARY KEY,
                workflow_name TEXT,
                data_json TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                duration_ms INTEGER DEFAULT 0,
                node_count INTEGER DEFAULT 0,
                failed_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'active',
                created_at REAL,
                closed_at REAL DEFAULT 0.0,
                metadata_json TEXT DEFAULT '{}',
                data_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS cost_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                model_id TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                fact_type TEXT DEFAULT 'semantic',
                confidence REAL DEFAULT 1.0,
                expires_at REAL DEFAULT 0.0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_cost_category ON cost_log(category);
            CREATE INDEX IF NOT EXISTS idx_cost_name ON cost_log(name);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(fact_type);
            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO _meta(key, value) VALUES ('schema_version', ?)",
            (str(self.SCHEMA_VERSION),),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def save_trace(self, trace: WorkflowTrace) -> str:
        """Persist a WorkflowTrace. Returns workflow_id."""
        data = trace.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO traces
               (workflow_id, workflow_name, data_json, duration_ms,
                node_count, failed_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                trace.workflow_id,
                trace.workflow_name,
                json.dumps(data, ensure_ascii=False),
                trace.summary.total_duration_ms,
                trace.summary.total_nodes,
                trace.summary.nodes_failed,
            ),
        )
        self._conn.commit()
        return trace.workflow_id

    def load_trace(self, workflow_id: str) -> WorkflowTrace | None:
        """Load a WorkflowTrace by ID."""
        row = self._conn.execute(
            "SELECT data_json FROM traces WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return WorkflowTrace._from_dict(json.loads(row["data_json"]))

    def list_traces(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """List recent trace summaries (no full data)."""
        rows = self._conn.execute(
            """SELECT workflow_id, workflow_name, duration_ms, node_count,
                      failed_count, created_at
               FROM traces ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_trace(self, workflow_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM traces WHERE workflow_id = ?", (workflow_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def save_session(self, session: StoredSession) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, status, created_at, closed_at, metadata_json, data_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.id, session.status, session.created_at,
                session.closed_at, session.metadata_json, session.data_json,
            ),
        )
        self._conn.commit()

    def load_session(self, session_id: str) -> StoredSession | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredSession(**dict(row))

    def list_sessions(
        self, status: str | None = None, limit: int = 20,
    ) -> list[StoredSession]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [StoredSession(**dict(r)) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close_session(self, session_id: str) -> StoredSession | None:
        s = self.load_session(session_id)
        if s is None:
            return None
        s.status = "closed"
        s.closed_at = time.time()
        self.save_session(s)
        return s

    # ------------------------------------------------------------------
    # Cost log
    # ------------------------------------------------------------------

    def record_cost(self, entry: StoredCostEntry) -> int:
        cur = self._conn.execute(
            """INSERT INTO cost_log (category, name, model_id, input_tokens,
                                     output_tokens, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.category, entry.name, entry.model_id,
                entry.input_tokens, entry.output_tokens,
                entry.cost_usd, entry.created_at,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def total_cost(
        self, category: str | None = None, name: str | None = None,
    ) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_log WHERE 1=1"
        params: list = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        if name:
            sql += " AND name = ?"
            params.append(name)
        row = self._conn.execute(sql, params).fetchone()
        return row[0]

    def cost_breakdown(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT category, COALESCE(SUM(cost_usd), 0) as total FROM cost_log GROUP BY category",
        ).fetchall()
        return {r["category"]: r["total"] for r in rows}

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def save_memory(self, mem: StoredMemory) -> int:
        cur = self._conn.execute(
            """INSERT INTO memories (session_id, subject, predicate, object,
                                     fact_type, confidence, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.session_id, mem.subject, mem.predicate, mem.object,
                mem.fact_type, mem.confidence, mem.expires_at, mem.created_at,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def query_memories(
        self,
        session_id: str | None = None,
        fact_type: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> list[StoredMemory]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if fact_type:
            sql += " AND fact_type = ?"
            params.append(fact_type)
        if keyword:
            sql += " AND (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)"
            kw = f"%{keyword}%"
            params.extend([kw, kw, kw])
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [StoredMemory(**dict(r)) for r in rows]

    def delete_expired_memories(self) -> int:
        """Remove memories with non-zero expires_at in the past. Returns count."""
        cur = self._conn.execute(
            "DELETE FROM memories WHERE expires_at > 0 AND expires_at < ?",
            (time.time(),),
        )
        self._conn.commit()
        return cur.rowcount

    def clear_memories(self, session_id: str | None = None) -> int:
        if session_id:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE session_id = ?", (session_id,),
            )
        else:
            cur = self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
