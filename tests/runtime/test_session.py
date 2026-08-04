"""Tests for agentflow.runtime.session — Session, SessionManager lifecycle."""

from __future__ import annotations

import tempfile

import pytest

from agentflow.runtime.session import Session, SessionManager, SessionStatus


class TestSession:
    def test_default_values(self):
        s = Session()
        assert len(s.id) == 12
        assert s.status == SessionStatus.ACTIVE
        assert s.created_at > 0
        assert s.is_active

    def test_to_dict_and_back(self):
        s = Session(metadata={"user": "alice"}, data={"key": "value"})
        d = s.to_dict()
        s2 = Session.from_dict(d)
        assert s2.id == s.id
        assert s2.metadata == {"user": "alice"}
        assert s2.data == {"key": "value"}

    def test_close(self):
        s = Session()
        s.close()
        assert s.status == SessionStatus.CLOSED
        assert s.closed_at > 0
        assert not s.is_active

    def test_mark_idle(self):
        s = Session()
        s.mark_idle()
        assert s.status == SessionStatus.IDLE
        assert s.is_active  # IDLE is still considered "not closed"


class TestSessionManager:
    @pytest.fixture
    def mgr(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield SessionManager(base_dir=tmp)

    def test_create_and_save(self, mgr):
        s = mgr.create(metadata={"user": "alice"})
        mgr.save(s)

        loaded = mgr.get(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded.metadata["user"] == "alice"

    def test_create_with_custom_id(self, mgr):
        s = mgr.create(session_id="my-custom-id")
        assert s.id == "my-custom-id"
        mgr.save(s)
        assert mgr.get("my-custom-id") is not None

    def test_get_nonexistent(self, mgr):
        assert mgr.get("nonexistent") is None

    def test_list_all(self, mgr):
        for i in range(3):
            s = mgr.create(metadata={"i": i})
            mgr.save(s)

        results = mgr.list()
        assert len(results) == 3

    def test_list_by_status(self, mgr):
        s1 = mgr.create()
        mgr.save(s1)
        s2 = mgr.create()
        s2.close()
        mgr.save(s2)

        active = mgr.list(status=SessionStatus.ACTIVE)
        closed = mgr.list(status=SessionStatus.CLOSED)
        assert len(active) == 1
        assert len(closed) == 1

    def test_list_respects_limit(self, mgr):
        for i in range(10):
            mgr.save(mgr.create())
        assert len(mgr.list(limit=3)) == 3

    def test_delete(self, mgr):
        s = mgr.create()
        mgr.save(s)
        assert mgr.delete(s.id) is True
        assert mgr.get(s.id) is None

    def test_delete_nonexistent(self, mgr):
        assert mgr.delete("nope") is False

    def test_close_session(self, mgr):
        s = mgr.create()
        mgr.save(s)

        closed = mgr.close_session(s.id)
        assert closed is not None
        assert closed.status == SessionStatus.CLOSED

        reloaded = mgr.get(s.id)
        assert reloaded.status == SessionStatus.CLOSED

    def test_close_session_nonexistent(self, mgr):
        assert mgr.close_session("nope") is None

    def test_session_isolation(self, mgr):
        """Sessions with different IDs are independent."""
        s1 = mgr.create(metadata={"role": "admin"})
        s2 = mgr.create(metadata={"role": "user"})
        mgr.save(s1)
        mgr.save(s2)

        assert mgr.get(s1.id).metadata["role"] == "admin"
        assert mgr.get(s2.id).metadata["role"] == "user"
