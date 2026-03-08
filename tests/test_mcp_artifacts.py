"""Tests for run artifact storage."""
import time
import pytest


class TestRunStore:
    def test_store_and_retrieve(self):
        from causal_copilot.mcp.artifacts import RunStore
        store = RunStore(ttl_seconds=60)
        run_id = store.save({"adj": [[0, 1], [0, 0]], "nodes": ["A", "B"]})
        assert run_id is not None
        result = store.get(run_id)
        assert result["nodes"] == ["A", "B"]

    def test_expired_returns_none(self):
        from causal_copilot.mcp.artifacts import RunStore
        store = RunStore(ttl_seconds=0)
        run_id = store.save({"test": True})
        time.sleep(0.01)
        assert store.get(run_id) is None

    def test_unknown_id_returns_none(self):
        from causal_copilot.mcp.artifacts import RunStore
        store = RunStore()
        assert store.get("nonexistent") is None

    def test_cleanup_removes_expired(self):
        from causal_copilot.mcp.artifacts import RunStore
        store = RunStore(ttl_seconds=0)
        store.save({"a": 1})
        store.save({"b": 2})
        time.sleep(0.01)
        store.cleanup()
        assert len(store._runs) == 0

    def test_singleton(self):
        from causal_copilot.mcp.artifacts import get_store
        s1 = get_store()
        s2 = get_store()
        assert s1 is s2
