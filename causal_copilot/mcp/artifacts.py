"""Run artifact storage for multi-step MCP workflows.

Each discover/run_algorithm call returns a run_id. Other tools
(refine_graph, estimate_effects, explain_result) can reference
a run_id to access cached results instead of re-parsing CSV.
"""

from __future__ import annotations

import time
import uuid
from threading import Lock


class RunStore:
    """In-memory store for run artifacts with TTL expiry."""

    def __init__(self, ttl_seconds=3600):
        self._runs = {}
        self._lock = Lock()
        self.ttl_seconds = ttl_seconds

    def save(self, data):
        """Store run data, return run_id."""
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._runs[run_id] = {
                "data": data,
                "created_at": time.time(),
            }
        return run_id

    def get(self, run_id):
        """Retrieve run data by ID. Returns None if expired or missing."""
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return None
            if time.time() - entry["created_at"] > self.ttl_seconds:
                del self._runs[run_id]
                return None
            return entry["data"]

    def cleanup(self):
        """Remove all expired entries."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._runs.items() if now - v["created_at"] > self.ttl_seconds]
            for k in expired:
                del self._runs[k]


# Module-level singleton
_store = RunStore()


def get_store():
    return _store
