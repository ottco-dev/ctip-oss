"""
backend.tasks.task_store — Persistent SQLite-backed background task store.

Design
------
Tasks are kept in an in-memory dict (_cache) for fast hot-path access while
a task is running (log lines appended at ~5–20 Hz).  SQLite writes happen at
three points only:

  1. Task creation      — INSERT with status="queued"
  2. Status transitions — UPDATE when status changes (queued→running→done/error)
  3. On finish          — full UPDATE with final log + elapsed

Logs are NOT written to SQLite on every append to avoid write amplification.
A periodic flush writes the current log every LOG_FLUSH_EVERY lines so that
a crash doesn't lose everything.

On startup the store loads all non-expired tasks from SQLite back into memory
so that the UI can continue polling after a backend restart.

Expiry
------
Tasks older than MAX_AGE_HOURS (default 24h) are deleted from both memory and
SQLite.  Expiry runs at startup and every EXPIRY_INTERVAL_HOURS thereafter.

Thread safety
-------------
The SQLite connection is NOT shared across threads.  All DB operations are
dispatched through asyncio.to_thread so they run in the thread-pool executor,
each call opening its own connection.

Schema
------
  tasks (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,       -- queued|running|done|error|port_conflict
    started_at    REAL NOT NULL,
    finished_at   REAL,
    ok            INTEGER,             -- 1=True, 0=False, NULL=unknown
    profile       TEXT NOT NULL DEFAULT 'annotation',
    log_json      TEXT NOT NULL DEFAULT '[]',
    port_conflict TEXT,                -- JSON object or NULL
    updated_at    REAL NOT NULL
  )
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

MAX_AGE_HOURS: int = 24
LOG_FLUSH_EVERY: int = 50       # write log to DB every N new lines during a run
EXPIRY_INTERVAL_HOURS: int = 6  # how often to prune expired tasks
MAX_LOG_LINES: int = 500        # cap per task to avoid unbounded growth


# ── Types (mirrored from containers.py to avoid circular imports) ──────────────

from typing import Literal

TaskStatus = Literal["queued", "running", "done", "error", "port_conflict"]


class PortConflictData:
    """Minimal plain-data class so task_store.py has no FastAPI dep."""
    __slots__ = ("port", "service", "env_var")

    def __init__(self, port: int, service: str, env_var: str) -> None:
        self.port = port
        self.service = service
        self.env_var = env_var

    def to_dict(self) -> dict[str, Any]:
        return {"port": self.port, "service": self.service, "env_var": self.env_var}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PortConflictData":
        return cls(port=d["port"], service=d["service"], env_var=d["env_var"])


class TaskRecord:
    """
    In-memory task record.  Mutated directly by background workers.
    Changes are flushed to SQLite at status transitions and on finish.
    """
    __slots__ = (
        "id", "status", "started_at", "finished_at", "ok",
        "profile", "log", "port_conflict",
        "_last_flush_line",
    )

    def __init__(
        self,
        id: str,
        profile: str,
        status: TaskStatus = "queued",
        started_at: float | None = None,
        finished_at: float | None = None,
        ok: bool | None = None,
        log: list[str] | None = None,
        port_conflict: PortConflictData | None = None,
    ) -> None:
        self.id = id
        self.profile = profile
        self.status: TaskStatus = status
        self.started_at = started_at or time.time()
        self.finished_at = finished_at
        self.ok = ok
        self.log: list[str] = log or []
        self.port_conflict = port_conflict
        self._last_flush_line = len(self.log)

    @property
    def elapsed_seconds(self) -> float | None:
        if self.finished_at:
            return round(self.finished_at - self.started_at, 1)
        if self.status in ("running", "queued"):
            return round(time.time() - self.started_at, 1)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "profile": self.profile,
            "log": self.log,
            "port_conflict": self.port_conflict.to_dict() if self.port_conflict else None,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id            TEXT PRIMARY KEY,
                status        TEXT NOT NULL,
                started_at    REAL NOT NULL,
                finished_at   REAL,
                ok            INTEGER,
                profile       TEXT NOT NULL DEFAULT 'annotation',
                log_json      TEXT NOT NULL DEFAULT '[]',
                port_conflict TEXT,
                updated_at    REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_started_at ON tasks(started_at DESC)
        """)
        conn.commit()


def _insert_task(db_path: Path, task: TaskRecord) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, status, started_at, finished_at, ok, profile, log_json, port_conflict, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id,
                task.status,
                task.started_at,
                task.finished_at,
                None if task.ok is None else int(task.ok),
                task.profile,
                json.dumps(task.log),
                json.dumps(task.port_conflict.to_dict()) if task.port_conflict else None,
                time.time(),
            ),
        )
        conn.commit()


def _update_task(db_path: Path, task: TaskRecord) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE tasks SET
               status=?, finished_at=?, ok=?, log_json=?, port_conflict=?, updated_at=?
               WHERE id=?""",
            (
                task.status,
                task.finished_at,
                None if task.ok is None else int(task.ok),
                json.dumps(task.log),
                json.dumps(task.port_conflict.to_dict()) if task.port_conflict else None,
                time.time(),
                task.id,
            ),
        )
        conn.commit()


def _load_recent(db_path: Path, max_age_hours: int) -> list[TaskRecord]:
    cutoff = time.time() - max_age_hours * 3600
    records: list[TaskRecord] = []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE started_at > ? ORDER BY started_at DESC LIMIT 100",
            (cutoff,),
        ).fetchall()
    for row in rows:
        pc_raw = row["port_conflict"]
        pc = PortConflictData.from_dict(json.loads(pc_raw)) if pc_raw else None
        ok_raw = row["ok"]
        task = TaskRecord(
            id=row["id"],
            profile=row["profile"],
            status=row["status"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            ok=None if ok_raw is None else bool(ok_raw),
            log=json.loads(row["log_json"]),
            port_conflict=pc,
        )
        records.append(task)
    return records


def _expire_old(db_path: Path, max_age_hours: int) -> int:
    cutoff = time.time() - max_age_hours * 3600
    with _connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM tasks WHERE started_at < ? AND status IN ('done','error','port_conflict')",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount


# ── TaskStore ──────────────────────────────────────────────────────────────────

class TaskStore:
    """
    Thread-safe, SQLite-backed task store.

    Usage
    -----
        store = TaskStore(db_path)
        await store.initialize()          # load history, run expiry
        task = store.create("annotation") # sync — called from async context
        task.log.append("starting…")
        await store.flush(task)           # async — dispatches DB write to thread pool
        task.status = "done"
        task.finished_at = time.time()
        await store.flush(task)
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._cache: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        """Load existing tasks from SQLite into memory, expire old ones."""
        db_path = self._db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_ensure_schema, db_path)
        expired = await asyncio.to_thread(_expire_old, db_path, MAX_AGE_HOURS)
        if expired:
            logger.info("Task store: expired old tasks", count=expired)
        records = await asyncio.to_thread(_load_recent, db_path, MAX_AGE_HOURS)
        with self._lock:
            for rec in records:
                self._cache[rec.id] = rec
        logger.info("Task store initialized", loaded=len(records), db=str(db_path))

    def create(self, profile: str, task_id: str | None = None) -> TaskRecord:
        """Create a new task record and persist it synchronously (fast INSERT)."""
        import uuid as _uuid
        tid = task_id or str(_uuid.uuid4())
        task = TaskRecord(id=tid, profile=profile)
        with self._lock:
            self._cache[tid] = task
        # Fire-and-forget insert — OK to be slightly async
        _insert_task(self._db_path, task)
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._cache.get(task_id)

    def list_recent(self, n: int = 20) -> list[TaskRecord]:
        with self._lock:
            tasks = sorted(self._cache.values(), key=lambda t: t.started_at, reverse=True)
        return tasks[:n]

    async def flush(self, task: TaskRecord) -> None:
        """Persist current task state to SQLite (runs in thread pool)."""
        await asyncio.to_thread(_update_task, self._db_path, task)
        task._last_flush_line = len(task.log)

    def should_flush_log(self, task: TaskRecord) -> bool:
        """Return True when enough new log lines have accumulated to warrant a DB write."""
        return len(task.log) - task._last_flush_line >= LOG_FLUSH_EVERY

    async def expire(self) -> int:
        """Delete old completed tasks from DB and memory. Returns deleted count."""
        deleted = await asyncio.to_thread(_expire_old, self._db_path, MAX_AGE_HOURS)
        cutoff = time.time() - MAX_AGE_HOURS * 3600
        with self._lock:
            expired_ids = [
                tid for tid, t in self._cache.items()
                if t.started_at < cutoff and t.status in ("done", "error", "port_conflict")
            ]
            for tid in expired_ids:
                del self._cache[tid]
        if deleted:
            logger.info("Task store: expired tasks", count=deleted)
        return deleted


# ── Module-level singleton ────────────────────────────────────────────────────

_store: TaskStore | None = None
_store_lock = threading.Lock()


def get_task_store(db_path: Path | None = None) -> TaskStore:
    """
    Return the module-level TaskStore singleton.

    The first call must supply db_path (done in containers.py at startup).
    Subsequent calls return the cached instance.
    """
    global _store
    with _store_lock:
        if _store is None:
            if db_path is None:
                raise RuntimeError("TaskStore not yet initialized — pass db_path on first call")
            _store = TaskStore(db_path)
        return _store
