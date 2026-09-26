"""
TaskStore — SQLite persistence layer.

Schema:
  tasks  — canonical task records
  logs   — append-only execution log entries
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

from tasks.task import Task, TaskStatus
from utils.logger import get_logger
from utils.helpers import utcnow

log = get_logger(__name__)


class TaskStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._init_schema()
        log.info("TaskStore opened: %s", db_path)

    # ── Connection ────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id           TEXT PRIMARY KEY,
                    description  TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    priority     INTEGER NOT NULL DEFAULT 1,
                    parent_id    TEXT,
                    created_at   TEXT,
                    started_at   TEXT,
                    completed_at TEXT,
                    retries      INTEGER NOT NULL DEFAULT 0,
                    result       TEXT,
                    error        TEXT,
                    metadata     TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id    TEXT NOT NULL,
                    event      TEXT NOT NULL,
                    detail     TEXT,
                    created_at TEXT NOT NULL
                );
            """)

    # ── Task CRUD ─────────────────────────────────────────────────────────

    def save_task(self, task: Task):
        d = task.to_dict()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO tasks
                    (id, description, status, priority, parent_id,
                     created_at, started_at, completed_at, retries,
                     result, error, metadata)
                VALUES
                    (:id, :description, :status, :priority, :parent_id,
                     :created_at, :started_at, :completed_at, :retries,
                     :result, :error, :metadata)
                ON CONFLICT(id) DO UPDATE SET
                    description  = excluded.description,
                    status       = excluded.status,
                    priority     = excluded.priority,
                    started_at   = excluded.started_at,
                    completed_at = excluded.completed_at,
                    retries      = excluded.retries,
                    result       = excluded.result,
                    error        = excluded.error,
                    metadata     = excluded.metadata
            """, d)

    def update_task(self, task: Task):
        self.save_task(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return Task.from_dict(dict(row)) if row else None

    def get_tasks_by_status(
        self,
        statuses: list[TaskStatus],
        limit: int = 100,
    ) -> list[Task]:
        placeholders = ",".join("?" * len(statuses))
        vals = [s.value for s in statuses]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE status IN ({placeholders}) "
                f"ORDER BY priority ASC, created_at ASC LIMIT ?",
                vals + [limit],
            ).fetchall()
        return [Task.from_dict(dict(r)) for r in rows]

    def get_all_tasks(self, limit: int = 200) -> list[Task]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at ASC LIMIT ?", (limit,)
            ).fetchall()
        return [Task.from_dict(dict(r)) for r in rows]

    def clear_all_tasks(self):
        """Wipe all task rows and event history for a clean session."""
        with self._conn() as conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM logs")

    # ── Logging ───────────────────────────────────────────────────────────

    def log_event(self, task_id: str, event: str, detail: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO logs (task_id, event, detail, created_at) VALUES (?,?,?,?)",
                (task_id, event, detail, utcnow()),
            )

    def get_logs(self, task_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM logs WHERE task_id = ? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        return [dict(r) for r in rows]