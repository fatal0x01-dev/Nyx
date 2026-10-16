"""SQLite-backed controls shared by the orchestrator and its frontend."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


class ControlPlane:
    def __init__(self, database_path: str):
        self._database_path = database_path
        os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS control_runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    paused INTEGER NOT NULL DEFAULT 0,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    current_task TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS control_guidance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES control_runs(run_id),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_run(self, run_id: str, goal: str) -> None:
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO control_runs (
                    run_id, goal, status, phase, paused, iteration, current_task, created_at, updated_at
                ) VALUES (?, ?, 'running', 'Starting', 0, 0, '', ?, ?)
                """,
                (run_id, goal, now, now),
            )

    def update_run(
        self,
        run_id: str,
        *,
        phase: str | None = None,
        iteration: int | None = None,
        current_task: str | None = None,
        status: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        for key, value in (
            ("phase", phase),
            ("iteration", iteration),
            ("current_task", current_task),
            ("status", status),
        ):
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.extend((self._now(), run_id))
        with self._connection() as connection:
            connection.execute(
                f"UPDATE control_runs SET {', '.join(fields)} WHERE run_id = ?",
                values,
            )

    def set_paused(self, run_id: str, paused: bool) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE control_runs
                SET paused = ?, status = ?, phase = ?, updated_at = ?
                WHERE run_id = ? AND status IN ('running', 'paused')
                """,
                (
                    int(paused),
                    "paused" if paused else "running",
                    "Paused" if paused else "Resuming",
                    self._now(),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("The orchestrator run is no longer active")

    def add_guidance(self, run_id: str, content: str) -> None:
        guidance = content.strip()
        if not guidance:
            raise ValueError("Guidance cannot be empty")
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM control_runs WHERE run_id = ? AND status IN ('running', 'paused')",
                (run_id,),
            ).fetchone()
            if exists is None:
                raise ValueError("The orchestrator run is no longer active")
            connection.execute(
                "INSERT INTO control_guidance (run_id, content, created_at) VALUES (?, ?, ?)",
                (run_id, guidance, self._now()),
            )
            connection.execute(
                "UPDATE control_runs SET updated_at = ? WHERE run_id = ?",
                (self._now(), run_id),
            )

    def guidance_text(self, run_id: str) -> str:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT content FROM control_guidance
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return "\n".join(f"- {row['content']}" for row in rows)

    def snapshot(self, run_id: str | None = None) -> dict[str, Any] | None:
        with self._connection() as connection:
            if run_id is None:
                row = connection.execute(
                    "SELECT * FROM control_runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM control_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            if row is None:
                return None
            state = dict(row)
            guidance = connection.execute(
                """
                SELECT content, created_at FROM control_guidance
                WHERE run_id = ? ORDER BY id DESC LIMIT 10
                """,
                (row["run_id"],),
            ).fetchall()
        state["paused"] = bool(state["paused"])
        state["guidance"] = [dict(item) for item in reversed(guidance)]
        return state

    def wait_if_paused(self, run_id: str, poll_interval: float = 0.25) -> str:
        """Wait at a safe checkpoint until resumed, returning current guidance."""
        while True:
            state = self.snapshot(run_id)
            if state is None or state["status"] not in {"running", "paused"}:
                return self.guidance_text(run_id)
            if not state["paused"]:
                return self.guidance_text(run_id)
            time.sleep(poll_interval)
