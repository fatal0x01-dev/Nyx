"""
TaskManager — in-memory task queue backed by SQLite via TaskStore.

Responsibilities:
- Add / update tasks
- Return the next task to run (by priority)
- Track pending / running / done counts
"""

from __future__ import annotations

from typing import Optional

import config.settings as cfg
from tasks.task import Task, TaskStatus
from memory.task_store import TaskStore
from utils.logger import get_logger
from utils.helpers import utcnow

log = get_logger(__name__)


class TaskManager:
    def __init__(self, store: TaskStore):
        self._store = store
        # In-memory queue mirrors the DB for fast access during a session
        self._queue: list[Task] = []
        self._load_pending()

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_pending(self):
        """Load previous-session tasks only when resume is explicitly enabled."""
        if not getattr(cfg, "RESUME_PREVIOUS_SESSION", False):
            self._store.clear_all_tasks()
            self._queue = []
            log.info("Previous-session task resume disabled; cleared stale pending/running tasks.")
            return

        tasks = self._store.get_tasks_by_status(
            [TaskStatus.PENDING, TaskStatus.RUNNING]
        )
        # Reset RUNNING → PENDING (crashed mid-execution)
        for t in tasks:
            if t.status == TaskStatus.RUNNING:
                log.warning("Task %s was RUNNING at shutdown — resetting to PENDING", t.id)
                t.status = TaskStatus.PENDING
                self._store.update_task(t)
        self._queue = sorted(tasks, key=lambda t: (t.priority, t.created_at))
        log.info("Loaded %d pending task(s) from previous session", len(self._queue))

    # ── Public API ────────────────────────────────────────────────────────

    def add_task(self, task: Task) -> Task:
        """Persist and enqueue a new task."""
        if not task.created_at:
            task.created_at = utcnow()
        self._store.save_task(task)
        self._queue.append(task)
        self._queue.sort(key=lambda t: (t.priority, t.created_at))
        log.info("Queued  %s", task.summary())
        return task

    def add_tasks(self, tasks: list[Task]) -> list[Task]:
        return [self.add_task(t) for t in tasks]

    def next_task(self) -> Optional[Task]:
        """Return the highest-priority pending task, or None if queue is empty."""
        for task in self._queue:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def mark_running(self, task: Task) -> Task:
        task.status = TaskStatus.RUNNING
        task.started_at = utcnow()
        self._store.update_task(task)
        log.info("Running %s", task.summary())
        return task

    def mark_completed(self, task: Task, result: str) -> Task:
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = utcnow()
        self._store.update_task(task)
        self._remove_from_queue(task.id)
        log.info("Done    %s", task.summary())
        return task

    def mark_failed(self, task: Task, error: str) -> Task:
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = utcnow()
        self._store.update_task(task)
        self._remove_from_queue(task.id)
        log.warning("Failed  %s | %s", task.summary(), error[:120])
        return task

    def mark_cancelled(self, task: Task, reason: str) -> Task:
        """Remove a pending task that is no longer relevant to the confirmed goal state."""
        task.status = TaskStatus.CANCELLED
        task.error = reason
        task.completed_at = utcnow()
        self._store.update_task(task)
        self._remove_from_queue(task.id)
        log.info("Cancelled %s | %s", task.summary(), reason[:120])
        return task

    def mark_retry(self, task: Task, error: str) -> Task:
        """Retry at the planner-assigned priority so prerequisites aren't bypassed."""
        task.retries += 1
        task.status = TaskStatus.PENDING
        task.error = error
        self._store.update_task(task)
        self._queue.sort(key=lambda t: (t.priority, t.created_at))
        log.warning("Retry   %s (attempt %d)", task.summary(), task.retries)
        return task

    def has_pending(self) -> bool:
        return any(t.status == TaskStatus.PENDING for t in self._queue)

    def pending_count(self) -> int:
        return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)

    def queue_snapshot(self) -> list[Task]:
        return list(self._queue)

    # ── Internal ──────────────────────────────────────────────────────────

    def _remove_from_queue(self, task_id: str):
        self._queue = [t for t in self._queue if t.id != task_id]

    def clear_all(self):
        """Remove all tasks from the queue and DB."""
        self._store.clear_all_tasks()
        self._queue = []
        log.warning("Cleared all queued tasks and persisted task state.")
