"""
History — convenience wrapper for reading completed task history.
Used by the Orchestrator to build planner context.
"""

from __future__ import annotations

from memory.task_store import TaskStore
from tasks.task import Task, TaskStatus
from utils.helpers import format_history_for_prompt


class History:
    def __init__(self, store: TaskStore):
        self._store = store

    def completed_tasks(self, limit: int = 20) -> list[Task]:
        return self._store.get_tasks_by_status(
            [TaskStatus.COMPLETED, TaskStatus.FAILED], limit=limit
        )

    def as_prompt_text(self, limit: int = 20) -> str:
        tasks = self.completed_tasks(limit=limit)
        return format_history_for_prompt(
            [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "result": t.result or t.error or "",
                }
                for t in tasks
            ]
        )