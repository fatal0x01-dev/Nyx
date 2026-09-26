"""
Core Task dataclass and TaskStatus enum.
This is the single definition used everywhere — DB, manager, planner, executor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1                          # lower number = higher priority
    parent_id: Optional[str] = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retries: int = 0
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["metadata"] = json.dumps(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        raw_meta = d.get("metadata", "{}")
        if isinstance(raw_meta, str):
            try:
                d["metadata"] = json.loads(raw_meta)
            except json.JSONDecodeError:
                d["metadata"] = {}
        return cls(**d)

    def summary(self) -> str:
        return (
            f"Task({self.id}) [{self.status.value.upper()}] "
            f"priority={self.priority} retries={self.retries} | {self.description[:80]}"
        )