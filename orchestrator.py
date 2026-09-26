"""
Orchestrator — central control loop.

Owns:
  - Global objective state
  - Task queue (via TaskManager)
  - Persistent history (via TaskStore)
  - Planner and Executor clients

Loop:
  1. Ask Planner what to do next
  2. Queue any new tasks
  3. Pull next pending task
  4. Send to Executor
  5. Save result
  6. Repeat until Planner signals goal_complete / goal_failed
     or the iteration cap is hit
"""

from __future__ import annotations

import os
import sys
import time

import config.settings as cfg
from memory.task_store import TaskStore
from memory.rag_store import RAGStore
from tasks.task import Task, TaskStatus
from tasks.task_manager import TaskManager
from tasks.history import History
from planner.planner import Planner
from executor.executor import Executor
from utils.logger import get_logger, get_reasoning_logger
from utils.helpers import new_id, utcnow

log = get_logger(__name__)
reasoning = get_reasoning_logger(__name__)


class Orchestrator:
    def __init__(self):
        # ── Persistence ───────────────────────────────────────────────────
        self._store = TaskStore(cfg.TASK_DB_PATH)
        self._rag = RAGStore()                  # stub in Phase 1

        # ── Task management ───────────────────────────────────────────────
        self._manager = TaskManager(self._store)
        self._history = History(self._store)

        # ── AI clients ────────────────────────────────────────────────────
        self._planner = Planner()
        self._executor = Executor()

        # ── Runtime state ─────────────────────────────────────────────────
        self._goal: str = ""
        self._iteration: int = 0
        self._start_time: float = 0.0

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self, goal: str):
        """Start the orchestration loop for a given goal."""
        self._goal = goal
        self._iteration = 0
        self._start_time = time.time()

        log.info("=" * 60)
        log.info("ORCHESTRATOR START")
        log.info("Goal: %s", goal)
        reasoning.info("User goal: %s", goal)
        log.info("=" * 60)

        self._preflight_checks()

        try:
            self._loop()
        except KeyboardInterrupt:
            log.warning("Interrupted by user — state saved, resume any time.")
            sys.exit(0)

    # ── Pre-flight ────────────────────────────────────────────────────────

    def _preflight_checks(self):
        from utils.tmux_client import TmuxClient

        log.info("Checking executor availability...")
        if not self._executor.is_available():
            log.warning(
                "Executor LM Studio not reachable at %s — "
                "make sure it's running before tasks are dispatched.",
                cfg.EXECUTOR_API_URL,
            )
        else:
            log.info("Executor model: OK")

        if not TmuxClient.is_installed():
            log.warning(
                "tmux is not installed — the executor cannot attach to or "
                "drive a terminal session until it is. Install with your "
                "package manager (e.g. `apt install tmux`)."
            )
        elif os.environ.get("TMUX"):
            log.info("Running inside tmux — executor will join THIS session.")
        else:
            log.info(
                "Not running inside tmux — executor will drive a detached "
                "session named '%s'. Run `tmux attach -t %s` to watch/join.",
                cfg.TMUX_SESSION_NAME,
                cfg.TMUX_SESSION_NAME,
            )

        resumed = self._manager.pending_count()
        if resumed:
            log.info("Resuming session — %d pending task(s) in queue", resumed)

    # ── Main loop ─────────────────────────────────────────────────────────

    @staticmethod
    def _completion_is_valid(plan: dict, pending_tasks: list[Task]) -> bool:
        """Only accept completion when no pending work remains and a summary is present."""
        if plan.get("action") != "goal_complete":
            return False
        if pending_tasks:
            return False
        summary = (plan.get("summary") or "").strip()
        return bool(summary)

    def _loop(self):
        while self._iteration < cfg.MAX_ITERATIONS:
            self._iteration += 1
            log.info("-" * 50)
            log.info("Iteration %d / %d", self._iteration, cfg.MAX_ITERATIONS)

            # ── Ask the Planner ───────────────────────────────────────────
            history_text = self._history.as_prompt_text(limit=20)
            pending_tasks = [
                t for t in self._manager.queue_snapshot()
                if t.status == TaskStatus.PENDING
            ]

            try:
                plan = self._planner.plan(
                    goal=self._goal,
                    history_text=history_text,
                    pending_tasks=pending_tasks,
                )
            except RuntimeError as e:
                log.error("Planner error: %s", e)
                log.info("Waiting 10s before retry...")
                time.sleep(10)
                continue

            action = plan.get("action")

            # ── Handle planner response ───────────────────────────────────
            if action == "goal_complete":
                self._retire_stale_retries(pending_tasks)
                pending_tasks = [
                    t for t in self._manager.queue_snapshot()
                    if t.status == TaskStatus.PENDING
                ]
                if not self._completion_is_valid(plan, pending_tasks):
                    reasoning.warning(
                        "Planner requested goal completion while %d task(s) are still pending; ignoring stale completion signal.",
                        len(pending_tasks),
                    )
                    log.warning(
                        "Ignoring goal_complete signal because %d pending task(s) remain.",
                        len(pending_tasks),
                    )
                    time.sleep(2)
                    continue

                reasoning.info("Planner completed the objective: %s", plan.get("summary", "No summary provided."))
                self._finish(success=True, message=plan.get("summary", ""))
                return

            if action == "goal_failed":
                reasoning.warning("Planner reported failure: %s", plan.get("reason", "Unknown planner failure."))
                self._finish(success=False, message=plan.get("reason", ""))
                return

            if action == "create_tasks":
                reasoning.info("Planner created %d task(s).", len(plan.get("tasks", [])))
                self._enqueue_planner_tasks(plan.get("tasks", []))

            else:
                reasoning.warning("Planner returned an unknown action: %s", action)
                log.warning("Unknown planner action '%s' — skipping", action)

            # ── Execute next pending task ─────────────────────────────────
            task = self._manager.next_task()
            if task is None:
                reasoning.info("No pending tasks yet; waiting for the planner to produce the next step.")
                log.info("No pending tasks — waiting for planner to create more...")
                time.sleep(2)
                continue

            reasoning.info("Executing task: %s", task.description)
            self._run_task(task)

        # ── Iteration cap hit ─────────────────────────────────────────────
        log.error("Reached maximum iterations (%d) without completion.", cfg.MAX_ITERATIONS)
        self._finish(
            success=False,
            message=f"Stopped after {cfg.MAX_ITERATIONS} iterations without goal_complete signal.",
        )

    def _retire_stale_retries(self, pending_tasks: list[Task]):
        """Cancel retried leftovers when the planner has confirmed the overall goal."""
        for task in pending_tasks:
            if task.retries <= 0:
                continue
            reason = "Superseded by planner-confirmed goal completion."
            self._manager.mark_cancelled(task, reason)
            self._store.log_event(task.id, "cancelled", reason)
            reasoning.info("Retired stale retry task %s after planner confirmed goal completion.", task.id)

    # ── Task execution ────────────────────────────────────────────────────

    def _validate_task_completion(self, task: Task, result: dict) -> bool:
        """Ask the planner to validate whether this task result is actually sufficient."""
        if not isinstance(result, dict):
            return False

        try:
            validation = self._planner.validate_task_result(
                goal=self._goal,
                task_description=task.description,
                result=result.get("result", ""),
                error=result.get("error", ""),
                history_text=self._history.as_prompt_text(limit=20),
            )
        except RuntimeError as e:
            log.warning("Planner validation rejected task %s because validation call failed: %s", task.id, e)
            return False

        if validation.get("action") == "rejected":
            reasoning.warning("Planner rejected task %s: %s", task.id, validation.get("reason", "unknown reason"))
            return False

        if validation.get("action") == "accepted":
            reasoning.info("Planner accepted task %s as complete: %s", task.id, validation.get("summary", "validated"))
            return True

        reasoning.warning("Planner validation for task %s returned an invalid action; rejecting as untrusted.", task.id)
        return False

    def _run_task(self, task: Task):
        self._manager.mark_running(task)
        self._store.log_event(task.id, "started")

        try:
            result = self._executor.execute(task)
        except Exception as e:
            log.exception("Unexpected executor error for task %s", task.id)
            result = {"status": "failed", "error": str(e), "result": ""}

        status = result.get("status", "failed")
        result_text = result.get("result", "")
        error_text = result.get("error", "")

        terminal_history = self._executor._tmux.capture_history(
            self._executor._tmux.get_target()[0],
            lines=cfg.TERMINAL_HISTORY_LINES,
        ) if self._executor else ""
        last_exit_code = None
        if isinstance(result, dict):
            last_exit_code = result.get("exit_code")

        if status == "completed" and not self._executor.validate_completion_status(
            status=status,
            result=result_text,
            error=error_text,
            terminal_history=terminal_history,
            last_exit_code=last_exit_code,
            step_log=getattr(self._executor, "_last_step_log", []),
            task_description=task.description,
        ):
            log.warning(
                "Task %s reported completed but terminal evidence was not sufficient; treating as failed.",
                task.id,
            )
            status = "failed"
            error_text = result_text or "Task completed signal was not validated by terminal evidence."
            result_text = ""

        if status == "completed":
            if not self._validate_task_completion(task, result):
                log.warning(
                    "Task %s was completed by the executor, but rejected by planner validation; retrying or failing the task.",
                    task.id,
                )
                status = "failed"
                error_text = result_text or "Task completion was not validated by the planner."
                result_text = ""

        if status == "completed":
            reasoning.info("Task %s succeeded: %s", task.id, result_text[:200] if result_text else "completed without additional output")
            self._manager.mark_completed(task, result_text)
            self._store.log_event(task.id, "completed", result_text[:500])
        else:
            reasoning.warning("Task %s failed or needs retry: %s", task.id, error_text[:200] if error_text else result_text[:200] if result_text else "unknown failure")
            if task.retries < cfg.MAX_TASK_RETRIES:
                self._manager.mark_retry(task, error_text)
                self._store.log_event(task.id, "retry", error_text[:300])
            else:
                self._manager.mark_failed(task, error_text)
                self._store.log_event(task.id, "failed", error_text[:300])
                log.error(
                    "Task %s exhausted %d retries — marked FAILED",
                    task.id,
                    cfg.MAX_TASK_RETRIES,
                )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _enqueue_planner_tasks(self, raw_tasks: list[dict]):
        existing_descriptions = {
            t.description.strip().lower()
            for t in self._manager.queue_snapshot()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        }

        for raw in raw_tasks:
            description = (raw.get("description") or "").strip()
            if not description:
                continue
            if description.lower() in existing_descriptions:
                log.info("Skipping duplicate planner task already pending/running: %s", description)
                continue

            task = Task(
                id=raw.get("id") or new_id(),
                description=description,
                priority=int(raw.get("priority", 1)),
                created_at=utcnow(),
            )
            self._manager.add_task(task)
            existing_descriptions.add(description.lower())

    def _finish(self, success: bool, message: str):
        elapsed = time.time() - self._start_time
        status = "COMPLETE" if success else "FAILED"
        log.info("=" * 60)
        log.info("GOAL %s after %d iterations (%.1fs)", status, self._iteration, elapsed)
        if message:
            log.info("Message: %s", message)
        log.info("=" * 60)

def main():
    """CLI entry point."""
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        goal = input("Enter goal: ").strip()

    if not goal:
        print("No goal provided.")
        sys.exit(1)

    orchestrator = Orchestrator()
    orchestrator.run(goal)


if __name__ == "__main__":
    main()