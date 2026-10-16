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

import argparse
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
from utils.ssh import SSHConfig
from utils.tmux_client import TmuxError
from utils.control_plane import ControlPlane
from utils.html_report import write_html_report

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
        self._remote_context: str = ""
        self._recent_terminal_context: list[str] = []
        self._safety_halt_reason: str = ""
        self._control = ControlPlane(cfg.CONTROL_DB_PATH)
        self._run_id: str = ""

    # ── Entry point ───────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        ssh_config: SSHConfig | None = None,
        auto_mode: bool = False,
        sudo_password: str | None = None,
    ):
        """Start the orchestration loop for a given goal."""
        self._goal = goal
        self._iteration = 0
        self._start_time = time.time()
        self._recent_terminal_context = []
        self._safety_halt_reason = ""
        self._run_id = new_id()
        self._control.start_run(self._run_id, goal)
        self._executor.configure_control_plane(self._control, self._run_id)
        self._executor.configure_input_modes(auto_mode, sudo_password)
        if auto_mode:
            log.warning(
                "AUTONOMOUS MODE ENABLED: Nyx will automatically accept detected prompts."
            )
        if sudo_password is not None:
            log.info("Automatic sudo password entry is enabled.")
        self._publish_report()

        log.info("=" * 60)
        log.info("ORCHESTRATOR START")
        log.info("Goal: %s", goal)
        reasoning.info("User goal: %s", goal)
        log.info("=" * 60)

        self._preflight_checks()

        if ssh_config is not None:
            try:
                self._control.update_run(self._run_id, phase="Connecting to SSH host")
                self._remote_context = self._executor.connect_ssh(ssh_config)
            except TmuxError as e:
                log.error("Could not establish requested SSH session: %s", e)
                self._finish(success=False, message=f"SSH connection failed: {e}")
                return

        try:
            self._loop()
        except KeyboardInterrupt:
            self._control.update_run(
                self._run_id,
                phase="Interrupted",
                status="interrupted",
            )
            self._publish_report()
            log.warning("Interrupted by user — state saved, resume any time.")
            sys.exit(0)
        except Exception:
            self._control.update_run(
                self._run_id,
                phase="Stopped after an unexpected error",
                status="failed",
            )
            self._publish_report()
            raise

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
            guidance_text = self._checkpoint("Preparing planning cycle")
            self._iteration += 1
            self._control.update_run(
                self._run_id,
                phase="Planning",
                iteration=self._iteration,
                current_task="",
            )
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
                    remote_context=self._remote_context,
                    guidance_text=guidance_text,
                    terminal_context=self._terminal_context_text(),
                )
            except RuntimeError as e:
                log.error("Planner error: %s", e)
                log.info("Waiting 10s before retry...")
                for _ in range(40):
                    self._checkpoint("Waiting to retry planning")
                    time.sleep(0.25)
                continue

            self._checkpoint("Reviewing plan")
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
            self._control.update_run(
                self._run_id,
                phase="Executing task",
                iteration=self._iteration,
                current_task=task.description,
            )
            self._run_task(task)
            safety_halt_reason = getattr(self, "_safety_halt_reason", "")
            if safety_halt_reason:
                self._finish(success=False, message=safety_halt_reason)
                return
            if self._connectivity_goal_satisfied(
                self._goal,
                getattr(self._executor, "_last_step_log", []),
            ):
                self._cancel_pending_tasks(
                    "Superseded: a successful ping confirmed the requested network connectivity."
                )
                message = "Network connectivity was confirmed by a successful ping."
                reasoning.info(message)
                self._finish(success=True, message=message)
                return

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

    @staticmethod
    def _connectivity_goal_satisfied(goal: str, step_log: list[dict]) -> bool:
        """Treat a successful ping as sufficient for a basic connectivity-only goal."""
        goal_lower = (goal or "").lower()
        connectivity_terms = (
            "connectivity",
            "network connection",
            "internet connection",
            "internet access",
        )
        additional_work_terms = (
            "diagnos",
            "troubleshoot",
            "repair",
            "fix ",
            "configure",
            "restart",
            "firewall",
            " dns",
        )
        if not any(term in goal_lower for term in connectivity_terms):
            return False
        if any(term in goal_lower for term in additional_work_terms):
            return False

        return Executor._has_successful_ping(step_log)

    def _cancel_pending_tasks(self, reason: str) -> None:
        """Cancel pending tasks that are no longer needed after goal completion."""
        for task in self._manager.queue_snapshot():
            if task.status != TaskStatus.PENDING:
                continue
            self._manager.mark_cancelled(task, reason)
            self._store.log_event(task.id, "cancelled", reason)
            reasoning.info("Cancelled task %s: %s", task.id, reason)

    def _checkpoint(self, phase: str) -> str:
        """Pause only between model or shell operations, then return current guidance."""
        self._control.update_run(
            self._run_id,
            phase=phase,
            iteration=self._iteration,
        )
        self._publish_report()
        guidance = self._control.wait_if_paused(self._run_id)
        return guidance

    def _publish_report(self) -> None:
        write_html_report(
            cfg.HTML_REPORT_PATH,
            self._control.snapshot(self._run_id or None),
            self._store.get_all_tasks(),
        )

    def _terminal_context_text(self) -> str:
        return "\n\n".join(self._recent_terminal_context[-8:])[-12000:]

    @staticmethod
    def _format_terminal_evidence(task: Task, step_log: list[dict]) -> str:
        if not step_log:
            return ""
        lines = [f"TASK {task.id}: {task.description}"]
        for step in step_log:
            if not isinstance(step, dict):
                continue
            command = str(step.get("command", "")).strip()
            output = str(step.get("output", "")).strip()
            if not command and not output:
                continue
            lines.append(f"$ {command or '(no command)'}")
            lines.append(
                f"exit_code={step.get('exit_code')} timed_out={step.get('timed_out', False)}"
            )
            if output:
                lines.append(output[-1600:])
        return "\n".join(lines)

    # ── Task execution ────────────────────────────────────────────────────

    def _validate_task_completion(self, task: Task, result: dict) -> bool:
        """Ask the planner to validate whether this task result is actually sufficient."""
        if not isinstance(result, dict):
            return False

        try:
            guidance_text = self._checkpoint("Validating task")
            validation = self._planner.validate_task_result(
                goal=self._goal,
                task_description=task.description,
                result=result.get("result", ""),
                error=result.get("error", ""),
                history_text=self._history.as_prompt_text(limit=20),
                remote_context=self._remote_context,
                guidance_text=guidance_text,
                terminal_context=self._terminal_context_text(),
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
        self._control.update_run(
            self._run_id,
            phase="Starting task",
            iteration=self._iteration,
            current_task=task.description,
        )
        self._manager.mark_running(task)
        self._store.log_event(task.id, "started")
        self._publish_report()

        try:
            result = self._executor.execute(task)
        except Exception as e:
            log.exception("Unexpected executor error for task %s", task.id)
            result = {"status": "failed", "error": str(e), "result": ""}
        if result.get("safety_halt"):
            self._safety_halt_reason = (
                "Nyx could not confirm that the unsafe terminal prompt was canceled. "
                "The orchestrator stopped without sending further commands; inspect the terminal manually."
            )
        elif result.get("safety_blocked"):
            self._safety_halt_reason = (
                "Nyx blocked or canceled an unsafe operation and stopped the run without retrying it. "
                "Review the goal and terminal before starting another run."
            )

        status = result.get("status", "failed")
        result_text = result.get("result", "")
        error_text = result.get("error", "")

        last_exit_code = None
        if isinstance(result, dict):
            last_exit_code = result.get("exit_code")

        if status == "completed" and not self._executor.validate_completion_status(
            status=status,
            result=result_text,
            error=error_text,
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
            if task.retries < cfg.MAX_TASK_RETRIES and not result.get("safety_blocked"):
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

        terminal_evidence = self._format_terminal_evidence(
            task,
            getattr(self._executor, "_last_step_log", []),
        )
        outcome = (
            f"TASK OUTCOME {task.id}: status={status}\n"
            f"task={task.description}\n"
            f"result={result_text or '(none)'}\n"
            f"error={error_text or '(none)'}"
        )
        task_context = "\n".join(part for part in (terminal_evidence, outcome) if part)
        self._recent_terminal_context.append(task_context)
        self._recent_terminal_context = self._recent_terminal_context[-8:]
        task.metadata["terminal_evidence"] = task_context
        self._publish_report()

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
            self._publish_report()

    def _finish(self, success: bool, message: str):
        elapsed = time.time() - self._start_time
        status = "COMPLETE" if success else "FAILED"
        log.info("=" * 60)
        log.info("GOAL %s after %d iterations (%.1fs)", status, self._iteration, elapsed)
        if message:
            log.info("Message: %s", message)
        log.info("=" * 60)
        if self._run_id:
            self._control.update_run(
                self._run_id,
                phase=status.title(),
                status="completed" if success else "failed",
                current_task="",
            )
            self._publish_report()

def parse_cli_args(
    argv: list[str] | None = None,
) -> tuple[str, SSHConfig | None, bool, str | None]:
    """Parse the goal and optional SSH connection parameters."""
    parser = argparse.ArgumentParser(description="Run Nyx against a local or SSH terminal.")
    parser.add_argument("--ssh", action="store_true", help="Connect to the IP supplied with --ip before planning.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Automatically accept detected terminal and executor approval prompts, "
            "including destructive confirmations."
        ),
    )
    parser.add_argument(
        "--password",
        metavar="PASSWORD",
        help=(
            "Automatically enter PASSWORD at sudo password prompts. "
            "Warning: command-line values may be visible in shell history and process listings."
        ),
    )
    parser.add_argument("--ip", help="Required IP address for the SSH destination.")
    parser.add_argument("--ssh-user", help=f"SSH username (default: {cfg.SSH_DEFAULT_USER})")
    parser.add_argument("--ssh-port", type=int, help="SSH port")
    parser.add_argument("--ssh-identity-file", help="SSH private key file")
    parser.add_argument(
        "--ssh-option",
        action="append",
        help="SSH -o option; when provided, replaces the default legacy-RSA compatibility options (repeatable)",
    )
    parser.add_argument("goal", nargs="*", help="High-level task goal")
    args = parser.parse_args(argv)

    supplied_ssh_options = (
        args.ip is not None
        or args.ssh_user is not None
        or args.ssh_port is not None
        or args.ssh_identity_file is not None
        or args.ssh_option is not None
    )
    if supplied_ssh_options and not args.ssh:
        parser.error("SSH connection parameters require --ssh")

    ssh_config = None
    if args.ssh:
        if not args.ip:
            parser.error("--ip is required when --ssh is supplied")
        try:
            ssh_config = SSHConfig(
                ip=args.ip,
                username=args.ssh_user or cfg.SSH_DEFAULT_USER,
                port=args.ssh_port,
                identity_file=args.ssh_identity_file,
                options=tuple(args.ssh_option) if args.ssh_option is not None else cfg.SSH_DEFAULT_OPTIONS,
            )
        except ValueError as e:
            parser.error(str(e))

    return " ".join(args.goal).strip(), ssh_config, args.auto, args.password


def main():
    """CLI entry point."""
    goal, ssh_config, auto_mode, sudo_password = parse_cli_args()
    if not goal:
        goal = input("Enter goal: ").strip()

    if not goal:
        print("No goal provided.")
        sys.exit(1)

    orchestrator = Orchestrator()
    orchestrator.run(
        goal,
        ssh_config=ssh_config,
        auto_mode=auto_mode,
        sudo_password=sudo_password,
    )


if __name__ == "__main__":
    main()