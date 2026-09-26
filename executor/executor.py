"""
Executor — joins a live tmux terminal session and drives it directly,
one shell command at a time, using terminal scrollback as context.

Replaces the old LM Studio MCP + mcp-docker integration: instead of
handing the task to an opaque tool-use sandbox, this attaches to the
real pane (the one you were already working in, if Nyx is launched
from inside tmux) and reasons over actual terminal history and output.
"""

from __future__ import annotations

import json
import re

import config.settings as cfg
from executor.shell_prompts import SYSTEM_PROMPT, build_step_prompt
from tasks.task import Task
from utils.helpers import parse_json_response, normalize_multiline_write_command
from utils.human_input import detect_human_input_need, request_human_input
from utils.lmstudio_client import LMStudioClient
from utils.logger import get_logger, get_reasoning_logger
from utils.tmux_client import TmuxClient, TmuxError

log = get_logger(__name__)
reasoning = get_reasoning_logger(__name__)


class Executor:
    def __init__(self):
        self._client = LMStudioClient(
            base_url=cfg.EXECUTOR_API_URL,
            model=cfg.EXECUTOR_MODEL,
            temperature=cfg.EXECUTOR_TEMPERATURE,
        )
        self._tmux = TmuxClient(session_name=cfg.TMUX_SESSION_NAME)
        self._last_step_log: list[dict] = []

        log.info(
            "Executor ready — %s model=%s (tmux-driven)",
            cfg.EXECUTOR_API_URL,
            cfg.EXECUTOR_MODEL,
        )

    @staticmethod
    def _should_stop_on_no_progress(step_log: list[dict]) -> bool:
        if len(step_log) < 2:
            return False

        last_two = step_log[-2:]
        commands = [str(step.get("command", "")).strip() for step in last_two]
        outputs = [str(step.get("output", "")).strip() for step in last_two]
        if commands[0] and commands[0] == commands[1] and outputs[0] == outputs[1]:
            return True
        return False

    @staticmethod
    def _normalize_action(action: dict, task_description: str) -> dict:
        """Accept common shell-agent aliases without weakening the action contract."""
        if not isinstance(action, dict):
            return action

        normalized = dict(action)
        kind = str(normalized.get("action", "")).strip()
        if kind in {"run", "command", "execute"}:
            normalized["action"] = "run_command"
        elif kind and kind not in {"run_command", "send_keys", "human_input", "done"}:
            if not normalized.get("command") and kind.startswith(("ping", "curl", "wget", "cat", "echo", "printf", "sed", "grep", "docker", "docker-compose")):
                normalized["action"] = "run_command"
                normalized["command"] = kind

        if Executor._is_network_check_task(task_description):
            command = str(normalized.get("command", ""))
            normalized["command"] = re.sub(r"\bping\s+-c\s+\d+", "ping -c 1", command, count=1)

        return normalized

    @staticmethod
    def _task_matches_compose_requirement(task_description: str, compose_text: str) -> bool:
        task_lower = (task_description or "").lower()
        compose_lower = (compose_text or "").lower()

        if "nginx" in task_lower:
            return "nginx" in compose_lower and "image: nginx" in compose_lower
        return True

    @staticmethod
    def _requires_file_write(task_description: str) -> bool:
        text = (task_description or "").lower()
        file_markers = ("docker-compose.yml", ".yml", ".yaml", "compose file")
        write_markers = ("write", "create", "fix", "update", "rewrite", "repair", "edit")
        return any(marker in text for marker in file_markers) and any(marker in text for marker in write_markers)

    @staticmethod
    def _is_file_write_command(command: str) -> bool:
        text = (command or "").lower()
        return (
            "cat >" in text
            or "tee " in text and ">" in text
            or "printf" in text and ">" in text
            or text.endswith(" > docker-compose.yml")
            or text.endswith(" > docker-compose.yaml")
            or "<<'eof'" in text.lower()
            or "<<eof" in text.lower()
        )

    @staticmethod
    def _has_write_step(step_log: list[dict]) -> bool:
        if not step_log:
            return False
        return any(
            isinstance(step, dict)
            and Executor._is_file_write_command(str(step.get("command", "")))
            for step in step_log
        )

    @staticmethod
    def _needs_write_before_retry(task_description: str, step_log: list[dict]) -> bool:
        if not Executor._requires_file_write(task_description):
            return False
        if Executor._has_write_step(step_log):
            return False
        if not step_log:
            return False

        last_command = str(step_log[-1].get("command", "")).lower()
        last_output = str(step_log[-1].get("output", "")).lower()
        return ("docker-compose config" in last_command or "docker-compose config" in last_output) and ("nginx" in (task_description or "").lower())

    @staticmethod
    def _is_validation_task(task_description: str) -> bool:
        text = (task_description or "").lower()
        return (
            ("validate" in text or "check" in text or "verify" in text or "inspect" in text)
            and ("docker-compose" in text or ".yml" in text or ".yaml" in text or "compose" in text)
        )

    @staticmethod
    def _is_network_check_task(task_description: str) -> bool:
        text = (task_description or "").lower()
        return "network" in text or "internet" in text or "connectivity" in text or "ping" in text

    @staticmethod
    def _has_file_read_step(step_log: list[dict]) -> bool:
        for step in step_log or []:
            if not isinstance(step, dict):
                continue
            command = str(step.get("command", "")).lower()
            if any(marker in command for marker in ("cat ", "sed -n", "head ", "tail ", "grep ", "less ", "more ", "wc -l")):
                return True
        return False

    @staticmethod
    def validate_completion_status(
        status: str,
        result: str,
        error: str,
        terminal_history: str,
        last_exit_code: int | None,
        step_log: list[dict] | None = None,
        task_description: str | None = None,
    ) -> bool:
        """Return True only when the terminal state genuinely suggests success.

        Prefer task-scoped step output when available; the full scrollback can be stale
        and contain unrelated historical failures from earlier work in the same tmux pane.
        """
        if status != "completed":
            return False

        if error and str(error).strip():
            return False

        if step_log:
            evidence_parts = [
                str(step.get("output", ""))
                for step in step_log
                if isinstance(step, dict) and step.get("output")
            ]
        else:
            evidence_parts = [terminal_history, result]

        text = "\n".join(part for part in evidence_parts if part).strip().lower()

        if task_description:
            task_lower = task_description.lower()
            if "nginx" in task_lower and "ubuntu" in text and "nginx" in text and "image: ubuntu" in text:
                if "ubuntu" not in task_lower:
                    return False

        if "nginx" in (result or "").lower() and "ubuntu" in text and "image: ubuntu:latest" in text:
            return False

        if Executor._is_network_check_task(task_description or "") and (
            "0% packet loss" in text
            or "packet loss" in text and "0%" in text
            or "google.com" in text and "64 bytes" in text
            or "reply from" in text and "time=" in text
            or "network connection to" in text and "active" in text
        ):
            return True

        if not text:
            return False

        failure_markers = (
            "traceback",
            "exception",
            "error:",
            "failed",
            "not found",
            "permission denied",
            "segmentation fault",
            "command not found",
            "no such file or directory",
            "continue? [y/n]",
            "password required",
            "enter password",
            "overwrite existing",
        )
        if any(marker in text for marker in failure_markers):
            return False

        if last_exit_code is not None and last_exit_code != 0:
            return False

        success_markers = (
            "success",
            "completed",
            "done",
            "listening on",
            "running",
            "ready",
            "ok",
            "available",
            "working",
            "finished",
            "healthy",
        )
        if any(marker in text for marker in success_markers):
            return True

        return bool(result and result.strip())

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def execute(self, task: Task) -> dict:
        """
        Execute a task by driving a live tmux session.

        Returns:
            {"status": "completed", "result": "..."}
            or
            {"status": "failed", "result": "...", "error": "..."}
        """
        self._last_step_log = []

        if not TmuxClient.is_installed():
            msg = "tmux is not installed — install it to enable terminal takeover (e.g. `apt install tmux`)"
            log.error(msg)
            return {"status": "failed", "result": "", "error": msg}

        try:
            target, attached_to_running = self._tmux.get_target()
        except TmuxError as e:
            log.error("Could not resolve tmux target: %s", e)
            return {"status": "failed", "result": "", "error": str(e)}

        if attached_to_running:
            log.info("Task %s: joined the currently running terminal session (%s)", task.id, target)
            reasoning.info("Task %s: joined the live tmux session to troubleshoot the issue.", task.id)
        else:
            log.info(
                "Task %s: no running session detected — driving detached session '%s'. "
                "Attach with: %s",
                task.id,
                target,
                self._tmux.attach_hint(target),
            )
            reasoning.info("Task %s: using detached tmux session '%s' to continue the task.", task.id, target)

        history = self._tmux.capture_history(target, lines=cfg.TERMINAL_HISTORY_LINES)
        step_log: list[dict] = []

        for step in range(1, cfg.MAX_SHELL_STEPS_PER_TASK + 1):
            if self._needs_write_before_retry(task.description, step_log):
                log.warning(
                    "Task %s: validation was repeated without an intervening config-file write; stopping to avoid a loop.",
                    task.id,
                )
                reasoning.warning(
                    "Task %s: file-edit task reached validation again without writing the file; stopping instead of blindly looping.",
                    task.id,
                )
                return {
                    "status": "failed",
                    "result": "",
                    "error": "Validation repeated without a write: the config file was not updated, so the task is stuck in a no-progress loop.",
                }

            if self._should_stop_on_no_progress(step_log):
                log.warning(
                    "Task %s: repeated the same failing command with identical output; stopping instead of looping.",
                    task.id,
                )
                reasoning.warning(
                    "Task %s: no progress after repeated identical failure; stopping to avoid blind looping.",
                    task.id,
                )
                return {
                    "status": "failed",
                    "result": "",
                    "error": "Repeated same failing command without new evidence; task failed by no-progress guard.",
                }

            action = self._decide_next_action(task, history, step_log, step)

            if action is None:
                return {
                    "status": "failed",
                    "result": "",
                    "error": "Executor model returned an unparseable response",
                }

            action = self._normalize_action(action, task.description)
            kind = action.get("action")

            if kind == "done":
                status = action.get("status", "completed")
                if status == "completed" and self._is_validation_task(task.description) and not self._has_file_read_step(step_log):
                    log.warning(
                        "Task %s: validation task ended without reading the target file; refusing completion.",
                        task.id,
                    )
                    reasoning.warning(
                        "Task %s: validation was claimed without inspecting the file contents; declining to mark complete.",
                        task.id,
                    )
                    return {
                        "status": "failed",
                        "result": "",
                        "error": "Validation task ended without reading the target file, so it had no evidence for a valid result.",
                    }

                log.info("Task %s: shell agent signaled %s", task.id, status)
                reasoning.info("Task %s: the terminal agent reported %s.", task.id, status)
                self._last_step_log = list(step_log)
                result = {"status": status, "result": action.get("result", "")}
                if status == "failed":
                    result["error"] = action.get("error", "")
                return result

            if kind == "send_keys":
                keys = action.get("keys", "")
                log.info("Task %s: sending raw keys: %r", task.id, keys)
                try:
                    self._tmux.send_raw_keys(target, keys)
                except TmuxError as e:
                    step_log.append({"command": f"<send_keys {keys!r}>", "output": str(e), "exit_code": None, "timed_out": False})
                    continue
                # give the pane a moment, then re-capture as the "output" of this step
                history = self._tmux.capture_history(target, lines=cfg.TERMINAL_HISTORY_LINES)
                step_log.append({"command": f"<send_keys {keys!r}>", "output": "(keys sent)", "exit_code": None, "timed_out": False})
                continue

            if kind == "human_input":
                input_kind = action.get("kind", "approval")
                prompt = action.get("prompt", "The command is waiting for input.")
                human_response = request_human_input(input_kind, prompt)
                try:
                    self._tmux.send_raw_keys(target, human_response)
                except TmuxError as e:
                    log.error("Task %s: failed to send human answer to tmux pane: %s", task.id, e)
                    return {"status": "failed", "result": "", "error": str(e)}
                history = self._tmux.capture_history(target, lines=cfg.TERMINAL_HISTORY_LINES)
                step_log.append({
                    "command": f"<human_input {input_kind}>",
                    "output": f"Human responded to: {prompt}",
                    "exit_code": None,
                    "timed_out": False,
                })
                continue

            if kind == "run_command":
                command = action.get("command", "").strip()
                if not command:
                    step_log.append({"command": "", "output": "(empty command skipped)", "exit_code": None, "timed_out": False})
                    continue

                if self._requires_file_write(task.description):
                    if not self._is_file_write_command(command) and not any(
                        probe in command.lower() for probe in ("cat ", "sed -n", "docker-compose config", "docker-compose up")
                    ):
                        log.warning(
                            "Task %s: non-write action attempted for file-editing task; refusing to continue loop.",
                            task.id,
                        )
                        return {
                            "status": "failed",
                            "result": "",
                            "error": "This task is about writing a config file, but the agent attempted a non-write action instead of editing the file.",
                        }

                command = normalize_multiline_write_command(command)
                log.info("Task %s: running `%s`", task.id, command)
                try:
                    result = self._tmux.run_command(
                        target,
                        command,
                        timeout=cfg.SHELL_COMMAND_TIMEOUT,
                    )
                except TmuxError as e:
                    result = {"output": str(e), "exit_code": None, "timed_out": False}

                if result.get("human_input_needed"):
                    input_info = detect_human_input_need(result.get("output", ""))
                    if input_info is not None:
                        log.warning("Task %s: command is waiting for %s input in tmux pane", task.id, input_info["kind"])
                        human_response = request_human_input(input_info["kind"], input_info["prompt"])
                        try:
                            self._tmux.send_raw_keys(target, human_response)
                        except TmuxError as e:
                            log.error("Task %s: failed to send prompted response: %s", task.id, e)
                            return {"status": "failed", "result": "", "error": str(e)}
                        history = self._tmux.capture_history(target, lines=cfg.TERMINAL_HISTORY_LINES)
                        step_log.append({
                            "command": command,
                            "output": f"Prompt required human {input_info['kind']} input: {input_info['prompt']}",
                            "exit_code": None,
                            "timed_out": False,
                        })
                        continue

                step_log.append({
                    "command": command,
                    "output": result["output"],
                    "exit_code": result["exit_code"],
                    "timed_out": result["timed_out"],
                })
                self._last_step_log = list(step_log)
                history = self._tmux.capture_history(target, lines=cfg.TERMINAL_HISTORY_LINES)
                continue

            log.warning("Task %s: unknown shell-agent action '%s' — skipping", task.id, kind)
            step_log.append({"command": f"<unknown action {kind}>", "output": "", "exit_code": None, "timed_out": False})

        self._last_step_log = list(step_log)
        log.warning("Task %s: hit MAX_SHELL_STEPS_PER_TASK (%d) without a done signal", task.id, cfg.MAX_SHELL_STEPS_PER_TASK)
        return {
            "status": "failed",
            "result": "",
            "error": f"Exceeded {cfg.MAX_SHELL_STEPS_PER_TASK} shell steps without completion",
        }

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────

    def _decide_next_action(
        self,
        task: Task,
        terminal_history: str,
        step_log: list[dict],
        step: int,
    ) -> dict | None:
        prompt = build_step_prompt(
            task_id=task.id,
            task_description=task.description,
            terminal_history=terminal_history,
            step_log=step_log,
            steps_taken=len(step_log),
            max_steps=cfg.MAX_SHELL_STEPS_PER_TASK,
        )

        try:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except RuntimeError as e:
            log.error("Executor model call failed on step %d: %s", step, e)
            return {"action": "done", "status": "failed", "error": str(e)}

        parsed = parse_json_response(raw)
        if parsed is None or not isinstance(parsed, dict):
            log.warning("Executor step %d: unparseable response: %s", step, raw[:300])
            return None

        return parsed

    def is_available(self) -> bool:
        return self._client.health_check()
