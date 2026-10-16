"""
TmuxClient — drives a real tmux pane: captures the visible screen, sends
keystrokes, and captures current-command output.

This is what lets Nyx "join" a terminal session instead of firing
one-shot task prompts at a sandboxed MCP tool. Two ways to get a target:

  1. Nyx is launched from *inside* the tmux session you want it to
     take over (the normal case: you're stuck, you run `nyx` from
     the same pane). We detect this via the $TMUX env var and attach
     to that exact pane.

  2. Nyx is launched outside tmux. We create/reuse a session named
     `cfg.TMUX_SESSION_NAME` and tell the user to run
     `tmux attach -t <name>` to watch/join it live.

Either way, a human can `tmux attach` to the same session at any time
and see exactly what the agent is doing — nothing hidden, and the
human can type into the same pane to take back control.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid

from utils.human_input import detect_human_input_need, request_human_input
from utils.logger import get_logger
from utils.ssh import SSHConfig, build_ssh_command

log = get_logger(__name__)


class TmuxError(RuntimeError):
    pass


class TmuxClient:
    def __init__(self, session_name: str = "nyx"):
        self._session_name = session_name

    # ── Availability ─────────────────────────────────────────────────

    @staticmethod
    def is_installed() -> bool:
        return shutil.which("tmux") is not None

    # ── Low-level tmux invocation ───────────────────────────────────

    @staticmethod
    def _run(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["tmux", *args],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError:
            raise TmuxError("tmux is not installed or not on PATH")
        except subprocess.TimeoutExpired:
            raise TmuxError(f"tmux command timed out: {' '.join(args)}")

        if result.returncode != 0:
            raise TmuxError(
                f"tmux {' '.join(args)} failed: {result.stderr.strip()}"
            )
        return result.stdout

    # ── Target resolution ────────────────────────────────────────────

    def get_target(self) -> tuple[str, bool]:
        """
        Resolve which pane to drive.

        Returns (target, attached_to_running_session):
          - Prefer the currently active tmux pane if one exists.
          - Otherwise, ensure a detached session named `self._session_name`
            exists and target its first pane so the user can attach.
        """
        try:
            target = self._run(
                ["display-message", "-p", "#{session_name}:#{window_index}.#{pane_index}"]
            ).strip()
            if target and target != "":
                log.info("Attaching to current running tmux pane: %s", target)
                return target, True
        except TmuxError:
            pass

        if os.environ.get("TMUX"):
            try:
                target = self._run(
                    ["display-message", "-p", "#{session_name}:#{window_index}.#{pane_index}"]
                ).strip()
                if target:
                    log.info("Attaching to current running tmux pane: %s", target)
                    return target, True
            except TmuxError:
                pass

        if not self._session_exists(self._session_name):
            log.info(
                "No active tmux pane detected — creating detached session '%s'. "
                "Run `tmux attach -t %s` in another terminal to watch/join.",
                self._session_name,
                self._session_name,
            )
            self._run(["new-session", "-d", "-s", self._session_name])
        target = f"{self._session_name}:0.0"
        return target, False

    def _session_exists(self, name: str) -> bool:
        try:
            self._run(["has-session", "-t", name])
            return True
        except TmuxError:
            return False

    def new_window(self, target: str) -> str:
        """Create a selected tmux window in the target pane's session."""
        session_target = f"{target.split(':', 1)[0]}:"
        window = self._run([
            "new-window",
            "-P",
            "-F",
            "#{session_name}:#{window_index}.#{pane_index}",
            "-t",
            session_target,
        ]).strip()
        if not window:
            raise TmuxError(f"tmux did not return the new window target for {session_target}")
        return window

    # ── Reading state ─────────────────────────────────────────────────

    def capture_screen(self, target: str) -> str:
        return self._run(["capture-pane", "-t", target, "-p"])

    def start_ssh(
        self,
        target: str,
        config: SSHConfig,
        timeout: float = 60.0,
        poll_interval: float = 0.4,
    ) -> None:
        """Open a persistent SSH shell in the target pane and wait for its ready marker."""
        token = uuid.uuid4().hex
        start_marker = f"__NYX_SSH_START_{token}__"
        ready_marker = f"__NYX_SSH_READY_{token}__"
        command = build_ssh_command(config, start_marker, ready_marker)
        self._run(["send-keys", "-t", target, command, "Enter"])

        deadline = time.time() + timeout
        last_prompt = ""
        last_output = ""
        failure_markers = (
            "could not resolve hostname",
            "connection refused",
            "connection timed out",
            "no route to host",
            "permission denied",
            "host key verification failed",
            "remote host identification has changed",
            "ssh: not found",
        )

        while time.time() < deadline:
            time.sleep(poll_interval)
            lines = self.capture_screen(target).splitlines()
            if any(line.strip() == ready_marker for line in lines):
                return

            start_index = next(
                (index for index, line in enumerate(lines) if line.strip() == start_marker),
                -1,
            )
            last_output = "\n".join(lines[start_index + 1:]) if start_index >= 0 else "\n".join(lines)
            lower_output = last_output.lower()
            failure = next((marker for marker in failure_markers if marker in lower_output), None)
            if failure:
                raise TmuxError(f"SSH connection failed: {last_output.strip()}")

            input_need = detect_human_input_need(last_output)
            if input_need is not None and input_need["prompt"] != last_prompt:
                last_prompt = input_need["prompt"]
                response = request_human_input(input_need["kind"], input_need["prompt"])
                self.send_raw_keys(target, response)

        message = (
            f"SSH connection to {config.destination} did not become ready within "
            f"{timeout:g}s. Last terminal output: {last_output.strip() or '(none)'}"
        )
        self.send_raw_keys(target, "C-c", enter=False)
        raise TmuxError(message)

    @staticmethod
    def _command_result(lines: list[str], start_marker: str, sentinel: str) -> tuple[str, int | None] | None:
        """Extract a command's visible output and exit code from the current screen."""
        start_index = next((i for i, line in enumerate(lines) if start_marker in line), -1)
        end_index = next((i for i, line in enumerate(lines) if sentinel in line), -1)
        if end_index < 0:
            return None

        output_start = start_index + 1 if 0 <= start_index < end_index else 0
        output = "\n".join(lines[output_start:end_index]).strip()
        try:
            exit_code = int(lines[end_index].split(f"{sentinel}_", 1)[1].strip())
        except (IndexError, ValueError):
            exit_code = None
        return output, exit_code

    @staticmethod
    def _is_pager_screen(output: str) -> bool:
        """Recognize common pager footers without treating arbitrary prompts as pagers."""
        lower_output = output.lower()
        return (
            "--more--" in lower_output
            or "(end)" in lower_output
            or bool(re.search(r"lines\s+\d+-\d+/\d+", lower_output))
            or "press q to quit" in lower_output
        )

    # ── Driving the session ──────────────────────────────────────────

    def run_command(
        self,
        target: str,
        command: str,
        timeout: float = 120.0,
        poll_interval: float = 0.4,
    ) -> dict:
        """
        Type `command` into the pane, press Enter, and wait for it to
        finish by watching for a unique sentinel this method appends.

        Returns {"output": "<visible output from this command>",
                 "exit_code": int | None, "timed_out": bool}
        """
        token = uuid.uuid4().hex
        start_marker = f"__NYX_START_{token}__"
        sentinel = f"__NYX_DONE_{token}__"

        wrapped = (
            f"printf '\\n%s\\n' '{start_marker}';\n"
            f"{command}\n"
            f"__nyx_status_{token}=$?; "
            f"printf '\\n%s_%s\\n' '{sentinel}' \"$__nyx_status_{token}\""
        )
        self._run(["send-keys", "-t", target, wrapped, "Enter"])

        deadline = time.time() + timeout
        exit_code = None
        timed_out = True
        new_text = ""
        pager_dismissed = False

        while time.time() < deadline:
            time.sleep(poll_interval)
            lines = self.capture_screen(target).splitlines()
            start_index = next((i for i, line in enumerate(lines) if start_marker in line), -1)
            current_output = "\n".join(lines[start_index + 1:]) if start_index >= 0 else ""

            input_need = detect_human_input_need(current_output)
            if input_need is not None:
                return {
                    "output": current_output,
                    "exit_code": None,
                    "timed_out": False,
                    "human_input_needed": True,
                    "human_input_prompt": input_need["prompt"],
                    "human_input_kind": input_need["kind"],
                }

            if not pager_dismissed and self._is_pager_screen(current_output):
                self.send_raw_keys(target, "q", enter=False)
                pager_dismissed = True
                continue

            command_result = self._command_result(lines, start_marker, sentinel)
            if command_result is not None:
                timed_out = False
                new_text, exit_code = command_result
                break

        return {
            "output": new_text,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "human_input_needed": False,
        }

    def send_raw_keys(self, target: str, keys: str, enter: bool = True):
        """For interactive prompts (y/n, passwords, editors) where you need
        to send keystrokes without the sentinel/wait machinery — e.g.
        answering a confirmation prompt mid-command."""
        args = ["send-keys", "-t", target, keys]
        if enter:
            args.append("Enter")
        self._run(args)

    def attach_hint(self, target: str) -> str:
        session = target.split(":")[0]
        return f"tmux attach -t {session}"
