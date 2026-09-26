"""
TmuxClient — drives a real tmux pane: reads scrollback history, sends
keystrokes, and captures command output.

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
import shutil
import subprocess
import time
import uuid

from utils.human_input import detect_human_input_need
from utils.logger import get_logger

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

    # ── Reading state ─────────────────────────────────────────────────

    def capture_history(self, target: str, lines: int = 4000) -> str:
        """Full scrollback (up to `lines`) so the model sees terminal history,
        not just the current screen."""
        return self._run(
            ["capture-pane", "-t", target, "-p", "-S", f"-{lines}"]
        )

    def capture_screen(self, target: str) -> str:
        return self._run(["capture-pane", "-t", target, "-p"])

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

        Returns {"output": "<new pane text since the command>",
                 "exit_code": int | None, "timed_out": bool}
        """
        sentinel = f"__NYX_DONE_{uuid.uuid4().hex[:8]}__"
        before = self.capture_screen(target)
        before_line_count = len(before.splitlines())

        wrapped = f"{command}; echo {sentinel}_$?"
        self._run(["send-keys", "-t", target, wrapped, "Enter"])

        deadline = time.time() + timeout
        exit_code = None
        timed_out = True
        new_text = ""

        while time.time() < deadline:
            time.sleep(poll_interval)
            screen = self.capture_history(target, lines=8000)
            lines = screen.splitlines()

            input_need = detect_human_input_need(screen)
            if input_need is not None:
                return {
                    "output": screen,
                    "exit_code": None,
                    "timed_out": False,
                    "human_input_needed": True,
                    "human_input_prompt": input_need["prompt"],
                    "human_input_kind": input_need["kind"],
                }

            if any(sentinel in ln for ln in lines):
                timed_out = False
                # Everything from just after the old screen up to the sentinel
                tail = lines[before_line_count:]
                out_lines = []
                for ln in tail:
                    if sentinel in ln:
                        try:
                            exit_code = int(ln.split(f"{sentinel}_")[-1].strip())
                        except ValueError:
                            exit_code = None
                        break
                    out_lines.append(ln)
                new_text = "\n".join(out_lines).strip()
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
