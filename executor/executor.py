"""
Executor — joins a live tmux terminal session and drives it directly,
one shell command at a time, using only current-task command output.

Replaces the old LM Studio MCP + mcp-docker integration: instead of
handing the task to an opaque tool-use sandbox, this attaches to the
real pane (the one you were already working in, if Nyx is launched
from inside tmux) and reasons over output from commands it runs.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable

import config.settings as cfg
from executor.file_operations import (
    MAX_FILE_BYTES,
    MISSING_FILE_SENTINEL,
    build_read_file_command,
    build_write_file_command,
    decode_read_file_output,
)
from executor.shell_prompts import SYSTEM_PROMPT, build_step_prompt
from tasks.task import Task
from utils.helpers import parse_json_response, normalize_multiline_write_command
from utils.human_input import detect_human_input_need, request_human_input
from utils.lmstudio_client import LMStudioClient
from utils.logger import get_logger, get_reasoning_logger
from utils.tmux_client import TmuxClient, TmuxError
from utils.ssh import SSHConfig, describe_ssh_target
from utils.control_plane import ControlPlane

log = get_logger(__name__)
reasoning = get_reasoning_logger(__name__)


class Executor:
    DISK_MUTATION_COMMANDS = {
        "blkdiscard",
        "fdisk",
        "parted",
        "pvcreate",
        "pvremove",
        "sfdisk",
        "sgdisk",
        "shred",
        "vgcreate",
        "vgremove",
        "lvcreate",
        "lvremove",
        "mdadm",
        "wipefs",
    }
    INTERACTIVE_COMMANDS = {
        "btop": "Use `top -b -n 1` for a one-shot process snapshot.",
        "htop": "Use `top -b -n 1` for a one-shot process snapshot.",
        "less": "Use `cat`, `sed -n`, or `head` for non-interactive output.",
        "man": "Use `command --help` or read the relevant documentation non-interactively.",
        "more": "Use `cat`, `sed -n`, or `head` for non-interactive output.",
        "nano": "Use a non-interactive file-edit command, or request human input.",
        "newgrp": (
            "It opens an interactive subshell that Nyx cannot safely control. "
            "Use `sg docker -c '<command>'` for one command, or reconnect the "
            "user session to activate updated group membership."
        ),
        "vi": "Use a non-interactive file-edit command, or request human input.",
        "vim": "Use a non-interactive file-edit command, or request human input.",
        "watch": "Run the watched command once without `watch`.",
    }

    def __init__(self):
        self._client = LMStudioClient(
            base_url=cfg.EXECUTOR_API_URL,
            model=cfg.EXECUTOR_MODEL,
            temperature=cfg.EXECUTOR_TEMPERATURE,
        )
        self._tmux = TmuxClient(session_name=cfg.TMUX_SESSION_NAME)
        self._last_step_log: list[dict] = []
        self._ssh_target: str | None = None
        self._remote_context: str = ""
        self._control_plane: ControlPlane | None = None
        self._control_run_id: str | None = None
        self._auto_mode = False
        self._sudo_password: str | None = None
        self._system_context = ""

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
    def _disable_systemctl_pager(command: str) -> str:
        """Add systemctl's global no-pager option, including when invoked via sudo."""
        return re.sub(
            r"(?P<systemctl>(?:/[\w./-]+/)?systemctl)(?!\s+--no-pager\b)",
            r"\g<systemctl> --no-pager",
            command,
            count=1,
        )

    @staticmethod
    def _high_risk_command_reason(command: str) -> str | None:
        """Refuse model-generated disk layout, filesystem, and raw-device mutations."""
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return "The command could not be safely parsed, so Nyx refused to run it."
        separators = {";", "&&", "||", "|", "|&", "&"}
        command_parts: list[list[str]] = [[]]
        for token in tokens:
            if token in separators:
                command_parts.append([])
            else:
                command_parts[-1].append(token)

        for part in command_parts:
            if not part:
                continue
            index = 0
            while index < len(part) and part[index].rsplit("/", 1)[-1] in {"sudo", "command"}:
                index += 1
                if part[index - 1].rsplit("/", 1)[-1] == "sudo":
                    while index < len(part) and part[index].startswith("-"):
                        if part[index] in {"-u", "-g", "-h", "-p"}:
                            index += 2
                        else:
                            index += 1
            if index < len(part) and part[index] == "env":
                index += 1
                while index < len(part) and "=" in part[index] and not part[index].startswith("-"):
                    index += 1
            if index < len(part) and part[index].endswith("=") and index == 0:
                index += 1
            if index >= len(part):
                continue
            executable = part[index].rsplit("/", 1)[-1].lower()
            arguments = part[index + 1:]

            if executable in {"sh", "bash", "dash", "zsh"} and "-c" in arguments:
                script_index = arguments.index("-c") + 1
                if script_index < len(arguments):
                    nested_reason = Executor._high_risk_command_reason(arguments[script_index])
                    if nested_reason:
                        return nested_reason

            # Permit only the explicit listing forms; all other parted/fdisk
            # invocations are refused because they can rewrite disk layouts.
            if executable in {"parted", "fdisk"}:
                read_only = bool(arguments) and arguments[0] in {"-l", "--list"} and all(
                    argument == "--" or argument.startswith("/dev/")
                    for argument in arguments[1:]
                )
                if read_only:
                    continue
                return (
                    f"`{executable}` can change disk partition layouts. "
                    "Nyx blocks partition-editing commands; inspect disks with `lsblk` "
                    "and make any required disk changes manually."
                )
            if executable.startswith("mkfs") or executable in Executor.DISK_MUTATION_COMMANDS:
                if executable.startswith("mkfs"):
                    detail = "format a filesystem"
                elif executable == "mdadm":
                    if not any(
                        option in arguments
                        for option in ("--create", "--zero-superblock", "--remove", "--stop")
                    ):
                        continue
                    detail = "create or modify software RAID metadata"
                elif executable in {"pvcreate", "vgcreate"}:
                    detail = "initialize or modify storage volumes"
                else:
                    detail = "erase or modify raw-disk data"
                return (
                    f"`{executable}` can {detail}. Nyx blocks destructive disk operations; "
                    "perform this operation manually after independently verifying the target."
                )
            if executable == "cryptsetup" and arguments and arguments[0] in {
                "luksFormat", "luksErase", "luksChangeKey", "erase", "reencrypt",
            }:
                return (
                    f"`cryptsetup {arguments[0]}` can destroy or replace encryption metadata. "
                    "Nyx blocks destructive disk operations."
                )
            if executable == "dd" and any(
                argument.startswith("of=/dev/") for argument in arguments
            ):
                return (
                    "`dd` targeting a device can overwrite a disk. Nyx blocks raw-device "
                    "writes; perform this operation manually after independently verifying the target."
                )
            if executable == "rm":
                recursive = any(
                    argument.startswith("-") and "r" in argument.lstrip("-")
                    for argument in arguments
                )
                system_targets = {
                    "/", "/*", "/boot", "/dev", "/etc", "/home", "/lib",
                    "/opt", "/root", "/sbin", "/sys", "/usr", "/var",
                }
                if recursive and any(Executor._is_system_wide_target(argument, system_targets) for argument in arguments):
                    return (
                        "`rm` recursively targeting a system-wide directory can destroy the OS "
                        "or user data. Nyx blocks this command."
                    )
            if executable in {"chmod", "chown"}:
                recursive = any(
                    argument.startswith("-") and "R" in argument.lstrip("-")
                    for argument in arguments
                )
                system_targets = {
                    "/", "/*", "/boot", "/dev", "/etc", "/home", "/lib",
                    "/opt", "/root", "/sbin", "/sys", "/usr", "/var",
                }
                if recursive and any(Executor._is_system_wide_target(argument, system_targets) for argument in arguments):
                    return (
                        f"`{executable}` recursively changing a system-wide directory can "
                        "break the OS or expose user data. Nyx blocks this command."
                    )
            if executable == "find" and "-delete" in arguments:
                system_targets = {
                    "/", "/boot", "/dev", "/etc", "/home", "/lib",
                    "/opt", "/root", "/sbin", "/sys", "/usr", "/var",
                }
                search_paths = [
                    argument for argument in arguments
                    if not argument.startswith("-") and "=" not in argument
                ]
                if any(Executor._is_system_wide_target(path, system_targets) for path in search_paths):
                    return (
                        "`find -delete` targeting system-wide paths can destroy the OS or "
                        "user data. Nyx blocks this command."
                    )
        return None

    @staticmethod
    def _is_system_wide_target(path: str, roots: set[str]) -> bool:
        normalized = path.rstrip("/") or "/"
        return normalized in roots or any(
            normalized.startswith(root.rstrip("/") + "/")
            or normalized.startswith(root.rstrip("/") + "/*")
            for root in roots
            if root != "/"
        )

    @staticmethod
    def _normalize_noninteractive_command(command: str) -> tuple[str, str | None]:
        """Convert supported interactive commands or reject commands with no safe batch mode."""
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return command, None
        if not tokens:
            return command, None

        command_index = 1 if tokens[0] == "sudo" and len(tokens) > 1 else 0
        executable = tokens[command_index].rsplit("/", 1)[-1].lower()
        shell_operators = {";", "&&", "||", "|", ">", ">>", "<"}
        if executable == "top" and not any(token in shell_operators for token in tokens):
            if "-b" not in tokens[command_index + 1:]:
                tokens.insert(command_index + 1, "-b")
            if "-n" not in tokens[command_index + 1:] and "-bn" not in tokens[command_index + 1:]:
                tokens.extend(("-n", "1"))
            return shlex.join(tokens), None

        reason = Executor.INTERACTIVE_COMMANDS.get(executable)
        if reason:
            return command, (
                f"`{executable}` is interactive and cannot be run by Nyx safely. {reason}"
            )
        if executable == "tail" and any(arg == "-f" or arg.startswith("--follow") for arg in tokens[command_index + 1:]):
            return command, "`tail -f` is a continuous interactive stream. Use a bounded `tail -n N` command instead."
        if executable == "journalctl" and any(arg in {"-f", "--follow"} for arg in tokens[command_index + 1:]):
            return command, "`journalctl -f` is a continuous stream. Use a bounded `journalctl --no-pager -n N` command instead."
        if executable == "ping" and not any(
            token == "-c" or token.startswith("--count") or re.fullmatch(r"-c\d+", token)
            for token in tokens[command_index + 1:]
        ):
            tokens[command_index + 1:command_index + 1] = ["-c", "1"]
            return shlex.join(tokens), None
        return command, None

    def connect_ssh(self, config: SSHConfig) -> str:
        """Connect through tmux and return the remote execution context."""
        if not TmuxClient.is_installed():
            raise TmuxError("tmux is not installed — SSH execution requires tmux")

        current_target, _ = self._tmux.get_target()
        target = self._tmux.new_window(current_target)
        log.info("Opening SSH connection to %s in tmux window %s", config.destination, target)
        self._tmux.start_ssh(target, config, timeout=cfg.SSH_CONNECT_TIMEOUT)
        self._ssh_target = target
        self._remote_context = describe_ssh_target(config)
        reasoning.info("Connected the tmux terminal to %s.", config.destination)
        return self._remote_context

    def configure_control_plane(self, control_plane: ControlPlane, run_id: str) -> None:
        self._control_plane = control_plane
        self._control_run_id = run_id

    def configure_input_modes(self, auto_mode: bool, sudo_password: str | None) -> None:
        self._auto_mode = auto_mode
        self._sudo_password = sudo_password
        self._tmux.configure_input_modes(auto_mode)

    def configure_system_context(
        self,
        system_context: str,
        on_refresh: Callable[[dict[str, str]], None] | None = None,
    ) -> None:
        self._system_context = system_context
        self._system_context_callback = on_refresh

    def capture_system_context(self, target: str | None = None) -> dict[str, str]:
        """Observe facts from the live shell Nyx will operate, not the local Python process."""
        if target is None:
            target = self._ssh_target
        if target is None:
            target, _ = self._tmux.get_target()
        command = (
            "printf 'current_directory=%s\\nhome_directory=%s\\nuser=%s\\nhostname=%s\\n"
            "os_kernel=%s\\n' \"$(pwd -P)\" \"$HOME\" \"$(id -un)\" "
            "\"$(hostname)\" \"$(uname -srm)\""
        )
        try:
            result = self._tmux.run_command(target, command, timeout=20)
        except TmuxError as error:
            raise TmuxError(f"Could not capture live system context: {error}") from error
        if result.get("timed_out") or result.get("exit_code") != 0:
            raise TmuxError(
                "Could not capture live system context: "
                f"{str(result.get('output', '')).strip() or 'terminal command failed'}"
            )
        context: dict[str, str] = {}
        for line in str(result.get("output", "")).splitlines():
            key, separator, value = line.partition("=")
            if separator and value.strip():
                context[key.strip()] = value.strip()
        required = {
            "current_directory",
            "home_directory",
            "user",
            "hostname",
            "os_kernel",
        }
        if not required.issubset(context):
            missing = ", ".join(sorted(required - context.keys()))
            raise TmuxError(f"Live system context was incomplete; missing: {missing}")
        return context

    def _control_checkpoint(self, task: Task, phase: str) -> str:
        if self._control_plane is None or self._control_run_id is None:
            return ""
        self._control_plane.update_run(
            self._control_run_id,
            phase=phase,
            current_task=task.description,
        )
        return self._control_plane.wait_if_paused(self._control_run_id)

    @staticmethod
    def _task_matches_compose_requirement(task_description: str, compose_text: str) -> bool:
        task_lower = (task_description or "").lower()
        compose_lower = (compose_text or "").lower()

        if "nginx" in task_lower:
            return "nginx" in compose_lower and "image: nginx" in compose_lower
        return True

    @staticmethod
    def _is_docker_setup_task(task_description: str) -> bool:
        text = (task_description or "").lower()
        return "docker" in text and any(
            term in text
            for term in (
                "setup",
                "set up",
                "configure",
                "container",
                "deploy",
                "run ",
                "launch",
                "start",
                "build",
                "image",
                "stop",
            )
        )

    @staticmethod
    def _explicitly_requests_compose(task_description: str) -> bool:
        text = (task_description or "").lower()
        return bool(
            re.search(r"\bdocker[- ]compose\b", text)
            or re.search(r"\bcompose(?:\.ya?ml)?\s+file\b", text)
            or re.search(r"\bcompose\.ya?ml\b", text)
        )

    @staticmethod
    def _is_compose_command(command: str) -> bool:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return False
        for index, token in enumerate(tokens):
            executable = token.rsplit("/", 1)[-1].lower()
            if executable == "docker-compose":
                return True
            if (
                executable == "docker"
                and index + 1 < len(tokens)
                and tokens[index + 1].lower() == "compose"
            ):
                return True
        return False

    @staticmethod
    def _is_compose_file_write_command(command: str) -> bool:
        if not Executor._is_file_write_command(command):
            return False
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return False
        compose_files = {
            "compose.yml",
            "compose.yaml",
            "docker-compose.yml",
            "docker-compose.yaml",
        }
        return any(token.rsplit("/", 1)[-1].lower() in compose_files for token in tokens)

    @staticmethod
    def _is_host_apache_command(command: str) -> bool:
        try:
            tokens = [token.lower() for token in shlex.split(command, posix=True)]
        except ValueError:
            return False
        if not tokens:
            return False
        command_index = 1 if tokens[0] == "sudo" and len(tokens) > 1 else 0
        executable = tokens[command_index].rsplit("/", 1)[-1]
        arguments = tokens[command_index + 1:]
        if executable in {"apache2", "apachectl", "httpd"}:
            return True
        return executable in {"systemctl", "service"} and any(
            argument.rsplit("/", 1)[-1] in {"apache2", "httpd"}
            for argument in arguments
        )

    @staticmethod
    def _is_host_port_investigation_command(command: str) -> bool:
        try:
            tokens = [token.lower() for token in shlex.split(command, posix=True)]
        except ValueError:
            return False
        if not tokens:
            return False
        command_index = 1 if tokens[0] == "sudo" and len(tokens) > 1 else 0
        executable = tokens[command_index].rsplit("/", 1)[-1]
        return executable in {"netstat", "ss", "lsof"}

    @staticmethod
    def _is_bulk_docker_stop_command(command: str) -> bool:
        normalized = " ".join(command.lower().split())
        return bool(
            re.search(
                r"\bdocker\s+stop\s+(?:['\"]?\$?\(\s*docker\s+ps\b|`docker\s+ps\b)",
                normalized,
            )
        )

    @staticmethod
    def _docker_port_conflict(output: str) -> bool:
        text = (output or "").lower()
        return any(
            marker in text
            for marker in (
                "port is already allocated",
                "address already in use",
                "bind: address already in use",
            )
        )

    @staticmethod
    def _requires_file_write(task_description: str) -> bool:
        text = (task_description or "").lower()
        file_markers = (
            " file",
            "file ",
            "configuration",
            "config file",
            "docker-compose.yml",
            "dockerfile",
            "compose file",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".md",
            ".html",
            ".css",
            ".sh",
            ".service",
            ".txt",
        )
        write_markers = (
            "write",
            "create",
            "fix",
            "update",
            "rewrite",
            "repair",
            "edit",
            "configure",
        )
        return any(marker in text for marker in file_markers) and any(marker in text for marker in write_markers)

    @staticmethod
    def _is_file_write_command(command: str) -> bool:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars="<>")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return True
        if any(token in {">", ">>", ">|", "<<", "<<<"} for token in tokens):
            return True
        for index, token in enumerate(tokens):
            executable = token.rsplit("/", 1)[-1].lower()
            arguments = tokens[index + 1:]
            if executable == "sed" and any(
                argument == "-i"
                or argument.startswith("-i") and argument != "-i"
                or argument == "--in-place"
                for argument in arguments
            ):
                return True
            if executable == "perl" and any(
                argument.startswith("-") and "i" in argument[1:]
                for argument in arguments
            ):
                return True
            if executable == "tee" and arguments:
                return True
        return False

    @staticmethod
    def _has_write_step(step_log: list[dict]) -> bool:
        if not step_log:
            return False
        return any(
            isinstance(step, dict)
            and (
                step.get("file_operation") == "write"
                or Executor._is_file_write_command(str(step.get("command", "")))
            )
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
    def _is_directory_creation_task(task_description: str) -> bool:
        return bool(
            re.search(
                r"\b(?:create|make|mkdir)\b.{0,80}director(?:y|ies)\b",
                task_description or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _directory_exists_conflict(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:already exists|file exists|directory exists|exists already)\b",
                text or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_network_check_task(task_description: str) -> bool:
        text = (task_description or "").lower()
        return "network" in text or "internet" in text or "connectivity" in text or "ping" in text

    @staticmethod
    def _is_basic_connectivity_task(task_description: str) -> bool:
        text = (task_description or "").lower()
        if not Executor._is_network_check_task(text):
            return False
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
        return not any(term in text for term in additional_work_terms)

    @staticmethod
    def _has_successful_ping(step_log: list[dict]) -> bool:
        return any(
            isinstance(step, dict)
            and re.search(
                r"(?:^|\s)(?:sudo\s+)?(?:[\w./-]+/)?ping(?:\s|$)",
                str(step.get("command", "")),
                re.IGNORECASE,
            )
            and step.get("exit_code") == 0
            for step in step_log
        )

    @staticmethod
    def _has_file_read_step(step_log: list[dict]) -> bool:
        for step in step_log or []:
            if not isinstance(step, dict):
                continue
            if step.get("file_operation") == "read":
                return True
            command = str(step.get("command", "")).lower()
            if any(marker in command for marker in ("cat ", "sed -n", "head ", "tail ", "grep ", "less ", "more ", "wc -l")):
                return True
        return False

    @staticmethod
    def validate_completion_status(
        status: str,
        result: str,
        error: str,
        last_exit_code: int | None,
        step_log: list[dict] | None = None,
        task_description: str | None = None,
    ) -> bool:
        """Return True only when current-task evidence genuinely suggests success."""
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
            evidence_parts = [result]

        text = "\n".join(part for part in evidence_parts if part).strip().lower()

        if Executor._is_directory_creation_task(task_description or ""):
            combined = f"{text}\n{result}".lower()
            if Executor._directory_exists_conflict(combined):
                return False
            created_by_mkdir = any(
                isinstance(step, dict)
                and step.get("exit_code") == 0
                and re.search(r"(?:^|\s)mkdir(?:\s|$)", str(step.get("command", "")))
                for step in step_log or []
            )
            return created_by_mkdir and bool(
                re.search(r"\bcreated\b", result, re.IGNORECASE)
            )

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

        if self._ssh_target is not None:
            target, attached_to_running = self._ssh_target, True
        else:
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

        step_log: list[dict] = []

        for step in range(1, cfg.MAX_SHELL_STEPS_PER_TASK + 1):
            self._control_checkpoint(task, f"Ready for step {step}")
            context_callback = getattr(self, "_system_context_callback", None)
            if context_callback is not None:
                try:
                    context = self.capture_system_context(target)
                except TmuxError as error:
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": str(error),
                    }
                self._system_context = self._format_system_context(context)
                context_callback(context)

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

            action = self._decide_next_action(task, step_log, step)

            if action is None:
                return {
                    "status": "failed",
                    "result": "",
                    "error": "Executor model returned an unparseable response",
                }

            action = self._normalize_action(action, task.description)
            kind = action.get("action")

            if kind in {"read_file", "write_file"}:
                path = action.get("path")
                if not isinstance(path, str) or not path.strip():
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"{kind} requires a non-empty path.",
                    }
                path = path.strip()

                if (
                    kind == "write_file"
                    and self._is_docker_setup_task(task.description)
                    and not self._explicitly_requests_compose(task.description)
                    and path.rsplit("/", 1)[-1].lower()
                    in {"compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml"}
                ):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "This Docker setup task did not request Docker Compose. "
                            "Write a goal-matched Dockerfile with write_file instead."
                        ),
                    }

                if not self._requires_file_write(task.description):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "Dedicated file editing is available only for tasks that explicitly "
                            "request creating or changing a file."
                        ),
                    }

                if kind == "read_file":
                    try:
                        command = build_read_file_command(path)
                    except ValueError as error:
                        return {"status": "failed", "result": "", "error": str(error)}
                    try:
                        file_result = self._tmux.run_command(
                            target,
                            command,
                            timeout=cfg.SHELL_COMMAND_TIMEOUT,
                            capture_history=True,
                        )
                    except TmuxError as error:
                        self._last_step_log = list(step_log)
                        return {
                            "status": "failed",
                            "result": "",
                            "error": f"Could not read file {path}: {error}",
                        }
                    output = str(file_result.get("output", ""))
                    if file_result.get("timed_out") or file_result.get("exit_code") != 0:
                        step_log.append({
                            "command": f"<read_file {path}>",
                            "output": output[-2000:],
                            "exit_code": file_result.get("exit_code"),
                            "timed_out": file_result.get("timed_out", False),
                            "file_operation": "read",
                            "file_path": path,
                        })
                        self._last_step_log = list(step_log)
                        return {
                            "status": "failed",
                            "result": "",
                            "error": f"Could not read complete file {path}: {output[-1000:]}",
                        }
                    if output.strip() == "__NYX_FILE_MISSING__":
                        content = ""
                        file_exists = False
                    else:
                        try:
                            content = decode_read_file_output(output)
                        except ValueError as error:
                            return {
                                "status": "failed",
                                "result": "",
                                "error": f"Could not read complete file {path}: {error}",
                            }
                        file_exists = True
                    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                        return {
                            "status": "failed",
                            "result": "",
                            "error": f"File {path} exceeds Nyx's {MAX_FILE_BYTES}-byte limit.",
                        }
                    step_log.append({
                        "command": f"<read_file {path}>",
                        "output": content if file_exists else "(file does not exist yet)",
                        "exit_code": 0,
                        "timed_out": False,
                        "file_operation": "read",
                        "file_path": path,
                        "file_exists": file_exists,
                        "file_read": True,
                    })
                    self._last_step_log = list(step_log)
                    continue

                if not any(
                    entry.get("file_operation") == "read"
                    and entry.get("file_path") == path
                    and entry.get("exit_code") == 0
                    for entry in step_log
                ):
                    message = (
                            f"Read {path} successfully before writing it. Nyx only permits "
                            "replacing the exact file reviewed in this task."
                    )
                    step_log.append({
                        "command": f"<write_file {path}>",
                        "output": message,
                        "exit_code": None,
                        "timed_out": False,
                        "policy_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    continue
                content = action.get("content")
                if not isinstance(content, str):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": "write_file requires the complete file contents as a string.",
                    }
                try:
                    command = build_write_file_command(path, content)
                except ValueError as error:
                    return {"status": "failed", "result": "", "error": str(error)}
                try:
                    write_result = self._tmux.run_command(
                        target,
                        command,
                        timeout=cfg.SHELL_COMMAND_TIMEOUT,
                    )
                except TmuxError as error:
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Could not atomically write {path}: {error}",
                    }
                step_log.append({
                    "command": f"<write_file {path}>",
                    "output": str(write_result.get("output", "")),
                    "exit_code": write_result.get("exit_code"),
                    "timed_out": write_result.get("timed_out", False),
                    "file_operation": "write",
                    "file_path": path,
                })
                self._last_step_log = list(step_log)
                if write_result.get("timed_out"):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Atomic write of {path} failed: {write_result.get('output', '')}",
                    }
                if write_result.get("exit_code") != 0:
                    step_log[-1]["output"] = (
                        f"{step_log[-1]['output']}\n"
                        f"Atomic write of {path} failed. Choose a corrected path "
                        "using CURRENT SYSTEM CONTEXT; do not root relative project "
                        "paths at /. Report a concrete blocker if no valid path exists."
                    ).strip()
                    self._last_step_log = list(step_log)
                    continue

                try:
                    verify_result = self._tmux.run_command(
                        target,
                        build_read_file_command(path),
                        timeout=cfg.SHELL_COMMAND_TIMEOUT,
                        capture_history=True,
                    )
                    verified_content = decode_read_file_output(
                        str(verify_result.get("output", ""))
                    )
                except (TmuxError, ValueError) as error:
                    step_log.append({
                        "command": f"<verify_file {path}>",
                        "output": f"Read-back verification failed: {error}",
                        "exit_code": None,
                        "timed_out": False,
                        "file_operation": "verify",
                        "file_path": path,
                    })
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Could not verify the complete contents of {path}: {error}",
                    }
                if verify_result.get("timed_out") or verify_result.get("exit_code") != 0:
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Could not verify the complete contents of {path}.",
                    }
                if verified_content != content:
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Read-back contents differ from the requested contents of {path}.",
                    }
                step_log.append({
                    "command": f"<verify_file {path}>",
                    "output": "Read-back verification matched the complete requested file contents.",
                    "exit_code": 0,
                    "timed_out": False,
                    "file_operation": "verify",
                    "file_path": path,
                })
                self._last_step_log = list(step_log)
                continue

            if kind == "done":
                status = action.get("status", "completed")
                if (
                    status == "completed"
                    and self._is_directory_creation_task(task.description)
                    and self._directory_exists_conflict(
                        str(action.get("result", ""))
                        + "\n"
                        + "\n".join(
                            str(entry.get("output", ""))
                            for entry in step_log
                            if isinstance(entry, dict)
                        )
                    )
                ):
                    error = (
                        "The requested directory already exists, so Nyx did not create "
                        "a new directory. Report this conflict to the planner, which "
                        "must choose a different unused directory name and update all "
                        "dependent task paths consistently."
                    )
                    self._last_step_log = list(step_log)
                    return {"status": "failed", "result": "", "error": error}
                if (
                    status == "completed"
                    and self._requires_file_write(task.description)
                    and not self._has_write_step(step_log)
                ):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "File-edit task ended without a verified write_file action. "
                            "Read the target, write the complete contents, and verify them."
                        ),
                    }
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
                step_log.append({"command": f"<send_keys {keys!r}>", "output": "(keys sent)", "exit_code": None, "timed_out": False})
                continue

            if kind == "human_input":
                input_kind = str(action.get("kind", "approval"))
                prompt = str(action.get("prompt", "The executor requested human approval."))
                if input_kind != "approval":
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "The executor requested input that was not prompted by the terminal. "
                            "Nyx will not type model-requested credentials or arbitrary input into the shell."
                        ),
                    }

                if (
                    self._is_directory_creation_task(task.description)
                    and self._directory_exists_conflict(prompt)
                ):
                    message = (
                        f"Directory conflict: {prompt} Do not treat an existing "
                        "directory as newly created; report failure so the planner "
                        "can choose a different unused directory name and update "
                        "dependent task paths."
                    )
                    step_log.append({
                        "command": "<directory_conflict>",
                        "output": message,
                        "exit_code": None,
                        "timed_out": False,
                        "policy_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": message,
                    }

                if any(
                    entry.get("command") == "<human_input approval>"
                    for entry in step_log
                ):
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "The executor requested approval more than once in one task. "
                            "Stopped to prevent repeated approval prompts."
                        ),
                    }

                if getattr(self, "_auto_mode", False):
                    human_response = "y"
                    log.warning(
                        "Task %s: --auto accepted the executor's approval request: %s",
                        task.id,
                        prompt,
                    )
                else:
                    human_response = request_human_input(input_kind, prompt).strip().lower()
                if human_response not in {"y", "yes", "approve", "approved"}:
                    step_log.append({
                        "command": "<human_input approval>",
                        "output": f"User declined: {prompt}",
                        "exit_code": None,
                        "timed_out": False,
                    })
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": "The user did not approve the requested action.",
                    }

                step_log.append({
                    "command": "<human_input approval>",
                    "output": f"User approved: {prompt}",
                    "exit_code": None,
                    "timed_out": False,
                })
                self._last_step_log = list(step_log)
                continue

            if kind == "run_command":
                command = action.get("command", "").strip()
                if not command:
                    step_log.append({"command": "", "output": "(empty command skipped)", "exit_code": None, "timed_out": False})
                    continue

                if (
                    self._is_docker_setup_task(task.description)
                    and not self._explicitly_requests_compose(task.description)
                    and (
                        self._is_compose_command(command)
                        or self._is_compose_file_write_command(command)
                    )
                ):
                    message = (
                        "This Docker setup task did not request Docker Compose. "
                        "Build a goal-matched local image from a Dockerfile, then use "
                        "`docker run`. Do not create a Compose file for this task."
                    )
                    log.warning("Task %s: %s", task.id, message)
                    step_log.append({
                        "command": command,
                        "output": message,
                        "exit_code": None,
                        "timed_out": False,
                        "policy_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    continue

                if (
                    self._is_docker_setup_task(task.description)
                    and not any(
                        term in task.description.lower()
                        for term in ("host apache", "host service", "system apache")
                    )
                    and (
                        self._is_host_apache_command(command)
                        or self._is_host_port_investigation_command(command)
                    )
                ):
                    message = (
                        "This task is setting up a Docker container, not host web "
                        "services. Do not investigate or change host Apache or inspect "
                        "host listeners; use a different published host port if Docker "
                        "reports a conflict, then verify the named container."
                    )
                    log.warning("Task %s: %s", task.id, message)
                    step_log.append({
                        "command": command,
                        "output": message,
                        "exit_code": None,
                        "timed_out": False,
                        "policy_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    continue

                if (
                    self._is_docker_setup_task(task.description)
                    and self._is_bulk_docker_stop_command(command)
                ):
                    message = (
                        "Stopping every Docker container is not allowed. Inspect "
                        "`docker ps -a` and stop only the specific relevant container "
                        "by its observed name or ID."
                    )
                    step_log.append({
                        "command": command,
                        "output": message,
                        "exit_code": None,
                        "timed_out": False,
                        "policy_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    continue

                if self._requires_file_write(task.description):
                    if self._is_file_write_command(command):
                        return {
                            "status": "failed",
                            "result": "",
                            "error": (
                                "Shell-based file modification is disabled for file-edit tasks. "
                                "Use read_file followed by write_file with the complete contents."
                            ),
                        }

                command = normalize_multiline_write_command(command)
                safety_error = self._high_risk_command_reason(command)
                if safety_error:
                    log.error("Task %s: blocked high-risk command: %s", task.id, safety_error)
                    step_log.append({
                        "command": command,
                        "output": safety_error,
                        "exit_code": None,
                        "timed_out": False,
                        "safety_blocked": True,
                    })
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": safety_error,
                        "safety_blocked": True,
                    }
                command, interactive_error = self._normalize_noninteractive_command(command)
                if interactive_error:
                    log.warning("Task %s: %s", task.id, interactive_error)
                    step_log.append({
                        "command": command,
                        "output": interactive_error,
                        "exit_code": None,
                        "timed_out": False,
                    })
                    self._last_step_log = list(step_log)
                    continue
                command = self._disable_systemctl_pager(command)
                log.info("Task %s: running `%s`", task.id, command)
                try:
                    result = self._tmux.run_command(
                        target,
                        command,
                        timeout=cfg.SHELL_COMMAND_TIMEOUT,
                    )
                except TmuxError as e:
                    log.error(
                        "Task %s: terminal state is unknown after tmux error; stopping safely: %s",
                        task.id,
                        e,
                    )
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": f"Terminal state could not be verified after tmux error: {e}",
                        "safety_blocked": True,
                        "safety_halt": True,
                    }

                if result.get("timed_out"):
                    log.warning(
                        "Task %s: command timed out; tmux interruption is handled by the command runner.",
                        task.id,
                    )
                if result.get("safety_interrupted"):
                    step_log.append({
                        "command": command,
                        "output": result["output"],
                        "exit_code": result.get("exit_code"),
                        "timed_out": result.get("timed_out", False),
                        "safety_interrupted": True,
                    })
                    self._last_step_log = list(step_log)
                    return {
                        "status": "failed",
                        "result": "",
                        "error": (
                            "Nyx canceled an unsafe interactive confirmation and stopped "
                            "this task without choosing an option."
                        ),
                        "safety_blocked": True,
                        "safety_halt": result.get("safety_halt", False),
                    }

                if result.get("human_input_needed"):
                    input_info = detect_human_input_need(result.get("output", ""))
                    if input_info is not None:
                        log.warning("Task %s: command is waiting for %s input in tmux pane", task.id, input_info["kind"])
                        is_password_prompt = re.search(
                            r"\b(?:password|passphrase|secret)\b",
                            input_info["prompt"],
                            re.IGNORECASE,
                        )
                        is_sudo_password_prompt = bool(
                            re.search(
                                r"(?:\[sudo\].{0,80}(?:password|passphrase)|"
                                r"(?:password|passphrase).{0,80}\bsudo\b)",
                                input_info["prompt"],
                                re.IGNORECASE | re.DOTALL,
                            )
                        )
                        if (
                            input_info["kind"] == "credential"
                            and is_password_prompt
                            and is_sudo_password_prompt
                        ):
                            sudo_password = getattr(self, "_sudo_password", None)
                            if sudo_password is None:
                                try:
                                    self._tmux.send_raw_keys(target, "C-c", enter=False)
                                except TmuxError as e:
                                    log.error(
                                        "Task %s: could not cancel sudo password prompt: %s",
                                        task.id,
                                        e,
                                    )
                                    return {
                                        "status": "failed",
                                        "result": "",
                                        "error": (
                                            "Could not cancel the sudo password prompt; "
                                            f"terminal state is unknown: {e}"
                                        ),
                                        "safety_halt": True,
                                    }
                                step_log.append({
                                    "command": command,
                                    "output": (
                                        "Sudo password prompt canceled because no --password "
                                        "value was supplied."
                                    ),
                                    "exit_code": 130,
                                    "timed_out": False,
                                    "credential_blocked": True,
                                })
                                self._last_step_log = list(step_log)
                                return {
                                    "status": "failed",
                                    "result": "",
                                    "error": (
                                        "A sudo password is required. Run Nyx with "
                                        "--password PASSWORD to enter it automatically."
                                    ),
                                }
                            human_response = sudo_password
                        else:
                            human_response = request_human_input(
                                input_info["kind"],
                                input_info["prompt"],
                            )
                        try:
                            if (
                                input_info["kind"] == "credential"
                                and is_password_prompt
                                and is_sudo_password_prompt
                            ):
                                self._tmux.send_secret(target, human_response)
                            else:
                                self._tmux.send_raw_keys(target, human_response)
                        except TmuxError as e:
                            log.error("Task %s: failed to send prompted response: %s", task.id, e)
                            return {"status": "failed", "result": "", "error": str(e)}
                        step_log.append({
                            "command": command,
                            "output": f"Prompt required human {input_info['kind']} input: {input_info['prompt']}",
                            "exit_code": None,
                            "timed_out": False,
                        })
                        continue

                step_log.append({
                    "command": command,
                    "output": (
                        f"{result['output']}\nDocker port conflict detected. "
                        "Retry the same container with a different free HOST_PORT "
                        "while keeping CONTAINER_PORT unchanged; do not investigate "
                        "or modify host web-server services."
                        if self._is_docker_setup_task(task.description)
                        and self._docker_port_conflict(str(result.get("output", "")))
                        else result["output"]
                    ),
                    "exit_code": result["exit_code"],
                    "timed_out": result["timed_out"],
                })
                self._last_step_log = list(step_log)
                if (
                    self._is_directory_creation_task(task.description)
                    and self._directory_exists_conflict(str(result.get("output", "")))
                ):
                    error = (
                        f"The directory creation command reported a name conflict: "
                        f"{str(result.get('output', '')).strip()}. Report this to the "
                        "planner, which must choose a different unused directory name "
                        "and update all dependent task paths consistently."
                    )
                    return {"status": "failed", "result": "", "error": error}
                if (
                    self._is_basic_connectivity_task(task.description)
                    and self._has_successful_ping(step_log)
                ):
                    result_text = "Network connectivity confirmed by a successful ping."
                    reasoning.info("Task %s: successful ping confirms basic connectivity; stopping further checks.", task.id)
                    return {"status": "completed", "result": result_text}
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
        step_log: list[dict],
        step: int,
    ) -> dict | None:
        guidance = self._control_checkpoint(task, f"Waiting for AI decision ({step})")
        prompt = build_step_prompt(
            task_id=task.id,
            task_description=task.description,
            step_log=step_log,
            steps_taken=len(step_log),
            max_steps=cfg.MAX_SHELL_STEPS_PER_TASK,
            remote_context=self._remote_context,
            guidance_text=guidance,
            system_context=getattr(self, "_system_context", ""),
            requires_file_write=self._requires_file_write(task.description),
            docker_setup=self._is_docker_setup_task(task.description),
            compose_requested=self._explicitly_requests_compose(task.description),
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

        self._control_checkpoint(task, f"AI decision received ({step})")
        parsed = parse_json_response(raw)
        if parsed is None or not isinstance(parsed, dict):
            log.warning("Executor step %d: unparseable response: %s", step, raw[:300])
            return None

        return parsed

    @staticmethod
    def _format_system_context(context: dict[str, str]) -> str:
        return "\n".join(f"{key}: {value}" for key, value in context.items())

    def is_available(self) -> bool:
        return self._client.health_check()
