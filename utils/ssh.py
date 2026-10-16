"""SSH connection settings and command construction."""

from __future__ import annotations

import ipaddress
import shlex
from dataclasses import dataclass

import config.settings as cfg


@dataclass(frozen=True)
class SSHConfig:
    ip: str
    username: str = cfg.SSH_DEFAULT_USER
    port: int | None = None
    identity_file: str | None = None
    options: tuple[str, ...] = cfg.SSH_DEFAULT_OPTIONS

    def __post_init__(self) -> None:
        try:
            ipaddress.ip_address(self.ip)
        except ValueError as e:
            raise ValueError("SSH destination must be a valid IP address") from e
        if not self.username or any(char.isspace() for char in self.username) or "@" in self.username:
            raise ValueError("SSH username contains unsupported characters")
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        if any(not option or "\n" in option or "\r" in option for option in self.options):
            raise ValueError("SSH options must be non-empty single-line values")

    @property
    def destination(self) -> str:
        return f"{self.username}@{self.ip}"


def build_ssh_command(config: SSHConfig, start_marker: str, ready_marker: str) -> str:
    """Build a safely quoted command that opens a persistent remote shell."""
    remote_shell = (
        f"printf '\\n%s\\n' {shlex.quote(ready_marker)}; "
        'exec "${SHELL:-/bin/sh}" -i'
    )
    args = ["ssh", "-tt"]
    for option in config.options:
        args.extend(("-o", option))
    if config.port is not None:
        args.extend(("-p", str(config.port)))
    if config.identity_file:
        args.extend(("-i", config.identity_file))
    args.extend((config.destination, remote_shell))

    invocation = shlex.join(args)
    return f"printf '\\n%s\\n' {shlex.quote(start_marker)}; {invocation}"


def describe_ssh_target(config: SSHConfig) -> str:
    """Return non-secret context for planning and task validation."""
    return f"SSH-connected remote host {config.destination}"
