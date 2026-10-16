import base64
import os
import shlex
import subprocess

import pytest

from executor.file_operations import (
    MAX_FILE_BYTES,
    MISSING_FILE_SENTINEL,
    build_read_file_command,
    build_write_file_command,
    decode_read_file_output,
)


def _run_generated_command(
    command: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        shlex.split(command),
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )


def test_atomic_write_replaces_complete_multiline_file_and_preserves_mode(tmp_path):
    path = tmp_path / "settings.ini"
    path.write_text("first=old\nsecond=old\nthird=keep\n", encoding="utf-8")
    path.chmod(0o640)
    new_contents = "first=new\nsecond=updated\nthird=keep\nfourth=added\n"

    command = build_write_file_command(str(path), new_contents)
    assert "\n" not in command
    result = _run_generated_command(command)

    assert result.returncode == 0, result.stderr
    assert "Atomic file replacement completed" in result.stdout
    assert path.read_text(encoding="utf-8") == new_contents
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.glob(".nyx-write-*")) == []


def test_read_file_returns_complete_utf8_content_without_losing_newlines(tmp_path):
    path = tmp_path / "multiline.txt"
    contents = "\n first line \nsecond line\n\n"
    path.write_text(contents, encoding="utf-8")

    command = build_read_file_command(str(path))
    assert "\n" not in command
    result = _run_generated_command(command)

    assert result.returncode == 0, result.stderr
    assert decode_read_file_output(result.stdout) == contents


def test_read_file_reports_missing_target_for_creation_tasks(tmp_path):
    path = tmp_path / "new-config.yml"

    command = build_read_file_command(str(path))
    assert "\n" not in command
    result = _run_generated_command(command)

    assert result.returncode == 0
    assert result.stdout == MISSING_FILE_SENTINEL


def test_file_operations_expand_home_relative_paths(tmp_path):
    home = tmp_path / "remote-home"
    project = home / "web-server"
    project.mkdir(parents=True)
    path = "~/web-server/Dockerfile"
    contents = "FROM httpd:latest\n"
    environment = {**os.environ, "HOME": str(home)}

    write_result = _run_generated_command(
        build_write_file_command(path, contents),
        env=environment,
    )
    read_result = _run_generated_command(
        build_read_file_command(path),
        env=environment,
    )

    assert write_result.returncode == 0, write_result.stderr
    assert read_result.returncode == 0, read_result.stderr
    assert decode_read_file_output(read_result.stdout) == contents


def test_write_file_refuses_symlink_target(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = _run_generated_command(
        build_write_file_command(str(link), "replacement")
    )

    assert result.returncode != 0
    assert "Refusing to replace a symlink" in result.stderr
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_write_file_enforces_utf8_text_size_and_nul_limits(tmp_path):
    path = tmp_path / "large.txt"

    with pytest.raises(ValueError, match="48 KiB"):
        build_write_file_command(str(path), "x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="NUL"):
        build_write_file_command(str(path), "bad\x00text")


def test_write_file_is_safe_for_shell_metacharacters_in_path_and_content(tmp_path):
    path = tmp_path / "config ; $(touch pwned).txt"
    contents = "line with 'quotes' and $() ;\nsecond line\n"

    result = _run_generated_command(build_write_file_command(str(path), contents))

    assert result.returncode == 0, result.stderr
    assert path.read_text(encoding="utf-8") == contents
    assert not (tmp_path / "pwned").exists()


def test_file_read_base64_round_trip_preserves_unicode():
    content = "naïve 雪\n"
    output = base64.b64encode(content.encode("utf-8")).decode("ascii")

    assert decode_read_file_output(output) == content
