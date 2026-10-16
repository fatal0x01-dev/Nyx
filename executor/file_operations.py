"""Build safe remote commands for complete text-file reads and atomic writes."""

from __future__ import annotations

import base64
import shlex

MAX_FILE_BYTES = 48 * 1024
MISSING_FILE_SENTINEL = "__NYX_FILE_MISSING__"

READ_FILE_SCRIPT = """import base64, os, stat, sys
path = sys.argv[1]
if os.path.islink(path):
    raise SystemExit("Refusing to read a symlink")
if not os.path.exists(path):
    sys.stdout.write("__NYX_FILE_MISSING__")
    raise SystemExit(0)
metadata = os.stat(path)
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("Target is not a regular file")
with open(path, "rb") as source:
    data = source.read(49153)
if len(data) > 49152:
    raise SystemExit("File exceeds Nyx's 48 KiB text-file limit")
try:
    data.decode("utf-8")
except UnicodeDecodeError:
    raise SystemExit("File is not valid UTF-8 text")
sys.stdout.write(base64.b64encode(data).decode("ascii"))
"""

WRITE_FILE_SCRIPT = """import base64, os, stat, sys, tempfile
path = os.path.abspath(sys.argv[1])
data = base64.b64decode(sys.argv[2], validate=True)
if len(data) > 49152:
    raise SystemExit("File exceeds Nyx's 48 KiB text-file limit")
data.decode("utf-8")
parent = os.path.dirname(path)
if not os.path.isdir(parent):
    raise SystemExit("Parent directory does not exist")
if os.path.islink(path):
    raise SystemExit("Refusing to replace a symlink")
try:
    metadata = os.lstat(path)
except FileNotFoundError:
    metadata = None
if metadata is not None and not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("Target is not a regular file")
if metadata is None:
    current_umask = os.umask(0)
    os.umask(current_umask)
    mode = 0o666 & ~current_umask
else:
    mode = stat.S_IMODE(metadata.st_mode)
fd, temporary = tempfile.mkstemp(prefix=".nyx-write-", dir=parent)
try:
    with os.fdopen(fd, "wb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print("Atomic file replacement completed")
"""


def build_read_file_command(path: str) -> str:
    if not path or "\x00" in path:
        raise ValueError("A valid file path is required")
    return _build_python_script_command(READ_FILE_SCRIPT, path)


def decode_read_file_output(output: str) -> str:
    try:
        content = base64.b64decode("".join(output.split()), validate=True)
        return content.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not decode complete UTF-8 file contents: {error}") from error


def build_write_file_command(path: str, content: str) -> str:
    if not path or "\x00" in path:
        raise ValueError("A valid file path is required")
    if "\x00" in content:
        raise ValueError("NUL bytes are not supported in text files")
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_FILE_BYTES:
        raise ValueError(
            f"File content exceeds the 48 KiB ({MAX_FILE_BYTES}-byte) limit"
        )
    payload = base64.b64encode(content_bytes).decode("ascii")
    command = _build_python_script_command(WRITE_FILE_SCRIPT, path)
    return f"{command} {shlex.quote(payload)}"


def _build_python_script_command(script: str, *arguments: str) -> str:
    """Encode helper source so tmux submits it as one terminal line."""
    encoded_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
    bootstrap = f"import base64;exec(base64.b64decode({encoded_script!r}))"
    command = f"python3 -c {shlex.quote(bootstrap)}"
    if arguments:
        command += " " + " ".join(shlex.quote(argument) for argument in arguments)
    return command
