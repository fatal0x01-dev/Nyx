"""
Misc helper functions used across the project.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    """Generate a short unique task ID."""
    return uuid.uuid4().hex[:8]


def utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_json_response(text: str) -> dict | list | None:
    """
    Attempt to extract and parse a JSON object or array from an LLM response.
    The model sometimes wraps JSON in markdown fences — this strips them first.
    """
    # Strip markdown code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} or [...] block
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def normalize_multiline_write_command(command: str) -> str:
    """Convert escaped-newline shell writes into a heredoc write that preserves real line breaks."""
    if not command:
        return command

    # Match patterns like: echo 'line1\nline2\n' > file
    match = re.search(r"echo\s+['\"](.+?)['\"]\s*>\s*([^\s]+)", command)
    if match:
        content = match.group(1)
        target = match.group(2)
        normalized = content.replace('\\n', '\n').replace('\\t', '\t')
        return f"cat > {target} <<'EOF'\n{normalized}\nEOF"

    # Leave other commands alone.
    return command


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate a string for safe logging."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… [+{len(text) - max_chars} chars]"


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Convert a list of task history dicts into a readable string for LLM prompts.
    """
    if not history:
        return "No tasks completed yet."

    lines = []
    for item in history:
        lines.append(
            f"[{item.get('status', '?').upper()}] Task {item.get('id', '?')}: "
            f"{item.get('description', '')}\n"
            f"  Result: {truncate(str(item.get('result', '')), 300)}"
        )
    return "\n\n".join(lines)