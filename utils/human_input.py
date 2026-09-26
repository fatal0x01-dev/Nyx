from __future__ import annotations

import re
import sys
from typing import Any

APPROVAL_PATTERNS = [
    re.compile(r"continue\??\s*(?:\[\s*y/?n\s*\]|\(\s*y/?n\s*\)|y/n|yes/no)", re.IGNORECASE),
    re.compile(r"overwrite.*(?:y/n|yes/no)", re.IGNORECASE),
    re.compile(r"proceed.*(?:y/n|yes/no)", re.IGNORECASE),
    re.compile(r"confirm.*(?:y/n|yes/no)", re.IGNORECASE),
]

CREDENTIAL_PATTERNS = [
    re.compile(r"(?:password|passphrase|secret)\s*(?:required|for|to|:)", re.IGNORECASE),
    re.compile(r"enter .*?(?:password|passphrase|secret)", re.IGNORECASE),
    re.compile(r"(?:username|login|user name)\s*[:?]", re.IGNORECASE),
]


def detect_human_input_need(output: str) -> dict[str, Any] | None:
    """Return a structured human-input request if terminal output is waiting for input."""
    if not output:
        return None

    cleaned = output.strip()
    if not cleaned:
        return None

    for pattern in APPROVAL_PATTERNS:
        if pattern.search(cleaned):
            return {
                "kind": "approval",
                "prompt": cleaned[:240],
                "field": "approval",
            }

    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(cleaned):
            return {
                "kind": "credential",
                "prompt": cleaned[:240],
                "field": "credential",
            }

    return None


def request_human_input(kind: str, prompt: str) -> str:
    """Ask the user for approval or a credential and return their response."""
    if kind == "approval":
        label = "Approval"
    elif kind == "credential":
        label = "Credential"
    else:
        label = "Input"

    print(f"\nNyx requires {label.lower()} input.", file=sys.stderr)
    print(f"Context: {prompt}", file=sys.stderr)

    while True:
        answer = input("Enter response: ").strip()
        if answer:
            return answer
        print("A response is required. Please enter a value.", file=sys.stderr)
