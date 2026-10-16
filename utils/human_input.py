from __future__ import annotations

import re
import sys
import getpass
from typing import Any

APPROVAL_PATTERNS = [
    re.compile(r"continue\??\s*(?:\[\s*y/?n\s*\]|\(\s*y/?n\s*\)|y/n|yes/no)", re.IGNORECASE),
    re.compile(r"are you sure.*(?:yes/no|fingerprint)", re.IGNORECASE),
    re.compile(r"overwrite.*(?:y/n|yes/no)", re.IGNORECASE),
    re.compile(r"proceed.*(?:y/n|yes/no)", re.IGNORECASE),
    re.compile(r"confirm.*(?:y/n|yes/no)", re.IGNORECASE),
]

PACKAGE_INSTALL_PATTERN = re.compile(
    r"can be installed with:\s*[\s\S]*?\bapt(?:-get)?\s+install\b"
    r"[\s\S]*?do you want to install it\?\s*\(\s*n\s*/\s*y\s*\)",
    re.IGNORECASE,
)

CREDENTIAL_PATTERNS = [
    re.compile(r"(?:password|passphrase|secret)\s*(?:required|for|to|:)", re.IGNORECASE),
    re.compile(r"(?:password|passphrase|secret)\s+is\s+required", re.IGNORECASE),
    re.compile(r"enter .*?(?:password|passphrase|secret)", re.IGNORECASE),
    re.compile(r"(?:username|login|user name)\s*[:?]", re.IGNORECASE),
]

UNSAFE_CONFIRMATION_PATTERNS = [
    re.compile(
        r"(?:\(\s*i\s*\)gnore|ignore)\s*/\s*(?:\(\s*c\s*\)ancel|cancel)\s*\??",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\(\s*f\s*\)ix|fix|repair)\s*/\s*(?:\(\s*i\s*\)gnore|ignore)\s*\??",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:write|apply)\s+changes?\s*\??", re.IGNORECASE),
    re.compile(r"\bproceed\s+with\s+(?:writing|modifying|formatting)\b", re.IGNORECASE),
]


def detect_human_input_need(output: str) -> dict[str, Any] | None:
    """Return a structured human-input request if terminal output is waiting for input."""
    if not output:
        return None

    cleaned = output.strip()
    if not cleaned:
        return None

    for pattern in UNSAFE_CONFIRMATION_PATTERNS:
        if pattern.search(cleaned):
            return {
                "kind": "unsafe_confirmation",
                "prompt": cleaned[:240],
                "field": "confirmation",
            }

    if PACKAGE_INSTALL_PATTERN.search(cleaned):
        return {
            "kind": "package_install_confirmation",
            "prompt": cleaned[-240:],
            "field": "package_install",
        }

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
        if kind == "credential":
            answer = getpass.getpass("Enter credential (input hidden): ").strip()
        else:
            answer = input("Enter response [y/n]: ").strip()
        if kind == "approval":
            normalized = answer.casefold()
            if normalized in {"y", "yes", "approve", "approved"}:
                return "y"
            if normalized in {"n", "no", "deny", "denied"}:
                return "n"
            print("Please answer y/yes or n/no.", file=sys.stderr)
            continue
        if answer:
            return answer
        print("A response is required. Please enter a value.", file=sys.stderr)
