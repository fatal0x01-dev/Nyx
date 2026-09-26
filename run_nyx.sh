#!/usr/bin/env bash
# Launch Nyx from the project root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip install httpx pydantic rich python-dotenv pyyaml tenacity pytest >/dev/null 2>&1 || true

cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
  python orchestrator.py "$@"
else
  python orchestrator.py
fi
