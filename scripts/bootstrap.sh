#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
REQUIREMENTS_MARKER="$VENV_DIR/.requirements.cksum"
REQUIREMENTS_FINGERPRINT="$(cksum "$REQUIREMENTS_FILE")"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3.12 -m venv "$VENV_DIR"
fi

if [[ ! -f "$REQUIREMENTS_MARKER" ]] || \
   [[ "$(<"$REQUIREMENTS_MARKER")" != "$REQUIREMENTS_FINGERPRINT" ]]; then
    "$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
    printf '%s\n' "$REQUIREMENTS_FINGERPRINT" > "$REQUIREMENTS_MARKER"
else
    echo "dependency-cache=valid"
fi
"$VENV_DIR/bin/python" -m pip check
