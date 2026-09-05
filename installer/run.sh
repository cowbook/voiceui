#!/usr/bin/env bash
set -euo pipefail

INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICEUI_DIR="$(cd "$INSTALLER_DIR/.." && pwd)"
VENV_DIR="$VOICEUI_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[voiceui-installer][error] Virtual environment missing. Run install.sh first." >&2
  exit 1
fi

if ! command -v openclaw >/dev/null 2>&1; then
  echo "[voiceui-installer][error] openclaw command not found in PATH." >&2
  echo "[voiceui-installer][error] Please install OpenClaw before running Voice UI." >&2
  exit 1
fi

cd "$VOICEUI_DIR"
exec "$VENV_DIR/bin/python" "$VOICEUI_DIR/voiceui.py" "$@"
