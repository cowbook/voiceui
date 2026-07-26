#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICEUI_DIR="$(cd "$SKILL_DIR/.." && pwd)"
VENV_DIR="$VOICEUI_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[voiceui-skill][error] Virtual environment missing. Run install.sh first." >&2
  exit 1
fi

cd "$VOICEUI_DIR"
exec "$VENV_DIR/bin/python" "$VOICEUI_DIR/voiceui.py" "$@"
