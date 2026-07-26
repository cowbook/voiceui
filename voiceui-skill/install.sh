#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICEUI_DIR="$(cd "$SKILL_DIR/.." && pwd)"
VENV_DIR="$VOICEUI_DIR/.venv"

print_ok() { printf "[voiceui-skill] %s\n" "$1"; }
print_warn() { printf "[voiceui-skill][warn] %s\n" "$1"; }
print_err() { printf "[voiceui-skill][error] %s\n" "$1" >&2; }

if ! command -v python3 >/dev/null 2>&1; then
  print_err "python3 not found. Please install Python 3.11+ first."
  exit 1
fi

PY_VER="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
print_ok "Detected Python $PY_VER"

if [[ ! -d "$VENV_DIR" ]]; then
  print_ok "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

print_ok "Installing dependencies"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$VOICEUI_DIR/requirements.txt"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  print_warn "DASHSCOPE_API_KEY is not set. ASR will fail until you export it."
  print_warn "Example: export DASHSCOPE_API_KEY='sk-...'"
else
  print_ok "DASHSCOPE_API_KEY detected"
fi

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
case "$OS_NAME" in
  Darwin)
    if command -v afplay >/dev/null 2>&1; then
      print_ok "TTS player detected: afplay"
    else
      print_warn "afplay not found. TTS playback may fail on macOS."
    fi
    ;;
  Linux)
    if command -v ffplay >/dev/null 2>&1 || \
       command -v mpv >/dev/null 2>&1 || \
       command -v mpg123 >/dev/null 2>&1 || \
       command -v mpg321 >/dev/null 2>&1 || \
       command -v cvlc >/dev/null 2>&1; then
      print_ok "A Linux TTS player was found"
    else
      print_warn "No Linux player found. Install one of: ffplay mpv mpg123 mpg321 vlc"
    fi
    ;;
  *)
    print_warn "If you are on Windows, run inside PowerShell/CMD and ensure powershell is available."
    ;;
esac

print_ok "Install completed. Start with: $SKILL_DIR/run.sh"
