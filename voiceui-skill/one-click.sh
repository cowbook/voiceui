#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "[voiceui-skill] DASHSCOPE_API_KEY is not set."
  read -r -s -p "Please enter DASHSCOPE_API_KEY: " DASH_KEY
  echo
  if [[ -z "$DASH_KEY" ]]; then
    echo "[voiceui-skill][error] Empty DASHSCOPE_API_KEY. Abort." >&2
    exit 1
  fi
  export DASHSCOPE_API_KEY="$DASH_KEY"
fi

"$SKILL_DIR/install.sh"
exec "$SKILL_DIR/run.sh" "$@"
