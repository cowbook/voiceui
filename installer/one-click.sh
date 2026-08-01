#!/usr/bin/env bash
set -euo pipefail

INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "[voiceui-installer] DASHSCOPE_API_KEY is not set."
  read -r -s -p "Please enter DASHSCOPE_API_KEY: " DASH_KEY
  echo
  if [[ -z "$DASH_KEY" ]]; then
    echo "[voiceui-installer][error] Empty DASHSCOPE_API_KEY. Abort." >&2
    exit 1
  fi
  export DASHSCOPE_API_KEY="$DASH_KEY"
fi

"$INSTALLER_DIR/install.sh"
exec "$INSTALLER_DIR/run.sh" "$@"
