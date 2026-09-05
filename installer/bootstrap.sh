#!/usr/bin/env bash
set -euo pipefail

REPO_TARBALL_URL="https://github.com/cowbook/voiceui/archive/refs/heads/main.tar.gz"
TARGET_DIR="${VOICEUI_HOME:-$HOME/.openclaw/apps/voiceui}"

if ! command -v tar >/dev/null 2>&1; then
  echo "[voiceui-bootstrap][error] tar not found." >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  FETCH_CMD=(curl -fsSL "$REPO_TARBALL_URL")
elif command -v wget >/dev/null 2>&1; then
  FETCH_CMD=(wget -qO- "$REPO_TARBALL_URL")
else
  echo "[voiceui-bootstrap][error] neither curl nor wget found." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[voiceui-bootstrap] Downloading latest source..."
"${FETCH_CMD[@]}" > "$TMP_DIR/voiceui.tar.gz"

echo "[voiceui-bootstrap] Extracting..."
tar -xzf "$TMP_DIR/voiceui.tar.gz" -C "$TMP_DIR"
SRC_DIR="$TMP_DIR/voiceui-main"

if [[ ! -d "$SRC_DIR/installer" ]]; then
  echo "[voiceui-bootstrap][error] installer directory not found in archive." >&2
  exit 1
fi

if [[ -e "$TARGET_DIR" ]]; then
  BACKUP_DIR="${TARGET_DIR}.bak.$(date +%Y%m%d%H%M%S)"
  echo "[voiceui-bootstrap] Existing install found. Backup -> $BACKUP_DIR"
  mv "$TARGET_DIR" "$BACKUP_DIR"
fi

mkdir -p "$(dirname "$TARGET_DIR")"
mv "$SRC_DIR" "$TARGET_DIR"

echo "[voiceui-bootstrap] Installed to $TARGET_DIR"
cd "$TARGET_DIR/installer"
exec ./one-click.sh "$@"
