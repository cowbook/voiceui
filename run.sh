#!/usr/bin/env bash
# run.sh — 启动海绵宝宝语音 UI
#
# 用法：
#   ./run.sh                 # 启动图形界面
#   ./run.sh --check         # 只检查依赖，不启动
#   ./run.sh --model <id>    # 指定对话模型（默认 minimax/MiniMax-M3）
#
# 首次运行会自动 pip install 缺失依赖。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY_BIN="${PY_BIN:-/opt/homebrew/bin/python3}"

# ---- 参数 ----
CHECK_ONLY=0
MODEL="minimax/MiniMax-M3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

# ---- venv ----
if [[ ! -d "$VENV" ]]; then
  echo "🐍 创建 venv..."
  "$PY_BIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- 依赖自检 / 自装 ----
need_install=0
for mod in PySide6 sounddevice numpy silero_vad onnxruntime dashscope websockets edge_tts; do
  if ! python3 -c "import ${mod}" >/dev/null 2>&1; then
    echo "❌ 缺: $mod"
    need_install=1
  fi
done

if [[ $need_install -eq 1 ]]; then
  echo "📦 安装依赖..."
  python3 -m pip install --upgrade pip wheel >/dev/null
  python3 -m pip install --quiet PySide6 sounddevice numpy silero-vad onnxruntime dashscope websockets edge-tts
  echo "✅ 依赖装好"
fi

# ---- 麦克风权限检查（macOS）----
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS 13+ 需要授权；这里只是提示，没有 API 可以无侵入式检查
  echo "🎤 提醒：macOS 会弹窗要求『麦克风』权限，给了才能用。"
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "✅ 依赖 OK，模型: $MODEL"
  exit 0
fi

# ---- 启动 ----
echo "🚀 启动 海绵宝宝 Voice UI · model=$MODEL"
exec python3 "$HERE/voiceui.py" \
  --model "$MODEL" \
  "$@"
