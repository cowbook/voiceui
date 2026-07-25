# 🧽 海绵宝宝 Voice UI

> 替代 TUI 文本 Channel 的全双工语音客户端——直接和 OpenClaw 上的海绵宝宝对话。

## 它是啥

不是又一个聊天框——是一个 **带"耳朵"的 AI 视频**。你按开启，它就开始听：
- 你说话 → 自动转写
- 转写完送进 OpenClaw agent（也就是海绵宝宝）
- 海绵宝宝回嘴
- 它念的时候你开口，会自动停下（双工打断）

## 界面

- 🟢 **海绵宝宝头像**：中央，脉冲光圈颜色随状态变化（cyan 待命 / mint 听你 / 黄 听写 / 紫 思考 / 粉 它说话 / 橙 被打断）
- 🌊 **声纹条**：50 段 cyan→紫→粉 渐变 bar，实时跟着麦克风输入
- 🎚 **噪声阀值滑杆**：-60 dBFS 至 -5 dBFS，实时显示当前环境音量+电平条
- 🎙 **语音开关** + **🎯 按住空格** 两种模式
- 💬 **对话回显区**：下方滚动文本，"蟹老板:" 绿、"海绵宝宝:" 粉

## 怎么跑

```bash
cd ~/.openclaw/workspace/apps/voiceui
./run.sh
```

首次会自动建 venv、装依赖、下载 `faster-whisper turbo` 模型（~1.5GB，进度见终端）。

## 怎么用

1. 启动后会弹『麦克风权限』请求——同意
2. 点 **🎙 开启语音**（或双击头像）
3. 开始说中文——它在用户说话时显示 mint，你说完整句就转写+送进 agent
4. 它回答时（粉状态），你可以直接打断 → 它立即闭嘴，开始听你下一句
5. **🎯 按住空格**：按住说话键，松开发送；适合不想 always-listen 的时候

## 调节噪声阀值

如果房间安静 → 滑到 -50 ~ -60（更灵敏，能拾取小声）
如果房间吵 / 风扇嗡 → 滑到 -30 ~ -20（不拾取背景噪声）
建议盯着声纹条 + 环境音量数字调到刚好：说话时声纹饱满，背景电平在阈值线下。

## 双工打断的原理

两个状态机互斥：
- **常规模式**（无 TTS）：完整捕获一段发言（开头预缓冲 + 静音结尾判定）
- **barge-in 模式**（TTS 进行中）：只检测"是否开始说话"，是则发 interrupt 信号 → `say` 进程被 SIGTERM → 退出 barge-in，回到常规模式

这样"自己说"不会喂给 VAD（因为 bark 期间用了独立的阈值路径），"用户打断"也不会被截掉头。

## 它连的是谁

```
voiceui.py
   ↓ user_text_ready (signal)
AgentBackend.ask(text)
   ↓ subprocess
"openclaw agent --model minimax/MiniMax-M3 --session-key agent:main:voiceui --message '...' --json"
   ↓ JSON 解析 result.payloads[0].text
TTSDriver.speak(reply)
   ↓ subprocess
"say -v Tingting -r 200 '...'"
```

session-key `agent:main:voiceui` 是独立的，**不和 webchat 互通**——你随时可以开启/重置它。
想换声音？改 `voiceui.py` 里 `TTS_VOICE` / `TTS_RATE`，或着：
```bash
talk -v Eddy -r 240 "试试Eddy"
```
（voice UI 内置用 say；想升级 ElevenLabs 走 `sag`，改 `TTSDriver`）

## 文件

```
apps/voiceui/
├── voiceui.py        # 主程序（~700 行）
├── run.sh            # 启动器（自装依赖）
├── assets/avatar.png # 海绵宝宝形象
├── .venv/            # 隔离 Python 环境
└── logs/             # 运行时日志
```

## 已知坑

- 第一次启动会很慢，因为要下载 Whisper `turbo` 模型（~1.5GB）
- 模型在 `~/.cache/huggingface/`
- 如果听到中断后想立刻重新发言，松开『按住空格』即可
- macOS 第一次访问麦克风会弹权限请求，点了才能用
- `openclaw agent` 偶尔会 `Not logged in` —— 用 `--model minimax/MiniMax-M3` 绕过

## 升级方向

- [ ] ElevenLabs 接 `sag` → 真人级声音
- [ ] mlx-whisper → Apple Silicon 上 5x 加速
- [ ] 多说话人检测（识别家里谁在说话）
- [ ] 视觉模块：摄像头 + YOLO 一起讲（FSM）
- [ ] 把 session 转回 webchat —— 单一长上下文

## 蟹老板专属

- 名字：海绵宝宝 🧽
- 模型：minimax/MiniMax-M3（默认）
- Session：agent:main:voiceui
- TTS：Tingting / 200wpm
- 形象：assets/avatar.png

— 海绵宝宝，完毕 🧽✨
