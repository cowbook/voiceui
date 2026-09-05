# OpenClaw Voice UI - 海绵宝宝

![SpongeBob](https://github.com/cowbook/voiceui/blob/main/assets/background-text-show.jpeg?raw=true)

把OpenClaw变成能听能说的AI Agent，全双工语音客户端，直接连接 OpenClaw agent。
当前版本的 ASR 固定为阿里云通义（DashScope Paraformer）。

平台说明：
- 语音采集、VAD、阿里云 ASR、OpenClaw 对话链路是跨平台能力（macOS / Linux / Windows）。
- TTS 已支持跨平台自动选择播放器：macOS 使用 `afplay`，Linux 自动探测 `ffplay/mpv/mpg123/mpg321/cvlc`，Windows 使用 `powershell` 媒体播放。
- 若系统缺少可用播放器，语音识别与对话仍可用，但不会有语音播报。


## 功能

- 语音采集 + VAD 自动分段
- 阿里云通义实时识别（流式 partial + final）
- TTS 播报与双工打断
- 空格按住说话（PTT）
- 对话回显与状态可视化

## 安装与启动

前置条件：已安装并可执行 `openclaw`（在 PATH 中）、`python3`、`bash`。

1. 一键安装并启动：

```bash
curl -fsSL https://raw.githubusercontent.com/cowbook/voiceui/main/installer/bootstrap.sh | bash
```

PowerShell 一键安装并启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/cowbook/voiceui/main/installer/bootstrap.ps1 | iex"
```

## 启动软件

```bash
cd ~/.openclaw/apps/voiceui/installer && ./run.sh
```


## 源代码安装

本项目是独立桌面程序，通过 OpenClaw CLI 完成对话能力接入。

说明：
- 如果未设置 `DASHSCOPE_API_KEY`，脚本会提示输入（仅当前终端会话生效）。
- 阿里云百炼 API Key 申请地址：https://bailian.console.aliyun.com/


```bash
git clone https://github.com/cowbook/voiceui.git && cd voiceui
cd installer
export DASHSCOPE_API_KEY='sk-...'
./install.sh
```

安装脚本会自动：检查 `openclaw`、创建 `.venv`、安装依赖、检查 TTS 播放器可用性。

## TTS音色

可选预设：`spongebob-lite`（默认）、`cartoon-bright`、`cartoon-energetic`、`calm`

直接覆盖参数tts-rate、tts-pitch，以更改音色：

```bash
.venv/bin/python voiceui.py --tts-preset cartoon-energetic --tts-rate +18% --tts-pitch +9Hz
```


## 使用

1. 启动后允许系统麦克风权限（macOS / Linux / Windows）
2. 点击开启语音，或按住空格说话
3. 说话结束后自动转写并发送给 OpenClaw agent
4. AI 回答时可直接开口打断

## 运行链路

1. 录音：sounddevice 采集 16k 单声道
2. 分段：VAD + dB 阈值决定语音起止
3. 识别：阿里云流式 ASR 输出 partial/final
4. 回复：openclaw agent 生成文本，TTS 播放

## 项目文件

- voiceui.py：主界面与音频状态机
- streaming_asr.py：阿里云实时 ASR WebSocket 客户端
- run.sh：依赖检查与启动脚本
- diagnose.py：麦克风/VAD/云端 ASR 诊断脚本

## 常见问题

- 启动无识别：先确认 DASHSCOPE_API_KEY 已设置
- 没有声音输入：检查系统麦克风权限与输入设备
- Linux 无法播报：请安装 `ffplay`、`mpv`、`mpg123`、`mpg321` 或 `cvlc` 中任意一个
- Windows 无法播报：请确认 `powershell` 可用，且系统具备 .NET 媒体组件
- 可说话但识别为空：提高输入音量，或下调阈值滑杆
