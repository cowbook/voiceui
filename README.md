# 海绵宝宝 Voice UI

全双工语音客户端，直接连接 OpenClaw agent。
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

## 安装（独立程序）

本项目是独立桌面程序，通过 OpenClaw CLI 完成对话能力接入。
适用于任意本地目录，不要求存在 `apps/voiceui` 结构。

一键安装并启动命令（可直接复制）：

```bash
git clone https://github.com/cowbook/voiceui.git && cd voiceui/installer && ./one-click.sh
```

适用条件：已安装并可执行 `openclaw`（在 PATH 中）、`python3`、`bash`。

说明：`one-click.sh` 会自动创建 `.venv`、安装依赖并启动程序。

0. 先获取项目代码（示例）：

```bash
git clone <你的仓库地址> voiceui
cd voiceui
```

一键安装并启动（推荐）：

```bash
cd voiceui/installer
./one-click.sh
```

如果未设置 `DASHSCOPE_API_KEY`，脚本会提示你输入并仅在当前终端会话生效。

1. 进入安装器目录：

```bash
cd installer
```

2. 设置阿里云 API Key（必需）：

```bash
export DASHSCOPE_API_KEY='sk-...'
```

3. 执行安装脚本：

```bash
./install.sh
```

安装脚本会自动：检查 `openclaw`、创建 `.venv`、安装依赖、检查 TTS 播放器可用性。

## 启动

```bash
cd installer
./run.sh
```

可选：使用“风格接近”TTS 预设（非角色克隆）

```bash
cd voiceui
.venv/bin/python voiceui.py --tts edge --tts-preset spongebob-lite
```

可选预设：`spongebob-lite`、`cartoon-bright`、`cartoon-energetic`、`calm`

默认即 `spongebob-lite`（儿童男声卡通感，接近海绵宝宝风格，参数约为 `+20% / +20Hz`）。

也可覆盖细调参数：

```bash
.venv/bin/python voiceui.py --tts-preset cartoon-energetic --tts-rate +18% --tts-pitch +9Hz
```

## 环境变量

必须设置阿里云百炼 API Key：

```bash
export DASHSCOPE_API_KEY='sk-...'
```

申请地址：
https://bailian.console.aliyun.com/

## 使用

1. 启动后允许系统麦克风权限（macOS / Linux / Windows）
2. 点击开启语音，或按住空格说话
3. 说话结束后自动转写并发送给 OpenClaw agent
4. AI 回答时可直接开口打断

## 运行链路

1. 录音：sounddevice 采集 16k 单声道
2. 分段：VAD + dB 阈值决定语音起止
3. 识别：阿里云流式 ASR 输出 partial/final
4. 兜底：无流式 final 时走阿里云 REST 转写
5. 回复：openclaw agent 生成文本，TTS 播放

## 项目文件

- voiceui.py：主界面与音频状态机
- streaming_asr.py：阿里云实时 ASR WebSocket 客户端
- cloud_asr.py：阿里云 REST 转写
- run.sh：依赖检查与启动脚本
- diagnose.py：麦克风/VAD/云端 ASR 诊断脚本

## 常见问题

- 启动无识别：先确认 DASHSCOPE_API_KEY 已设置
- 没有声音输入：检查系统麦克风权限与输入设备
- Linux 无法播报：请安装 `ffplay`、`mpv`、`mpg123`、`mpg321` 或 `cvlc` 中任意一个
- Windows 无法播报：请确认 `powershell` 可用，且系统具备 .NET 媒体组件
- 可说话但识别为空：提高输入音量，或下调阈值滑杆
