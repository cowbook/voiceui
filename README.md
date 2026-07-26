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

## 安装（OpenClaw Skill）

本项目已转成 OpenClaw Skill，适用于任意本地目录，不要求存在 `apps/voiceui` 结构。

0. 先获取项目代码（示例）：

```bash
git clone <你的仓库地址> voiceui
cd voiceui
```

一键安装并启动（推荐）：

```bash
cd voiceui/voiceui-skill
./one-click.sh
```

如果未设置 `DASHSCOPE_API_KEY`，脚本会提示你输入并仅在当前终端会话生效。

1. 进入 skill 目录：

```bash
cd voiceui-skill
```

2. 设置阿里云 API Key（必需）：

```bash
export DASHSCOPE_API_KEY='sk-...'
```

3. 执行安装脚本：

```bash
./install.sh
```

安装脚本会自动：创建 `.venv`、安装依赖、检查 TTS 播放器可用性。

## 启动

```bash
cd voiceui-skill
./run.sh
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
