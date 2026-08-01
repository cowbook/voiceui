# VoiceUI Installer

Independent installer/launcher scripts for the SpongeBob Voice UI app.

## What it installs

- Python virtual environment under `.venv`
- All dependencies from `requirements.txt`
- Runtime prerequisites check (`openclaw`, audio player)

## Quick start

```bash
cd installer
./install.sh
./run.sh
```

If `DASHSCOPE_API_KEY` is not exported, set it first before install/start.

## Manual install

```bash
cd installer
export DASHSCOPE_API_KEY='sk-...'
./install.sh
```

## Start

```bash
cd installer
./run.sh
```

## Notes

- macOS: allow microphone permission when prompted.
- Linux: install one player if missing (`ffplay`, `mpv`, `mpg123`, `mpg321`, or `vlc`).
- Windows: ensure `powershell` is available for playback.
