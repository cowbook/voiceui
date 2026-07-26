# VoiceUI Skill

OpenClaw Skill for the SpongeBob Voice UI app.

## What this skill installs

- Python virtual environment under `apps/voiceui/.venv`
- All dependencies from `apps/voiceui/requirements.txt`
- Cross-platform TTS playback support through the app runtime

## Install

Quick one-click install + run:

```bash
cd voiceui-skill
./one-click.sh
```

If `DASHSCOPE_API_KEY` is not exported, the script will prompt for it.

Manual install:

```bash
cd ~/.openclaw/workspace/apps/voiceui/voiceui-skill
cp .env.example .env
# Edit .env and set DASHSCOPE_API_KEY, then:
export DASHSCOPE_API_KEY='sk-...'
./install.sh
```

## Start

```bash
cd ~/.openclaw/workspace/apps/voiceui/voiceui-skill
./run.sh
```

## Notes

- macOS: allow microphone permission when prompted.
- Linux: install one player if missing (`ffplay`, `mpv`, `mpg123`, `mpg321`, or `vlc`).
- Windows: ensure `powershell` is available for playback.
