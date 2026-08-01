#!/usr/bin/env python3
"""
watch_download.py — 守护蟹老板的模型下完了通知

逻辑：
  · 监控 HF cache 里 faster-whisper-large-v3-turbo 的 .incomplete 文件
  · 一旦 .incomplete 都没了，且主权重文件 ≥ 1.4GB，触发加载验证
  · 加载成功 → `shuo` 语音通知 + 写 marker 文件 + 退出
  · 加载失败 → 继续等（可能 HF 还在做最后 commit）
  · 默认每 20 秒轮询一次；最长等 30 分钟自动退出

跑：  ./watch_download.py
"""
import os, sys, time, subprocess, signal
from pathlib import Path

HF_DIR = Path.home() / ".cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/blobs"
MARKER = Path("/tmp/voiceui_whisper_ready.marker")
LOG    = Path(__file__).parent / "logs" / "download_watch.log"
SHUO   = Path.home() / ".openclaw/workspace/scripts/shuo"
APP    = Path(__file__).parent
VENV   = APP / ".venv"
MAX_WAIT_S = 30 * 60

# 让 log 写出，父进程 detach 后也能写
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def is_incomplete_present() -> bool:
    return any(HF_DIR.glob("*.incomplete"))

def main_weight_size() -> int:
    """找最大的（>10MB）blob——主权重"""
    sizes = [(p.stat().st_size, p.name) for p in HF_DIR.glob("*") if p.is_file() and not p.suffix == ".incomplete" and p.name == p.name]
    sizes = [(s, n) for (s, n) in sizes if s > 10_000_000]
    return max((s for (s, _) in sizes), default=0)

def try_load():
    """试一次 import + 实例化模型，看是否能加载"""
    try:
        # 使用 venv 的 python
        py = str(VENV / "bin/python3")
        r = subprocess.run(
            [py, "-c",
             "from faster_whisper import WhisperModel; "
             "import time; t=time.time(); "
             "m = WhisperModel('turbo', device='cpu', compute_type='int8'); "
             "import numpy as np; "
             "segs, _ = m.transcribe(np.zeros(16000, dtype='float32'), language='zh', vad_filter=False); "
             "_=list(segs); "
             f"print('OK_LOAD', time.time()-t)"],
            capture_output=True, text=True, timeout=300,
        )
        ok = "OK_LOAD" in r.stdout
        log(f"  load.py stdout: {r.stdout.strip()[:200]}")
        if r.stderr.strip():
            log(f"  load.py stderr: {r.stderr.strip()[:300]}")
        return ok
    except subprocess.TimeoutExpired:
        log("  load.py 超时")
        return False
    except Exception as e:
        log(f"  load.py 异常: {e}")
        return False

def notify():
    msg = "蟹老板，海绵宝宝报告：Whisper turbo 模型下载并加载完成！STT 全栈就绪，说话就能识别。"
    try:
        SHUO_PY = SHUO
        if SHUO_PY.exists():
            subprocess.run([str(SHUO_PY), msg], check=False, timeout=15)
        else:
            # 退化到 say
            subprocess.run(["say", "-v", "Tingting", "-r", "200", msg], check=False, timeout=15)
    except Exception as e:
        log(f"  notify 失败: {e}")

def main():
    t0 = time.time()
    log(f"👀 开始守模型下载 (HF_DIR={HF_DIR})")
    log(f"   当前 .incomplete 数: {len(list(HF_DIR.glob('*.incomplete')))}")
    log(f"   当前最大 blob: {main_weight_size() // 1024 // 1024} MB")
    
    last_report = 0
    while time.time() - t0 < MAX_WAIT_S:
        if is_incomplete_present():
            # 还在下，进度汇报（每 2 分钟一次）
            now = time.time()
            if now - last_report > 120:
                incomp = sum(p.stat().st_size for p in HF_DIR.glob("*.incomplete"))
                log(f"   ⏳ 还在下：{incomp // 1024 // 1024} MB 未完成 / 主权重 {main_weight_size() // 1024 // 1024} MB")
                last_report = now
            time.sleep(20)
            continue
        
        # .incomplete 没了，检查权重
        mw = main_weight_size()
        log(f"   ✨ .incomplete 已清，权重 {mw // 1024 // 1024} MB，开始验证加载...")
        if mw < 1_400_000_000:  # 小于 1.4GB
            log(f"   权重还小，再等等...")
            time.sleep(20)
            continue
        
        # 试加载
        log("   🧪 验证模型可用...")
        if try_load():
            log("✅✅✅ 模型就绪！")
            MARKER.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            notify()
            log("🔔 已语音通知")
            return 0
        else:
            log("   加载失败，等 30s 再试...")
            time.sleep(30)
    
    log("⚠ 超过 30 分钟，自动退出")
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("🛑 用户中断")
        sys.exit(2)
