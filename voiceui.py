#!/usr/bin/env python3
"""
voiceui.py — 海绵宝宝 · 语音交互客户端
======================================

功能：
  · 全双工语音对话（代替 TUI 文本 Channel）
  · 实时动态声纹（50 段渐变 bar）
  · 可视化 VAD：滑杆调节噪声阀值，实时显示环境音量
  · 双工打断（barge-in）：蟹老板说话立即停止 AI 语音
  · 不会把自己说的当用户说的（speech 状态机隔离）
  · 科技感皮肤：暗底 + 青/紫/粉霓虹渐变

依赖：
  · PySide6、sounddevice、numpy
  · silero-vad（VAD，轻量 ONNX）
  · faster-whisper（本地 STT，中文友好）
  · macOS `say` 命令（TTS，可被 SIGTERM 杀掉以实现打断）
  · `openclaw agent --json`（连到海绵宝宝——也就是我 😎）

启动：
  ./run.sh
"""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Qt ---
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QSize
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient, QFont,
    QRadialGradient, QPixmap, QFontDatabase, QIcon,
    QShortcut, QKeySequence, QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QTextEdit, QFrame, QSizePolicy,
    QStatusBar, QGraphicsDropShadowEffect, QSystemTrayIcon, QMenu,
)

# --- Audio / ML ---
import sounddevice as sd
import silero_vad
from silero_vad import load_silero_vad, VADIterator
from faster_whisper import WhisperModel

# --- Constants ---------------------------------------------------------------

APP_DIR  = Path(__file__).resolve().parent
ASSETS   = APP_DIR / "assets"
AVATAR   = ASSETS / "avatar.png"
LOG_DIR  = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SESSION_KEY = "agent:main:voiceui"   # 持久化语音 session（与 webchat 隔离）
DEFAULT_MODEL = "minimax/MiniMax-M3" # 默认模型（与 webchat 同源）

# TTS 默认值
TTS_BACKEND_DEFAULT = "edge"     # "edge" 在线 / "say" 本地
EDGE_VOICE_DEFAULT  = "zh-CN-XiaoxiaoNeural"
SAY_VOICE_DEFAULT   = "Tingting"
SAY_RATE_DEFAULT    = "200"

# Audio
SAMPLE_RATE    = 16_000
FRAME_SAMPLES  = 512                 # silero-vad 要求 16kHz 下 512/1024/1536
CHANNELS       = 1

# VAD
SILENCE_END_FRAMES  = 20             # ~640ms 静音结尾（中文说话默认是这样）
MIN_SPEECH_FRAMES   = 4              # ~128ms 最短有效发言
PRE_BUFFER_FRAMES   = 6              # 发言开始前预缓冲 ~200ms，防切音头
DB_FLOOR, DB_CEIL   = -60.0, 0.0     # dB 显示范围

# TTS
TTS_VOICE  = "Tingting"
TTS_RATE   = "200"


# --- Audio worker (QThread) --------------------------------------------------

class AudioWorker(QThread):
    """后台线程：麦克风采集 → VAD → STT；同时支持双工打断（barge-in）"""
    
    level_changed       = Signal(float)  # 0..1 归一化音量（用于波形/电平）
    speech_started      = Signal()
    speech_ended        = Signal()
    partial_text        = Signal(str)    # 实时字幕（每 2s 更新）
    user_text_ready     = Signal(str)
    interrupt_detected  = Signal()
    model_ready         = Signal()       # Whisper 加载完
    model_failed        = Signal(str)    # Whisper 加载失败
    error               = Signal(str)
    
    def __init__(self):
        # 故意不传 parent：避免 VoiceUIMain 销毁时被 Qt 连带析构
        super().__init__()
        self._lock = threading.Lock()
        self.threshold_db   = -30.0        # 默认门限；启用监听时会自动校准
        self.listening_on   = False        # 总开关
        self.barge_in       = False        # 是否处于"双工打断"模式（TTS 期间）
        # Whisper 模型懒加载
        self._whisper: WhisperModel | None = None
        self._whisper_lock = threading.Lock()
        # 内部缓冲
        self._pre_buf  : list[np.ndarray] = []   # 发言前的预缓冲
        self._speech_buf: list[np.ndarray] = []
        self._speech_active = False
        self._silence_count = 0
        self._speech_frames = 0
        self._frame_count = 0
        self._running = True
        # ASR 后端选择：hybrid / cloud / local
        self.asr_mode = "hybrid"
        self.cloud_provider = None
        self._cloud = None
        self._cloud_lock = threading.Lock()
        # 云端实时 ASR session（不占云 REST 限额；首选 hybrid / cloud 模式）
        self._stream_session = None
        self._stream_final_text: str = ""
        self._stream_final_lock = threading.Lock()
        self._silence_count = 0
        self._speech_frames = 0
        self._frame_count = 0
        self._running = True
        # PTT（Push-to-Talk）状态
        self._ptt_active = False
        self._ptt_lock = threading.Lock()
        # Partial STT（实时字幕）状态
        self._partial_running = False
        self._partial_lock = threading.Lock()
        # Silero VAD（v6 API）
        try:
            self._vad_model = load_silero_vad()
        except Exception as e:
            print(f"[AudioWorker] VAD 模型加载失败: {e}", file=sys.stderr)
            self._vad_model = None
        # 设置 stdout 不缓冲
        try: sys.stdout.reconfigure(line_buffering=True)
        except Exception: pass
    
    # ---- controls from main thread ----
    def set_threshold(self, db: float):
        with self._lock:
            self.threshold_db = float(db)
    
    def set_listening(self, on: bool):
        with self._lock:
            self.listening_on = bool(on)
            # 关闭时重置内部状态
            if not on:
                self._speech_active = False
                self._speech_buf.clear()
                self._pre_buf.clear()
                self._silence_count = 0
                self._speech_frames = 0
    
    def calibrate(self, duration_s: float = 1.5):
        """安静期采样 ~1.5s，自动设阈值 = max(环境噪声 dBFS) + 6dB"""
        import sounddevice as sd
        import math
        sr = SAMPLE_RATE
        block = FRAME_SAMPLES
        levels = []
        try:
            stream = sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=block)
            with stream:
                needed = int(duration_s * sr / block)
                for _ in range(needed):
                    chunk, _ = stream.read(block)
                    rms = float(np.sqrt(np.mean(chunk**2)) + 1e-10)
                    db = 20 * math.log10(rms)
                    levels.append(db)
        except Exception as e:
            print(f"[calibrate] 采样失败: {e}", file=sys.stderr)
            return None, None
        if not levels:
            return None, None
        levels.sort()
        idx = int(len(levels) * 0.95)
        p95 = levels[min(idx, len(levels)-1)]
        new_threshold = max(min(p95 + 6.0, -10.0), -55.0)
        with self._lock:
            self.threshold_db = new_threshold
        return new_threshold, p95

    def set_barge_in(self, on: bool):
        with self._lock:
            self.barge_in = bool(on)

    def start_ptt(self):
        """按住发言：启动 PTT 捕获（必须同时 listenting）"""
        with self._ptt_lock:
            self._ptt_active = True
        with self._lock:
            if not self.listening_on:
                self.listening_on = True  # PTT 期间隐式开麦
            # 清旧 buf，准备新一段
            self._speech_active = False
            self._speech_buf.clear()
            self._silence_count = 0
            self._speech_frames = 0
        self.start_partial_stt()  # PTT 也上实时字幕
    
    def end_ptt(self) -> bool:
        """松开：立刻把当前 buf 送 STT
        Returns: 是否真的送去了 STT（buf 空 返回 False）
        """
        with self._ptt_lock:
            self._ptt_active = False
        with self._lock:
            if not self._speech_active or not self._speech_buf:
                self._speech_active = False
                self._speech_buf.clear()
                self.stop_partial_stt()
                # 云端 session 也要结束
                self._stop_cloud_stream()
                return False
            audio = np.concatenate(self._speech_buf)
            self._speech_buf.clear()
            self._speech_active = False
            self._silence_count = 0
            self._speech_frames = 0
        self.stop_partial_stt()
        # 云端：先 拿 stream_final_text（调用 stop session 需点击几秒）
        cloud_text = self._stop_cloud_stream()
        if cloud_text:
            self.user_text_ready.emit(cloud_text)
            self.speech_ended.emit()
            return True
        self.speech_ended.emit()
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()
        return True
    
    def stop(self):
        self._running = False
        # 打断任何阻塞的 sd.sleep
        try:
            import _portaudio as _pa  # type: ignore
        except Exception:
            pass
    
    # ---- main loop ----
    def run(self):
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
                callback=self._on_audio,
            )
            with stream:
                while self._running:
                    # 100ms 顶嘴念测，没有 sd.sleep 依赖
                    time.sleep(0.05)
        except Exception as e:
            self.error.emit(f"音频采集失败: {e}")
    
    # ---- audio callback ----
    def _on_audio(self, indata, frames, time_info, status):
        if status:
            # InputOverflow 通常无害
            pass
        
        audio = indata.flatten().astype(np.float32)
        
        # 计算 dB
        rms = float(np.sqrt(np.mean(audio * audio)) + 1e-10)
        db  = 20.0 * math.log10(rms)
        normalized = max(0.0, min(1.0, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)))
        self.level_changed.emit(normalized)
        
        # 读取当前阈值（线程安全）
        with self._lock:
            threshold_db = self.threshold_db
            listening_on = self.listening_on
            barge_in     = self.barge_in
        
        # 是否超过门限（鼠标可调）
        above_threshold = db > threshold_db
        
        # VAD 使用 silero-vad v6（不受鼠标阈值影响，用于稳健检测）
        try:
            if self._vad_model is not None:
                speech_prob = float(self._vad_model(audio, SAMPLE_RATE).item())
            else:
                speech_prob = 0.0
        except Exception as e:
            speech_prob = 0.0
        
        # 综合判断：超过鼠标阀值 且（VAD 高概率 OR VAD 不可用）
        # 实际中 silero-vad 对背景噪声环境经常错判（iMac 内置麦更是）
        # 默认仅按 dB 阀值判，让用户可以手动调门槛。silero 作为可选“提示”使用
        is_active = above_threshold
        
        # 预缓冲（始终保留最近的 N 帧用于发言开头补帧）
        self._frame_count += 1
        self._pre_buf.append(audio.copy())
        if len(self._pre_buf) > PRE_BUFFER_FRAMES:
            self._pre_buf.pop(0)
        
        # 模式分支
        if not listening_on:
            return

        # PTT 模式：最高优先级，只要按住就持续录
        with self._ptt_lock:
            ptt = self._ptt_active
        if ptt:
            if above_threshold:
                if not self._speech_active:
                    self._speech_active = True
                    self._silence_count = 0
                    self._speech_frames = 0
                    self._speech_buf = list(self._pre_buf)
                    self.start_partial_stt()
                    self._start_cloud_stream()
                    self.speech_started.emit()
                self._speech_buf.append(audio.copy())
                self._speech_frames += 1
                self._silence_count = 0
            else:
                # 静音帧：不算发言结束，还是收着
                if self._speech_active:
                    self._speech_buf.append(audio.copy())
            return

        if barge_in:
            # 双工模式：检测打断 + 累积语音帧，过渡到 normal 模式时不丢首音节
            if is_active:
                if not self._speech_active:
                    # 第一次检测到用户声音 → 转为“捕获+打断”模式
                    self._speech_active = True
                    self._silence_count = 0
                    self._speech_frames = 1
                    self._speech_buf = list(self._pre_buf)   # 拼上预缓冲
                    self._speech_buf.append(audio.copy())
                    self.interrupt_detected.emit()
                else:
                    # 已触发，继续累积（让 GUI 来得及关 TTS）
                    self._speech_buf.append(audio.copy())
                    self._speech_frames += 1
                    self._silence_count = 0
            return
        
        # 正常捕获模式
        if is_active:
            if not self._speech_active:
                # 发言开始：把预缓冲拼上
                self._speech_active = True
                self._silence_count = 0
                self._speech_frames = 0
                self._speech_buf = list(self._pre_buf)
                self.start_partial_stt()
                self._start_cloud_stream()
                self.speech_started.emit()
            self._speech_buf.append(audio.copy())
            self._speech_frames += 1
            self._silence_count = 0
        else:
            if self._speech_active:
                # 还在捕获阶段：记静音帧，过阈值则结束
                self._speech_buf.append(audio.copy())
                self._silence_count += 1
                if self._silence_count >= SILENCE_END_FRAMES:
                    self._speech_active = False
                    duration_frames = len(self._speech_buf)
                    if duration_frames >= MIN_SPEECH_FRAMES:
                        audio_segment = np.concatenate(self._speech_buf)
                        self._speech_buf.clear()
                        self.stop_partial_stt()
                        self.speech_ended.emit()
                        # STT 在工作线程跑（避免阻塞 callback）
                        threading.Thread(
                            target=self._transcribe, args=(audio_segment,),
                            daemon=True,
                        ).start()
                    else:
                        self._speech_buf.clear()
                    self._silence_count = 0
                    self._speech_frames = 0
    
    def _transcribe(self, audio: np.ndarray):
        """最终转写：根据 self.asr_mode 选 在线 / 本地 / hybrid。
        注意：如果 _stop_cloud_stream 前面已经在主线程拿到 final text，则 _transcribe 不多走一趟。
        """
        # 检查云端 stream final 是否已拿到
        with self._stream_final_lock:
            stream_final = self._stream_final_text
            # 清空，避免影响下一次
            self._stream_final_text = ""
        if stream_final and self.asr_mode in ("cloud", "hybrid"):
            self.user_text_ready.emit(stream_final)
            return
        
        text = ""
        err = None
        if self.asr_mode in ("cloud", "hybrid"):
            try:
                if self._cloud is None:
                    with self._cloud_lock:
                        if self._cloud is None:
                            from cloud_asr import cloud_asr_for, _detect_provider
                            provider = self.cloud_provider or _detect_provider()
                            if not provider:
                                raise RuntimeError("没云端 key")
                            self._cloud = cloud_asr_for(provider)
                text = self._cloud.transcribe(audio, SAMPLE_RATE).strip()
            except Exception as e:
                err = f"云 {self.cloud_provider}: {e}"
        if not text and self.asr_mode in ("local", "hybrid"):
            try:
                if self._whisper is None:
                    with self._whisper_lock:
                        if self._whisper is None:
                            self._whisper = WhisperModel(
                                "turbo", device="cpu", compute_type="int8"
                            )
                            self.model_ready.emit()
                segments, info = self._whisper.transcribe(
                    audio, language="zh", beam_size=5, vad_filter=False,
                )
                text = "".join(seg.text for seg in segments).strip()
            except Exception as e:
                err = err or f"本地 Whisper: {e}"
        if text:
            self.user_text_ready.emit(text)
        elif err:
            self.error.emit(f"STT 失败: {err}")
    
    def start_partial_stt(self):
        """发言期间背景跳 2 秒一次的实时识别。"""
        with self._partial_lock:
            if self._partial_running:
                return
            self._partial_running = True
        t = threading.Thread(target=self._partial_stt_loop, daemon=True)
        t.start()

    def stop_partial_stt(self):
        with self._partial_lock:
            self._partial_running = False

    def _start_cloud_stream(self):
        """如果 asr_mode 含 cloud，开一个实时 ASR session（不占云 REST 限额）。"""
        if self.asr_mode not in ("cloud", "hybrid"):
            return
        if self._stream_session is not None:
            return
        try:
            from streaming_asr import AliyunStreamingASR
            sess = AliyunStreamingASR(
                on_partial=lambda t: self.partial_text.emit(t),
                on_final=lambda t: self._on_stream_final(t),
                on_error=lambda e: None,
            )
            sess.start()  # 阻塞到 task-started
            self._stream_session = sess
        except Exception:
            self._stream_session = None

    def _on_stream_final(self, text: str):
        with self._stream_final_lock:
            self._stream_final_text = text

    def _stop_cloud_stream(self) -> str:
        """结束 stream session，拿 final 文本。"""
        if self._stream_session is None:
            return ""
        sess = self._stream_session
        self._stream_session = None
        try:
            text = sess.finish(timeout_s=15)
        except Exception:
            text = ""
        with self._stream_final_lock:
            if text and not self._stream_final_text:
                self._stream_final_text = text
            return self._stream_final_text
        return text or ""

    def _partial_stt_loop(self):
        """后台 partial STT：每 2s 对最近 1.2s 音频转一次，emit partial_text。"""
        import math
        PARTIAL_INTERVAL = 2.0
        PARTIAL_WINDOW = 20  # 帧数×512/16000 ~ 0.6s
        while True:
            with self._partial_lock:
                if not self._partial_running:
                    return
            time.sleep(PARTIAL_INTERVAL)
            with self._lock:
                if not self._speech_active or not self._speech_buf:
                    continue
                buf_copy = list(self._speech_buf[-PARTIAL_WINDOW:])
            if len(buf_copy) < 4:
                continue
            audio = np.concatenate(buf_copy).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio * audio)) + 1e-10)
            db = 20 * math.log10(rms)
            with self._lock:
                if db < self.threshold_db - 3:
                    continue
            try:
                with self._whisper_lock:
                    if self._whisper is None:
                        self._whisper = WhisperModel(
                            "turbo", device="cpu", compute_type="int8",
                        )
                        self.model_ready.emit()
                segments, _ = self._whisper.transcribe(
                    audio, language="zh", beam_size=1,
                    vad_filter=False, without_timestamps=True,
                )
                text = "".join(s.text for s in segments).strip()
                if text:
                    self.partial_text.emit(text)
            except Exception:
                pass

    def prewarm_whisper(self):
        """后台预热模型，避免首次 STT 时才加载造成冷启动跳跃"""
        def _go():
            try:
                if self._whisper is None:
                    self._whisper = WhisperModel(
                        "turbo", device="cpu", compute_type="int8"
                    )
                self.model_ready.emit()
            except Exception as e:
                self.model_failed.emit(str(e))
        threading.Thread(target=_go, daemon=True).start()


# --- Custom widgets ----------------------------------------------------------

class WaveformWidget(QWidget):
    """50 段渐变 bar 实时声纹"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self._history = [0.0] * 50
        self._bar_count = 50
        self.setMinimumHeight(140)
        self.setStyleSheet("background: transparent;")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30fps
    
    def set_level(self, lvl: float):
        self._level = float(lvl)
    
    def _tick(self):
        # 平滑：新值更靠近旧值（一个简单的低通）
        prev = self._history[-1]
        nval = prev * 0.4 + self._level * 0.6
        self._history = self._history[1:] + [nval]
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 40
        bar_w = 5
        n = self._bar_count
        gap = max(2, (w - 2 * margin - bar_w * n) // (n - 1))
        
        # baseline line
        p.setPen(QPen(QColor(40, 80, 120, 100), 1))
        p.drawLine(margin, h // 2, w - margin, h // 2)
        
        for i, lvl in enumerate(self._history):
            x = margin + i * (bar_w + gap)
            # 双侧（mirror）
            amp = max(2.0, lvl * (h - 30))
            y_top = h // 2 - amp / 2
            y_bot = h // 2 + amp / 2
            
            # 渐变：cyan → purple → pink
            grad = QLinearGradient(0, y_top, 0, y_bot)
            grad.setColorAt(0.0, QColor(80, 230, 255))    # cyan
            grad.setColorAt(0.5, QColor(140, 90, 240))    # purple
            grad.setColorAt(1.0, QColor(255, 100, 200))  # pink
            
            p.fillRect(int(x), int(y_top), bar_w, int(amp), grad)
            
            # 微光
            p.setPen(QPen(QColor(180, 240, 255, 60), 1))
            p.drawRect(int(x) - 1, int(y_top) - 1, bar_w + 2, int(amp) + 2)


class AvatarWidget(QWidget):
    """头像 + 脉冲光圈"""
    
    def __init__(self, avatar_path: Path, parent=None):
        super().__init__(parent)
        self._avatar = QPixmap(str(avatar_path))
        if not self._avatar.isNull():
            self._avatar = self._avatar.scaled(
                200, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._phase = 0
        self._state_color = QColor(80, 230, 255)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)
        self.setMinimumHeight(260)
        self.setStyleSheet("background: transparent;")
    
    def set_state_color(self, color: QColor):
        self._state_color = QColor(color)
    
    def _tick(self):
        self._phase = (self._phase + 1) % 100
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 - 10
        r = 110
        
        # 脉冲光圈：3 圈
        pulse = 0.5 + 0.5 * math.sin(self._phase * 0.0628)
        for i, mult in enumerate([1.05, 1.22, 1.42]):
            radius = int(r * mult + pulse * 8)
            op = int((1.0 - i * 0.25) * (0.10 + 0.20 * pulse))
            c = QColor(self._state_color); c.setAlpha(op)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        
        # 头像
        if not self._avatar.isNull():
            aw = self._avatar.width()
            ah = self._avatar.height()
            p.drawPixmap(cx - aw // 2, cy - ah // 2, self._avatar)
        else:
            # fallback：绘一个圆形占位
            grad = QRadialGradient(cx, cy, r)
            grad.setColorAt(0, QColor(255, 230, 100))
            grad.setColorAt(1, QColor(140, 90, 240))
            p.setBrush(grad); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        
        # glow ring overlay
        rg = QRadialGradient(cx, cy, r)
        c1 = QColor(self._state_color); c1.setAlpha(110)
        c2 = QColor(self._state_color); c2.setAlpha(0)
        rg.setColorAt(0.7, c1); rg.setColorAt(1.0, c2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(rg)
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)


# --- TTS drivers -------------------------------------------------------------

class _TTSBase(QObject):
    """TTS 驱动接口。所有驱动实现同协议：speak/stop/is_speaking"""
    
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
    
    def _kill_locked(self):
        if self._proc and self._proc.poll() is None:
            try:
                import os, signal as _sig
                os.killpg(self._proc.pid, _sig.SIGTERM)
            except Exception:
                try: self._proc.terminate()
                except Exception: pass
            try: self._proc.wait(timeout=0.5)
            except Exception: pass
        self._proc = None
    
    def stop(self):
        with self._lock:
            self._kill_locked()
    
    def is_speaking(self) -> bool:
        with self._lock:
            # 生成阶段 也视为“还在说话” — 看护轮询需要这个
            if not self._gen_done.is_set():
                return True
            return self._proc is not None and self._proc.poll() is None
    
    def speak(self, text: str):
        raise NotImplementedError


class SayTTSDriver(_TTSBase):
    """本地 fallback：macOS `say`，可被 SIGTERM 杀掉以实现双工打断"""
    
    def __init__(self, voice: str = TTS_VOICE, rate: str = TTS_RATE):
        super().__init__()
        self.voice = voice
        self.rate  = rate
    
    def speak(self, text: str):
        with self._lock:
            self._kill_locked()
            if not text.strip():
                return
            try:
                self._proc = subprocess.Popen(
                    ["say", "-v", self.voice, "-r", self.rate, text],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                print(f"[SayTTS] 启动失败: {e}", file=sys.stderr)
                self._proc = None


class EdgeTTSDriver(_TTSBase):
    """在线 TTS：Microsoft Edge neural voices（免费、免 key、中文超自然）
    
    默认 zh-CN-XiaoxiaoNeural：年轻女声、活泼自然，超级貼合海绵宝宝风格。
    备选：YunxiNeural（男）、XiaoyiNeural（暖女）、YunyangNeural（播音）、XiaobeiNeural（东北）、XiaoniNeural（陕）。
    """
    
    SUPPORTED_VOICES = [
        "zh-CN-XiaoxiaoNeural",   # 女 / 活泼甜 → 默认
        "zh-CN-YunxiNeural",      # 男 / 有诉诸热情
        "zh-CN-XiaoyiNeural",     # 女 / 暖
        "zh-CN-YunyangNeural",    # 男 / 播报
        "zh-CN-YunjianNeural",    # 男 / 温和
        "zh-CN-liaoning-XiaobeiNeural",  # 东北女
        "zh-CN-shaanxi-XiaoniNeural",    # 陕西女
    ]
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", volume: str = "+0%"):
        super().__init__()
        self.voice = voice
        self.rate = rate      # 如 "+10%"、"-20%"
        self.volume = volume  # 如 "+5%"、"-10%"
        self._tmpfile = "/tmp/voiceui_edge_tts.mp3"
        self._gen_thread: threading.Thread | None = None
        self._gen_done = threading.Event()
        self._gen_done.set()  # 初始：有生成义务吗？没有
    
    def speak(self, text: str):
        with self._lock:
            self._kill_locked()
            if not text.strip():
                return
            # 等上一次生成结束，避免覆写
            self._gen_done.clear()
        # 生成 + 播放 都在后台线程
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
    
    def _run(self, text: str):
        try:
            # 1) 在线合成
            import asyncio
            import edge_tts
            async def gen():
                communicate = edge_tts.Communicate(
                    text, voice=self.voice, rate=self.rate, volume=self.volume,
                )
                with open(self._tmpfile, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
            asyncio.run(gen())
        except Exception as e:
            print(f"[EdgeTTS] 合成失败: {e}", file=sys.stderr)
            with self._lock:
                self._gen_done.set()
            return
        # 2) 播放
        with self._lock:
            self._gen_done.set()
            try:
                self._proc = subprocess.Popen(
                    ["afplay", self._tmpfile],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                print(f"[EdgeTTS] afplay 失败: {e}", file=sys.stderr)
                self._proc = None


# 旧名兼容：默认使用 EdgeTTS（云端、中文超自然）
TTSDriver = EdgeTTSDriver


# --- Agent backend -----------------------------------------------------------

class AgentBackend(QObject):
    """通过 `openclaw agent --json` 跟海绵宝宝聊天"""
    
    response_ready = Signal(str)
    error          = Signal(str)
    
    def __init__(self, session_key: str = SESSION_KEY, model: str = DEFAULT_MODEL):
        super().__init__()
        self.session_key = session_key
        self.model = model
    
    def ask(self, text: str):
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
    
    def _run(self, text: str):
        try:
            r = subprocess.run(
                [
                    "openclaw", "agent",
                    "--model", self.model,
                    "--session-key", self.session_key,
                    "--message", text,
                    "--json",
                ],
                capture_output=True, text=True, timeout=180,
            )
            try:
                data = json.loads(r.stdout)
                reply = data.get("result", {}).get("payloads", [{}])[0].get("text") or r.stdout.strip()
            except Exception:
                reply = r.stdout.strip()
            if not reply:
                reply = "(空回复)"
            self.response_ready.emit(reply)
        except subprocess.TimeoutExpired:
            self.error.emit("请求超时（>180s）")
        except Exception as e:
            self.error.emit(f"agent 调用失败: {e}")


# --- Main window -------------------------------------------------------------

class VoiceUIMain(QMainWindow):
    
    STATE_COLORS = {
        "IDLE":          QColor(80, 230, 255),    # cyan
        "USER_SPEAKING": QColor(120, 255, 180),   # mint
        "TRANSCRIBING":  QColor(255, 230, 120),   # 暖黄
        "THINKING":      QColor(200, 120, 255),   # 紫
        "AI_SPEAKING":   QColor(255, 100, 200),   # 粉
        "INTERRUPTED":   QColor(255, 160, 100),   # 橙
        "ERROR":         QColor(255, 80, 80),     # 红
    }
    STATE_LABELS = {
        "IDLE":          "🎤  待命（点击右下方开启）",
        "USER_SPEAKING": "🗣️  蟹老板在说话…",
        "TRANSCRIBING":  "🧠  听写中…",
        "THINKING":      "🦐  海绵宝宝在想…",
        "AI_SPEAKING":   "💬  海绵宝宝在回答（可以说打断）",
        "INTERRUPTED":   "⏸️  被打断，重新听",
        "ERROR":         "⚠️  出错了",
        "INIT":          "🔥 预热模型…",
    }
    
    def __init__(self, tts_backend: str = "edge", tts_voice: str = "zh-CN-XiaoxiaoNeural"):
        super().__init__()
        self.setWindowTitle("🧽 海绵宝宝 · Voice Agent")
        self.setMinimumSize(1024, 720)
        self.resize(1180, 780)
        
        self._state = "IDLE"
        self._ptt_active = False
        # TTS backend: "edge"（在线神经语音，默认）还是 "say"（本地）
        if tts_backend == "say":
            self._tts = SayTTSDriver()
        else:
            self._tts = EdgeTTSDriver(voice=tts_voice)
        self._agent = AgentBackend()
        
        self._build_ui()
        self._wire()
        self._install_space_ptt()
        self._set_state("IDLE")
    
    # ---- ui ----
    def _build_ui(self):
        central = QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        central.setStyleSheet(self._stylesheet())
        
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 14, 20, 14)
        root.setSpacing(10)
        
        # Title row
        title_row = QHBoxLayout()
        title = QLabel("🧽 海绵宝宝 · Voice Agent")
        title.setObjectName("title")
        sub   = QLabel("⚡ 全双工语音 / 动态声纹 / VAD 智能识别")
        sub.setObjectName("sub")
        title_row.addWidget(title); title_row.addStretch(); title_row.addWidget(sub)
        root.addLayout(title_row)
        
        # Center: 头像 + 声纹 (左) / 控制台 (右)
        center = QHBoxLayout()
        center.setSpacing(16)
        
        # ---- left: avatar + waveform ----
        left = QFrame(); left.setObjectName("panel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(20, 14, 20, 14)
        ll.setSpacing(4)
        
        self.avatar = AvatarWidget(AVATAR)
        ll.addWidget(self.avatar, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 实时字幕：有内容时亮 + 描边变色；无内容时灰提示
        self.live_caption = QLabel("🗣  说话中…实时字幕会在这里跳字")
        self.live_caption.setWordWrap(True)
        self.live_caption.setMinimumHeight(48)
        self.live_caption.setMaximumHeight(96)
        self.live_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_caption.setStyleSheet("""
            QLabel {
                background: rgba(10, 18, 32, 220);
                border: 1px solid #1a4060;
                border-radius: 8px;
                padding: 8px 14px;
                color: #80a0c0;
                font-size: 14px;
                font-family: 'PingFang SC','Microsoft YaHei';
            }
            QLabel[live="true"] {
                background: rgba(20, 50, 80, 220);
                border-color: #50c0ff;
                color: #ffffff;
                font-size: 15px;
            }
        """)
        self.live_caption.setProperty("live", False)
        ll.addWidget(self.live_caption)
        
        wave_title = QLabel("◉  LIVE  WAVEFORM")
        wave_title.setStyleSheet("color:#5090c0; font-size:10px; letter-spacing:3px; padding-left:6px;")
        ll.addWidget(wave_title)
        
        self.wave = WaveformWidget()
        ll.addWidget(self.wave)
        
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center.addWidget(left, 3)
        
        # ---- right: controls ----
        right = QFrame(); right.setObjectName("panel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(18, 14, 18, 14)
        rl.setSpacing(10)
        
        rl.addWidget(self._small_title("⚙  控制台"))
        
        # state
        self.state_lbl = QLabel()
        self.state_lbl.setObjectName("state")
        rl.addWidget(self.state_lbl)
        
        # buttons row
        btn_row = QHBoxLayout()
        self.btn_listen = QPushButton("🎙 开启语音")
        self.btn_listen.setCheckable(True)
        btn_row.addWidget(self.btn_listen)
        self.btn_ptt = QPushButton("🎯 按住空格说话")
        self.btn_ptt.setCheckable(True)
        btn_row.addWidget(self.btn_ptt)
        btn_row.addWidget(self._make_btn("🧹 清屏", self._clear_transcript))
        rl.addLayout(btn_row)
        
        # slider
        rl.addWidget(self._small_title("🎚  噪声阀值（越低越灵敏）"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-60); self.slider.setMaximum(-10)
        self.slider.setValue(-40)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)
        rl.addWidget(self.slider)
        
        self.thresh_lbl = QLabel()
        self.thresh_lbl.setObjectName("subinfo")
        rl.addWidget(self.thresh_lbl)
        
        self.live_lbl = QLabel()
        self.live_lbl.setObjectName("subinfo")
        rl.addWidget(self.live_lbl)
        
        # 简易电平条
        bar_frame = QFrame(); bar_frame.setObjectName("meter")
        bfl = QVBoxLayout(bar_frame); bfl.setContentsMargins(8, 6, 8, 6)
        self.meter_lbl = QLabel("环境: — dBFS  ·  阈值门: OFF")
        self.meter_lbl.setStyleSheet("color:#80c0ff; font-size:11px;")
        bfl.addWidget(self.meter_lbl)
        self.meter = QFrame(); self.meter.setMinimumHeight(6)
        self.meter.setStyleSheet("background:#142030; border-radius:3px;")
        bfl.addWidget(self.meter)
        self.meter_fill = QFrame(self.meter)
        self.meter_fill.setStyleSheet("background: qlineargradient(x1:0,x2:1,stop:0 #50e6ff, stop:1 #ff64c8); border-radius:3px;")
        self.meter_fill.setFixedHeight(6)
        rl.addWidget(bar_frame)
        
        rl.addStretch()
        
        hint = QLabel(
            "💡 双击头像 = 切换语音总开关\n"
            "💡 空格按住说话（Push-to-Talk）\n"
            "💡 海绵宝宝说话时直接开口 → 自动停下"
        )
        hint.setStyleSheet("color:#406080; font-size:11px;")
        rl.addWidget(hint)
        
        right.setMaximumWidth(360)
        right.setMinimumWidth(300)
        center.addWidget(right, 1)
        
        root.addLayout(center, 1)
        
        # ---- transcript ----
        root.addWidget(self._small_title("💬  对话回显"))
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setMaximumHeight(220)
        root.addWidget(self.transcript)
        
        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage("就绪 · 待命")
    
    def _make_btn(self, text, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        return b
    
    def _small_title(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#5090c0; font-size:10px; letter-spacing:2px;")
        return l
    
    def _stylesheet(self) -> str:
        return """
        QWidget#central { background: #07091a; }
        QWidget { color: #d8e8ff; font-family: 'Menlo','PingFang SC','Microsoft YaHei'; }
        QLabel#title { font-size: 22px; font-weight: bold; color: #80e6ff; letter-spacing: 2px; }
        QLabel#sub   { font-size: 12px; color: #6080a0; }
        QLabel#state { font-size: 15px; font-weight: 600; padding: 6px 4px; }
        QLabel#subinfo { color: #6080a0; font-size: 12px; }
        QFrame#panel {
            background: rgba(18, 28, 44, 200);
            border: 1px solid #1a2a40;
            border-radius: 10px;
        }
        QFrame#meter { background: rgba(15, 22, 36, 200); border: 1px solid #1a2a40; border-radius: 6px; }
        QPushButton {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a2a40, stop:1 #0e1828);
            color: #b8d8ff; border: 1px solid #305070; border-radius: 6px;
            padding: 8px 14px; font-size: 13px;
        }
        QPushButton:hover { background: #1e3252; border-color: #5090d0; color: #d8eeff; }
        QPushButton:checked { background: #2a4a78; border-color: #80c0ff; color: #fff; }
        QSlider::groove:horizontal {
            background: #142030; height: 8px; border-radius: 4px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0,x2:1, stop:0 #50e6ff, stop:1 #ff64c8);
            border-radius: 4px;
        }
        QSlider::add-page:horizontal { background: #0e1828; border-radius: 4px; }
        QSlider::handle:horizontal {
            background: #ffffff; border: 2px solid #80c0ff;
            width: 18px; height: 18px; margin: -7px 0; border-radius: 9px;
        }
        QTextEdit {
            background: #0a1020; color: #d8e8ff;
            border: 1px solid #1a2a40; border-radius: 6px;
            padding: 10px; font-size: 14px;
            font-family: 'Menlo','PingFang SC','Microsoft YaHei';
        }
        QStatusBar { background: #050818; color: #5090c0; }
        """
    
    # ---- wire ----
    def _wire(self):
        # worker：故意不传 parent，由 closeEvent 显式 wait
        self.worker = AudioWorker()
        self.worker.start()
        
        self.worker.level_changed.connect(self._on_level)
        self.worker.speech_started.connect(lambda: self._set_state("USER_SPEAKING"))
        self.worker.speech_started.connect(self._on_speech_started)
        self.worker.speech_ended.connect(lambda: self._set_state("TRANSCRIBING"))
        self.worker.speech_ended.connect(self._on_speech_ended)
        self.worker.partial_text.connect(self._on_partial_text)
        self.worker.user_text_ready.connect(self._on_user_text)
        self.worker.interrupt_detected.connect(self._on_interrupt)
        self.worker.error.connect(lambda e: self._set_state("ERROR") or self.statusBar().showMessage(e))
        
        # slider / buttons
        self.slider.valueChanged.connect(self._on_slider)
        self.btn_listen.toggled.connect(self._on_listen_toggle)
        
        # agent
        self._agent.response_ready.connect(self._on_agent_reply)
        self._agent.error.connect(lambda e: (self._tts.stop(), self._set_state("ERROR"), self._append("系统", e, "#ff8080")))
        
        # 双击头像
        self.avatar.mouseDoubleClickEvent = lambda e: self.btn_listen.toggle()
        
        # 空格按住说话
        self.btn_ptt.setAutoRepeat(False)
        self.btn_ptt.pressed.connect(self._on_ptt_press)
        self.btn_ptt.released.connect(self._on_ptt_release)
    
    # ---- slots ----
    def _on_level(self, lvl: float):
        self.wave.set_level(lvl)
        db = lvl * (DB_CEIL - DB_FLOOR) + DB_FLOOR
        self.live_lbl.setText(f"环境音量: {db:+5.1f} dBFS")
        # 更新电平条
        width = max(2, int(lvl * self.meter.width()))
        self.meter_fill.setFixedWidth(width)
        # 阈值门指示
        listening = self.btn_listen.isChecked()
        self.meter_lbl.setText(
            f"环境: {db:+5.1f} dBFS  ·  阈值门: {'ON' if listening else 'OFF'}"
        )
    
    def _on_slider(self, val: int):
        self.worker.set_threshold(val)
        self.thresh_lbl.setText(f"当前阀值: {val} dBFS")
    
    def _on_model_ready(self):
        self.statusBar().showMessage("🧠 Whisper 模型就绪 — 点 '开启语音' 开始对话", 5000)
    
    def _on_model_failed(self, err):
        self.statusBar().showMessage(f"⚠ Whisper 加载失败: {err[:80]}")
    
    def _on_listen_toggle(self, on: bool):
        self.btn_listen.setText("🎙 关闭语音" if on else "🎙 开启语音")
        self.worker.set_listening(on)
        if on:
            self._set_state("IDLE")
            self._append("系统", "🎙 正在校准环境噪声…", "#80c0ff")
            # 后台自动校准：以环境噪声 p95 + 6dB 作阈值
            def _calib():
                res = self.worker.calibrate(1.5)
                if res and res[0] is not None:
                    thr, p95 = res
                    self.slider.setValue(int(thr))
                    self.thresh_lbl.setText(f"🔧 已校准：门槛={thr:+.0f} dB / 环境噪声 {p95:+.0f} dB")
                    self._append("系统", f"🔧 校准完成：环境 {p95:+.0f} dB，门槛 {thr:+.0f} dB（可手动拉滑杆调）", "#80ffb0")
                else:
                    self._append("系统", "⚠ 校准失败，使用默认门槛", "#ff8080")
            threading.Thread(target=_calib, daemon=True).start()
        else:
            self._tts.stop()
            self.worker.set_barge_in(False)
            self._set_state("IDLE")
            self._append("系统", "⏹ 语音已关闭", "#6080a0")
    
    def _on_ptt_press(self):
        """按住空格/按住按钮 → 立即进入录音"""
        if self._ptt_active:
            return  # 幂等
        self._ptt_active = True
        # 中断任何还在说的 TTS
        self._tts.stop()
        self.worker.set_barge_in(False)
        # 隐式开麦 + 重置 capture buf
        self.worker.start_ptt()
        # GUI 状态
        self.btn_ptt.setChecked(True)
        self._set_state("USER_SPEAKING")
        self._append("系统", "🎯 PTT 录音中…（松开送出）", "#80c0ff")
    
    def _on_ptt_release(self):
        """松开空格/松开按钮 → 立即送 ASR + 发给海绵宝宝"""
        if not self._ptt_active:
            return
        self._ptt_active = False
        self.btn_ptt.setChecked(False)
        sent = self.worker.end_ptt()  # 强制 buf → STT
        if sent:
            self._set_state("TRANSCRIBING")
            self._append("系统", "⏏ 松开 → 送 ASR 中…", "#80c0ff")
        else:
            # buf 空：什么都不送
            self._set_state("IDLE")
            self._append("系统", "🎯 PTT 未检测到录音", "#6080a0")
    
    def _is_text_input_focused(self) -> bool:
        """Qt输入控件获得焦点时，空格不应被 PTT 抢走"""
        from PySide6.QtWidgets import QTextEdit, QLineEdit
        fw = self.focusWidget()
        return isinstance(fw, (QTextEdit, QLineEdit))
    
    def keyPressEvent(self, event):
        """空格 = 按住说话（但输入控件里不被抢）"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self._is_text_input_focused():
                self._on_ptt_press()
                event.accept()
                return
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """松开空格 = 结束 PTT"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self._is_text_input_focused():
                self._on_ptt_release()
                event.accept()
                return
        super().keyReleaseEvent(event)
    
    def _install_space_ptt(self):
        """用 QShortcut (ApplicationShortcut) 全局接管空格作为 PTT。
        优先于原生 keyPressEvent，能绕过按钮 / 文本控件对空格的处理。"""
        from PySide6.QtCore import QEvent
        # Press shortcut
        self._sp_press = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._sp_press.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sp_press.activated.connect(self._on_ptt_press)
        # Release shortcut：用 eventFilter 监听 KeyRelease，因为 QShortcut 不区分 press/release
        self.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """接 KeyRelease Space，转 PTT release。
        Press 已由 QShortcut 处理。"""
        from PySide6.QtCore import QEvent as _QE
        if event.type() == _QE.Type.KeyRelease:
            ev = event  # type: QKeyEvent
            if (ev.key() == Qt.Key.Key_Space
                and not ev.isAutoRepeat()
                and not self._is_text_input_focused()
                and self._ptt_active):
                self._on_ptt_release()
                ev.accept()
                return False
        return super().eventFilter(obj, event)
    
    def _on_interrupt(self):
        if self._state != "AI_SPEAKING":
            return
        self._tts.stop()
        self._set_state("INTERRUPTED")
        self._append("系统", "⏸ 被打断了，听你说…", "#ffa060")
        # 切换到普通捕获模式
        self.worker.set_barge_in(False)
        # ⚠️ 不要重置 _speech_active / _speech_buf
        # ——barge 模式已在累积，normal 模式会无缝接管
    
    def _on_speech_started(self):
        self.live_caption.setProperty("live", True)
        self.live_caption.setText("🗣  · · · · ·")
        self.live_caption.style().unpolish(self.live_caption)
        self.live_caption.style().polish(self.live_caption)
    
    def _on_speech_ended(self):
        self.live_caption.setProperty("live", False)
        self.live_caption.setText("🧠  送 STT 转写中…")
        self.live_caption.style().unpolish(self.live_caption)
        self.live_caption.style().polish(self.live_caption)
    
    def _on_partial_text(self, text: str):
        self.live_caption.setText(f"🗣 {text}")
        self.live_caption.setProperty("live", True)
        self.live_caption.style().unpolish(self.live_caption)
        self.live_caption.style().polish(self.live_caption)
    
    def _on_user_text(self, text: str):
        # reset 实况字幕，送 agent
        self.live_caption.setProperty("live", False)
        self.live_caption.setText("🎤  待命")
        self.live_caption.style().unpolish(self.live_caption)
        self.live_caption.style().polish(self.live_caption)
        self._append("蟹老板", text, "#80ffb0")
        self._set_state("THINKING")
        self._agent.ask(text)
    
    def _on_agent_reply(self, reply: str):
        self._append("海绵宝宝", reply, "#ff90d0")
        # 进入 AI_SPEAKING：开启 barge_in
        self.worker.set_barge_in(True)
        self._set_state("AI_SPEAKING")
        self._tts.speak(reply)
        # 监听 TTS 是否自然结束（轮询计时器）
        self._tts_watchdog = QTimer(self)
        self._tts_watchdog.setSingleShot(False)
        self._tts_watchdog.timeout.connect(self._check_tts_done)
        self._tts_watchdog.start(150)
    
    def _check_tts_done(self):
        if self._tts.is_speaking():
            return
        # 自然结束：退出 barge_in，回 IDLE（如果还在监听）
        self._tts_watchdog.stop()
        self.worker.set_barge_in(False)
        if self.btn_listen.isChecked():
            self._set_state("IDLE")
        else:
            self._set_state("IDLE")
    
    def _set_state(self, s: str):
        self._state = s
        self.state_lbl.setText(self.STATE_LABELS.get(s, s))
        c = self.STATE_COLORS.get(s, QColor(80, 230, 255))
        self.state_lbl.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()}); font-size:15px; font-weight:600;")
        self.avatar.set_state_color(c)
        self.statusBar().showMessage(self.STATE_LABELS.get(s, s))
    
    def _clear_transcript(self):
        self.transcript.clear()
    
    def _append(self, who: str, text: str, color: str = "#a0d0ff"):
        sb = self.transcript.verticalScrollBar()
        at_bottom = (sb.value() >= sb.maximum() - 4)
        # 转义 HTML
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        html = (
            f'<div style="margin:6px 0;">'
            f'<span style="color:{color}; font-weight:bold;">{who}:</span> '
            f'<span style="color:#e0e8f0;">{safe}</span>'
            f'</div>'
        )
        self.transcript.append(html)
        if at_bottom:
            sb.setValue(sb.maximum())
    
    # ---- close ----
    def closeEvent(self, e):
        self._tts.stop()
        self.worker.stop()
        super().closeEvent(e)
        if self.worker.isRunning():
            # 给到 5s：够 sd 收成声卡 + callback 结束
            if not self.worker.wait(5000):
                self.worker.terminate()
                if not self.worker.wait(2000):
                    pass
        self.worker.deleteLater()
        self.worker = None


# --- Main --------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", choices=["edge", "say"], default="edge",
                    help="TTS 后端：edge=微软神经在线 / say=macOS 本地")
    ap.add_argument("--voice", default="zh-CN-XiaoxiaoNeural",
                    help="edge 模式下选声音 (zh-CN-XiaoxiaoNeural/zh-CN-YunxiNeural/...)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="对话模型 id")
    ap.add_argument("--asr", choices=["hybrid", "cloud", "local"], default="hybrid",
                    help="ASR 模式：hybrid=partial 本地 + final 云端（默认），"
                         "cloud=全云端，local=全本地 Whisper")
    ap.add_argument("--cloud-provider", choices=["aliyun", "openai"], default=None,
                    help="强制云端 provider (默认看环境变量)")
    args = ap.parse_args()
    
    app = QApplication(sys.argv)
    app.setApplicationName("海绵宝宝 Voice UI")
    
    win = VoiceUIMain(tts_backend=args.tts, tts_voice=args.voice)
    win._agent.model = args.model
    win.worker.asr_mode = args.asr
    win.worker.cloud_provider = args.cloud_provider
    win.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
