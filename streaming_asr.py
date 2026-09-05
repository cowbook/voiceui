"""
streaming_asr.py — 直接 WebSocket 协议级对接阿里云 Paraformer-Realtime
======================================================================

跳开 dashscope SDK 的 __str__ bug，自己用 websockets 实现 QPS 协议。
对应文档：
  阿里云百炼 → 语音识别 → 实时语音识别 (Paraformer-Realtime V2)
  接入指南：dashscope.aliyuncs.com/api-ws/v1/inference

协议：
  客户端 → run-task (json)            启动任务
  服务端 → task-started event         就绪
  客户端 → binary pcm                 音频帧
  服务端 → result-generated event     partial 结果（sentence.text 增量）
  客户端 → finish-task (json)         结束
  服务端 → task-finished event        最终文本（含 sentence 数组）
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import websockets

# 打开看协议调试；可可关
DEBUG = bool(os.environ.get("VOICEUI_DEBUG_STREAM", "0") == "1")

API_KEY_ENV = "DASHSCOPE_API_KEY"
WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
MODEL_REALTIME = "paraformer-realtime-v2"


def _extract_text(sent) -> str:
    """Aliyun Paraformer-Realtime 的 sentence 字段：可能为 dict、list of dicts、嵌套结构。
    返回所有 text 拼接。"""
    if isinstance(sent, dict):
        return sent.get("text", "") or ""
    if isinstance(sent, list):
        out = []
        for item in sent:
            if isinstance(item, dict):
                t = item.get("text", "")
                if t:
                    out.append(t)
            elif isinstance(item, str):
                out.append(item)
        return "".join(out)
    return ""


def _extract_output_text(output: dict) -> str:
    """兼容不同返回结构，尽可能提取文本。"""
    if not isinstance(output, dict):
        return ""
    # 常见：sentence / sentences
    text = _extract_text(output.get("sentence"))
    if text:
        return text
    text = _extract_text(output.get("sentences"))
    if text:
        return text
    # 兼容 Transcription 风格：transcripts[].sentences[].text
    transcripts = output.get("transcripts")
    if isinstance(transcripts, list):
        out = []
        for tr in transcripts:
            if not isinstance(tr, dict):
                continue
            t = _extract_text(tr.get("sentence")) or _extract_text(tr.get("sentences"))
            if t:
                out.append(t)
                continue
            sents = tr.get("sentences", [])
            if isinstance(sents, list):
                for s in sents:
                    if isinstance(s, dict) and s.get("text"):
                        out.append(s["text"])
        text = "".join(out).strip()
        if text:
            return text
    # 兜底字段
    if isinstance(output.get("text"), str):
        return output.get("text", "").strip()
    return ""


class AliyunStreamingASR:
    """同步外壳 + 内部 asyncio event loop。
    
    AudioWorker 用法：
      sess = AliyunStreamingASR(on_partial=..., on_final=..., on_error=...)
      sess.start()
      # audio callback 里：
      sess.feed(pcm_bytes_int16)
      # speech_end 里：
      sess.finish_and_wait()  # 阻塞到 final_text 返回
    """
    
    def __init__(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        model: str = MODEL_REALTIME,
        sample_rate: int = 16000,
        language_hints: list[str] | None = None,
    ):
        api_key = os.environ.get(API_KEY_ENV) or os.environ.get("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError(f"{API_KEY_ENV} 未设")
        if api_key.startswith("YOUR_") or "DASHSCOPE" in api_key:
            raise ValueError("DASHSCOPE_API_KEY 还是占位符，请换成真 key")
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.language_hints = language_hints or ["zh", "en"]
        self.on_partial = on_partial or (lambda t: None)
        self.on_final = on_final or (lambda t: None)
        self.on_error = on_error or (lambda e: None)
        # 音频发送队列 + 持续排水循环（避免丢弃帧）
        self._send_q: asyncio.Queue = None  # asyncio.Queue 在 _main() 里创建
        self._send_task: Optional[asyncio.Future] = None
        self._task_id = uuid.uuid4().hex
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None  # set in loop
        self._ready_event = threading.Event()
        self._closed_event = threading.Event()
        self._final_text = ""
        self._started_event = threading.Event()  # 是否已经收到 task-started
        self._finished_event = threading.Event()
        self._error_str = ""
        self._last_partial = ""
        self._event_trace = deque(maxlen=64)
        self._fed_total = 0
        self._sent_total = 0
        self._sent_frames = 0
    
    def start(self):
        """启动后台事件循环 + WebSocket。阻塞到收到 task-started。"""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=10):
            raise RuntimeError("WebSocket 握手超时")
        if self._error_str:
            raise RuntimeError(f"WebSocket 失败：{self._error_str}")
        if not self._started_event.wait(timeout=10):
            events = ", ".join(self._event_trace) or "<none>"
            raise RuntimeError(f"未收到 task-started（events: {events}）")
    
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self._error_str = str(e)
            self.on_error(self._error_str)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
    
    async def _main(self):
        headers = [("Authorization", f"Bearer {self.api_key}")]
        try:
            async with websockets.connect(
                WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            ) as ws:
                self._ws = ws
                # run-task
                start_payload = {
                    "model": self.model,
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "input": {},
                    "parameters": {
                        "format": "pcm",
                        "sample_rate": self.sample_rate,
                        "language_hints": self.language_hints,
                    },
                }
                msg = {
                    "header": {
                        "action": "run-task",
                        "task_id": self._task_id,
                        "streaming": "duplex",
                    },
                    "payload": start_payload,
                }
                await ws.send(json.dumps(msg))
                # 启动 _send_loop + 准备 asyncio.Queue
                self._send_q = asyncio.Queue(maxsize=4000)  # ~25s @16k
                self._send_task = asyncio.create_task(self._send_loop(ws))
                self._ready_event.set()
                # 接收循环
                async for raw in ws:
                    if DEBUG:
                        snippet = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                        if len(snippet) < 800:
                            print(f"[raw] {snippet!r}", flush=True)
                        else:
                            print(f"[raw] {snippet[:400]!r}...{snippet[-200:]!r}", flush=True)
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    evt = (msg.get("event") or
                           msg.get("header", {}).get("event") or
                           "")
                    hdr = msg.get("header", {})
                    if evt:
                        self._event_trace.append(evt)
                    if evt == "task-started":
                        self._started_event.set()
                    elif evt == "result-generated":
                        # sentence 可以是 list of sentence objects 或 dict
                        out = msg.get("payload", {}).get("output", {}) or {}
                        text = _extract_output_text(out)
                        if text and text != self._last_partial:
                            self._last_partial = text
                            try:
                                self.on_partial(text)
                            except Exception as e:
                                if DEBUG: print(f"[partial err] {e}", flush=True)
                    elif evt == "task-finished":
                        out = msg.get("payload", {}).get("output", {}) or {}
                        text = _extract_output_text(out)
                        self._final_text = text
                        try:
                            self.on_final(text)
                        except Exception as e:
                            if DEBUG: print(f"[final err] {e}", flush=True)
                        self._finished_event.set()
                        break
                    elif evt == "task-failed":
                        err = (hdr.get("error_message") or
                               msg.get("payload", {}).get("error_message") or
                               "unknown")
                        self._error_str = err
                        try:
                            self.on_error(err)
                        except Exception:
                            pass
                        self._finished_event.set()
                        break
        except Exception as e:
            self._error_str = f"WebSocket 异常: {e}"
            self._ready_event.set()  # 让 start() 返回
    
    def feed(self, pcm_bytes: bytes):
        """送一段 PCM 音频（16-bit signed LE mono）—— 入队，不阻塞。"""
        if not self._loop or self._ws is None or self._send_q is None:
            if DEBUG: print(f"[feed] skip (loop/ws 未就绪): {len(pcm_bytes) if pcm_bytes else 0}B", flush=True)
            return
        if not pcm_bytes:
            return
        chunk = bytes(pcm_bytes)
        self._fed_total += len(chunk)
        # 注意：feed 可能从非 event-loop 线程调用，必须 thread-safe 入队。
        try:
            self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)
        except Exception:
            pass

    def _enqueue_chunk(self, chunk: bytes):
        """在 event-loop 线程中执行的安全入队。"""
        if self._send_q is None:
            return
        try:
            self._send_q.put_nowait(chunk)
            if DEBUG: print(f"[feed] qsize={self._send_q.qsize()}, +{len(chunk)}B", flush=True)
        except queue.Full:
            pass

    async def _send_loop(self, ws):
        """持续从 asyncio.Queue 取 PCM发送。"""
        bytes_per_sec = self.sample_rate * 2
        if DEBUG: print(f"[send_loop] start qsize={self._send_q.qsize()}", flush=True)
        try:
            sent_total = 0
            frame_count = 0
            consecutive_err = 0
            while True:
                try:
                    chunk = await self._send_q.get()
                except asyncio.CancelledError:
                    break
                t0 = time.time()
                # websockets 16.x：bytes → binary 帧
                try:
                    await ws.send(chunk)
                except Exception as e:
                    consecutive_err += 1
                    if DEBUG: print(f"[send err #{consecutive_err}] {type(e).__name__}: {e!r}", flush=True)
                    if consecutive_err >= 5:
                        raise
                    await asyncio.sleep(0.05)
                    continue
                consecutive_err = 0
                sent_total += len(chunk)
                frame_count += 1
                self._sent_total = sent_total
                self._sent_frames = frame_count
                if DEBUG and (frame_count <= 5 or frame_count % 10 == 0):
                    print(f"[send_loop] #{frame_count} sent={sent_total}B qsize={self._send_q.qsize()}", flush=True)
                # Pacing
                elapsed = time.time() - t0
                expected = len(chunk) / bytes_per_sec
                if expected > elapsed:
                    await asyncio.sleep(expected - elapsed)
        except asyncio.CancelledError:
            if DEBUG: print(f"[send_loop] cancelled (sent {sent_total}B in {frame_count} frames)", flush=True)
        except Exception as e:
            if DEBUG: print(f"[send_loop err] {e}", flush=True)
    
    def finish(self, timeout_s: float = 30.0) -> str:
        """结束会话 + 拿最终结果（阻塞）"""
        if not self._loop or not self._ws:
            return ""
        # 1) 先尽量排空待发送队列，避免尾音在 finish 前被截断。
        async def _drain_pending(max_wait_s: float = 2.0):
            if self._send_q is None:
                return
            t0 = time.time()
            while self._send_q.qsize() > 0 and (time.time() - t0) < max_wait_s:
                await asyncio.sleep(0.02)

        async def _cleanup_send_task():
            if self._send_task and not self._send_task.done():
                self._send_task.cancel()
                try:
                    await self._send_task
                except Exception:
                    pass
        try:
            asyncio.run_coroutine_threadsafe(_drain_pending(2.0), self._loop).result(timeout=3)
        except Exception:
            pass
        # 2) 发 finish
        try:
            asyncio.run_coroutine_threadsafe(self._send_finish(), self._loop).result(timeout=5)
        except Exception as e:
            self._error_str = str(e)
            if DEBUG: print(f"[finish err] {e}", flush=True)
        # 等 final
        self._finished_event.wait(timeout=timeout_s)
        # task-finished 可能偶发丢失，至少返回最后一次 partial
        text = self._final_text or self._last_partial or ""
        if not text and DEBUG:
            print(f"[finish] empty result, events={list(self._event_trace)}", flush=True)
        # 3) 清理发送任务 + 关闭连接，释放资源
        try:
            asyncio.run_coroutine_threadsafe(_cleanup_send_task(), self._loop).result(timeout=2)
        except Exception:
            pass
        try:
            asyncio.run_coroutine_threadsafe(self._close_ws(), self._loop).result(timeout=5)
        except Exception:
            pass
        return text
    
    async def _send_finish(self):
        if self._ws:
            try:
                finish_msg = {
                    "header": {
                        "action": "finish-task",
                        "task_id": self._task_id,
                        "streaming": "duplex",
                    },
                    "payload": {"input": {}},
                }
                await self._ws.send(json.dumps(finish_msg))
            except Exception:
                pass

    async def _close_ws(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass


# --- 自检 ---
if __name__ == "__main__":
    print("=== AliyunStreamingASR 自检 ===")
    print(f"  Key: {'***已设***' if os.environ.get(API_KEY_ENV) else '未设'}")
    
    if not os.environ.get(API_KEY_ENV):
        print("\n  ⚠ 没法测，请设 DASHSCOPE_API_KEY")
    else:
        # 用真音频测
        import sounddevice as sd
        import time
        
        partial_seen = []
        final_seen = []
        errors = []
        
        sess = AliyunStreamingASR(
            on_partial=lambda t: partial_seen.append(t),
            on_final=lambda t: final_seen.append(t),
            on_error=lambda e: errors.append(e),
        )
        
        print("\n启动 session...")
        try:
            sess.start()
            print("  ✓ WS 已连上")
        except Exception as e:
            print(f"  ❌ 启动失败: {e}")
            sys.exit(1)
        
        SR = 16000
        print(f"\n🎤 采 4s 请说话...")
        buf = []
        t0 = time.time()
        frame_idx = [0]
        def cb(a, n, t, s):
            frame_idx[0] += 1
            buf.append(a.copy())
            # 本地录音调试：每 10 帧打印一次 dB 和峰值，确认 mic 真在进声。
            if frame_idx[0] % 10 == 0:
                rms = float(np.sqrt(np.mean(a * a)) + 1e-10)
                db = 20.0 * np.log10(rms)
                peak = float(np.max(np.abs(a)))
                print(f"\r  [mic] frame={frame_idx[0]:03d} db={db:+6.1f} peak={peak:.4f}", end="", flush=True)
            # 进 cloud
            pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
            sess.feed(pcm)
        
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=1024, callback=cb):
                time.sleep(4)
        except Exception as e:
            print(f"  ❌ mic: {e}")
        print()
        
        elapsed = time.time() - t0
        print(f"\n  采了 {elapsed:.1f}s")
        if buf:
            audio = np.concatenate(buf).flatten().astype(np.float32)
            rms = float(np.sqrt(np.mean(audio * audio)) + 1e-10)
            db = 20.0 * np.log10(rms)
            peak = float(np.max(np.abs(audio)))
            print(f"  本地录音统计: samples={len(audio)}, rms={db:+.1f} dBFS, peak={peak:.4f}")

            # 保存 wav 供回放和频谱排查。
            logs_dir = Path(__file__).resolve().parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            wav_path = logs_dir / f"stream_selftest_{int(time.time())}.wav"
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(pcm16.tobytes())
            print(f"  已保存录音: {wav_path}")
        else:
            print("  ⚠ 本地未采到任何音频帧")

        print(f"  partials ({len(partial_seen)}):")
        for p in partial_seen[-5:]:
            print(f"    {p!r}")
        if not partial_seen:
            print("    (无 partial — 可能没说话，或延迟还没到)")
        print(f"  events: {list(sess._event_trace)}")
        print(f"  feed/sent: fed={sess._fed_total}B, sent={sess._sent_total}B, frames={sess._sent_frames}")
        
        print("\n⏹ finish + 等 final...")
        text = sess.finish(timeout_s=15)
        print(f"  Final text: {text!r}")
        
        if errors:
            print(f"\n  errors: {errors}")
