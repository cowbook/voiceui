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
import sys
import threading
import time
import uuid
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
    
    def start(self):
        """启动后台事件循环 + WebSocket。阻塞到收到 task-started。"""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=10):
            raise RuntimeError("WebSocket 握手超时")
        if self._error_str:
            raise RuntimeError(f"WebSocket 失败：{self._error_str}")
    
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
                self._ready_event.set()
                # 启动 _send_loop + 准备 asyncio.Queue
                self._send_q = asyncio.Queue(maxsize=4000)  # ~25s @16k
                self._send_task = asyncio.create_task(self._send_loop(ws))
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
                    if evt == "task-started":
                        self._started_event.set()
                    elif evt == "result-generated":
                        # sentence 可以是 list of sentence objects 或 dict
                        out = msg.get("payload", {}).get("output", {}) or {}
                        sent = out.get("sentence")
                        text = self._extract_text(sent)
                        if text and text != self._last_partial:
                            self._last_partial = text
                            try:
                                self.on_partial(text)
                            except Exception as e:
                                if DEBUG: print(f"[partial err] {e}", flush=True)
                    elif evt == "task-finished":
                        out = msg.get("payload", {}).get("output", {}) or {}
                        sents = out.get("sentence", [])
                        text = _extract_text(sents)
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
        if not self._loop or self._ws is None:
            if DEBUG: print(f"[feed] skip (loop/ws 未就绪): {len(pcm_bytes) if pcm_bytes else 0}B", flush=True)
            return
        if not pcm_bytes:
            return
        try:
            self._send_q.put_nowait(bytes(pcm_bytes))
            if DEBUG: print(f"[feed] qsize={self._send_q.qsize()}, +{len(pcm_bytes)}B", flush=True)
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
        # 1) 停 send_loop
        async def _cleanup():
            if self._send_task and not self._send_task.done():
                self._send_task.cancel()
                try: await self._send_task
                except Exception: pass
        try:
            asyncio.run_coroutine_threadsafe(_cleanup(), self._loop).result(timeout=2)
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
        return self._final_text
    
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
        def cb(a, n, t, s):
            buf.append(a.copy())
            # 进 cloud
            pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()
            sess.feed(pcm)
        
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=1024, callback=cb):
                time.sleep(4)
        except Exception as e:
            print(f"  ❌ mic: {e}")
        
        elapsed = time.time() - t0
        print(f"\n  采了 {elapsed:.1f}s")
        print(f"  partials ({len(partial_seen)}):")
        for p in partial_seen[-5:]:
            print(f"    {p!r}")
        if not partial_seen:
            print("    (无 partial — 可能没说话，或延迟还没到)")
        
        print("\n⏹ finish + 等 final...")
        text = sess.finish(timeout_s=15)
        print(f"  Final text: {text!r}")
        
        if errors:
            print(f"\n  errors: {errors}")
