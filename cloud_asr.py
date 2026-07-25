"""
cloud_asr.py — 在线 ASR 驱动
==============================

为海绵宝宝语音助手提供"比本地 Whisper 更准"的中文识别。

支持两个后端：
  · aliyun  : 阿里云 Paraformer-v2（中文识别之王）
              需要 DASHSCOPE_API_KEY（百炼/ModelStudio 同一个 key）
              申请：https://bailian.console.aliyun.com/
              通常有免费试用积分

  · openai  : OpenAI Whisper API（兼容 Groq / Azure OpenAI）
              需要 OPENAI_API_KEY
              申请：https://platform.openai.com/api-keys

接口：
  from cloud_asr import cloud_asr_for
  asr = cloud_asr_for("aliyun")
  text = asr.transcribe(audio_np_float32_16k, sample_rate=16000)

音频：numpy float32 mono 16kHz。
"""
from __future__ import annotations

import io
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np


# ---------- 工具：把 numpy 转 wav bytes ----------

def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """numpy float32[-1,1] → 16-bit PCM mono WAV bytes"""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ---------- 基类 ----------

class CloudASRBase:
    name: str = "abstract"
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        raise NotImplementedError


# ---------- 阿里云 Paraformer-v2 ----------

class AliyunDashScopeASR(CloudASRBase):
    """阿里云百炼 Paraformer-v2 ASR
    中文识别精度业界顶级，对带口音、噪声有较强鲁棒性。
    价格：约 ¥0.0008/秒，免费试用 100 万 token。
    """
    name = "aliyun"
    MODELS = [
        "paraformer-v2",      # 推荐：16k 中文多模态，准确度最高
        "paraformer-realtime-v2",  # 流式专用
        "paraformer-v1",      # 老版本
    ]
    
    def __init__(self):
        import dashscope
        self._dashscope = dashscope
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIYUN_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置。\n"
                "申请：https://bailian.console.aliyun.com/\n"
                "→ API-KEY 管理 → 创建 → 复制 key → 设到环境变量。"
            )
        if api_key == "YOUR_DASHSCOPE_API_KEY":
            raise ValueError(
                "DASHSCOPE_API_KEY 还是占位符 'YOUR_DASHSCOPE_API_KEY'，请换成真 key。"
            )
        dashscope.api_key = api_key
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        # 写到临时文件，DashScope Transcription 从文件读
        import tempfile
        from dashscope.audio.asr import Transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        try:
            r = Transcription.call(
                model="paraformer-v2",
                file_urls=[f"file://{tmp_path}"],
                language_hints=["zh", "en"],
            )
            if r.status_code == 200:
                # Transcription 返回结构：output.transcripts[].sentences[].text
                transcripts = (r.output or {}).get("transcripts", []) or []
                texts = []
                for tr in transcripts:
                    for s in tr.get("sentences", []) or []:
                        texts.append(s.get("text", ""))
                return "".join(texts).strip()
            else:
                raise RuntimeError(f"Aliyun ASR 失败 {r.status_code}: {r.message}")
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass


# ---------- OpenAI Whisper API（也兼容 Groq） ----------

class OpenAIWhisperASR(CloudASRBase):
    """OpenAI Whisper API（whisper-1）
    也可用任何 OpenAI-兼容 endpoint（Groq / Azure / 自建），改 base_url 即可。
    """
    name = "openai"
    
    def __init__(self, base_url: str | None = None):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置。https://platform.openai.com/api-keys")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.base_url = base_url or "https://api.openai.com/v1"
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        from openai import OpenAI as _Cls
        # OpenAI SDK 直接吃 BytesIO 也可以
        from io import BytesIO
        file_tuple = ("audio.wav", BytesIO(wav_bytes), "audio/wav")
        # 新版 API: client.audio.transcriptions.create(model=..., file=(...))
        resp = self._client.audio.transcriptions.create(
            model="whisper-1",
            file=file_tuple,
            language="zh",
            response_format="text",
        )
        # text 响应：直接拿 str
        if isinstance(resp, str):
            return resp.strip()
        return getattr(resp, "text", "").strip()


# ---------- dispatcher ----------

def _detect_provider() -> str | None:
    """看环境变量，挑第一个有真 key 的 provider。"""
    dq = os.environ.get("DASHSCOPE_API_KEY")
    if dq and dq != "YOUR_DASHSCOPE_API_KEY":
        return "aliyun"
    oa = os.environ.get("OPENAI_API_KEY")
    if oa:
        return "openai"
    return None


def cloud_asr_for(provider: str | None = None) -> CloudASRBase:
    """构造 cloud ASR。provider=None 时自动检测。"""
    if provider is None:
        provider = _detect_provider()
        if not provider:
            raise ValueError(
                "没找到可用 API key。请设 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 后重试。"
            )
    p = provider.lower()
    if p in ("aliyun", "dashscope"):
        return AliyunDashScopeASR()
    if p in ("openai", "groq"):
        return OpenAIWhisperASR()
    raise ValueError(f"未知 provider: {provider}")


# ---------- 模块自检 ----------

if __name__ == "__main__":
    print("=== cloud_asr 自检 ===")
    dq = os.environ.get("DASHSCOPE_API_KEY", "")
    oa = os.environ.get("OPENAI_API_KEY", "")
    print(f"  DASHSCOPE_API_KEY: {'已设' if dq and dq != 'YOUR_DASHSCOPE_API_KEY' else '未设/占位符'}")
    print(f"  OPENAI_API_KEY:     {'已设' if oa else '未设'}")
    
    p = _detect_provider()
    print(f"  自动检测到: {p}")
    
    if not p:
        print()
        print("⚠ 没可用 key——架构已经就绪，给你两个选项：")
        print()
        print("A) 阿里云百炼 Paraformer-v2 (中文识别之王)")
        print("   1. https://bailian.console.aliyun.com/ → 注册")
        print("   2. API-KEY 管理 → 创建 → 复制")
        print("   3. export DASHSCOPE_API_KEY='sk-...'")
        print()
        print("B) OpenAI Whisper API (通用)")
        print("   1. https://platform.openai.com/api-keys")
        print("   2. 创建 key")
        print("   3. export OPENAI_API_KEY='sk-...'")
    else:
        try:
            asr = cloud_asr_for(p)
            print(f"  ✅ {p} 初始化 OK：{type(asr).__name__}")
        except Exception as e:
            print(f"  ❌ {p} 初始化失败：{e}")
