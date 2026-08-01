"""Aliyun Tongyi (DashScope) cloud ASR only."""
from __future__ import annotations

import io
import os
import wave

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


def cloud_asr_for() -> CloudASRBase:
    """Construct Aliyun DashScope ASR only."""
    return AliyunDashScopeASR()


# ---------- 模块自检 ----------

if __name__ == "__main__":
    print("=== cloud_asr 自检 ===")
    dq = os.environ.get("DASHSCOPE_API_KEY", "")
    print(f"  DASHSCOPE_API_KEY: {'已设' if dq and dq != 'YOUR_DASHSCOPE_API_KEY' else '未设/占位符'}")
    if not dq or dq == "YOUR_DASHSCOPE_API_KEY":
        print()
        print("⚠ 没可用 key，请配置阿里云百炼 DashScope key:")
        print()
        print("1. https://bailian.console.aliyun.com/ → 注册")
        print("2. API-KEY 管理 → 创建 → 复制")
        print("3. export DASHSCOPE_API_KEY='sk-...'")
    else:
        try:
            asr = cloud_asr_for()
            print(f"  ✅ aliyun 初始化 OK：{type(asr).__name__}")
        except Exception as e:
            print(f"  ❌ aliyun 初始化失败：{e}")
