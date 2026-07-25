#!/usr/bin/env python3
"""
diagnose.py — 语音 UI 排错脚本（不启动 GUI）

依次检查：
  1) sounddevice 看得到哪些输入设备
  2) 默认输入设备能不能采到音频 + 实时 dB
  3) silero-vad 在真实音频上的反应
  4) faster-whisper 能不能顺利转写

跑：  ./diagnose.py        # 3 秒短测
      ./diagnose.py 8      # 8 秒长测
"""
import sys, time, os
from pathlib import Path
import numpy as np

SR = 16000

def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    print(f"🔊 诊断：采 {dur:.1f}s 音频 + VAD + 听写测试")
    print("=" * 50)
    
    # 1) 设备列表
    import sounddevice as sd
    print("\n📡 1) 输入设备列表（输入列只关心 IN > 0 的）")
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        for i, d in enumerate(devices):
            if d.get('max_input_channels', 0) > 0:
                marker = " ★ default" if i == default_in else ""
                print(f"   [{i}] {d['name']} (IN:{d['max_input_channels']} OUT:{d['max_output_channels']}){marker}")
        print(f"\n   默认输入设备 id = {default_in}")
    except Exception as e:
        print(f"   ❌ 列设备失败: {e}")
        return
    print()
    
    # 2) 实时 dB 监控
    print(f"🎤 2) 用默认输入设备采 {dur:.1f}s，看实时 dB：")
    print("   （请对着麦克风说几句话，或者弄点声音）")
    print()
    
    frames = []
    levels = []
    def cb(indata, frames_, ti, status):
        if status:
            print(f"\n   ⚠ status: {status}", flush=True)
        a = indata.copy()
        frames.append(a)
        rms = float(np.sqrt(np.mean(a * a)) + 1e-10)
        db = 20.0 * np.log10(rms)
        levels.append(db)
        bar = int(max(0, min(40, (db + 60) * 0.66)))
        marker = "🎙️" if db > -40 else "  "
        print(f"\r   {marker} {db:+6.1f} dB  [{'#' * bar:<40}]  frames={len(frames)}", end="", flush=True)
    
    try:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=512, callback=cb)
    except Exception as e:
        print(f"   ❌ 开流失败: {e}")
        print("   提示：检查 系统设置 → 隐私与安全 → 麦克风，看 Python/sounddevice 是否被允许")
        return
    
    with stream:
        t0 = time.time()
        while time.time() - t0 < dur:
            time.sleep(0.05)
    
    print("\n")
    
    if not frames:
        print("   ❌ 没采到任何帧！")
        print("   最大可能：")
        print("     · 系统设置 → 隐私与安全 → 麦克风 没打开")
        print("     · 你选了带静音键的外接耳机？")
        print("     · 别的 app 占用着麦克风独占模式")
        return
    
    audio = np.concatenate(frames).flatten().astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    rms_db = float(20 * np.log10(np.sqrt(np.mean(audio * audio)) + 1e-10))
    max_db = float(20 * np.log10(peak + 1e-10)) if peak > 0 else -120
    
    print(f"\n📊 3) {dur:.1f}s 统计：")
    print(f"   帧数: {len(frames)}")
    print(f"   平均 RMS dB:  {rms_db:+5.1f}")
    print(f"   峰值 dB:      {max_db:+5.1f}")
    print(f"   峰值幅度:     {peak:.4f}")
    
    # 3) VAD 测试
    print(f"\n🧠 4) silero-vad 在这 {len(frames)} 帧上的反应：")
    try:
        import silero_vad
        vad = silero_vad.SileroVAD()
        probs = [float(vad(s.flatten().astype(np.float32), SR)) for s in frames]
        avg_p = float(np.mean(probs))
        max_p = float(np.max(probs))
        n_speech = sum(1 for p in probs if p > 0.4)
        print(f"   平均概率:     {avg_p:.3f}")
        print(f"   最大概率:     {max_p:.3f}")
        print(f"   高于 0.4 的帧: {n_speech}/{len(probs)}")
        if max_p < 0.3:
            print("   ⚠ VAD 完全没识别到语音成分——可能太安静 / 麦克风没收到声音")
        elif max_p < 0.5:
            print("   ⚠ VAD 概率较低，可能是不在说话或者太小声")
    except Exception as e:
        print(f"   ❌ VAD 加载/调用失败: {e}")
    
    # 4) 听写测试
    print(f"\n✍️  5) faster-whisper 转写测试（turbo 模型）：")
    print("   首次会下模型 ~1.5GB，等几分钟…")
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel("turbo", device="cpu", compute_type="int8")
        segments, info = m.transcribe(audio, language="zh", vad_filter=False)
        text = "".join(s.text for s in segments).strip()
        if text:
            print(f"   ✅ 听写结果: 「{text}」")
        else:
            print("   (无识别结果，可能确实太安静 / 没说话)")
    except Exception as e:
        print(f"   ❌ 听写失败: {e}")
    
    print("\n" + "=" * 50)
    print("✅ 诊断结束")
    print()
    print("下一步怎么调？")
    
    if max_db < -50:
        print("  ⛔ 麦克风基本没声音 → 检查 系统设置 → 隐私与安全性 → 麦克风")
    elif max_db < -30:
        print("  🟡 麦克风能采到，但声音太轻")
        print("     · 检查 Mac 输入音量（系统设置 → 声音 → 输入）")
        print("     · 调近麦克风")
    else:
        print("  ✅ 麦克风信号正常")
    
    if 'probs' in dir() and max_p < 0.4:
        print("  ⛔ VAD 几乎不认这是语音")
        print("     · 试一个更长的「你好」再跑一次")
        print("     · voiceui.py 里可以调低 threshold_db 或调小 vad 阈值")
    elif 'probs' in dir() and max_p > 0.6:
        print("  ✅ VAD 能识别语音 — 接下来该看 GUI 那边")

if __name__ == "__main__":
    main()
