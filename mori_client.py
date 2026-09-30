import sounddevice as sd
import numpy as np
import whisper
import requests
import warnings

warnings.filterwarnings("ignore")


def record_audio_to_array(duration=5, fs=16000):
    """直接录制为 Whisper 要求的 16kHz 浮点数组，跳过文件保存"""
    print(f"\n🎤 [开始录音...] 请对着电脑说话，你有 {duration} 秒的时间")
    try:
        # 直接录制 16kHz, float32, 单声道格式
        myrecording = sd.rec(
            int(duration * fs), samplerate=fs, channels=1, dtype="float32"
        )
        sd.wait()
        print("✅ 录音结束！")
        return myrecording.flatten()  # 转成 Whisper 需要的一维数组
    except Exception as e:
        print(f"❌ 录音失败: {e}")
        return None


def main():
    print("=" * 50)
    print("  Mori 前端感知模块 - MVP 测试版 (内存极速版)")
    print("=" * 50)

    print("\n⏳ 正在加载 Whisper 模型...")
    model = whisper.load_model("base")
    print("✅ 模型加载完成！")

    # 拿到内存里的录音数据
    audio_array = record_audio_to_array(duration=5)
    if audio_array is None:
        return

    print(f"\n⏳ 正在进行语音识别...")
    # 魔法：直接把 numpy 数组传给 Whisper，彻底告别 ffmpeg 依赖
    result = model.transcribe(audio_array)
    user_text = result["text"].strip()

    if not user_text:
        print("⚠️ 未检测到有效声音，请重试。")
        return

    print(f"\n🗣️  识别结果: 【{user_text}】")

    print("\n⏳ 准备发送给 Mori ...")
    mori_api_url = "http://师弟的服务器IP地址:端口/api/chat"

    if "师弟的服务器IP地址" in mori_api_url:
        print("⚠️ 提示: 尚未配置真实的 API 地址。等拿到接口后替换 URL 即可。")
    else:
        payload = {"user_id": "test_001", "message": user_text}
        try:
            response = requests.post(mori_api_url, json=payload, timeout=10)
            print(f"🤖 Mori 的回复: {response.json().get('reply', '无回复')}")
        except Exception as e:
            print(f"❌ 接口连接失败: {e}")


if __name__ == "__main__":
    main()
