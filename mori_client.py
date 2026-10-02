import sounddevice as sd
import numpy as np
import whisper
import requests
import time
import os
import json
import logging


# ==========================================
# 1. 关键配置集中管理
# ==========================================
class Config:
    # API 设定
    # 如果还没有真实接口，保持为空，代码会自动走 Mock 逻辑
    MORI_API_URL = os.getenv("MORI_API_URL", "")
    API_TIMEOUT = 10  # 秒
    USER_ID = "test_user_001"

    # 录音设定
    RECORD_DURATION = 5  # 秒
    SAMPLE_RATE = 16000


# ==========================================
# 2. 日志配置 (替代 print 和全局抑制警告)
# ==========================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# 3. 核心功能模块
# ==========================================
class MoriVoiceClient:
    def __init__(self):
        logger.info("⏳ 正在加载 Whisper 模型...")
        self.model = whisper.load_model("base")
        logger.info("✅ 模型加载完成！")

    def record_audio(self):
        """录音模块：固定时长，内存流转"""
        logger.info(f"🎤 [开始录音] 请说话，限时 {Config.RECORD_DURATION} 秒...")
        try:
            audio_data = sd.rec(
                int(Config.RECORD_DURATION * Config.SAMPLE_RATE),
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            logger.info("✅ 录音结束。")
            return audio_data.flatten()
        except Exception as e:
            logger.error(f"❌ 录音失败: {e}")
            return None

    def process_asr(self, audio_array):
        """ASR 模块：包含耗时统计"""
        logger.info("⏳ 开始语音识别 (ASR)...")
        start_time = time.time()
        try:
            result = self.model.transcribe(audio_array)
            latency = time.time() - start_time
            return result["text"].strip(), latency
        except Exception as e:
            logger.error(f"❌ ASR 识别异常: {e}")
            return "", time.time() - start_time

    def call_mori_api(self, text):
        """API 通信模块：包含 Mock 逻辑、耗时统计和状态码检查"""
        if not text:
            return None, 0.0

        payload = {"user_id": Config.USER_ID, "message": text}

        logger.info("⏳ 正在请求 Mori API...")
        start_time = time.time()

        # Mock 逻辑：如果没配 URL，走本地模拟返回
        if not Config.MORI_API_URL:
            logger.warning("未配置 MORI_API_URL，使用 Mock API 返回数据。")
            time.sleep(0.5)  # 模拟网络延迟
            latency = time.time() - start_time
            return {"reply": f"[Mock 响应] 我收到了你的话：{text}"}, latency

        # 真实网络请求逻辑
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                Config.MORI_API_URL,
                json=payload,
                headers=headers,
                timeout=Config.API_TIMEOUT,
            )
            response.raise_for_status()  # 检查 HTTP 状态码
            latency = time.time() - start_time
            return response.json(), latency

        except requests.exceptions.Timeout:
            logger.error(f"❌ API 请求超时 (>{Config.API_TIMEOUT}s)")
        except requests.exceptions.HTTPError as err:
            logger.error(f"❌ API HTTP 错误: {err}")
        except Exception as e:
            logger.error(f"❌ API 请求异常: {e}")

        return None, time.time() - start_time


# ==========================================
# 4. 主干链路 (Pipeline)
# ==========================================
def main():
    client = MoriVoiceClient()

    while True:
        input("\n按 [Enter] 键开始一次测试 (输入 q 退出): ")

        audio_array = client.record_audio()
        if audio_array is None:
            continue

        # 1. 执行 ASR
        user_text, asr_latency = client.process_asr(audio_array)
        if not user_text:
            logger.warning("⚠️️ 未检测到有效输入，跳过 API 请求。")
            continue

        logger.info(f"🗣️ 识别文本: 【{user_text}】")

        # 2. 调用 API
        api_response, api_latency = client.call_mori_api(user_text)

        if api_response:
            logger.info(f"🤖 Mori 回复: {api_response.get('reply', '无回复内容')}")

        # 3. 性能度量打印
        total_latency = asr_latency + api_latency
        logger.info(
            f"📊 [性能度量] ASR 耗时: {asr_latency:.3f}s | API 耗时: {api_latency:.3f}s | 总处理耗时: {total_latency:.3f}s"
        )
        print("-" * 50)


if __name__ == "__main__":
    main()
