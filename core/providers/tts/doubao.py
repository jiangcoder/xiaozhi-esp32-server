import os
import uuid
import json
import base64
import requests
from datetime import datetime
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.appid = config.get("appid")
        self.access_token = config.get("access_token")
        self.cluster = config.get("cluster")
        self.voice = config.get("voice")
        self.logger = setup_logging()
        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.header = {"Authorization": f"Bearer;{self.access_token}"}

    def generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    async def text_to_speak(self, text, output_file):
        request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": self.cluster
            },
            "user": {
                "uid": "1"
            },
            "audio": {
                "voice_type": self.voice,
                "encoding": "ogg_opus",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
                "rate": 16000
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        resp = requests.post(self.api_url, json.dumps(request_json), headers=self.header)
        if "data" in resp.json():
            duration = resp.json()["addition"]["duration"]
            data = resp.json()["data"]
            
            # 保存音频数据
            with open(output_file, "wb") as f:
                f.write(base64.b64decode(data))
            
            # 保存duration信息，去掉.opus后缀
            base_path = output_file.rsplit('.opus', 1)[0]
            duration_file = base_path + '.duration'
            with open(duration_file, "w") as f:
                f.write(str(duration))
                
            self.logger.bind(tag=TAG).info(f"音频文件生成成功: {text}")

    def get_audio_duration(self, file_path):
        """从duration文件中读取音频时长"""
        try:
            base_path = file_path.rsplit('.opus', 1)[0]
            duration_file = base_path + '.duration'
            with open(duration_file, "r") as f:
                duration = float(f.read().strip()) / 1000  # 转换为秒
            return duration
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"读取音频时长失败: {e}")
            return 0
