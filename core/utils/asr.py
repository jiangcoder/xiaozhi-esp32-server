import time
import wave
import os
import sys
import io
from abc import ABC, abstractmethod
from config.logger import setup_logging
from typing import Optional, Tuple, List
import uuid
import pyogg
import base64

import opuslib_next
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

TAG = __name__
logger = setup_logging()

# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())

class ASR(ABC):
    @abstractmethod
    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """解码Opus数据并保存为WAV文件"""
        pass

    @abstractmethod
    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass


class FunASR(ASR):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        with CaptureOutput():
            self.model = AutoModel(
                model=self.model_dir,
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
                hub="hf"
                # device="cuda:0",  # 启用GPU加速
            )

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """将Opus音频数据解码并保存为WAV文件"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)
        opus_file_name = f"asr_{session_id}_{uuid.uuid4()}.opus"
        opus_file_path = os.path.join(self.output_dir, opus_file_name)

        # 保存原始Opus数据
        # 直接保存原始Opus数据
        """将Opus数据封装为Ogg Opus文件, 并Base64编码"""
        opus_file_name = f"asr_{session_id}_{uuid.uuid4()}.opus"  # We will create Ogg Opus file, but extension .opus is conventional
        opus_file_path = os.path.join(self.output_dir, opus_file_name)

        # try:
        #     # 使用 pyogg 创建 Ogg Opus 文件
        #     with pyogg.OggOpusWriter(opus_file_path, sample_rate=16000,
        #                              channels=1) as writer:  # 假设 16kHz, 单声道，根据你的硬件终端实际参数调整
        #         for opus_packet in opus_data:
        #             writer.write(opus_packet)  # 直接写入 Opus 数据包
        #
        #     logger.bind(tag=TAG).info(f"Ogg Opus 文件已保存: {opus_file_path}")
        #
        #     # 读取 Ogg Opus 文件为二进制数据
        #     with open(opus_file_path, "rb") as f:
        #         opus_binary_data = f.read()
        #
        #     # Base64 编码二进制数据
        #     base64_opus_string = base64.b64encode(opus_binary_data).decode('utf-8')  # 编码为 base64 字符串 (文本)
        #     logger.bind(tag=TAG).info(f"Ogg Opus 文件已 Base64 编码:, {base64_opus_string}")
        #
        # except Exception as e:
        #     logger.bind(tag=TAG).error(f"Ogg Opus 封装或 Base64 编码失败: {e}", exc_info=True)
        #     return None  # 或根据你的需求返回错误指示

        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []

        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).info(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 语音识别
            start_time = time.time()
            result = self.model.generate(
                input=file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            logger.bind(tag=TAG).info(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"文件删除失败: {file_path} | 错误: {e}")


def create_instance(class_name: str, *args, **kwargs) -> ASR:
    """工厂方法创建ASR实例"""
    cls_map = {
        "FunASR": FunASR,
        # 可扩展其他ASR实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的ASR类型: {class_name}")
