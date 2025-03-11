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

import struct
import subprocess
import tempfile

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
        """将Opus音频数据解码并保存为WAV文件和标准的Ogg Opus文件"""
        base_name = f"asr_{session_id}_{uuid.uuid4()}"
        wav_path = os.path.join(self.output_dir, f"{base_name}.wav")
        opus_path = os.path.join(self.output_dir, f"{base_name}.opus")
        
        # 解码Opus数据为PCM并保存为WAV文件
        decoder = opuslib_next.Decoder(16000, 1)  # 16kHz, 单声道
        pcm_data = []
    
        for opus_packet in opus_data:
            try:
                pcm_frame = decoder.decode(opus_packet, 960)  # 960 samples = 60ms
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus解码错误: {e}", exc_info=True)
    
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))
        
        # 使用FFmpeg将WAV转换为标准Ogg Opus文件
        try:
            cmd = [
                'ffmpeg',
                '-i', wav_path,  # 输入WAV文件
                '-c:a', 'libopus',  # 使用libopus编码器
                '-b:a', '32k',  # 比特率
                opus_path  # 输出文件
            ]
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.bind(tag=TAG).error(f"FFmpeg转换失败: {result.stderr}")
            else:
                logger.bind(tag=TAG).info(f"已保存标准Ogg Opus文件: {opus_path}")
                
                # 将Opus文件转换为Base64字符串
                with open(opus_path, 'rb') as f:
                    opus_binary = f.read()
                    opus_base64 = base64.b64encode(opus_binary).decode('utf-8')
                    logger.bind(tag=TAG).info(f"Opus文件Base64编码长度: {len(opus_base64)}")
                    # 可以在这里存储或返回base64字符串
                    
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存Ogg Opus文件失败: {e}", exc_info=True)
        
        return wav_path

    def get_opus_base64(self, file_path: str) -> Optional[str]:
        """获取Opus文件的Base64编码字符串"""
        opus_path = file_path.replace('.wav', '.opus')
        if not os.path.exists(opus_path):
            logger.bind(tag=TAG).error(f"Opus文件不存在: {opus_path}")
            return None
            
        try:
            with open(opus_path, 'rb') as f:
                opus_binary = f.read()
                return base64.b64encode(opus_binary).decode('utf-8')
        except Exception as e:
            logger.bind(tag=TAG).error(f"Opus文件Base64编码失败: {e}", exc_info=True)
            return None

    def speech_to_text(self, opus_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """语音转文本主处理逻辑，返回识别文本、WAV文件路径和Opus Base64编码"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(opus_data, session_id)
            logger.bind(tag=TAG).info(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")
    
            # 获取Opus文件的Base64编码
            opus_base64 = self.get_opus_base64(file_path)
    
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
    
            return text, file_path, opus_base64
    
        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}", exc_info=True)
            return None, None, None
    
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
