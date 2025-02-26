import asyncio
from config.logger import setup_logging
import os
import numpy as np
import opuslib_next
from pydub import AudioSegment
from abc import ABC, abstractmethod
import time

TAG = __name__
logger = setup_logging()


class TTSProviderBase(ABC):
    def __init__(self, config, delete_audio_file):
        self.delete_audio_file = delete_audio_file
        self.output_file = config.get("output_file")

    @abstractmethod
    def generate_filename(self):
        pass

    def to_tts(self, text):
        tmp_file = self.generate_filename()
        try:
            max_repeat_time = 5
            while not os.path.exists(tmp_file) and max_repeat_time > 0:
                asyncio.run(self.text_to_speak(text, tmp_file))
                if not os.path.exists(tmp_file):
                    max_repeat_time = max_repeat_time - 1
                    logger.bind(tag=TAG).error(f"语音生成失败: {text}:{tmp_file}，再试{max_repeat_time}次")

            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(f"语音生成成功: {text}:{tmp_file}，重试{5 - max_repeat_time}次")

            return tmp_file
        except Exception as e:
            logger.bind(tag=TAG).info(f"Failed to generate TTS file: {e}")
            return None

    @abstractmethod
    async def text_to_speak(self, text, output_file):
        pass

    def get_opus_data(self, file_path):
        """直接从opus文件获取数据和时长"""
        try:
            # 读取opus文件
            with open(file_path, 'rb') as f:
                opus_data = f.read()
            
            # 获取音频时长
            duration = self.get_audio_duration(file_path)
            
            opus_datas = []
            current_pos = 0
            
            while current_pos < len(opus_data):
                # 读取帧长度（前2个字节）
                if current_pos + 2 > len(opus_data):
                    break
                    
                frame_length = int.from_bytes(opus_data[current_pos:current_pos + 2], 'little')
                current_pos += 2
                
                # 确保有足够的数据读取
                if current_pos + frame_length > len(opus_data):
                    break
                    
                # 读取帧数据
                frame_data = opus_data[current_pos:current_pos + frame_length]
                opus_datas.append(frame_data)
                current_pos += frame_length
            
            return opus_datas, duration
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理opus文件失败: {e}")
            return [], 0
    
    @abstractmethod
    def get_audio_duration(self, file_path):
        """获取音频时长的抽象方法，由具体实现类提供"""
        pass

    def wav_to_opus_data(self, file_path):
        """保持原有接口兼容"""
        if file_path.endswith('.opus'):
            return self.get_opus_data(file_path)

        file_type = os.path.splitext(file_path)[1]
        if file_type:
            file_type = file_type.lstrip('.')
        audio = AudioSegment.from_file(file_path, format=file_type)

        duration = len(audio) / 1000.0

        # 转换为单声道和16kHz采样率（确保与编码器匹配）
        audio = audio.set_channels(1).set_frame_rate(16000)

        # 获取原始PCM数据（16位小端）
        raw_data = audio.raw_data

        # 初始化Opus编码器
        encoder = opuslib_next.Encoder(16000, 1, opuslib_next.APPLICATION_AUDIO)

        # 编码参数
        frame_duration = 60  # 60ms per frame
        frame_size = int(16000 * frame_duration / 1000)  # 960 samples/frame

        opus_datas = []
        # 按帧处理所有音频数据（包括最后一帧可能补零）
        for i in range(0, len(raw_data), frame_size * 2):  # 16bit=2bytes/sample
            # 获取当前帧的二进制数据
            chunk = raw_data[i:i + frame_size * 2]

            # 如果最后一帧不足，补零
            if len(chunk) < frame_size * 2:
                chunk += b'\x00' * (frame_size * 2 - len(chunk))

            # 转换为numpy数组处理
            np_frame = np.frombuffer(chunk, dtype=np.int16)

            # 编码Opus数据
            opus_data = encoder.encode(np_frame.tobytes(), frame_size)
            opus_datas.append(opus_data)

        return opus_datas, duration
