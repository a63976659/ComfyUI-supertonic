"""
ComfyUI Node for Supertonic TTS
"""

import os
import numpy as np
import torch
import soundfile as sf
from io import BytesIO

# Import helper functions
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(current_dir, "py")
if py_dir not in sys.path:
    sys.path.insert(0, py_dir)

from helper import load_text_to_speech, load_voice_style, Style


class SupertonicTTS:
    """
    Supertonic 文本转语音节点
    超快速本地 TTS，支持自然文本处理
    """
    
    def __init__(self):
        self.text_to_speech = None
        self.current_model_dir = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的音色
        voice_styles_dir = os.path.join(current_dir, "assets", "voice_styles")
        voice_styles = []
        voice_map = {}
        
        if os.path.exists(voice_styles_dir):
            files = [f[:-5] for f in os.listdir(voice_styles_dir) if f.endswith('.json')]
            # 创建中文映射
            for f in files:
                if f == "M1":
                    voice_map["男声1"] = "M1"
                elif f == "M2":
                    voice_map["男声2"] = "M2"
                elif f == "F1":
                    voice_map["女声1"] = "F1"
                elif f == "F2":
                    voice_map["女声2"] = "F2"
            voice_styles = list(voice_map.keys())
        
        if not voice_styles:
            voice_styles = ["男声1", "男声2", "女声1", "女声2"]
        
        return {
            "required": {
                "输入文本": ("STRING", {
                    "multiline": True,
                    "default": "今天早上我在公园散步，鸟鸣和微风的声音让人心旷神怡。"
                }),
                "音色选择": (voice_styles, {
                    "default": voice_styles[0] if voice_styles else "男声1"
                }),
                "推理步数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "步数越多质量越高但速度越慢。推荐: 2(快速) 5(默认) 10(高质量)"
                }),
                "语速倍数": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "1.0为正常速度，大于1.0加快，小于1.0减慢"
                }),
                "句间停顿": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "句子之间的停顿时长(秒)"
                }),
            },
            "optional": {
                "使用GPU": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否使用GPU加速(当前建议使用CPU)"
                }),
                "输出采样率": (["44100 Hz (原始)", "48000 Hz (Opus兼容)"], {
                    "default": "44100 Hz (原始)",
                    "tooltip": "如果使用Opus格式保存，请选择48000 Hz避免发音和时长问题"
                }),
                "分句策略": (["标准（。！？）", "增强（。！？，、；：……）"], {
                    "default": "标准（。！？）",
                    "tooltip": "选择分句规则，增强模式会在逗号等处也加入停顿"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频输出",)
    FUNCTION = "generate_speech"
    CATEGORY = "🎙️ Supertonic 语音合成/语音合成"
    
    def generate_speech(self, 输入文本, 音色选择, 推理步数, 语速倍数, 句间停顿, 使用GPU=False, 输出采样率="44100 Hz (原始)", 分句策略="标准（。！？）"):
        """使用 Supertonic TTS 生成语音"""
        
        try:
            # 中文到英文音色映射
            voice_map = {
                "男声1": "M1",
                "男声2": "M2",
                "女声1": "F1",
                "女声2": "F2"
            }
            voice_style = voice_map.get(音色选择, 音色选择)
            text = 输入文本
            total_steps = 推理步数
            speed = 语速倍数
            silence_duration = 句间停顿
            use_gpu = 使用GPU
            # 分句策略映射
            segmentation_strategy = "standard" if 分句策略.startswith("标准") else "enhanced"
            
            # Load model if not loaded or model directory changed
            onnx_dir = os.path.join(current_dir, "assets", "onnx")
            
            if not os.path.exists(onnx_dir):
                raise FileNotFoundError(
                    f"ONNX 模型未找到: {onnx_dir}\n"
                    "请下载模型: git clone https://huggingface.co/Supertone/supertonic assets"
                )
            
            if self.text_to_speech is None or self.current_model_dir != onnx_dir:
                self.text_to_speech = load_text_to_speech(onnx_dir, use_gpu)
                self.current_model_dir = onnx_dir
            
            # Load voice style
            voice_style_path = os.path.join(current_dir, "assets", "voice_styles", f"{voice_style}.json")
            
            if not os.path.exists(voice_style_path):
                raise FileNotFoundError(
                    f"音色文件 '{voice_style}' 未找到: {voice_style_path}\n"
                    "请确保音色文件已下载到 assets/voice_styles/ 目录"
                )
            
            style = load_voice_style([voice_style_path], verbose=False)
            
            # 生成语音
            import time
            start_time = time.time()
            print(f"[Supertonic TTS] 文本: {text}")
            
            wav, duration = self.text_to_speech(
                text, 
                style, 
                total_steps, 
                speed,
                silence_duration,
                segmentation_strategy
            )
            
            # Trim to actual duration
            wav_trimmed = wav[0, :int(self.text_to_speech.sample_rate * duration[0].item())]
            
            # 音频归一化处理：确保在 [-1, 1] 范围内
            max_val = np.abs(wav_trimmed).max()
            if max_val > 1.0:
                wav_trimmed = wav_trimmed / max_val
            elif max_val < 0.01:
                wav_trimmed = wav_trimmed / max_val * 0.5
            
            # 处理输出采样率
            target_sample_rate = self.text_to_speech.sample_rate  # 默认 44100
            if "输出采样率" in locals() and 输出采样率 == "48000 Hz (Opus兼容)":
                target_sample_rate = 48000
                # 使用 torchaudio 重采样
                import torchaudio
                wav_tensor = torch.from_numpy(wav_trimmed).float().unsqueeze(0)  # [1, samples]
                wav_resampled = torchaudio.functional.resample(
                    wav_tensor, 
                    self.text_to_speech.sample_rate, 
                    target_sample_rate
                )
                wav_trimmed = wav_resampled.squeeze(0).numpy()
            
            # Convert to ComfyUI audio format
            # ComfyUI expects audio as a dict with 'waveform' and 'sample_rate'
            # waveform shape: [batch, channels, samples]
            waveform = torch.from_numpy(wav_trimmed).float().unsqueeze(0).unsqueeze(0)
            
            audio_output = {
                "waveform": waveform,  # [1, 1, samples] - mono audio
                "sample_rate": target_sample_rate
            }
            
            elapsed_time = time.time() - start_time
            print(f"[Supertonic TTS] 生成完成，耗时: {elapsed_time:.2f}秒")
            
            return (audio_output,)
            
        except Exception as e:
            print(f"[Supertonic TTS] ❌ 错误: {str(e)}")
            raise


class SupertonicBatchTTS:
    """
    Supertonic 批量文本转语音节点
    同时处理多个文本以提高效率
    """
    
    def __init__(self):
        self.text_to_speech = None
        self.current_model_dir = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的音色
        voice_styles_dir = os.path.join(current_dir, "assets", "voice_styles")
        voice_styles = []
        voice_map = {}
        
        if os.path.exists(voice_styles_dir):
            files = [f[:-5] for f in os.listdir(voice_styles_dir) if f.endswith('.json')]
            for f in files:
                if f == "M1":
                    voice_map["男声1"] = "M1"
                elif f == "M2":
                    voice_map["男声2"] = "M2"
                elif f == "F1":
                    voice_map["女声1"] = "F1"
                elif f == "F2":
                    voice_map["女声2"] = "F2"
            voice_styles = list(voice_map.keys())
        
        if not voice_styles:
            voice_styles = ["男声1", "男声2", "女声1", "女声2"]
        
        return {
            "required": {
                "文本1": ("STRING", {
                    "multiline": True,
                    "default": "第一段要合成的文本。"
                }),
                "音色1": (voice_styles, {
                    "default": voice_styles[0] if voice_styles else "男声1"
                }),
                "推理步数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "步数越多质量越高但速度越慢"
                }),
                "语速倍数": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "1.0为正常速度"
                }),
            },
            "optional": {
                "文本2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选的第二段文本"
                }),
                "音色2": (voice_styles, {
                    "default": voice_styles[0] if voice_styles else "男声1"
                }),
                "文本3": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选的第三段文本"
                }),
                "音色3": (voice_styles, {
                    "default": voice_styles[0] if voice_styles else "男声1"
                }),
                "使用GPU": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否使用GPU加速"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频列表",)
    FUNCTION = "generate_batch_speech"
    CATEGORY = "🎙️ Supertonic 语音合成/语音合成"
    OUTPUT_IS_LIST = (True,)
    
    def generate_batch_speech(self, 文本1, 音色1, 推理步数, 语速倍数, 
                             文本2="", 音色2="男声1", 
                             文本3="", 音色3="男声1",
                             使用GPU=False):
        """批量生成多个文本的语音"""
        
        # 中文到英文音色映射
        voice_map = {
            "男声1": "M1",
            "男声2": "M2",
            "女声1": "F1",
            "女声2": "F2"
        }
        
        text_1 = 文本1
        text_2 = 文本2
        text_3 = 文本3
        voice_style_1 = voice_map.get(音色1, 音色1)
        voice_style_2 = voice_map.get(音色2, 音色2)
        voice_style_3 = voice_map.get(音色3, 音色3)
        total_steps = 推理步数
        speed = 语速倍数
        use_gpu = 使用GPU
        
        # 收集非空文本及其音色
        texts = [text_1]
        voice_styles = [voice_style_1]
        
        if text_2.strip():
            texts.append(text_2)
            voice_styles.append(voice_style_2)
        
        if text_3.strip():
            texts.append(text_3)
            voice_styles.append(voice_style_3)
        
        # 加载模型
        onnx_dir = os.path.join(current_dir, "assets", "onnx")
        
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(
                f"ONNX 模型未找到: {onnx_dir}\n"
                "请下载模型: git clone https://huggingface.co/Supertone/supertonic assets"
            )
        
        if self.text_to_speech is None or self.current_model_dir != onnx_dir:
            self.text_to_speech = load_text_to_speech(onnx_dir, use_gpu)
            self.current_model_dir = onnx_dir
        
        # 加载音色文件
        voice_style_paths = []
        for vs in voice_styles:
            path = os.path.join(current_dir, "assets", "voice_styles", f"{vs}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"音色文件 '{vs}' 未找到: {path}")
            voice_style_paths.append(path)
        
        style = load_voice_style(voice_style_paths, verbose=False)
        
        # 批量生成语音
        import time
        start_time = time.time()
        print(f"[Supertonic TTS Batch] 批量生成 {len(texts)} 段文本")
        
        wav, duration = self.text_to_speech.batch(texts, style, total_steps, speed)
        
        # Convert to ComfyUI audio format (list of audio dicts)
        audio_outputs = []
        for i in range(len(texts)):
            wav_trimmed = wav[i, :int(self.text_to_speech.sample_rate * duration[i].item())]
            
            # 音频归一化处理
            max_val = np.abs(wav_trimmed).max()
            if max_val > 1.0:
                wav_trimmed = wav_trimmed / max_val
            elif max_val < 0.01:
                wav_trimmed = wav_trimmed / max_val * 0.5
            
            waveform = torch.from_numpy(wav_trimmed).float().unsqueeze(0).unsqueeze(0)
            audio_output = {
                "waveform": waveform,  # [1, 1, samples] - mono audio
                "sample_rate": self.text_to_speech.sample_rate
            }
            audio_outputs.append(audio_output)
        
        elapsed_time = time.time() - start_time
        print(f"[Supertonic TTS Batch] 批量生成完成，耗时: {elapsed_time:.2f}秒")
        
        return (audio_outputs,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "SupertonicTTS": SupertonicTTS,
    "SupertonicBatchTTS": SupertonicBatchTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SupertonicTTS": "🎙️ Supertonic 语音合成",
    "SupertonicBatchTTS": "🎙️ Supertonic 批量语音合成",
}
