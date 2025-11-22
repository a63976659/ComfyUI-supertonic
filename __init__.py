"""
Supertonic TTS 插件 for ComfyUI
超快速本地文本转语音
"""

import os
import sys

# 添加 py 目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(current_dir, "py")
if py_dir not in sys.path:
    sys.path.insert(0, py_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Web 目录用于前端扩展
WEB_DIRECTORY = "./web"
