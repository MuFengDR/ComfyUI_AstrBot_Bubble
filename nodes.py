# -*- coding: utf-8 -*-
"""Custom ComfyUI nodes used by astrbot_plugin_comfyui_bubble.

The AstrBot plugin scans these class_type names and injects values by the
explicit 1-based `index` field. Keep class names stable.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, PngImagePlugin

try:
    import folder_paths
except Exception:  # pragma: no cover - only available inside ComfyUI
    folder_paths = None


def _clean_base64(value: str) -> str:
    text = str(value or "").strip()
    if "," in text and text.lower().startswith("data:"):
        return text.split(",", 1)[1].strip()
    return text


def _blank_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _resolve_video_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    if folder_paths is None:
        return None
    for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
        getter = getattr(folder_paths, getter_name, None)
        if not getter:
            continue
        candidate = Path(getter()) / raw
        if candidate.exists():
            return candidate
    return None


def _output_dir() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return Path.cwd()


class AstrBubbleTextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "文本输入"}),
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, label: str, text: str):
        return (str(text or ""),)


class AstrBubbleImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "图片输入"}),
                "image_base64": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, label: str, image_base64: str):
        payload = _clean_base64(image_base64)
        if not payload:
            return (_blank_image(),)
        image = Image.open(io.BytesIO(base64.b64decode(payload)))
        return (_pil_to_tensor(image),)


class AstrBubbleVideoInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "视频输入"}),
                "video": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, label: str, video: str):
        return (str(video or ""),)


class AstrBubbleTextOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "文本输出"}),
                "text": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def save(self, index: int, label: str, text: str):
        return {"ui": {"text": [str(text or "")]}, "result": ()}


class AstrBubbleImageOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "图片输出"}),
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "AstrBubble"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def save(
        self,
        index: int,
        label: str,
        images: torch.Tensor,
        filename_prefix: str = "AstrBubble",
        prompt: Any = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ):
        output_dir = _output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        if folder_paths is not None:
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix, str(output_dir), images[0].shape[1], images[0].shape[0]
                )
            )
            full_output_folder = Path(full_output_folder)
        else:
            full_output_folder = output_dir
            filename = filename_prefix
            counter = 1
            subfolder = ""

        for batch_number, image in enumerate(images):
            array = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            metadata = PngImagePlugin.PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", str(prompt))
            if extra_pnginfo:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(str(key), str(value))
            batch_filename = filename.replace("%batch_num%", str(batch_number))
            file_name = f"{batch_filename}_{counter:05}_.png"
            pil_image.save(full_output_folder / file_name, pnginfo=metadata, compress_level=4)
            results.append({"filename": file_name, "subfolder": subfolder, "type": "output"})
            counter += 1

        return {"ui": {"images": results}, "result": ()}


class AstrBubbleVideoOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "label": ("STRING", {"default": "视频输出"}),
                "video": ("STRING", {"default": "", "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "AstrBubble"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def save(self, index: int, label: str, video: str, filename_prefix: str = "AstrBubble"):
        source = _resolve_video_path(video)
        if source is None:
            return {"ui": {"gifs": []}, "result": ()}
        output_dir = _output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix or ".mp4"
        safe_prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename_prefix)
        target_name = f"{safe_prefix}_{index}_{os.getpid()}_{source.stem}{suffix}"
        target = output_dir / target_name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return {
            "ui": {
                "gifs": [
                    {
                        "filename": target.name,
                        "subfolder": "",
                        "type": "output",
                        "format": f"video/{suffix.lstrip('.')}",
                    }
                ]
            },
            "result": (),
        }


NODE_CLASS_MAPPINGS = {
    "AstrBubble_TextInput": AstrBubbleTextInput,
    "AstrBubble_ImageInput": AstrBubbleImageInput,
    "AstrBubble_VideoInput": AstrBubbleVideoInput,
    "AstrBubble_TextOutput": AstrBubbleTextOutput,
    "AstrBubble_ImageOutput": AstrBubbleImageOutput,
    "AstrBubble_VideoOutput": AstrBubbleVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AstrBubble_TextInput": "Bubble Text Input",
    "AstrBubble_ImageInput": "Bubble Image Input",
    "AstrBubble_VideoInput": "Bubble Video Input",
    "AstrBubble_TextOutput": "Bubble Text Output",
    "AstrBubble_ImageOutput": "Bubble Image Output",
    "AstrBubble_VideoOutput": "Bubble Video Output",
}
