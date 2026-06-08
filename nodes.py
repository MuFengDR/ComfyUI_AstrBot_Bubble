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
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, PngImagePlugin

try:
    import folder_paths
except Exception:  # pragma: no cover - only available inside ComfyUI
    folder_paths = None


class AnyType(str):
    def __ne__(self, other: object) -> bool:
        return False


ASTRBUBBLE_ANY = AnyType("*")
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


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


def _find_sibling_video(path: Path) -> Path | None:
    for suffix in (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _resolve_video_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return path
        return _find_sibling_video(path)
    if folder_paths is None:
        return None
    for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
        getter = getattr(folder_paths, getter_name, None)
        if not getter:
            continue
        candidate = Path(getter()) / raw
        if candidate.exists() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate
        if candidate.exists():
            replacement = _find_sibling_video(candidate)
            if replacement is not None:
                return replacement
    return None


def _resolve_audio_path(value: Any) -> Path | None:
    for candidate in _iter_video_candidates(value):
        raw = str(candidate or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.suffix.lower() in AUDIO_SUFFIXES:
            return path
        if folder_paths is None:
            continue
        for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
            getter = getattr(folder_paths, getter_name, None)
            if not getter:
                continue
            candidate_path = Path(getter()) / raw
            if candidate_path.exists() and candidate_path.suffix.lower() in AUDIO_SUFFIXES:
                return candidate_path
    return None


def _ffmpeg_exe() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _video_codec_args(container: str, crf: int, pix_fmt: str) -> tuple[str, list[str], str]:
    value = str(container or "video/h264-mp4").lower()
    if "webm" in value:
        return ".webm", ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"], "video/webm"
    return ".mp4", ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pix_fmt], "video/h264-mp4"


def _save_frames_video(
    images: torch.Tensor,
    target: Path,
    fps: float,
    container: str,
    crf: int,
    pix_fmt: str,
    audio: Any = None,
) -> None:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError("images must be an IMAGE tensor")
    frames = torch.clamp(images.detach().cpu(), 0, 1).numpy()
    frames = (frames * 255.0).astype(np.uint8)
    height, width = frames.shape[1], frames.shape[2]
    suffix, codec_args, _ = _video_codec_args(container, crf, pix_fmt)
    ffmpeg = _ffmpeg_exe()
    audio_path = _resolve_audio_path(audio)

    encode_target = target
    temp_video = None
    if audio_path is not None:
        tmp = tempfile.NamedTemporaryFile(prefix="astrbubble_video_", suffix=suffix, delete=False)
        tmp.close()
        temp_video = Path(tmp.name)
        encode_target = temp_video

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(1e-3, float(fps or 24))),
        "-i",
        "-",
        *codec_args,
        str(encode_target),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(b"".join(frame.tobytes() for frame in frames))
    if proc.returncode != 0:
        raise RuntimeError(err.decode("utf-8", "ignore")[-1000:] or "ffmpeg encode failed")

    if audio_path is not None and temp_video is not None:
        mux_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ]
        mux = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        temp_video.unlink(missing_ok=True)
        if mux.returncode != 0:
            raise RuntimeError(mux.stderr.decode("utf-8", "ignore")[-1000:] or "ffmpeg mux failed")


def _video_ref(filename: str, subfolder: str = "", file_type: str = "output", fmt: str = "") -> dict[str, str]:
    suffix = Path(filename).suffix.lstrip(".") or "mp4"
    item = {
        "filename": Path(filename).name,
        "subfolder": str(subfolder or "").replace("\\", "/"),
        "type": str(file_type or "output"),
        "format": fmt or f"video/{suffix}",
    }
    return item


def _is_video_filename(value: str) -> bool:
    return Path(str(value or "")).suffix.lower() in VIDEO_SUFFIXES


def _output_ref_from_path(path: Path) -> dict[str, str] | None:
    if folder_paths is None:
        return None
    try:
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        resolved = path.resolve()
        rel = resolved.relative_to(output_dir)
    except Exception:
        return None
    subfolder = "" if rel.parent == Path(".") else rel.parent.as_posix()
    return _video_ref(rel.name, subfolder, "output")


def _output_ref_from_name(value: str) -> dict[str, str] | None:
    source = _resolve_video_path(value)
    if source is None:
        return None
    return _output_ref_from_path(source)


def _iter_video_refs(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        filename = value.get("filename")
        file_type = str(value.get("type") or "output")
        if filename and file_type == "output" and _is_video_filename(str(filename)):
            yield _video_ref(
                str(filename),
                str(value.get("subfolder") or ""),
                file_type,
                str(value.get("format") or ""),
            )
        elif filename and file_type == "output":
            ref = _output_ref_from_name(str(Path(str(value.get("subfolder") or "")) / str(filename)))
            if ref is not None:
                yield ref
        for key in ("file", "path", "video", "name"):
            item = value.get(key)
            if item:
                yield from _iter_video_refs(item)
        for key in ("files", "videos", "gifs"):
            item = value.get(key)
            if item:
                yield from _iter_video_refs(item)
        return
    if isinstance(value, (str, os.PathLike)):
        ref = _output_ref_from_name(str(value))
        if ref is not None:
            yield ref
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_video_refs(item)


def _iter_video_candidates(value: Any):
    if value is None:
        return
    if isinstance(value, (str, os.PathLike)):
        text = str(value).strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for key in ("filename", "file", "path", "video", "name"):
            item = value.get(key)
            if item:
                yield from _iter_video_candidates(item)
        for key in ("files", "videos", "gifs"):
            item = value.get(key)
            if item:
                yield from _iter_video_candidates(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_video_candidates(item)


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
                "explain": ("STRING", {"default": "文本输入"}),
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, explain: str, text: str):
        return (str(text or ""),)


class AstrBubbleImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "explain": ("STRING", {"default": "图片输入"}),
                "image_base64": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, explain: str, image_base64: str):
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
                "explain": ("STRING", {"default": "视频输入"}),
                "video": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "AstrBubble/Input"

    def load(self, index: int, explain: str, video: str):
        return (str(video or ""),)


class AstrBubbleTextOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "explain": ("STRING", {"default": "文本输出"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "optional": ("BOOLEAN", {"default": False}),
                "text": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def save(self, index: int, explain: str, enabled: bool, optional: bool, text: str):
        return {"ui": {"text": [str(text or "")]}, "result": ()}


class AstrBubbleImageOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "explain": ("STRING", {"default": "图片输出"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "optional": ("BOOLEAN", {"default": False}),
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
        explain: str,
        enabled: bool,
        optional: bool,
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
                "explain": ("STRING", {"default": "视频输出"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "optional": ("BOOLEAN", {"default": False}),
                "video": (ASTRBUBBLE_ANY, {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "AstrBubble"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def save(self, index: int, explain: str, enabled: bool, optional: bool, video: Any, filename_prefix: str = "AstrBubble"):
        refs = list(_iter_video_refs(video) or [])
        if refs:
            return {"ui": {"gifs": refs}, "result": ()}

        source = None
        for candidate in _iter_video_candidates(video):
            source = _resolve_video_path(candidate)
            if source is not None:
                break
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
        ref = _output_ref_from_path(target) or _video_ref(target.name, "", "output", f"video/{suffix.lstrip('.')}")
        return {
            "ui": {
                "gifs": [ref]
            },
            "result": (),
        }


class AstrBubbleVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "explain": ("STRING", {"default": "视频输出"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "optional": ("BOOLEAN", {"default": False}),
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
                "format": (["video/h264-mp4", "video/webm"], {"default": "video/h264-mp4"}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AstrBubble"}),
            },
            "optional": {
                "audio": (ASTRBUBBLE_ANY,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "AstrBubble/Output"

    def combine(
        self,
        index: int,
        explain: str,
        enabled: bool,
        optional: bool,
        images: torch.Tensor,
        fps: float,
        format: str,
        pix_fmt: str,
        crf: int,
        filename_prefix: str = "AstrBubble",
        audio: Any = None,
    ):
        output_dir = _output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix, _, mime = _video_codec_args(format, crf, pix_fmt)
        safe_prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename_prefix)
        target_name = f"{safe_prefix}_{index}_{os.getpid()}{suffix}"
        target = output_dir / target_name
        _save_frames_video(images, target, fps, format, crf, pix_fmt, audio)
        ref = _output_ref_from_path(target) or _video_ref(target.name, "", "output", mime)
        return {"ui": {"gifs": [ref]}, "result": (target.name,)}


NODE_CLASS_MAPPINGS = {
    "AstrBubble_TextInput": AstrBubbleTextInput,
    "AstrBubble_ImageInput": AstrBubbleImageInput,
    "AstrBubble_VideoInput": AstrBubbleVideoInput,
    "AstrBubble_TextOutput": AstrBubbleTextOutput,
    "AstrBubble_ImageOutput": AstrBubbleImageOutput,
    "AstrBubble_VideoOutput": AstrBubbleVideoOutput,
    "AstrBubble_VideoCombine": AstrBubbleVideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AstrBubble_TextInput": "Bubble Text Input",
    "AstrBubble_ImageInput": "Bubble Image Input",
    "AstrBubble_VideoInput": "Bubble Video Input",
    "AstrBubble_TextOutput": "Bubble Text Output",
    "AstrBubble_ImageOutput": "Bubble Image Output",
    "AstrBubble_VideoOutput": "Bubble Video Output",
    "AstrBubble_VideoCombine": "Bubble Video Combine",
}




