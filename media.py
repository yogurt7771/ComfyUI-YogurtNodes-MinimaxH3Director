"""Director media: editor uploads and graph inputs, adapted to the 24 fps H3 timeline."""

import os

import av
import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.audio
import comfy.model_management
import folder_paths
import node_helpers
from comfy_extras.nodes_audio import load as load_audio_file
from comfy_extras.nodes_minimax_h3 import _resize

from .cache import tensor_digest
from .project import FPS


def resolve_path(item):
    base = os.path.realpath(folder_paths.get_directory_by_type(item["type"]))
    path = os.path.realpath(os.path.join(base, item["subfolder"], item["filename"]))
    try:
        inside = os.path.commonpath([path, base]) == base
    except ValueError:
        inside = False
    if not inside:
        raise ValueError(f"素材路径越出 {item['type']} 目录：{item['filename']}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到素材 {item['type']}/{item['subfolder']}/{item['filename']}，请重新上传。")
    return path


def fit(frames, width, height, crop="center"):
    """[N, H, W, C] -> [N, height, width, 3]; ``crop`` follows the native H3 nodes."""
    if frames.shape[1:3] == (height, width):
        return frames[..., :3]
    return torch.cat([_resize(frames[i:i + 16], width, height, crop) for i in range(0, frames.shape[0], 16)])


class Media:
    """Execution-scoped access to the project's media; decoded audio is kept only for this run."""

    def __init__(self, images=None, masks=None):
        self.inputs = {**(images or {}), **(masks or {})}
        self.inputs = {k: v for k, v in self.inputs.items() if v is not None}
        self._identity = {}
        self._audio = {}

    def graph(self, item):
        value = self.inputs.get(item["input"])
        if value is None:
            raise ValueError(f"导演台引用了图输入 {item['input']}，但该输入没有连接。")
        return value

    def identity(self, item):
        if item is None:
            return None
        if "input" in item:
            key = item["input"]
            if key not in self._identity:
                self._identity[key] = tensor_digest(self.graph(item))
            return ["input", self._identity[key]]
        stat = os.stat(resolve_path(item))
        return [item["type"], item["subfolder"], item["filename"], stat.st_size, stat.st_mtime_ns]

    def image(self, item):
        if "input" in item:
            return self.graph(item)[:1, ..., :3].float()
        image = node_helpers.pillow(Image.open, resolve_path(item))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None]

    def mask(self, item):
        """[F, H, W] in 0..1 where 1 marks pixels to regenerate; a painted PNG is a single frame."""
        if "input" in item:
            value = self.graph(item).float()
            return value if value.ndim == 3 else value[None]
        image = ImageOps.exif_transpose(node_helpers.pillow(Image.open, resolve_path(item)))
        if "A" in image.getbands() and image.getextrema()[-1][0] < 255:
            plane = image.getchannel("A")
        else:
            plane = image.convert("L")
        return torch.from_numpy(np.array(plane).astype(np.float32) / 255.0)[None]

    def video(self, item, start, count, width, height, crop="center"):
        """``count`` frames from 24 fps frame ``start``; each source frame is held until the next one starts."""
        if "input" in item:
            frames = self.graph(item)[start:start + count]
            if frames.shape[0] == 0:
                raise ValueError(f"图输入 {item['input']} 在第 {start} 帧之后没有画面。")
            frames = fit(frames.float(), width, height, crop)
        else:
            frames = self._decode_video(resolve_path(item), start, count, width, height, crop)
        if frames.shape[0] < count:
            frames = torch.cat([frames, frames[-1:].expand(count - frames.shape[0], -1, -1, -1)])
        return frames

    def _decode_video(self, path, start, count, width, height, crop):
        def convert(frame):
            pixels = torch.from_numpy(frame.to_ndarray(format="rgb24")).float().div_(255)[None]
            return fit(pixels, width, height, crop)

        frames, slot, current, converted = [], start, None, None
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            origin = float((stream.start_time or 0) * stream.time_base)
            if start > 0:
                container.seek(int((origin + start / FPS) / stream.time_base), stream=stream, backward=True)
            for decoded in container.decode(stream):
                comfy.model_management.throw_exception_if_processing_interrupted()
                seconds = float(decoded.pts * decoded.time_base) - origin
                while current is not None and slot / FPS < seconds - 1e-4 and len(frames) < count:
                    converted = convert(current) if converted is None else converted
                    frames.append(converted)
                    slot += 1
                if len(frames) >= count:
                    break
                current, converted = decoded, None
        if current is None:
            raise ValueError(f"源视频在第 {start} 帧（{start / FPS:.2f} 秒）之后没有画面：{os.path.basename(path)}")
        if len(frames) < count:
            frames.append(convert(current) if converted is None else converted)
        return torch.cat(frames)

    def audio(self, item, start, count):
        """Audio of 24 fps frames [start, start + count) on the file's video timeline, zero-padded."""
        path = resolve_path(item) if "input" not in item else None
        if path is None:
            raise ValueError("音频只能来自上传的音频或视频文件。")
        if path not in self._audio:
            self._audio[path] = self._decode_audio(path)
        wave, rate = self._audio[path]
        a, b = round(start * rate / FPS), round((start + count) * rate / FPS)
        clip = torch.zeros((wave.shape[0], b - a), dtype=wave.dtype)
        lo, hi = max(a, 0), min(b, wave.shape[-1])
        if hi > lo:
            clip[:, lo - a:hi - a] = wave[:, lo:hi]
        return {"waveform": clip[None], "sample_rate": rate}

    @staticmethod
    def _decode_audio(path):
        with av.open(path) as container:
            if not container.streams.audio:
                return torch.zeros((2, 0)), 32000
            audio = container.streams.audio[0]
            origin = float((container.streams.video[0].start_time or 0) * container.streams.video[0].time_base) if container.streams.video else 0.0
            offset = float((audio.start_time or 0) * audio.time_base) - origin
        wave, rate = load_audio_file(path)
        shift = round(offset * rate)
        if shift > 0:
            wave = torch.nn.functional.pad(wave, (shift, 0))
        elif shift < 0:
            wave = wave[:, -shift:]
        return wave, rate


def match_rate(audio, rate):
    """Waveform [1, 2, S] at ``rate``; mono is duplicated so every shot shares one stereo layout."""
    wave = audio["waveform"]
    if int(audio["sample_rate"]) != rate:
        wave = comfy.audio.resample(wave, int(audio["sample_rate"]), rate)
    if wave.shape[1] == 1:
        wave = wave.expand(-1, 2, -1)
    return wave[:, :2]

