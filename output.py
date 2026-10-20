"""Frame storage: lossless FFV1 shot caches and the MP4 written while shots are assembled."""

from fractions import Fraction
import os

import av
import numpy as np
import torch

import comfy.model_management
from comfy_api.latest._input_impl.video_types import BT709_NCL, set_video_color_properties

from . import cache
from .project import FPS


def _rgb24(frame):
    pixels = frame.detach().cpu()[..., :3]
    if pixels.dtype != torch.uint8:
        pixels = pixels.float().clamp(0, 1).mul(255).round().to(torch.uint8)
    return pixels.contiguous().numpy()


def write_frames(key, frames):
    path, temporary = cache.temporary_path(key, ".mkv")
    try:
        with av.open(str(temporary), mode="w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=Fraction(FPS))
            stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "bgr0"
            for frame in frames:
                comfy.model_management.throw_exception_if_processing_interrupted()
                container.mux(stream.encode(av.VideoFrame.from_ndarray(_rgb24(frame), format="rgb24")))
            container.mux(stream.encode(None))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_frames(key):
    path = cache.path_for(key, ".mkv")
    with av.open(str(path)) as container:
        frames = [torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]
    return torch.stack(frames).float().div_(255)


class Mp4Writer:
    """H.264 + AAC at 24 fps; video is encoded shot by shot, the full soundtrack once at the end."""

    def __init__(self, path, width, height, audio_rate):
        self.container = av.open(path, mode="w", format="mp4", options={"movflags": "+faststart"})
        self.video = self.container.add_stream("h264", rate=Fraction(FPS))
        self.video.width, self.video.height, self.video.pix_fmt = width, height, "yuv420p"
        self.video.options = {"crf": "18"}
        set_video_color_properties(self.video.codec_context, "sRGB")
        self.audio = self.container.add_stream("aac", rate=audio_rate, layout="stereo")
        self.audio_rate = audio_rate

    def write(self, frames):
        for frame in frames:
            comfy.model_management.throw_exception_if_processing_interrupted()
            picture = av.VideoFrame.from_ndarray(_rgb24(frame), format="rgb24").reformat(format="yuv420p", dst_colorspace=BT709_NCL)
            set_video_color_properties(picture, "sRGB")
            self.container.mux(self.video.encode(picture))

    def finish(self, waveform):
        self.container.mux(self.video.encode(None))
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform.float().cpu().numpy()), format="fltp", layout="stereo")
        frame.sample_rate, frame.pts = self.audio_rate, 0
        self.container.mux(self.audio.encode(frame))
        self.container.mux(self.audio.encode(None))
        self.container.close()

    def abort(self):
        self.container.close()
