"""成片超分: NVIDIA RTX Video Super Resolution on the finished frames, the last step before the MP4.

It runs in milliseconds per frame and keeps the picture as generated, so H3 renders at width x height / scale and RTX
brings it to the film size. Needs an NVIDIA RTX GPU and the nvidia-vfx SDK.
"""

from contextlib import contextmanager

import torch

import comfy.model_management

try:
    from nvvfx import VideoSuperRes
except ImportError:  # NVIDIA RTX only; the stage asks for the package when it is switched on
    VideoSuperRes = None

QUALITIES = ("ULTRA", "HIGH", "MEDIUM", "LOW", "HIGHBITRATE_ULTRA", "HIGHBITRATE_HIGH")


@contextmanager
def upscaler(quality, width, height):
    """``run(frames)``: [F, H, W, C] in [0, 1] -> [F, height, width, 3] on the CPU; the effect loads once per film."""
    if VideoSuperRes is None:
        raise RuntimeError("成片超分需要 NVIDIA 的 nvidia-vfx（仅 RTX 显卡）：用运行 ComfyUI 的 Python 执行 "
                           "pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com，然后重启 ComfyUI。")
    device = comfy.model_management.get_torch_device()
    with VideoSuperRes(quality=VideoSuperRes.QualityLevel[quality], device=device.index or 0) as effect:
        effect.output_width, effect.output_height = width, height
        effect.load()

        def run(frames):
            out = torch.empty((frames.shape[0], height, width, 3))
            for k, frame in enumerate(frames):
                image = frame[..., :3].movedim(-1, 0).to(device, torch.float32).contiguous()
                # The result lives in SDK memory that the next call reuses: copy it out at once.
                out[k] = torch.from_dlpack(effect.run(image).image).movedim(0, -1).cpu()
            return out

        yield run
