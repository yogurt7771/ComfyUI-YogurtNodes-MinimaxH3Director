"""Test support: load the plugin as a package and fake H3 models with the real tensor layouts."""

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from comfy.ldm.minimax.model import FRAME_PER_TOKEN  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402


def load():
    name = "yogurt_h3_director"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def token_spans(frames):
    spans, start, k = [], 0, 0
    while start < frames:
        count = FRAME_PER_TOKEN[k % 5]
        spans.append((start, min(start + count, frames)))
        start += count
        k += 1
    return spans


class FakeVAE:
    """Channels 0-2 hold the 16x16 / per-token average colour, so decode(encode(x)) is a blurred x."""

    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        pixels = pixels[..., :3].float()
        self.encoded.append(tuple(pixels.shape))
        pooled = F.avg_pool2d(pixels.movedim(-1, 1), 16)
        tokens = torch.stack([pooled[a:b].mean(0) for a, b in token_spans(pixels.shape[0])], dim=1)
        latent = torch.zeros((1, 24, tokens.shape[1], *tokens.shape[2:]))
        latent[0, :3] = tokens
        return latent

    def decode(self, latent):
        frames = []
        for k in range(latent.shape[2]):
            image = F.interpolate(latent[0, :3, k][None], scale_factor=16, mode="nearest")[0].movedim(0, -1)
            frames += [image] * FRAME_PER_TOKEN[k % 5]
        return torch.stack(frames)[None]

    def spacial_compression_encode(self):
        return 16


class FakeAudioVAE:
    audio_sample_rate = 32000
    first_stage_model = SimpleNamespace(output_sample_rate=32000)

    def encode(self, wave):
        samples = wave.shape[1]
        count = math.ceil(samples / 800)
        padded = F.pad(wave, (0, 0, 0, count * 800 - samples))
        latent = torch.zeros((1, 32, 2, count))
        latent[0, 0] = padded[0].reshape(count, 800, 2).mean(1).movedim(0, 1)
        return latent

    def decode(self, latent):
        return latent[:, 0].movedim(1, -1).repeat_interleave(800, dim=1)


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, text, images=None, minimax_ref_items=None):
        self.calls.append({"text": text, "images": len(images or []), "refs": [i["type"] for i in minimax_ref_items or []]})
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 4, 8)), {"text": tokens["text"]}]]


class FakeSampler:
    """Records every sampling call; regenerated tokens become a seed-derived constant, masked tokens stay."""

    def __init__(self):
        self.calls = []

    def schedule(self, model, steps, scheduler, denoise=1.0, custom=None):
        return torch.linspace(denoise, 0.0, steps + 1)

    def sample(self, model, positive, negative, cfg, sampler, sigmas, latent, seed):
        video, audio = latent["samples"].unbind()
        mask = latent.get("noise_mask")
        video_mask, audio_mask = mask.unbind() if mask is not None else (torch.ones(1), torch.ones(1))
        self.calls.append({"model": model, "prompt": positive[0][1]["text"], "seed": seed, "shape": tuple(video.shape),
                           "keyframes": [k["resolved_frame_index"] for k in positive[0][1].get("minimax_keyframes", [])],
                           "refs": [r["kind"] for r in positive[0][1].get("minimax_refs", [])],
                           "video_mask": video_mask.clone(), "audio_mask": audio_mask.clone(), "steps": sigmas.shape[-1] - 1,
                           "start": round(float(sigmas[0]), 4)})
        fill = (seed % 89 + 5) / 100.0
        return NestedTensor((video + video_mask * (fill - video), audio + audio_mask * (fill - audio)))


def models(module, **kwargs):
    values = dict(model="model", clip=FakeClip(), vae=FakeVAE(), audio_vae=FakeAudioVAE(), ref_model=None, refine_model=None,
                  inpaint_patch=None, upscale_model=None, sampler="euler", sigmas=None, seed=11, steps=6, cfg=1.0, scheduler="simple")
    values.update(kwargs)
    return module.render.Models(**values)


def write_video(path, frames, width, height, fps=24, audio_seconds=None):
    """Colour-ramp test clip; frame i is filled with i/frames so frame selection is observable."""
    import av
    import numpy as np
    from fractions import Fraction

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv444p"
        stream.options = {"crf": "0"}
        audio = None
        if audio_seconds:
            audio = container.add_stream("aac", rate=32000, layout="stereo")
        for i in range(frames):
            value = int(255 * i / max(frames - 1, 1))
            image = np.full((height, width, 3), value, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode(None))
        if audio is not None:
            samples = int(32000 * audio_seconds)
            wave = (0.25 * np.sin(np.arange(samples) / 32000 * 2 * np.pi * 440)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.stack([wave, wave]), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = 32000, 0
            container.mux(audio.encode(frame))
            container.mux(audio.encode(None))
