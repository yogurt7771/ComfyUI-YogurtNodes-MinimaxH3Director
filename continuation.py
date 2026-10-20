"""Direct AV-latent continuation between MiniMax H3 segments.

Adapted from Songssx/ComfyUI-MiniMaxH3-TimelineDirector (GPL-3.0). Its
Drift-Control AV schedule derives from ethanfel/ComfyUI-MiniMaxH3-Contex-Loop
(GPL-3.0). The previous segment's sampled latent tail is copied into the next
segment's opening, kept clean by a noise mask, and released toward generation
by a mask that follows the sampler's sigma schedule.
"""

from __future__ import annotations

import math

import torch
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS, video_latent_t

DRIFT_CONTROL_KEY = "yogurt_h3_drift_control_av"
_TAPER_STEPS = 4
_AUDIO_RELEASE_TOKENS = 8


def av_streams(latent, label):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError(f"{label} 必须是 MiniMax H3 音视频嵌套 latent。")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4 or audio.shape[1] != 32:
        raise ValueError(f"{label} 不是有效的 MiniMax H3 音视频 latent。")
    return video, audio


def carry_latent_tail(target_latent, previous_latent, overlap_frames, continue_audio):
    """Copy the previous segment's tail into the target opening and mask it clean.

    The mask is compact (one channel, 1x1 spatial); ComfyUI expands it per stream.
    0 keeps the carried latent, 1 generates. Carried audio is released over its
    final eight tokens with a half-cosine so the seam sound continues smoothly.
    """
    target_video, target_audio = av_streams(target_latent, "目标 latent")
    source_video, source_audio = av_streams(previous_latent, "上一段 latent")
    if target_video.shape[3:] != source_video.shape[3:]:
        raise ValueError(f"上一段 latent 尺寸 {tuple(source_video.shape[3:])} 与本段 {tuple(target_video.shape[3:])} 不一致。")
    video_tokens = video_latent_t(overlap_frames)
    if video_tokens >= target_video.shape[2] or video_tokens > source_video.shape[2]:
        raise ValueError(f"{overlap_frames} 帧重叠需要 {video_tokens} 个视频 token，超出了分段长度。")

    video = target_video.clone()
    video[:, :, :video_tokens] = source_video[:1, :, -video_tokens:].to(video)
    video_mask = torch.ones((1, 1, video.shape[2], 1, 1), dtype=torch.float32, device=video.device)
    video_mask[:, :, :video_tokens] = 0.0

    audio = target_audio.clone()
    existing_mask = target_latent.get("noise_mask")
    locked_audio = (getattr(existing_mask, "is_nested", False)
                    and bool(torch.all(existing_mask.unbind()[1] == 0)))
    audio_mask = torch.zeros((1, 1, 1, audio.shape[-1]), dtype=torch.float32, device=audio.device) if locked_audio else torch.ones(
        (1, 1, 1, audio.shape[-1]), dtype=torch.float32, device=audio.device)
    if continue_audio and not locked_audio:
        audio_tokens = min(round(overlap_frames * AUDIO_LATENT_FPS / FPS), source_audio.shape[-1], audio.shape[-1])
        audio[..., :audio_tokens] = source_audio[:1, ..., -audio_tokens:].to(audio)
        release = min(_AUDIO_RELEASE_TOKENS, audio_tokens)
        ramp = torch.zeros(audio_tokens, dtype=torch.float32, device=audio.device)
        steps = torch.arange(1, release + 1, dtype=torch.float32, device=audio.device)
        ramp[-release:] = 0.5 - 0.5 * torch.cos(torch.pi * steps / release)
        audio_mask[..., :audio_tokens] = ramp

    output = dict(target_latent)
    output["samples"] = NestedTensor((video, audio))
    output["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return output, video_tokens


def _schedule_values(sigmas):
    values = sigmas.detach().float().reshape(-1).cpu().tolist() if torch.is_tensor(sigmas) else list(sigmas or ())
    return tuple(sorted({v for v in values if math.isfinite(v) and v >= 0.0}, reverse=True))


def _matched_noise_ratio(current, schedule):
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tolerance = max(1e-7, abs(current) * 1e-6)
    following = next((value for value in schedule if value < current - tolerance), 0.0)
    return max(0.0, min(1.0, following / current))


def _prefix_weights(prefix_steps):
    taper = min(_TAPER_STEPS, prefix_steps)
    return [1.0] * (prefix_steps - taper) + [(taper - offset - 1) / taper for offset in range(taper)]


class DriftControlState:
    """Live video-prefix mask shared by the sampler blend and H3's per-token timesteps."""

    def __init__(self, video_shape, audio_shape, sigmas, prefix_steps):
        self.video_shape = tuple(video_shape)
        self.audio_shape = tuple(audio_shape)
        self.schedule = _schedule_values(sigmas)
        self.prefix_steps = int(prefix_steps)
        self.packed_mask = None
        self.video_mask = None
        self.audio_mask = None

    def _update(self, sigma, packed_mask):
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        output = packed_mask.clone()
        video_elements = math.prod(self.video_shape[1:])
        video = output[..., :video_elements].reshape(self.video_shape)
        weights = torch.tensor(_prefix_weights(self.prefix_steps), device=video.device, dtype=video.dtype)
        video[:, :, :self.prefix_steps] = (weights * _matched_noise_ratio(current, self.schedule)).view(1, 1, -1, 1, 1)
        # Match H3's token-mask quantization so the sampler blend and the model's
        # per-row timestep labels describe the same state.
        self.video_mask = torch.ceil(video[:, :1].float() * 256.0) / 256.0
        self.audio_mask = output[..., video_elements:].reshape(self.audio_shape)[:, :1].clone()
        self.packed_mask = output
        return output

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        return self._update(sigma, denoise_mask)

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.video_mask is not None:
            kwargs["denoise_mask"] = self.video_mask
            kwargs["audio_denoise_mask"] = self.audio_mask
        return executor(*args, **kwargs)


def install_drift_control(model, latent, sigmas, prefix_steps):
    """Clone an H3 model with a schedule-matched continuation-prefix mask."""
    if model.model.__class__.__name__ != "MiniMaxH3":
        raise ValueError("段间 latent 续接需要 MiniMax H3 模型。")
    if callable(model.model_options.get("denoise_mask_function")):
        raise ValueError("该模型已有动态去噪遮罩补丁，无法叠加段间 latent 续接。")
    video, audio = av_streams(latent, "续接 latent")
    patched = model.clone()
    state = DriftControlState(video.shape, audio.shape, sigmas, prefix_steps)
    patched.set_model_denoise_mask_function(state.denoise_mask_function)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, DRIFT_CONTROL_KEY, state.apply_model_wrapper)
    patched.model_options[DRIFT_CONTROL_KEY] = state
    return patched
