"""Sampling helpers shared by the base, repaint, refine and detail passes."""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlNetApply, _empty_av_latent


def schedule(model, steps, scheduler, denoise=1.0, custom=None):
    """``steps`` sigmas starting at noise level ``denoise`` (1 = pure noise).

    A step fraction (BasicScheduler's denoise) would start far too noisy here: H3's shift of 12 keeps
    sigma near 1 for most of the schedule, so its last 35% of steps still begins at sigma 0.87.
    A connected SIGMAS keeps its own steps from the first sigma at or below ``denoise``.
    """
    if custom is not None:
        if denoise >= 1.0:
            return custom
        start = min(int((custom > denoise + 1e-6).sum()), custom.shape[-1] - 2)
        return custom[start:]
    model_sampling = model.get_model_object("model_sampling")
    total = steps
    while True:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total).cpu()
        if denoise >= 1.0 or sigmas[total - steps] <= denoise + 1e-6 or total >= steps * 2000:
            return sigmas[total - steps:]
        total = max(total + 1, total * 9 // 8)


def split(sigmas, steps):
    """One schedule for two-pass 二采: (draft, tail). The draft runs all but the last ``steps`` steps and lands on 0,
    the clean estimate that gets upscaled; the tail re-noises it to where the draft stopped and runs those last steps."""
    k = max(1, sigmas.shape[-1] - 1 - steps)
    return torch.cat([sigmas[:k], sigmas.new_zeros(1)]), sigmas[k:]


def trajectory_noise(model, last, sigma):
    """Noise the audio's flow path was on when the last step started at ``sigma`` (state = sigma * noise + (1 - sigma)
    * clean), in stored-latent units: the sampler carries the audio scaled by the model's audio_scale."""
    state, clean = last["x"].unbind()[1], last["x0"].unbind()[1]
    noise = (state - (1.0 - sigma) * clean) / (sigma * model.get_model_object("model_sampling").audio_scale)
    return noise.to(comfy.model_management.intermediate_device())


def sample(model, positive, negative, cfg, sampler, sigmas, latent, seed, audio_noise=None, last=None):
    """``audio_noise`` (see trajectory_noise) starts the audio stream on that path instead of fresh noise;
    ``last``, when given, receives the state and clean estimate the final step started from."""
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    samples = latent["samples"]
    noise = comfy.sample.prepare_noise(samples, seed)
    if audio_noise is not None:
        model_sampling = model.get_model_object("model_sampling")
        video_noise, fresh = noise.unbind()
        noise = NestedTensor((video_noise, (audio_noise * model_sampling.audio_scale / model_sampling.noise_scale).to(fresh)))
    preview = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)

    def callback(step, x0, x, total_steps):
        if last is not None and step == total_steps - 1:
            last.update(x=x, x0=x0)
        preview(step, x0, x, total_steps)

    out = guider.sample(noise, samples, sampler, sigmas, denoise_mask=latent.get("noise_mask"), callback=callback,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    return out.to(comfy.model_management.intermediate_device())


def empty_latent(width, height, frames):
    return _empty_av_latent(width, height, frames)[0]


def streams(latent):
    return latent["samples"].unbind() if isinstance(latent, dict) else latent.unbind()


def av_latent(video, audio, video_mask=None, audio_mask=None):
    latent = {"samples": NestedTensor((video, audio))}
    if video_mask is not None or audio_mask is not None:
        if video_mask is None:
            video_mask = torch.ones((1, 1, video.shape[2], 1, 1))
        if audio_mask is None:
            audio_mask = torch.ones((1, 1, 1, audio.shape[-1]))
        latent["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return latent


def locked_audio_mask(audio):
    return torch.zeros((1, 1, 1, audio.shape[-1]))


def token_mask(pixels, latent_t):
    """[F, H, W] pixel mask -> [1, 1, T, H/16, W/16]; a latent token regenerates if any of its pixels does."""
    pooled = F.max_pool2d(pixels[:, None].float(), 16)[:, 0]
    rows, start = [], 0
    for k in range(latent_t):
        count = FRAME_PER_TOKEN[k % 5]
        rows.append(pooled[start:start + count].amax(0))
        start += count
    return torch.stack(rows)[None, None]


def grow(mask, pixels):
    if pixels <= 0:
        return mask
    return F.max_pool2d(mask[:, None], 2 * pixels + 1, stride=1, padding=pixels)[:, 0]


def feather(mask, pixels):
    """Separable Gaussian blur of [F, H, W] with a ~``pixels`` wide transition."""
    if pixels <= 0:
        return mask
    radius = int(pixels)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * max(radius / 2.0, 0.5) ** 2))
    kernel = (kernel / kernel.sum()).to(mask)
    out = F.conv2d(F.pad(mask[:, None], (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return out[:, 0].clamp(0, 1)


def window_mask(mask, gen, a, b, width, height):
    """Place a [F, h, w] mask on generation frames [a, b); one frame is held, a sequence is padded with its last."""
    mask = comfy.utils.common_upscale(mask[:, None].float(), width, height, "bilinear", "disabled")[:, 0].clamp(0, 1)
    out = torch.zeros((gen, height, width))
    count = b - a
    if mask.shape[0] == 1:
        out[a:b] = mask[0]
    else:
        mask = mask[:count]
        out[a:a + mask.shape[0]] = mask
        out[a + mask.shape[0]:b] = mask[-1]
    return out


def inpaint_model(model, patch, vae, mask, source):
    """Fun ControlNet inpaint patch: ``mask`` [F, H, W] marks regenerated pixels, ``source`` is the video behind it."""
    return MiniMaxH3FunControlNetApply.execute(model, patch, vae, 1.0, 0.0, 1.0, mask=mask, source_video=source).result[0]

