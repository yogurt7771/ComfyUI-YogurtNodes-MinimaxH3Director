"""MiniMax H3 latent upscaler: a learned 3D network that enlarges the 24-channel video latent, skipping the VAE round trip.

Inference port of LatentResizer3D from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT). Weights live in
ComfyUI's models/latent_upscale_models; the default checkpoint is downloaded there on first use.
"""

import hashlib
import logging
import os
import re
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

FOLDER = "latent_upscale_models"
DEFAULT = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
# Pinned to one revision so the download is always the file the checksum describes.
SOURCE = ("LBH-123-AI/Minimax_h3_latent_Upscaler", "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6",
          f"minimax_h3_latent_upscaler_3d_conv_v1/{DEFAULT}", 690592672,
          "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2")
CHUNK = 32  # latent frames per temporal chunk, as upstream

# Per-channel statistics of the H3 video latents the network was trained on.
MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
        -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
        -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
        -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
        -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
       1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
       0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
       0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
       2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
       3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]

ops = comfy.ops.manual_cast


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dtype=None, device=None):
        super().__init__()
        self.in_layers = nn.Sequential(ops.GroupNorm(32, channels, dtype=dtype, device=device), nn.SiLU(),
                                       ops.Conv3d(channels, channels, 3, padding=1, dtype=dtype, device=device))
        self.emb_layers = nn.Sequential(nn.SiLU(), ops.Linear(emb_channels, 2 * channels, dtype=dtype, device=device))
        self.out_norm = ops.GroupNorm(32, channels, dtype=dtype, device=device)
        # Slot 1 held training dropout; Identity keeps the checkpoint keys.
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Identity(), ops.Conv3d(channels, channels, 3, padding=1, dtype=dtype, device=device))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None, None].chunk(2, dim=1)
        return x + self.out_layers(torch.addcmul(shift, self.out_norm(h), 1 + scale))


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel, dtype=None, device=None):
        super().__init__()
        self.norm = ops.GroupNorm(32, channels, dtype=dtype, device=device)
        self.dwconv = ops.Conv3d(channels, channels, (kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels, dtype=dtype, device=device)
        self.pwconv = ops.Conv3d(channels, channels, 1, dtype=dtype, device=device)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """conv_in -> blocks -> trilinear resize -> blocks -> conv_out; a TemporalConv follows every other ResBlock."""

    def __init__(self, in_channels, channels, in_blocks, out_blocks, temporal_kernel, dtype=None, device=None):
        super().__init__()
        self.temporal_kernel = temporal_kernel
        self.conv_in = ops.Conv3d(in_channels, channels, 3, padding=1, dtype=dtype, device=device)
        self.embed = nn.Sequential(ops.Linear(1, 64, dtype=dtype, device=device), nn.SiLU(), ops.Linear(64, 64, dtype=dtype, device=device))

        def stage(blocks):
            layers = nn.ModuleList()
            for b in range(blocks):
                layers.append(ResBlock(channels, 64, dtype, device))
                if b % 2 == 0:
                    layers.append(TemporalConv(channels, temporal_kernel, dtype, device))
            return layers

        self.in_blocks, self.out_blocks = stage(in_blocks), stage(out_blocks)
        self.norm_out = ops.GroupNorm(32, channels, dtype=dtype, device=device)
        self.conv_out = ops.Conv3d(channels, in_channels, 3, padding=1, dtype=dtype, device=device)

    def forward(self, x, size, scale):
        """Normalized latent [B, C, T, h, w] -> [B, C, *size]; ``scale`` is the enlargement the network is told about."""
        emb = self.embed(torch.full((x.shape[0], 1), scale - 1.0, dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, emb) if isinstance(block, ResBlock) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, emb) if isinstance(block, ResBlock) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def choices():
    """Upscaler files for the node: the default checkpoint (downloaded on first use) and every local file."""
    return list(dict.fromkeys([DEFAULT] + folder_paths.get_filename_list(FOLDER)))


def identity(name):
    """What refine caches key on: the pinned checksum of the default file (present or not yet downloaded),
    the size and time of any other file."""
    if name == DEFAULT:
        return SOURCE[4]
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        return None
    stat = os.stat(path)
    return [name, stat.st_size, stat.st_mtime_ns]


def _download(name):
    repo, revision, path, size, sha256 = SOURCE
    folder = folder_paths.get_folder_paths(FOLDER)[0]
    target = os.path.join(folder, name)
    temporary = target + ".part"
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{repo}/resolve/{revision}/{path}"
    logging.info(f"下载 H3 latent 放大模型（{size / 2**20:.0f} MB）：{url} -> {target}")
    os.makedirs(folder, exist_ok=True)
    hasher, done = hashlib.sha256(), 0
    progress = comfy.utils.ProgressBar(size)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
            while chunk := response.read(1 << 22):
                comfy.model_management.throw_exception_if_processing_interrupted()
                out.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                progress.update_absolute(done)
        if done != size or hasher.hexdigest() != sha256:
            raise ValueError(f"H3 latent 放大模型下载不完整或校验失败（{done}/{size} 字节），请重试或手动下载到 {folder}。")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def load(name):
    """ModelPatcher for upscaler file ``name`` in models/latent_upscale_models; the default one is fetched if missing."""
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        if name != DEFAULT:
            raise FileNotFoundError(f"找不到 latent 放大模型 {name}，请放到 models/{FOLDER}。")
        path = _download(name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if "conv_in.weight" not in sd or "in_blocks.1.dwconv.weight" not in sd:
        raise ValueError(f"{name} 不是 MiniMax H3 3D latent 放大模型（LBH-123-AI LatentResizer3D）。")

    def blocks(stage):
        return len({m.group(1) for key in sd if (m := re.match(rf"{stage}\.(\d+)\.in_layers\.", key))})

    channels, in_channels = sd["conv_in.weight"].shape[:2]
    with torch.device("meta"):
        model = LatentResizer3D(in_channels, channels, blocks("in_blocks"), blocks("out_blocks"),
                                sd["in_blocks.1.dwconv.weight"].shape[2], dtype=sd["conv_in.weight"].dtype)
    model.load_state_dict(sd, strict=True, assign=True)
    return comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(),
                                            offload_device=comfy.model_management.unet_offload_device())


def _resize(model, x, size, scale):
    """Clips longer than CHUNK latent frames run in overlapping chunks blended linearly across the overlaps (as upstream)."""
    frames = x.shape[2]
    if frames <= CHUNK:
        return model(x, (frames, *size), scale)
    overlap = model.temporal_kernel
    padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
    out = x.new_zeros((*x.shape[:3], *size))
    weight = x.new_zeros((1, 1, frames, 1, 1))
    for start in range(0, frames, CHUNK):
        end = min(frames, start + CHUNK)
        a, b = max(0, start - overlap), min(frames, end + overlap)
        lo, hi = max(0, a - overlap), min(frames + 2 * overlap, b + overlap)
        part = model(padded[:, :, lo:hi], (hi - lo, *size), scale)[:, :, a + overlap - lo:b + overlap - lo]
        ramp = torch.ones(b - a, dtype=x.dtype, device=x.device)
        if start > a:
            ramp[:start - a] = torch.arange(1, start - a + 1, dtype=x.dtype, device=x.device) / (start - a + 1)
        if b > end:
            ramp[end - b:] = torch.arange(b - end, 0, -1, dtype=x.dtype, device=x.device) / (b - end + 1)
        out[:, :, a:b] += part * ramp.view(1, 1, -1, 1, 1)
        weight[:, :, a:b] += ramp.view(1, 1, -1, 1, 1)
    return out / weight


def upscale(patcher, video, height, width):
    """Stored H3 video latent [B, 24, T, h, w] -> [B, 24, T, height, width] (latent units: pixels / 16)."""
    device = patcher.load_device
    dtype = comfy.model_management.vae_dtype(device, [torch.float16, torch.bfloat16, torch.float32])
    # Measured peak: 7.5-11 full-size activation frames per output frame; reserve 12 for one temporal chunk.
    model = patcher.model
    frames = min(video.shape[2], CHUNK) + 2 * model.temporal_kernel
    per_frame = 12 * model.conv_in.weight.shape[0] * height * width * dtype.itemsize
    comfy.model_management.load_models_gpu([patcher], memory_required=per_frame * frames)
    mean = torch.tensor(MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    scale = (width / video.shape[-1] + height / video.shape[-2]) / 2
    out = _resize(model, (video.to(device, dtype) - mean) / std, (height, width), scale)
    return torch.addcmul(mean, out, std).to(comfy.model_management.intermediate_device(), video.dtype)
