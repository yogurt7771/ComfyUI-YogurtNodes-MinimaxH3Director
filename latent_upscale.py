"""MiniMax H3 latent upscaler: a learned 3D network that enlarges the 24-channel video latent, skipping the VAE round trip.

The network and its loader are ComfyUI's own (Load Latent Upscale Model, MiniMax H3 Latent Upscale With Model); this module
picks the weights in models/latent_upscale_models and downloads the default checkpoint there on first use.
"""

import hashlib
import logging
import os
import urllib.request

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel

FOLDER = "latent_upscale_models"
DEFAULT = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
# Pinned to one revision so the download is always the file the checksum describes.
SOURCE = ("LBH-123-AI/Minimax_h3_latent_Upscaler", "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6",
          f"minimax_h3_latent_upscaler_3d_conv_v1/{DEFAULT}", 690592672,
          "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2")


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
    """ComfyUI's latent upscale model for file ``name`` in models/latent_upscale_models; the default one is fetched if missing."""
    if folder_paths.get_full_path(FOLDER, name) is None:
        if name != DEFAULT:
            raise FileNotFoundError(f"找不到 latent 放大模型 {name}，请放到 models/{FOLDER}。")
        _download(name)
    patcher = LatentUpscaleModelLoader.execute(name).result[0]
    # The folder also holds other families' upscalers (LTX, HunyuanVideo), which take other latents.
    if not isinstance(patcher.model, MiniMaxH3LatentUpscaler):
        raise ValueError(f"{name} 不是 MiniMax H3 latent 放大模型。")
    return patcher


def upscale(patcher, video, height, width):
    """Stored H3 video latent [B, 24, T, h, w] -> [B, 24, T, height, width] (latent units: pixels / 16)."""
    return MiniMaxH3LatentUpscaleWithModel.execute(patcher, {"samples": video}, width * 16, height * 16).result[0]["samples"]
