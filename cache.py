"""Lossless per-window checkpoints; all paths are derived from content hashes."""

import hashlib
import json
import os
from pathlib import Path
import uuid

import folder_paths
import safetensors.torch
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def tensor_digest(tensor):
    hasher = hashlib.sha256()
    hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
    # Bound staging memory when hashing a long batch.
    for frame in tensor:
        hasher.update(frame.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def video_digest(video):
    source = video.get_stream_source()
    hasher = hashlib.sha256()
    if isinstance(source, str):
        with open(source, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
    else:
        source.seek(0)
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
        source.seek(0)
    return digest([hasher.hexdigest(), video.get_active_trim_window(), video.get_dimensions()])


def upstream_signature(prompt, node_id):
    """Hash render dependencies, excluding unrelated output/UI settings and secrets."""
    prompt = prompt or {}
    root = prompt.get(str(node_id), {}).get("inputs", {})

    def resolve(value):
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
            node = prompt[str(value[0])]
            inputs = {k: resolve(v) for k, v in node.get("inputs", {}).items()}
            files = {}
            for k, folder in (("unet_name", "diffusion_models"), ("clip_name", "text_encoders"),
                              ("vae_name", "vae"), ("lora_name", "loras")):
                if isinstance(inputs.get(k), str):
                    path = folder_paths.get_full_path(folder, inputs[k])
                    if path:
                        stat = os.stat(path)
                        files[k] = [stat.st_size, stat.st_mtime_ns]
            return [node["class_type"], inputs, files, value[1]]
        return value

    sources = {k: root[k] for k in ("model", "clip", "vae", "audio_vae", "sampler", "sigmas") if k in root}
    # Refine models reach the sampler through the plan; follow only the plan chain, not prompt or API-key nodes.
    link = root.get("plan")
    while isinstance(link, list) and str(link[0]) in prompt:
        inputs = prompt[str(link[0])].get("inputs", {})
        if "refine" in inputs:
            sources["refine"] = inputs["refine"]
            break
        link = inputs.get("plan")
    return digest({k: resolve(v) for k, v in sources.items()})


def checkpoint_path(key):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid segment cache key")
    return Path(folder_paths.get_output_directory()) / "yogurt_h3_cache" / key[:2] / (key + ".safetensors")


def read_checkpoint(key):
    path = checkpoint_path(key)
    if not path.is_file():
        return None
    return safetensors.torch.load_file(str(path))


def write_checkpoint(key, tensors):
    path = checkpoint_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    safetensors.torch.save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def selected_segments(text, count):
    selected = set()
    for item in text.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        ends = item.split("-")
        start = int(ends[0])
        end = int(ends[-1])
        if len(ends) > 2 or start < 1 or end < start or end > count:
            raise ValueError(f"重跑段号需在 1–{count} 内，例如 2,4-6。")
        selected.update(range(start, end + 1))
    return selected
