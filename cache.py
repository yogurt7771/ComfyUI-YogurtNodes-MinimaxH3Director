"""Content-addressed director cache under output/yogurt_h3_director/cache; every path is a key digest."""

import hashlib
import json
import os
from pathlib import Path
import uuid

import folder_paths
import safetensors.torch
import torch

MODEL_FOLDERS = ("diffusion_models", "checkpoints", "text_encoders", "vae", "loras", "upscale_models", "model_patches")
MODEL_EXTENSIONS = (".safetensors", ".sft", ".pt", ".pth", ".ckpt", ".bin", ".gguf")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def tensor_digest(tensor):
    hasher = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    for frame in tensor.reshape(-1, *tensor.shape[-2:]) if tensor.ndim > 2 else [tensor]:
        hasher.update(frame.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _file_stamp(name):
    if not name.lower().endswith(MODEL_EXTENSIONS):
        return None
    for folder in MODEL_FOLDERS:
        path = folder_paths.get_full_path(folder, name)
        if path:
            stat = os.stat(path)
            return [folder, stat.st_size, stat.st_mtime_ns]
    return None


def upstream_signature(prompt, node_id, names):
    """Hash the graph feeding ``names`` of this node, including the size/time of every model file it loads."""
    prompt = prompt or {}
    root = prompt.get(str(node_id), {}).get("inputs", {})
    seen = {}

    def resolve(value):
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
            node_key = str(value[0])
            if node_key not in seen:
                seen[node_key] = None
                node = prompt[node_key]
                inputs = {k: resolve(v) for k, v in node.get("inputs", {}).items()}
                files = {k: _file_stamp(v) for k, v in inputs.items() if isinstance(v, str)}
                seen[node_key] = [node["class_type"], inputs, {k: v for k, v in files.items() if v}]
            return [seen[node_key], value[1]]
        return value

    return digest({name: resolve(root[name]) for name in names if name in root})


def root():
    return Path(folder_paths.get_output_directory()) / "yogurt_h3_director" / "cache"


def path_for(key, suffix):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid director cache key")
    return root() / key[:2] / (key + suffix)


def exists(key, suffix=".safetensors"):
    return path_for(key, suffix).is_file()


def read_tensors(key):
    path = path_for(key, ".safetensors")
    return safetensors.torch.load_file(str(path)) if path.is_file() else None


def write_tensors(key, tensors):
    path = path_for(key, ".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    safetensors.torch.save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def write_json(key, value):
    path, temporary = temporary_path(key, ".json")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def read_json(key):
    return json.loads(path_for(key, ".json").read_text(encoding="utf-8"))


def temporary_path(key, suffix):
    path = path_for(key, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
