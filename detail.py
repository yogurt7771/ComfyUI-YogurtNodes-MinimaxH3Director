"""局部精修: crop a (tracked) region of the decoded shot, resample it at a higher resolution, stitch it back.

Tracking and stitching follow the approach of ComfyUI-H3-FaceRefine (MIT) as integrated in
ComfyUI_MiniMaxH3_Director (Apache-2.0): per-frame detection, gap interpolation, smoothed
trajectory, sub-pixel affine crop, feathered paste with colour matching.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths

from .sampling import feather

DETECTOR_FOLDERS = ("ultralytics_bbox", "ultralytics")


def detectors():
    names = set()
    for folder in DETECTOR_FOLDERS:
        if folder in folder_paths.folder_names_and_paths:
            names.update(n for n in folder_paths.get_filename_list(folder) if n.endswith(".pt"))
    bbox = os.path.join(folder_paths.models_dir, "ultralytics", "bbox")
    if os.path.isdir(bbox):
        names.update(n for n in os.listdir(bbox) if n.endswith(".pt"))
    return sorted(names)


def _detector_path(name):
    if os.path.basename(name) != name:
        raise ValueError(f"检测器名称无效：{name}")
    for folder in DETECTOR_FOLDERS:
        if folder in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(folder, name)
            if path:
                return path
    path = os.path.join(folder_paths.models_dir, "ultralytics", "bbox", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"人脸跟踪需要检测器 {name}，请放到 models/ultralytics/bbox/。")
    return path


def crop_canvas(box_w, box_h, megapixels):
    """Crop resolution with the box aspect at ``megapixels``, on H3's 32 px grid."""
    aspect = box_w / max(box_h, 1e-6)
    area = megapixels * 1e6
    width = max(256, round(math.sqrt(area * aspect) / 32) * 32)
    height = max(256, round(math.sqrt(area / aspect) / 32) * 32)
    return width, height


def _smooth(values, window):
    if window <= 1 or len(values) < 3:
        return values
    window = min(window | 1, len(values) - (1 - len(values) % 2))
    pad = window // 2
    x = np.arange(window, dtype=np.float64) - pad
    kernel = np.exp(-(x ** 2) / (2.0 * max(window / 6.0, 0.5) ** 2))
    return np.convolve(np.pad(values, pad, mode="reflect"), kernel / kernel.sum(), mode="valid")


def track(frames, item, aspect):
    """Per-frame crop boxes (x, y, w, h) in pixels with the crop canvas aspect."""
    count, height, width = frames.shape[:3]
    x, y, w, h = item["box"]
    cx, cy = np.full(count, (x + w / 2) * width), np.full(count, (y + h / 2) * height)
    size = np.full(count, h * height)
    if item["follow"] == "face":
        cx, cy, size = _follow_face(frames, item, cx[0], cy[0], size[0])
    boxes = []
    for i in range(count):
        bh = min(size[i], height, width / aspect)
        bw = bh * aspect
        bx = min(max(cx[i] - bw / 2, 0.0), width - bw)
        by = min(max(cy[i] - bh / 2, 0.0), height - bh)
        boxes.append([float(bx), float(by), float(bw), float(bh)])
    return boxes


def _follow_face(frames, item, cx0, cy0, size0):
    """Follow the face nearest the drawn box; the drawn box size relative to that face sets the framing."""
    from ultralytics import YOLO

    detector = YOLO(_detector_path(item["detector"]))
    count = frames.shape[0]
    cx, cy, fh = np.zeros(count), np.zeros(count), np.zeros(count)
    found = np.zeros(count, dtype=bool)
    lx, ly, reach = cx0, cy0, size0
    for i in range(count):
        comfy.model_management.throw_exception_if_processing_interrupted()
        image = (frames[i, ..., :3].clamp(0, 1) * 255).byte().numpy()[..., ::-1].copy()
        result = detector.predict(image, conf=0.35, verbose=False)[0]
        best = None
        for x0, y0, x1, y1 in result.boxes.xyxy.tolist():
            distance = math.hypot((x0 + x1) / 2 - lx, (y0 + y1) / 2 - ly)
            if distance <= reach and (best is None or distance < best[0]):
                best = (distance, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0)
        if best is not None:
            _, cx[i], cy[i], fh[i] = best
            found[i] = True
            lx, ly, reach = cx[i], cy[i], max(fh[i], 16.0)
    if not found.any():
        return np.full(count, cx0), np.full(count, cy0), np.full(count, size0)
    index = np.arange(count)
    cx, cy, fh = (np.interp(index, index[found], v[found]) for v in (cx, cy, fh))
    scale = size0 / max(float(fh[found][0]), 1.0)
    return _smooth(cx, 15), _smooth(cy, 15), _smooth(fh, 31) * scale


def crop(frames, boxes, width, height):
    """Affine crop + resize: [N, H, W, 3] -> [N, height, width, 3]."""
    count, frame_h, frame_w = frames.shape[:3]
    theta = torch.tensor([[[bw / frame_w, 0.0, (2 * bx + bw) / frame_w - 1.0],
                           [0.0, bh / frame_h, (2 * by + bh) / frame_h - 1.0]] for bx, by, bw, bh in boxes])
    out = []
    for i in range(0, count, 16):
        grid = F.affine_grid(theta[i:i + 16], (theta[i:i + 16].shape[0], 3, height, width), align_corners=False)
        out.append(F.grid_sample(frames[i:i + 16, ..., :3].movedim(-1, 1).float(), grid, mode="bicubic",
                                 padding_mode="border", align_corners=False).clamp(0, 1).movedim(1, -1))
    return torch.cat(out)


def paste_mask(width, height, ellipse, edge):
    """Crop-space blend mask: an ellipse for faces, otherwise the crop with a feathered border."""
    if ellipse:
        yy = torch.linspace(-1, 1, height).view(-1, 1)
        xx = torch.linspace(-1, 1, width).view(1, -1)
        mask = ((xx ** 2 + yy ** 2) <= 0.8 ** 2).float()
    else:
        mask = torch.zeros(height, width)
        mask[edge:height - edge, edge:width - edge] = 1.0
    return feather(mask[None], edge)[0]


def stitch(frames, crops, boxes, ellipse):
    """Warp ``crops`` back into ``frames`` (in place) and blend with colour matching inside the mask."""
    count, frame_h, frame_w = frames.shape[:3]
    crop_h, crop_w = crops.shape[1:3]
    mask = paste_mask(crop_w, crop_h, ellipse, max(4, min(crop_w, crop_h) // 12))
    for i in range(0, count, 8):
        chunk = boxes[i:i + 8]
        theta = torch.tensor([[[frame_w / bw, 0.0, (frame_w - 2 * bx) / bw - 1.0],
                               [0.0, frame_h / bh, (frame_h - 2 * by) / bh - 1.0]] for bx, by, bw, bh in chunk])
        grid = F.affine_grid(theta, (len(chunk), 3, frame_h, frame_w), align_corners=False)
        patch = F.grid_sample(crops[i:i + 8].movedim(-1, 1).float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        weight = F.grid_sample(mask.expand(len(chunk), 1, -1, -1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        patch, weight = patch.movedim(1, -1), weight.movedim(1, -1)
        base = frames[i:i + 8]
        total = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        base_mean = (base * weight).sum(dim=(1, 2), keepdim=True) / total
        patch_mean = (patch * weight).sum(dim=(1, 2), keepdim=True) / total
        base_std = ((base - base_mean) ** 2 * weight).sum(dim=(1, 2), keepdim=True).div(total).sqrt()
        patch_std = ((patch - patch_mean) ** 2 * weight).sum(dim=(1, 2), keepdim=True).div(total).sqrt().clamp_min(1e-4)
        patch = ((patch - patch_mean) * (base_std / patch_std) + base_mean).clamp(0, 1)
        frames[i:i + 8] = base + weight * (patch - base)
    return frames
