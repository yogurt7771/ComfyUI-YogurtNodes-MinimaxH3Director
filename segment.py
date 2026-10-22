"""局部重绘 text targets: SAM 3 tracks a described object over a shot's frames through ComfyUI's native SAM 3 nodes."""

from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack


def track_mask(model, clip, frames, target):
    """([N, H, W] union mask of what SAM 3 tracks for ``target`` over [N, H, W, 3] ``frames``, number of objects)."""
    if model is None or clip is None:
        raise ValueError("文字识别遮罩需要 SAM 3：用 CheckpointLoaderSimple 加载 SAM 3 检查点（如 sam3.1_multiplex_fp16.safetensors），"
                         "MODEL 接导演台的 sam3、CLIP 接 sam3_clip。")
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(target.get("english") or target["text"]))
    track = SAM3_VideoTrack.execute(frames, model, conditioning=conditioning, detection_threshold=target["threshold"],
                                    max_objects=target["objects"]).result[0]
    packed = track["packed_masks"]
    return SAM3_TrackToMask.execute(track, "").result[0].float().cpu(), 0 if packed is None else packed.shape[1]
