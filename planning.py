"""Pure frame planning helpers for MiniMax H3."""

from __future__ import annotations

import math


H3_FPS = 24
H3_MIN_FRAMES = 5
H3_FRAME_STEP = 17


def align_h3_frames(frame_count: int) -> int:
    """Round a positive frame count up to MiniMax H3's 17k+5 lattice."""

    frame_count = max(H3_MIN_FRAMES, int(frame_count))
    return H3_MIN_FRAMES + math.ceil((frame_count - H3_MIN_FRAMES) / H3_FRAME_STEP) * H3_FRAME_STEP


def segment_frame_limit(segment_seconds: int) -> int:
    """Largest legal H3 frame count not exceeding the requested whole seconds."""

    requested = max(H3_MIN_FRAMES, int(segment_seconds) * H3_FPS)
    return H3_MIN_FRAMES + max(0, (requested - H3_MIN_FRAMES) // H3_FRAME_STEP) * H3_FRAME_STEP


def canvas(ratio_width, ratio_height, megapixels: float) -> tuple[int, int]:
    """Width and height with the given aspect ratio and pixel count, each snapped to 32."""

    scale = math.sqrt(float(megapixels) * 1e6 / (ratio_width * ratio_height))
    return tuple(max(32, round(side * scale / 32) * 32) for side in (ratio_width, ratio_height))


def normalize_overlap(overlap_frames: int, window_frames: int) -> int:
    overlap = min(max(0, int(overlap_frames)), window_frames - 1)
    if overlap < 5:
        return min(overlap, 1)
    return 5 + (overlap - 5) // 17 * 17


def build_segments(target_frames: int, segment_seconds: int, overlap_frames: int = 0, window_lengths=None) -> list[dict[str, int | float]]:
    target_frames = int(target_frames)
    if target_frames < 1:
        raise ValueError("源视频没有可用画面。")
    limit = segment_frame_limit(segment_seconds)
    overlap = normalize_overlap(overlap_frames, limit)
    segments = []
    start = 0
    head = 0
    while start < target_frames:
        # Windows end on the 17k+5 lattice so the next window can carry this latent tail.
        requested = int((window_lengths or {}).get(start, limit))
        requested = H3_MIN_FRAMES + max(0, requested - H3_MIN_FRAMES) // H3_FRAME_STEP * H3_FRAME_STEP
        content_frames = min(max(align_h3_frames(head + 1), requested), target_frames - start)
        # Absorb a tiny final remainder rather than create a near-empty window.
        if 0 < target_frames - start - content_frames < H3_MIN_FRAMES:
            content_frames = target_frames - start
        generation_frames = align_h3_frames(content_frames)
        segments.append({
            "index": len(segments),
            "start_frame": start,
            "content_frames": content_frames,
            "generation_frames": generation_frames,
            "overlap_frames": head,
            "output_start_frame": start + head,
            "output_frames": content_frames - head,
            "start_seconds": start / H3_FPS,
            "content_seconds": content_frames / H3_FPS,
            "generation_seconds": generation_frames / H3_FPS,
        })
        if start + content_frames >= target_frames:
            break
        head = normalize_overlap(overlap, content_frames)
        start += content_frames - head
    return segments


def timeline_segments(target_frames, segment_seconds, overlap_frames, cuts=(), edits=()):
    boundaries = [0] + sorted({int(x) for x in cuts if 0 < int(x) < target_frames}) + [target_frames]
    segments = []
    by_start = {int(edit["start_frame"]): edit for edit in edits}
    for scene, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        window_lengths = {start-lo: int(edit["window_frames"]) for start, edit in by_start.items()
                          if lo <= start < hi and edit.get("window_frames", 0) > 0}
        for item in build_segments(hi-lo, segment_seconds, overlap_frames, window_lengths):
            item.update(index=len(segments), scene=scene)
            item["start_frame"] += lo
            item["output_start_frame"] += lo
            item["start_seconds"] = item["start_frame"] / H3_FPS
            segments.append(item)
    for item in segments:
        edit = by_start.get(item["start_frame"], {})
        for name in ("prompt", "role_index", "target_description", "seed_offset", "bypass", "window_frames"):
            if name in edit:
                item[name] = edit[name]
    # A new character, a new shot, or an unedited window must not inherit the old identity.
    for previous, current in zip(segments, segments[1:]):
        if current.get("role_index", 0) != previous.get("role_index", 0) or previous.get("bypass") or current.get("bypass"):
            current["guide_reset"] = True
    return segments
