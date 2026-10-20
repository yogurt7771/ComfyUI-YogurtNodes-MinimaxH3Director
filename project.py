"""Director project: the JSON written by the timeline editor, normalized into H3 generation windows.

Every shot is generated on H3's 17k+5 frame grid at 24 fps. A shot that continues the previous one
re-generates the previous ``overlap`` frames from that shot's latent tail (``head``) and drops them
from the output, so the timeline only receives ``keep`` new frames per shot.
"""

from dataclasses import dataclass
import json

from comfy_extras.nodes_minimax_h3 import align_frame_count

FPS = 24
OVERLAPS = (0, 5, 22, 39, 56)
MEDIA_TYPES = ("input", "output", "temp")
AUDIO_MODES = ("generate", "soundtrack", "source", "silent")
MAX_REF_IMAGES, MAX_REF_VIDEOS, MAX_REF_AUDIOS = 9, 3, 3

DEFAULT_PROJECT = {
    "version": 1,
    "width": 832,
    "height": 480,
    "overlap": 22,
    "negative_prompt": "",
    "ref_image_size": "match",
    "soundtrack": None,
    "story": "",
    "rewrite": {"enabled": False, "notes": "", "seed": 0},
    "refine": {"enabled": False, "scale": 1.5, "upscaler": "lanczos", "steps": 8, "denoise": 0.35},
    "detail": {"steps": 8},
    "shots": [],
}


@dataclass
class Window:
    index: int
    shot: dict
    head: int
    gen: int
    keep: int
    start: int
    carry: bool

    def frame(self, at):
        """Generation-window index of kept frame ``at`` (negative counts from the last kept frame)."""
        at = self.keep + at if at < 0 else at
        return self.head + min(max(at, 0), self.keep - 1)

    def span(self, start, end):
        """Generation-window range [a, b) of kept frames start..end inclusive; end < 0 counts from the end."""
        return self.frame(start), self.frame(end) + 1


def _int(value, default, lo, hi):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(value, lo), hi)


def _float(value, default, lo, hi):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(value, lo), hi)


def media(value):
    """A file under input/output/temp, or a graph input such as ``image_1`` / ``mask_2``."""
    if not isinstance(value, dict):
        return None
    if value.get("input"):
        return {"input": str(value["input"])}
    if not value.get("filename"):
        return None
    kind = value.get("type") or "input"
    if kind not in MEDIA_TYPES:
        raise ValueError(f"素材目录只能是 {'/'.join(MEDIA_TYPES)}，收到 {kind}。")
    return {"filename": str(value["filename"]), "subfolder": str(value.get("subfolder") or ""), "type": kind}


def _media_list(values, limit):
    return [m for m in (media(v) for v in (values or [])) if m is not None][:limit]


def _shot(raw, index):
    shot_id = str(raw.get("id") or f"shot{index + 1}")
    kind = "source" if raw.get("kind") == "source" else "generate"
    refs = raw.get("refs") or {}
    videos = []
    for item in (refs.get("videos") or [])[:MAX_REF_VIDEOS]:
        if isinstance(item, dict) and media(item.get("media")):
            videos.append({"media": media(item["media"]), "audio": bool(item.get("audio"))})
    source = None
    if kind == "source":
        raw_source = raw.get("source") or {}
        source = {"media": media(raw_source.get("media")), "start": _int(raw_source.get("start"), 0, 0, 10**7),
                  "regen": bool(raw_source.get("regen")), "strength": _float(raw_source.get("strength"), 1.0, 0.05, 1.0)}
        if source["media"] is None:
            raise ValueError(f"第 {index + 1} 镜是源视频镜头，但还没有选择源视频。")
    audio = raw.get("audio") if raw.get("audio") in AUDIO_MODES else ("source" if kind == "source" else "generate")
    if audio == "source" and kind != "source":
        audio = "generate"
    keyframes = []
    for item in raw.get("keyframes") or []:
        if isinstance(item, dict) and media(item.get("media")):
            keyframes.append({"at": _int(item.get("at"), 0, -1, 10**6), "media": media(item["media"])})
    repaints = []
    for j, item in enumerate(raw.get("repaints") or []):
        if not isinstance(item, dict) or media(item.get("mask")) is None or item.get("enabled") is False:
            continue
        repaints.append({"id": str(item.get("id") or f"{shot_id}-r{j}"), "mask": media(item["mask"]), "start": _int(item.get("start"), 0, 0, 10**6),
                         "end": _int(item.get("end"), -1, -1, 10**6), "strength": _float(item.get("strength"), 1.0, 0.05, 1.0),
                         "prompt": str(item.get("prompt") or ""), "grow": _int(item.get("grow"), 8, 0, 256),
                         "feather": _int(item.get("feather"), 8, 0, 256), "seed": _int(item.get("seed"), 0, 0, 2**31)})
    details = []
    for j, item in enumerate(raw.get("details") or []):
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        box = item.get("box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        x, y, w, h = (_float(v, 0.0, 0.0, 1.0) for v in box)
        if w < 0.01 or h < 0.01:
            continue
        details.append({"id": str(item.get("id") or f"{shot_id}-d{j}"), "box": [x, y, min(w, 1.0 - x), min(h, 1.0 - y)], "follow": "face" if item.get("follow") == "face" else "none",
                        "detector": str(item.get("detector") or "face_yolov8m.pt"),
                        "start": _int(item.get("start"), 0, 0, 10**6), "end": _int(item.get("end"), -1, -1, 10**6),
                        "prompt": str(item.get("prompt") or ""), "denoise": _float(item.get("denoise"), 0.4, 0.05, 1.0),
                        "megapixels": _float(item.get("megapixels"), 0.5, 0.1, 1.0), "seed": _int(item.get("seed"), 0, 0, 2**31)})
    return {
        "id": shot_id,
        "kind": kind,
        "frames": _int(raw.get("frames"), 121, 1, 24 * 600),
        "prompt": str(raw.get("prompt") or ""),
        "continue": bool(raw.get("continue", True)),
        "seed": _int(raw.get("seed"), 0, 0, 2**31),
        "keyframes": keyframes,
        "refs": {"images": _media_list(refs.get("images"), MAX_REF_IMAGES), "videos": videos,
                 "audios": _media_list(refs.get("audios"), MAX_REF_AUDIOS)},
        "source": source,
        "audio": audio,
        "repaints": repaints,
        "details": details,
        "refine": bool(raw.get("refine", True)),
        "rewrite": bool(raw.get("rewrite", True)),
    }


def load_project(text):
    try:
        data = json.loads(text) if text and text.strip() else {}
    except json.JSONDecodeError as error:
        raise ValueError(f"导演台工程数据损坏：{error}") from error
    if not isinstance(data, dict):
        raise ValueError("导演台工程数据必须是 JSON 对象。")
    refine = {**DEFAULT_PROJECT["refine"], **(data.get("refine") or {})}
    detail = {**DEFAULT_PROJECT["detail"], **(data.get("detail") or {})}
    rewrite = {**DEFAULT_PROJECT["rewrite"], **(data.get("rewrite") or {})}
    project = {
        "width": _int(data.get("width"), 832, 64, 4096) // 32 * 32,
        "height": _int(data.get("height"), 480, 64, 4096) // 32 * 32,
        "overlap": max(v for v in OVERLAPS if v <= _int(data.get("overlap"), 22, 0, 56)),
        "negative_prompt": str(data.get("negative_prompt") or ""),
        "ref_image_size": "max" if data.get("ref_image_size") == "max" else "match",
        "soundtrack": media(data.get("soundtrack")),
        "story": str(data.get("story") or ""),
        "rewrite": {"enabled": bool(rewrite["enabled"]), "notes": str(rewrite["notes"] or ""), "seed": _int(rewrite["seed"], 0, 0, 2**31)},
        "refine": {"enabled": bool(refine["enabled"]), "scale": _float(refine["scale"], 1.5, 1.0, 4.0),
                   "upscaler": "model" if refine["upscaler"] == "model" else "lanczos",
                   "steps": _int(refine["steps"], 8, 1, 200), "denoise": _float(refine["denoise"], 0.35, 0.05, 1.0)},
        "detail": {"steps": _int(detail["steps"], 8, 1, 200)},
        "shots": [_shot(raw, i) for i, raw in enumerate(data.get("shots") or []) if isinstance(raw, dict)],
    }
    if not project["shots"]:
        raise ValueError("导演台里还没有镜头：在时间线上添加至少一个镜头。")
    if any(s["audio"] == "soundtrack" for s in project["shots"]) and project["soundtrack"] is None:
        raise ValueError("有镜头使用“配乐音轨”，但工程还没有设置配乐音轨。")
    return project


def plan_windows(project):
    shots, overlap = project["shots"], project["overlap"]
    windows, start = [], 0
    carries = [False] * len(shots)
    for i, shot in enumerate(shots):
        # Only generated shots can continue a tail; a source shot's content is defined by its video.
        carries[i] = i > 0 and overlap > 0 and shot["continue"] and shot["kind"] == "generate"
    gens = []
    for i, shot in enumerate(shots):
        head = overlap if carries[i] else 0
        gens.append(align_frame_count(head + shot["frames"]))
        if carries[i] and gens[i - 1] < overlap:
            carries[i], gens[i] = False, align_frame_count(shot["frames"])
    for i, shot in enumerate(shots):
        head = overlap if carries[i] else 0
        # The next shot copies this window's latent tail, and an end keyframe pins the last generated
        # frame, so both keep the whole window instead of cropping the grid padding.
        fixed_tail = (i + 1 < len(shots) and carries[i + 1]) or any(k["at"] == -1 for k in shot["keyframes"])
        keep = gens[i] - head if fixed_tail else shot["frames"]
        windows.append(Window(i, shot, head, gens[i], keep, start, carries[i]))
        start += keep
    return windows
