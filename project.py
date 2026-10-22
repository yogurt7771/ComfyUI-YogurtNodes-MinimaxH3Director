"""Director project: the JSON written by the timeline editor, normalized into H3 generation windows.

Every shot is generated on H3's 17k+5 frame grid at 24 fps. A shot that continues the previous one
re-generates the previous ``overlap`` frames from that shot's latent tail (``head``) and drops them
from the output, so the timeline only receives ``keep`` new frames per shot.
"""

from dataclasses import dataclass
import json

from comfy_extras.nodes_minimax_h3 import align_frame_count

from .latent_upscale import DEFAULT as LATENT_UPSCALER

FPS = 24
OVERLAPS = (0, 5, 22, 39, 56)
MEDIA_TYPES = ("input", "output", "temp")
AUDIO_MODES = ("generate", "soundtrack", "source", "silent")
REFINE_UPSCALERS = ("h3_latent", "lanczos", "model")
# How an image meets the canvas; "auto" is the native H3 rule (references keep their aspect, the first keyframe
# stretches, later keyframes cover-crop).
FITS = ("auto", "crop", "pad", "stretch")
# Where a 局部重绘 layer's mask comes from: painted in the editor, a masks socket, a SAM 3 text target, a followed
# face or a fixed box.
MASK_SOURCES = ("paint", "input", "text", "face", "box")

DEFAULT_PROJECT = {
    "version": 1,
    "soundtrack": None,
    "refs": {"images": [], "videos": [], "audios": []},
    "story": "",
    "rewrite": {"enabled": False, "notes": "", "seed": 0},
    "shots": [],
}

# Render settings are director node widgets, not part of the editor's project JSON.
SETTINGS = {
    "width": 832,
    "height": 480,
    "overlap": 22,
    "ref_image_size": "match",
    "negative_prompt": "",
    "two_pass": False,
    "two_pass_scale": 2.0,
    "two_pass_steps": 2,
    "two_pass_upscaler": "h3_latent",
    "two_pass_latent_model": LATENT_UPSCALER,
    "rerender": False,
    "rerender_scale": 1.0,
    "rerender_steps": 2,
    "rerender_denoise": 0.35,
    "rerender_upscaler": "h3_latent",
    "rerender_latent_model": LATENT_UPSCALER,
    "detail": True,
    "detail_steps": 8,
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
    """A file under input/output/temp, or a graph input such as ``image_1`` / ``mask_2`` / ``video_0`` / ``audio_0``."""
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


def _media_list(values):
    return [m for m in (media(v) for v in (values or [])) if m is not None]


def _fit(value):
    return value if value in FITS else "auto"


def _privacy(raw):
    """Whether the prompt-rewriting model may see this media, and the user's optional words for it instead."""
    return {"share": bool(raw.get("share", True)), "description": str(raw.get("description") or "")}


def _refs(raw):
    raw = raw or {}
    # ``start`` is the 24 fps frame of the material where the reference clip begins.
    videos = [{"media": media(item["media"]), "audio": bool(item.get("audio")), "start": _int(item.get("start"), 0, 0, 10**7), **_privacy(item)}
              for item in raw.get("videos") or [] if isinstance(item, dict) and media(item.get("media"))]
    # Reference images used to be stored as bare media; both forms are read.
    images = [{"media": media(item.get("media", item)), "fit": _fit(item.get("fit")), **_privacy(item)}
              for item in raw.get("images") or [] if isinstance(item, dict) and media(item.get("media", item))]
    return {"images": images, "videos": videos, "audios": _media_list(raw.get("audios"))}


def _shot(raw, index, global_refs):
    shot_id = str(raw.get("id") or f"shot{index + 1}")
    kind = "source" if raw.get("kind") == "source" else "generate"
    # Global references come first in every inheriting shot, so their labels are the same in every shot.
    inherit_refs = bool(raw.get("inherit_refs", True))
    inherited = global_refs if inherit_refs else _refs(None)
    own = _refs(raw.get("refs"))
    source = None
    if kind == "source":
        raw_source = raw.get("source") or {}
        source = {"media": media(raw_source.get("media")), "start": _int(raw_source.get("start"), 0, 0, 10**7),
                  "regen": bool(raw_source.get("regen")), "strength": _float(raw_source.get("strength"), 1.0, 0.05, 1.0), **_privacy(raw_source)}
        if source["media"] is None:
            raise ValueError(f"第 {index + 1} 镜是源视频镜头，但还没有选择源视频。")
    audio = raw.get("audio") if raw.get("audio") in AUDIO_MODES else ("source" if kind == "source" else "generate")
    if audio == "source" and kind != "source":
        audio = "generate"
    keyframes = []
    for item in raw.get("keyframes") or []:
        if isinstance(item, dict) and media(item.get("media")):
            keyframes.append({"at": _int(item.get("at"), 0, -1, 10**6), "media": media(item["media"]), "fit": _fit(item.get("fit")), **_privacy(item)})
    layers = [_layer(item, str(item.get("id") or f"{shot_id}-r{j}"), False) for j, item in enumerate(raw.get("repaints") or [])
              if isinstance(item, dict) and item.get("enabled") is not False]
    layers += [_layer(item, str(item.get("id") or f"{shot_id}-d{j}"), True) for j, item in enumerate(raw.get("details") or [])
               if isinstance(item, dict) and item.get("enabled") is not False]
    # Numbered as the editor lists them: its layers, then former 局部精修 regions.
    for number, layer in enumerate(layers, 1):
        if layer is not None:
            layer["number"] = number
    layers = [layer for layer in layers if layer is not None]
    return {
        "id": shot_id,
        "kind": kind,
        "frames": _int(raw.get("frames"), 121, 1, 24 * 600),
        "prompt": str(raw.get("prompt") or ""),
        "continue": bool(raw.get("continue", True)),
        "seed": _int(raw.get("seed"), 0, 0, 2**31),
        "keyframes": keyframes,
        "refs": {kind: inherited[kind] + own[kind] for kind in own},
        "global_refs": {kind: len(inherited[kind]) for kind in own},
        "inherit_refs": inherit_refs,
        "inherit_prompt": bool(raw.get("inherit_prompt", True)),
        # How closely the shot follows its references and keyframes (H3's visual condition strength).
        "ref_strength": _float(raw.get("ref_strength"), 1.0, 0.1, 1.0),
        "source": source,
        "audio": audio,
        # 局部重绘 layers: in place on the draft before 二采, or cropped, enlarged and pasted back after it.
        "repaints": [layer for layer in layers if not layer["crop"]],
        "details": [layer for layer in layers if layer["crop"]],
        "refine": bool(raw.get("refine", True)),
        "rewrite": bool(raw.get("rewrite", True)),
    }


def _upscaler(settings, prefix):
    upscaler = settings[f"{prefix}_upscaler"]
    return {"upscaler": upscaler if upscaler in REFINE_UPSCALERS else "lanczos",
            "latent_model": str(settings[f"{prefix}_latent_model"] or LATENT_UPSCALER)}


def _box(value):
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        return None
    x, y, w, h = (_float(v, 0.0, 0.0, 1.0) for v in value)
    return [x, y, min(w, 1.0 - x), min(h, 1.0 - y)] if w >= 0.01 and h >= 0.01 else None


def _target(raw):
    """SAM 3 text target: comma-separated phrases (``name:N`` keeps up to N of that kind), score threshold, object cap."""
    text = str(raw.get("target") or "").strip()
    if not text:
        return None
    return {"text": text, "threshold": _float(raw.get("threshold"), 0.5, 0.05, 1.0), "objects": _int(raw.get("objects"), 4, 0, 64)}


def _layer(raw, layer_id, legacy_detail):
    """A 局部重绘 layer, or None when its mask source has nothing to use. ``legacy_detail`` reads a former 局部精修
    region: a 放大贴回 layer on a fixed or face-following box."""
    painted = raw.get("mask") if isinstance(raw.get("mask"), dict) else {}
    if legacy_detail:
        source = "face" if raw.get("follow") == "face" else "box"
    else:
        source = raw.get("source") if raw.get("source") in MASK_SOURCES else ("input" if painted.get("input") else "paint")
    layer = {"id": layer_id, "source": source, "mask": None, "target": None, "box": None,
             "detector": str(raw.get("detector") or "face_yolov8m.pt"),
             "start": _int(raw.get("start"), 0, 0, 10**6), "end": _int(raw.get("end"), -1, -1, 10**6),
             "strength": _float(raw.get("denoise" if legacy_detail else "strength"), 0.4 if legacy_detail else 1.0, 0.05, 1.0),
             "prompt": str(raw.get("prompt") or ""), "grow": _int(raw.get("grow"), 8, 0, 256),
             "feather": _int(raw.get("feather"), 8, 0, 256), "seed": _int(raw.get("seed"), 0, 0, 2**31),
             "crop": bool(raw.get("crop", legacy_detail)), "megapixels": _float(raw.get("megapixels"), 0.5, 0.1, 1.0)}
    if source == "paint":
        layer["mask"] = media(painted)
    elif source == "input":
        layer["mask"] = media({"input": raw.get("input") or painted.get("input")})
    elif source == "text":
        layer["target"] = _target(raw)
    else:
        layer["box"] = _box(raw.get("box"))
    return layer if layer["mask"] or layer["target"] or layer["box"] else None


def load_project(text, settings=None):
    """The editor's project JSON plus the node's render ``settings`` (widget values, see SETTINGS)."""
    try:
        data = json.loads(text) if text and text.strip() else {}
    except json.JSONDecodeError as error:
        raise ValueError(f"导演台工程数据损坏：{error}") from error
    if not isinstance(data, dict):
        raise ValueError("导演台工程数据必须是 JSON 对象。")
    s = {**SETTINGS, **(settings or {})}
    rewrite = {**DEFAULT_PROJECT["rewrite"], **(data.get("rewrite") or {})}
    project = {
        "width": _int(s["width"], 832, 64, 4096) // 32 * 32,
        "height": _int(s["height"], 480, 64, 4096) // 32 * 32,
        "overlap": max(v for v in OVERLAPS if v <= _int(s["overlap"], 22, 0, 56)),
        "negative_prompt": str(s["negative_prompt"] or ""),
        "ref_image_size": "max" if s["ref_image_size"] == "max" else "match",
        "soundtrack": media(data.get("soundtrack")),
        "refs": _refs(data.get("refs")),
        "story": str(data.get("story") or ""),
        "rewrite": {"enabled": bool(rewrite["enabled"]), "notes": str(rewrite["notes"] or ""), "seed": _int(rewrite["seed"], 0, 0, 2**31)},
        # 二采 passes, either or both; each lifts the canvas by its own scale and the last one ends at width x height.
        "refine": {"two_pass": {"scale": _float(s["two_pass_scale"], 2.0, 1.0, 4.0), "steps": _int(s["two_pass_steps"], 2, 1, 200),
                                **_upscaler(s, "two_pass")} if s["two_pass"] else None,
                   "rerender": {"scale": _float(s["rerender_scale"], 1.0, 1.0, 4.0), "steps": _int(s["rerender_steps"], 2, 1, 200),
                                "denoise": _float(s["rerender_denoise"], 0.35, 0.05, 1.0), **_upscaler(s, "rerender")} if s["rerender"] else None},
        "detail": {"steps": _int(s["detail_steps"], 8, 1, 200)},
    }
    project["shots"] = [_shot(raw, i, project["refs"]) for i, raw in enumerate(data.get("shots") or []) if isinstance(raw, dict)]
    if not s["detail"]:
        for shot in project["shots"]:
            shot["details"] = []
    # The editor's 预览 runs one layer's mask instead of the film.
    preview = data.get("mask_preview")
    project["preview"] = {"layer": str(preview["layer"])} if isinstance(preview, dict) and preview.get("layer") else None
    if not project["shots"]:
        raise ValueError("导演台里还没有镜头：在时间线上添加至少一个镜头。")
    if any(s["audio"] == "soundtrack" for s in project["shots"]) and project["soundtrack"] is None:
        raise ValueError("有镜头的声音选了“配乐音轨”，但还没有设置配乐：在导演台“镜头”页的声音旁选择配乐。")
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
