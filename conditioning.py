"""Per-shot H3 conditioning with the semantics of MiniMaxH3ImageToVideo / AddGuide / ReferenceToVideo.

The native nodes build one conditioning per graph run; the director needs the same payload at two
canvases (first pass and refine), so the keyframe images are kept and re-encoded for the refine canvas.
"""

from dataclasses import dataclass, field
import math

import node_helpers
from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP
from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, FPS, REF_IMAGE_SHORT_EDGE, _encode_ref_audio, adapt_canvas

from .media import fit


@dataclass
class ShotConditioning:
    texts: dict
    keyframe_images: list
    refs: list = field(default_factory=list)
    uses_refs: bool = False
    strength: float = 1.0

    def conds(self, prompt, keyframes):
        """(positive, negative) for ``prompt`` with keyframe latents for the current canvas."""
        values = {}
        if keyframes:
            values["minimax_keyframes"] = keyframes
        if self.refs:
            values["minimax_refs"] = self.refs
        if self.strength < 1.0:
            # H3 mixes this much of each visual condition (references, keyframes) with noise; 0.999 is its default.
            values["minimax_visual_cond_noise_aug"] = VISUAL_COND_TIMESTEP * self.strength
        positive, negative = self.texts[prompt]
        if not values:
            return positive, negative
        return node_helpers.conditioning_set_values(positive, values), node_helpers.conditioning_set_values(negative, values)


def _round(value):
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def place(image, width, height, mode):
    """[1, H, W, C] image on a ``width`` x ``height`` canvas: "crop" covers, "pad" fits inside black bars, "stretch" distorts."""
    if mode != "pad":
        return fit(image, width, height, "center" if mode == "crop" else "disabled")
    h, w = image.shape[1:3]
    scale = min(width / w, height / h)
    inner_w, inner_h = max(1, round(w * scale)), max(1, round(h * scale))
    out = image.new_zeros((1, height, width, 3))
    top, left = (height - inner_h) // 2, (width - inner_w) // 2
    out[:, top:top + inner_h, left:left + inner_w] = fit(image, inner_w, inner_h, "disabled")
    return out


def keyframe_images(window, media):
    """[(generation index, image, fit mode)]; a later keyframe at the same frame replaces an earlier one.

    "auto" follows MiniMaxH3ImageToVideo: the first frame stretches (geometry anchor), the rest cover-crop."""
    frames = {}
    for item in window.shot["keyframes"]:
        index = window.frame(item["at"])
        mode = item["fit"] if item["fit"] != "auto" else "stretch" if index == 0 else "crop"
        frames[index] = (media.image(item["media"]), mode)
    return [(index, image, mode) for index, (image, mode) in sorted(frames.items(), key=lambda kv: kv[0])]


def encode_keyframes(images, vae, width, height):
    return [{"resolved_frame_index": index, "latent": vae.encode(place(image, width, height, mode))} for index, image, mode in images]


def references(media, window, project, source_frames, vae, audio_vae):
    """Presentation items and DiT ref blocks in ReferenceToVideo order: images, videos, standalone audio.

    "match" sizes reference images to the canvas H3 ends on, also for the draft: the same references
    condition the full-size passes, where their detail matters."""
    width, height = project["canvas"][-1]
    refs = window.shot["refs"]
    items, blocks = [], []
    for item in refs["images"]:
        image = media.image(item["media"])
        h, w = image.shape[1:3]
        # Down only, to the canvas area ("match") or a 2048 short edge ("max"), as ReferenceToVideo does.
        if project["ref_image_size"] == "match":
            scale = min(1.0, math.sqrt(width * height / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        if item["fit"] == "auto":
            tw, th = _round(w * scale), _round(h * scale)
            resized = fit(image, tw, th, "disabled")
        else:
            # Crop / pad / stretch take the canvas aspect at the same pixel budget.
            area = w * h * scale * scale
            tw, th = _round(math.sqrt(area * width / height)), _round(math.sqrt(area * height / width))
            resized = place(image, tw, th, item["fit"])
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(resized)})

    videos = [(source_frames, None)] if source_frames is not None else []
    for ref in refs["videos"]:
        vw, vh = media.video_size(ref["media"])
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw, ch = _round(vw), _round(vh)
        frames = media.video(ref["media"], ref["start"], window.gen, cw, ch, "disabled")
        soundtrack = media.audio(ref["media"], ref["start"], window.gen) if ref["audio"] else None
        videos.append((frames, soundtrack))
    for frames, soundtrack in videos:
        n = min(frames.shape[0], window.gen)
        if n < 5:
            raise ValueError(f"第 {window.index + 1} 镜的参考视频不足 5 帧。")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        if soundtrack is not None:
            items.append({"type": "audio"})
        sampled = list(range(0, n, FPS // 2))
        items.append({"type": "video", "data": frames[sampled], "timestamps": [i / 2.0 for i in range(len(sampled))]})
        latent = vae.encode(frames)
        audio_latent, audio_t = _encode_ref_audio(audio_vae, soundtrack) if soundtrack is not None else (None, 0)
        blocks.append({"kind": "video_audio" if audio_t else "video", "latent_t": latent.shape[2],
                       "latent_h": frames.shape[1] // 16, "latent_w": frames.shape[2] // 16,
                       "ref_audio_t": audio_t, "latent": latent, "audio_latent": audio_latent})

    for item in refs["audios"]:
        items.append({"type": "audio"})
        audio_latent, audio_t = _encode_ref_audio(audio_vae, media.audio(item, 0, window.gen))
        blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": audio_latent})
    return items, blocks


def encode_text(clip, prompt, images, ref_items):
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        tokens = clip.tokenize(prompt, images=images)
    return clip.encode_from_tokens_scheduled(tokens)


def shot_conditioning(window, project, media, clip, vae, audio_vae, prompts, cfg, width, height, source_frames=None):
    """Encode every prompt the shot samples with (base, repaint overrides) against one shared presentation on the
    first-pass ``width`` x ``height`` canvas."""
    images = keyframe_images(window, media)
    ref_items, ref_blocks = references(media, window, project, source_frames, vae, audio_vae)
    # fl2va presents the first/last keyframes to the text encoder; middle guides and ref2va do not.
    presented = [place(image, width, height, mode) for index, image, mode in images if index in (0, window.gen - 1)]
    texts = {}
    for prompt in dict.fromkeys(prompts):
        positive = encode_text(clip, prompt, presented, ref_items)
        negative = encode_text(clip, project["negative_prompt"], presented, ref_items) if cfg != 1.0 else positive
        texts[prompt] = (positive, negative)
    return ShotConditioning(texts, images, ref_blocks, bool(ref_items), window.shot["ref_strength"])
