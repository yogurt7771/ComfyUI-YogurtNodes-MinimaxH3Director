"""Director render: cache keys and needs first, then encode / sample / decode work grouped by model.

Stages per shot: base (generate, or encode the source video) -> repaint layers -> 二采 refine ->
decoded frames -> 局部精修 crops -> assembly. With 二采 the base and repaints run on a draft canvas and each pass that
is on lifts every shot by its own scale, then samples it there, in this order: two-pass stops the draft before the
last steps of its schedule and runs those steps on its canvas, the low-noise rerender resamples the result from a low
noise level; the last pass ends at the output size. Every stage result is cached under a key over everything that
affects it, so editing one shot re-runs that shot and the shots continuing from it, nothing else.
"""

from dataclasses import dataclass
import json
import os
import zlib

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import InputImpl
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio, align_frame_count, temporal_shape, video_latent_t
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from server import PromptServer

from . import cache, detail, latent_upscale, sampling
from .cache import digest
from .conditioning import encode_keyframes, encode_text, shot_conditioning
from .continuation import carry_latent_tail, install_drift_control
from .media import fit, match_rate
from .output import Mp4Writer, read_frames, write_frames
from .project import FPS, plan_windows

VERSION = "director-1"
AUDIO_RATE = 48000
EVENT = "yogurt_h3.director"


@dataclass
class Models:
    model: object
    clip: object
    vae: object
    audio_vae: object
    ref_model: object
    refine_model: object
    inpaint_patch: object
    upscale_model: object
    sampler: object
    sigmas: object
    seed: int
    steps: int
    cfg: float
    scheduler: str


def send_event(node_id, shot_id, stage, prompt=None):
    server = getattr(PromptServer, "instance", None)
    if server is not None:
        server.send_sync(EVENT, {"node": node_id, "shot": shot_id, "stage": stage, "prompt": prompt})


def _round32(value):
    return max(64, round(value / 32) * 32)


def _decode(vae, video):
    images = vae.decode(video)
    return images.reshape(-1, *images.shape[-3:]).float().cpu()


def _latent(tensors):
    return sampling.av_latent(tensors["video"], tensors["audio"])


def _store(key, latent):
    video, audio = sampling.streams(latent)
    extra = {"audio_noise": latent["audio_noise"]} if "audio_noise" in latent else {}
    cache.write_tensors(key, {"video": video, "audio": audio, **extra})


class Render:
    def __init__(self, node_id, prompt, project, models, media, settings, report=()):
        self.node_id, self.project, self.m, self.media = node_id, project, models, media
        self.windows = plan_windows(project)
        self.out_w, self.out_h = project["width"], project["height"]
        refine = project["refine"]
        self.passes = [(kind, refine[kind]) for kind in ("two_pass", "rerender") if refine[kind]]
        self.refine = refine if self.passes else None
        # canvas[p] is what pass p starts from and canvas[p + 1] what it samples on; canvas[0] is the draft.
        self.canvas = [(self.out_w, self.out_h)]
        for _, settings in reversed(self.passes):
            w, h = self.canvas[0]
            self.canvas.insert(0, (_round32(w / settings["scale"]), _round32(h / settings["scale"])))
        self.width, self.height = self.canvas[0]
        signature = lambda names: cache.upstream_signature(prompt, node_id, names)
        base = [VERSION, signature(["model", "ref_model", "clip", "vae", "audio_vae", "sampler", "sigmas"]), settings,
                self.width, self.height, project["overlap"], project["negative_prompt"], project["ref_image_size"]]
        lift = self.refine and [signature(["refine_model", "upscale_model"]), self.refine["upscaler"], self.refine["latent_model"],
                                self.refine["upscaler"] == "h3_latent" and latent_upscale.identity(self.refine["latent_model"])]
        self.env = {"base": digest(base), "inpaint": signature(["inpaint_patch"]),
                    "passes": [digest([base, lift, kind, settings, self.canvas[p], self.canvas[p + 1]]) for p, (kind, settings) in enumerate(self.passes)],
                    "detail": digest([base, signature(["refine_model"]), project["detail"]])}
        self.report = list(report)
        self.final, self.refined, self.conds, self.keyframes, self.detail_conds = {}, {}, {}, {}, {}
        # Two-pass drafts keep the noise their generated audio was on, so the full-size pass continues that path.
        self.audio_noise = {}

    def event(self, index, stage):
        send_event(self.node_id, self.windows[index].shot["id"], stage)

    # ------------------------------------------------------------------ shot properties

    @staticmethod
    def uses_refs(shot):
        refs = shot["refs"]
        return bool(refs["images"] or refs["videos"] or refs["audios"] or (shot["source"] and shot["source"]["regen"]))

    @staticmethod
    def edits_source(shot):
        """A source video kept pixel-exact outside the repaint masks."""
        return shot["kind"] == "source" and not shot["source"]["regen"]

    def passthrough(self, shot):
        return self.edits_source(shot) and not shot["repaints"]

    def audio_mode(self, shot):
        return "source" if self.edits_source(shot) and shot["audio"] == "generate" else shot["audio"]

    def seed(self, *parts):
        """Noise seed per shot / layer / region, keyed by their stable ids so equal-length shots never share noise."""
        return (self.m.seed + zlib.crc32(json.dumps(parts).encode())) % 2**63

    def shot_model(self, shot):
        return self.m.ref_model if self.uses_refs(shot) and self.m.ref_model is not None else self.m.model

    def tail_steps(self, shot):
        """Steps of the shot's schedule that two-pass 二采 leaves for its own canvas (0: the draft runs them all)."""
        return self.refine["two_pass"]["steps"] if self.refine and self.refine["two_pass"] and shot["refine"] else 0

    def schedule(self, shot):
        """The shot's own schedule: from noise, from a source video's strength, or for a source edit from its strongest repaint."""
        if self.edits_source(shot):
            denoise = max(layer["strength"] for layer in shot["repaints"])
        else:
            denoise = shot["source"]["strength"] if shot["kind"] == "source" else 1.0
        return sampling.schedule(self.shot_model(shot), self.m.steps, self.m.scheduler, denoise, self.m.sigmas)

    # ------------------------------------------------------------------ planning

    def _spec(self, window):
        shot, identity = window.shot, self.media.identity
        refs, source = shot["refs"], shot["source"]
        lock = [identity(self.project["soundtrack"]), window.start - window.head] if self.audio_mode(shot) == "soundtrack" else None
        return {"kind": shot["kind"], "prompt": shot["prompt"], "seed": self.seed(shot["id"], shot["seed"]), "head": window.head, "gen": window.gen,
                "keyframes": [[window.frame(k["at"]), identity(k["media"]), k["fit"]] for k in shot["keyframes"]],
                "refs": [[[identity(m["media"]), m["fit"]] for m in refs["images"]], [[identity(v["media"]), v["audio"], v["start"]] for v in refs["videos"]],
                         [identity(m) for m in refs["audios"]]],
                "source": source and [identity(source["media"]), source["start"], source["regen"], source["strength"]],
                "audio": [self.audio_mode(shot), lock], "ref_strength": shot["ref_strength"]}

    def plan(self):
        self.keys = []
        for i, window in enumerate(self.windows):
            shot = window.shot
            previous = self.keys[i - 1] if window.carry else None
            layers = [digest(["base", self.env["base"], self._spec(window), self.tail_steps(shot), previous and previous["final"]])]
            for layer in shot["repaints"]:
                layers.append(digest(["repaint", layers[-1], {**layer, "mask": self.media.identity(layer["mask"])}, self.env["inpaint"]]))
            source = shot["source"]
            encoded = lambda size: digest(["source", self.media.identity(source["media"]), source["start"], window.gen, *size])
            passes = []
            for p in range(len(self.passes)):
                if self.passthrough(shot):
                    passes.append(encoded(self.canvas[p + 1]))
                else:
                    passes.append(digest(["refine", passes[-1] if passes else layers[-1], self.env["passes"][p], shot["refine"],
                                          previous and previous["passes"][p]]))
            refined = passes[-1] if passes else encoded(self.canvas[0]) if self.passthrough(shot) else layers[-1]
            frames = digest(["frames", refined, window.head, window.keep])
            details = [digest(["detail", frames, item, self.env["detail"]]) for item in shot["details"]]
            self.keys.append({"layers": layers, "final": layers[-1], "passes": passes, "refined": refined, "frames": frames, "details": details})

        # Backward pass: a shot's latent is needed for its own output or for the tail the next shot copies.
        n = len(self.windows)
        self.need = [None] * n
        for i in reversed(range(n)):
            shot, keys = self.windows[i].shot, self.keys[i]
            following = self.need[i + 1] if i + 1 < n and self.windows[i + 1].carry else {}
            passthrough = self.passthrough(shot)
            frames = not cache.exists(keys["frames"], ".mkv")
            # Each pass feeds the next one of the same shot (a passthrough encodes its source instead) and the same
            # pass of a shot continuing from it.
            passes = [False] * len(self.passes)
            for p in reversed(range(len(self.passes))):
                wanted = (passes[p + 1] if p + 1 < len(passes) else frames) and not passthrough
                passes[p] = not cache.exists(keys["passes"][p]) and (wanted or bool(following) and following["passes"][p])
            final = ((frames and not passes and not passthrough) or (passes and passes[0] and not passthrough)
                     or following.get("base", False))
            last = len(keys["layers"]) - 1
            resume = max((j for j, key in enumerate(keys["layers"]) if cache.exists(key)), default=-1) if final else last
            self.need[i] = {"frames": frames, "passes": passes, "refine": any(passes), "final": final, "resume": resume,
                            "repaint": final and resume < last, "base": final and resume < 0,
                            "details": [not cache.exists(key, ".mkv") for key in keys["details"]]}

    # ------------------------------------------------------------------ media helpers

    def source_frames(self, window, width, height):
        source = window.shot["source"]
        return self.media.video(source["media"], source["start"], window.gen, width, height)

    def encode_audio(self, audio, frames):
        latent, _ = _encode_ref_audio(self.m.audio_vae, audio)
        count = temporal_shape(frames)[2]
        if latent.shape[-1] < count:
            latent = torch.nn.functional.pad(latent, (0, count - latent.shape[-1]))
        return latent[..., :count].to(comfy.model_management.intermediate_device())

    def silence(self, frames):
        return self.encode_audio({"waveform": torch.zeros((1, 2, round(frames * 32000 / FPS))), "sample_rate": 32000}, frames)

    def lock_audio(self, window):
        """Encoded audio the shot is locked to, or None when the shot generates its own sound."""
        mode, shot = self.audio_mode(window.shot), window.shot
        if mode == "generate":
            return None
        if mode == "silent":
            return self.silence(window.gen)
        if mode == "soundtrack":
            audio = self.media.audio(self.project["soundtrack"], window.start - window.head, window.gen)
        else:
            audio = self.media.audio(shot["source"]["media"], shot["source"]["start"], window.gen)
        return self.encode_audio(audio, window.gen)

    def repaint_pixels(self, window, layer, width, height):
        """Grow and feather are in output pixels, the size the editor paints on."""
        a, b = window.span(layer["start"], layer["end"])
        mask = sampling.window_mask(self.media.mask(layer["mask"]), window.gen, a, b, width, height)
        return sampling.grow(mask, round(layer["grow"] * width / self.out_w))

    def composite_mask(self, window, width, height):
        """Feathered union of the repaint masks; regenerated pixels are pasted over the original video with it."""
        masks = [sampling.feather(self.repaint_pixels(window, layer, width, height), round(layer["feather"] * width / self.out_w))
                 for layer in window.shot["repaints"]]
        return torch.stack(masks).amax(0)

    # ------------------------------------------------------------------ stages

    def encode(self):
        """Text encoder and VAE work for every shot that samples in this run."""
        for i, window in enumerate(self.windows):
            flags, shot = self.need[i], window.shot
            layers = range(max(flags["resume"], 0), len(shot["repaints"])) if flags["repaint"] else ()
            refines = flags["refine"] and shot["refine"] and not self.passthrough(shot)
            generates = flags["base"] and not self.edits_source(shot)
            if not (generates or layers or refines):
                continue
            comfy.model_management.throw_exception_if_processing_interrupted()
            self.event(i, "encoding")
            prompts = [shot["prompt"]] + [shot["repaints"][j]["prompt"] or shot["prompt"] for j in layers]
            source = self.source_frames(window, self.width, self.height) if shot["kind"] == "source" and shot["source"]["regen"] else None
            self.conds[i] = shot_conditioning(window, self.project, self.media, self.m.clip, self.m.vae, self.m.audio_vae,
                                              prompts, self.m.cfg, self.width, self.height, source)
            self.keyframes[i] = encode_keyframes(self.conds[i].keyframe_images, self.m.vae, self.width, self.height)
        for i, window in enumerate(self.windows):
            for j, item in enumerate(window.shot["details"]):
                if self.need[i]["details"][j]:
                    prompt = item["prompt"] or window.shot["prompt"]
                    positive = encode_text(self.m.clip, prompt, [], None)
                    negative = encode_text(self.m.clip, self.project["negative_prompt"], [], None) if self.m.cfg != 1.0 else positive
                    self.detail_conds[(i, j)] = (positive, negative)

    def sample_shots(self):
        for i, window in enumerate(self.windows):
            flags, keys, shot = self.need[i], self.keys[i], window.shot
            if not flags["final"]:
                continue
            comfy.model_management.throw_exception_if_processing_interrupted()
            if flags["base"]:
                self.event(i, "sampling")
                latent = self.base(i)
                _store(keys["layers"][0], latent)
                noise = latent.get("audio_noise")
                done = "生成" if not self.edits_source(shot) else "编码源视频"
            else:
                latent = _latent(cache.read_tensors(keys["layers"][flags["resume"]]))
                noise = cache.read_tensors(keys["layers"][0]).get("audio_noise") if self.tail_steps(shot) else None
                done = "读取缓存"
            if noise is not None:
                self.audio_noise[i] = noise
            for j in range(max(flags["resume"], 0), len(shot["repaints"])):
                self.event(i, "repainting")
                latent = self.repaint(i, latent, j)
                _store(keys["layers"][j + 1], latent)
                done += f"，重绘第 {j + 1} 层"
            self.final[i] = latent
            self.report.append(f"第 {i + 1} 镜：{done}")

    def base(self, i):
        window, shot, m = self.windows[i], self.windows[i].shot, self.m
        lock = self.lock_audio(window)
        if self.edits_source(shot):
            video = m.vae.encode(self.source_frames(window, self.width, self.height)).to(comfy.model_management.intermediate_device())
            return sampling.av_latent(video, lock)
        video, audio = sampling.streams(sampling.empty_latent(self.width, self.height, window.gen))
        denoise = shot["source"]["strength"] if shot["kind"] == "source" else 1.0
        if denoise < 1.0:
            video = m.vae.encode(self.source_frames(window, self.width, self.height)).to(video)
        if lock is not None:
            latent = sampling.av_latent(video, lock, audio_mask=sampling.locked_audio_mask(lock))
        else:
            latent = sampling.av_latent(video, audio)
        model = self.shot_model(shot)
        sigmas = self.schedule(shot)
        if self.tail_steps(shot):
            sigmas = sampling.split(sigmas, self.tail_steps(shot))[0]
        if window.carry:
            latent, tokens = carry_latent_tail(latent, self.final[i - 1], window.head, continue_audio=True)
            model = install_drift_control(model, latent, sigmas, tokens)
        positive, negative = self.conds[i].conds(shot["prompt"], self.keyframes[i])
        last = {} if self.tail_steps(shot) and lock is None else None
        out = {"samples": sampling.sample(model, positive, negative, m.cfg, m.sampler, sigmas, latent, self.seed(shot["id"], shot["seed"]), last=last)}
        if last:
            # The draft's last step started at sigmas[-2]; its audio continues from there in the full-size pass.
            out["audio_noise"] = sampling.trajectory_noise(model, last, float(sigmas[-2]))
        return out

    def repaint(self, i, latent, j):
        window, shot, m = self.windows[i], self.windows[i].shot, self.m
        layer = shot["repaints"][j]
        pixels = self.repaint_pixels(window, layer, self.width, self.height)
        video, audio = sampling.streams(latent)
        video_mask = sampling.token_mask(pixels, video.shape[2])
        video_mask[:, :, :video_latent_t(window.head) if window.head else 0] = 0.0
        model = self.shot_model(shot)
        if m.inpaint_patch is not None:
            behind = self.source_frames(window, self.width, self.height) if self.edits_source(shot) else _decode(m.vae, video)
            model = sampling.inpaint_model(model, m.inpaint_patch, m.vae, (pixels > 0.5).float(), behind)
        sigmas = sampling.schedule(model, m.steps, m.scheduler, layer["strength"], m.sigmas)
        positive, negative = self.conds[i].conds(layer["prompt"] or shot["prompt"], self.keyframes[i])
        masked = sampling.av_latent(video, audio, video_mask, sampling.locked_audio_mask(audio))
        seed = self.seed(shot["id"], shot["seed"], layer["id"], layer["seed"])
        return {"samples": sampling.sample(model, positive, negative, m.cfg, m.sampler, sigmas, masked, seed)}

    def upscale(self, frames, width, height):
        out = []
        for start in range(0, frames.shape[0], 8):
            chunk = frames[start:start + 8]
            if self.refine["upscaler"] == "model" and self.m.upscale_model is not None:
                chunk = ImageUpscaleWithModel.execute(self.m.upscale_model, chunk).result[0].float().cpu()
            out.append(fit(chunk, width, height))
        return torch.cat(out)

    def refine_shots(self):
        results = self.final
        for p in range(len(self.passes)):
            results = self.refine_pass(p, results)
        self.refined = results

    def refine_pass(self, p, inputs):
        """Lift every shot that takes pass ``p`` to its canvas first (learned latent upscaler, or VAE decode / upscale /
        encode), then sample them in order (UNET). ``inputs`` holds the previous stage's latents of this run; the
        returned results are what the next pass and the decoder read."""
        kind, settings = self.passes[p]
        (w0, h0), (w1, h1) = self.canvas[p], self.canvas[p + 1]
        prepared, lift = {}, None
        for i, window in enumerate(self.windows):
            if not self.need[i]["passes"][p]:
                continue
            comfy.model_management.throw_exception_if_processing_interrupted()
            self.event(i, "upscaling")
            shot, mask, keyframes = window.shot, None, None
            if self.passthrough(shot):
                video = self.m.vae.encode(self.source_frames(window, w1, h1))
                audio = torch.zeros((1, 32, 2, temporal_shape(window.gen)[2]))
            else:
                latent = inputs[i] if i in inputs else _latent(cache.read_tensors(self.keys[i]["passes"][p - 1]))
                video, audio = sampling.streams(latent)
                # Scale 1 keeps the canvas; only source edits still go through their pixels.
                at_size = (w0, h0) == (w1, h1)
                if self.refine["upscaler"] == "h3_latent" and not self.edits_source(shot) and not at_size:
                    lift = lift or latent_upscale.load(self.refine["latent_model"])
                    video = latent_upscale.upscale(lift, video, h1 // 16, w1 // 16)
                elif self.edits_source(shot) or not at_size:
                    # Source edits are composited with the source pixels, so they always take the pixel path.
                    frames = self.upscale(_decode(self.m.vae, video), w1, h1)
                    if self.edits_source(shot):
                        frames = torch.lerp(self.source_frames(window, w1, h1), frames, self.composite_mask(window, w1, h1)[..., None])
                        pixels = torch.stack([self.repaint_pixels(window, layer, w1, h1) for layer in shot["repaints"]])
                        mask = sampling.token_mask(pixels.amax(0), video.shape[2])
                    video = self.m.vae.encode(frames)
                if i in self.conds:
                    keyframes = encode_keyframes(self.conds[i].keyframe_images, self.m.vae, w1, h1)
            prepared[i] = (video.to(comfy.model_management.intermediate_device()), audio, mask, keyframes)

        out = {}
        for i, window in enumerate(self.windows):
            following = i + 1 < len(self.windows) and self.windows[i + 1].carry and self.need[i + 1]["passes"][p]
            if i not in prepared:
                if following:
                    out[i] = _latent(cache.read_tensors(self.keys[i]["passes"][p]))
                continue
            shot, m = window.shot, self.m
            video, audio, mask, keyframes = prepared.pop(i)
            latent = sampling.av_latent(video, audio)
            if shot["refine"] and not self.passthrough(shot):
                comfy.model_management.throw_exception_if_processing_interrupted()
                model = m.refine_model if m.refine_model is not None else self.shot_model(shot)
                if kind == "two_pass":
                    self.event(i, "refining")
                    name, sigmas = "两段式二采", sampling.split(self.schedule(shot), settings["steps"])[1]
                    # The draft stopped early, so generated sound takes the last steps too, on the path it was already on.
                    free_audio = self.audio_mode(shot) == "generate"
                    audio_noise = self.audio_noise.get(i) if free_audio else None
                    if free_audio and audio_noise is None:
                        self.report.append(f"第 {i + 1} 镜：首采没有留下声音的噪声（采样器未回报或旧缓存），二采声音重新加噪")
                    seed = self.seed(shot["id"], shot["seed"])
                else:
                    self.event(i, "rerendering")
                    name, sigmas = "低噪重绘", sampling.schedule(model, settings["steps"], m.scheduler, settings["denoise"])
                    free_audio, audio_noise = False, None
                    # Its own seed: after two-pass the shot's seed would add back the exact noise the tail started from.
                    seed = self.seed(shot["id"], shot["seed"], "rerender")
                latent = sampling.av_latent(video, audio, mask, None if free_audio else sampling.locked_audio_mask(audio))
                if window.carry:
                    latent, tokens = carry_latent_tail(latent, out[i - 1], window.head, continue_audio=free_audio)
                    model = install_drift_control(model, latent, sigmas, tokens)
                positive, negative = self.conds[i].conds(shot["prompt"], keyframes)
                latent = {"samples": sampling.sample(model, positive, negative, m.cfg, m.sampler, sigmas, latent, seed, audio_noise=audio_noise)}
                size = f"{w1}×{h1}" if (w0, h0) == (w1, h1) else f"{w0}×{h0} → {w1}×{h1}"
                self.report.append(f"第 {i + 1} 镜：{name} {size}（{sigmas.shape[-1] - 1} 步，从 {float(sigmas[0]):.2f} 起）")
            _store(self.keys[i]["passes"][p], latent)
            out[i] = latent
        return out

    def decode_shots(self):
        for i, window in enumerate(self.windows):
            keys, shot = self.keys[i], window.shot
            if not self.need[i]["frames"]:
                continue
            comfy.model_management.throw_exception_if_processing_interrupted()
            self.event(i, "decoding")
            audio = None
            if self.passthrough(shot):
                frames = self.source_frames(window, self.out_w, self.out_h)
            else:
                latent = (self.refined if self.refine else self.final).get(i)
                if latent is None:
                    latent = _latent(cache.read_tensors(keys["refined"] if self.refine else keys["final"]))
                video, audio = sampling.streams(latent)
                frames = _decode(self.m.vae, video)
                if self.edits_source(shot):
                    frames = torch.lerp(self.source_frames(window, self.out_w, self.out_h), frames,
                                        self.composite_mask(window, self.out_w, self.out_h)[..., None])
            if self.audio_mode(shot) == "generate":
                wave = self.m.audio_vae.decode(audio).movedim(-1, 1).float().cpu()
                rate = int(self.m.audio_vae.first_stage_model.output_sample_rate)
                cache.write_tensors(keys["frames"], {"waveform": match_rate({"waveform": wave, "sample_rate": rate}, AUDIO_RATE)[0]})
            write_frames(keys["frames"], frames[window.head:window.head + window.keep])

    def detail_shots(self):
        """Crop and encode every refinement region (VAE), sample them all (UNET), then decode (VAE)."""
        jobs = []
        for i, window in enumerate(self.windows):
            frames = None
            for j, item in enumerate(window.shot["details"]):
                if not self.need[i]["details"][j]:
                    continue
                comfy.model_management.throw_exception_if_processing_interrupted()
                self.event(i, "detailing")
                frames = read_frames(self.keys[i]["frames"]) if frames is None else frames
                a, b = window.frame(item["start"]) - window.head, window.frame(item["end"]) - window.head + 1
                canvas = detail.crop_canvas(item["box"][2] * self.out_w, item["box"][3] * self.out_h, item["megapixels"])
                boxes = detail.track(frames[a:b], item, canvas[0] / canvas[1])
                crops = detail.crop(frames[a:b], boxes, *canvas)
                length = align_frame_count(crops.shape[0])
                padded = torch.cat([crops, crops[-1:].expand(length - crops.shape[0], -1, -1, -1)])
                video = self.m.vae.encode(padded).to(comfy.model_management.intermediate_device())
                jobs.append((i, j, item, boxes, crops.shape[0], video, self.silence(length)))
        sampled = []
        model = self.m.refine_model if self.m.refine_model is not None else self.m.model
        for i, j, item, boxes, count, video, silent in jobs:
            comfy.model_management.throw_exception_if_processing_interrupted()
            self.event(i, "detailing")
            latent = sampling.av_latent(video, silent, audio_mask=sampling.locked_audio_mask(silent))
            sigmas = sampling.schedule(model, self.project["detail"]["steps"], self.m.scheduler, item["denoise"])
            positive, negative = self.detail_conds[(i, j)]
            shot = self.windows[i].shot
            seed = self.seed(shot["id"], shot["seed"], item["id"], item["seed"])
            sampled.append((i, j, boxes, count, sampling.sample(model, positive, negative, self.m.cfg, self.m.sampler, sigmas, latent, seed)))
        for i, j, boxes, count, samples in sampled:
            crops = _decode(self.m.vae, sampling.streams(samples)[0])[:count]
            key = self.keys[i]["details"][j]
            cache.write_json(key, boxes)
            write_frames(key, crops)
            self.report.append(f"第 {i + 1} 镜：局部精修 {j + 1}（{crops.shape[2]}×{crops.shape[1]}）")

    # ------------------------------------------------------------------ assembly

    def shot_audio(self, window):
        """(waveform [2, S] at AUDIO_RATE, timeline frame where it starts), or (None, _) for silence."""
        shot = window.shot
        mode = self.audio_mode(shot)
        if mode == "generate":
            wave = cache.read_tensors(self.keys[window.index]["frames"])["waveform"]
            previous = self.windows[window.index - 1].shot if window.index else None
            # Carried audio already releases into this shot, so its head replaces the previous shot's tail.
            if window.head and self.audio_mode(previous) == "generate":
                return wave, window.start - window.head
            return wave[:, round(window.head * AUDIO_RATE / FPS):], window.start
        if mode == "silent":
            return None, window.start
        if mode == "soundtrack":
            audio = self.media.audio(self.project["soundtrack"], window.start, window.keep)
        else:
            audio = self.media.audio(shot["source"]["media"], shot["source"]["start"], window.keep)
        return match_rate(audio, AUDIO_RATE)[0], window.start

    def assemble(self, filename_prefix):
        total = sum(w.keep for w in self.windows)
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), self.out_w, self.out_h)
        name = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(folder, name)
        temporary = path + ".partial"
        waveform = torch.zeros((2, round(total * AUDIO_RATE / FPS)))
        writer = Mp4Writer(temporary, self.out_w, self.out_h, AUDIO_RATE)
        try:
            for window in self.windows:
                keys = self.keys[window.index]
                frames = read_frames(keys["frames"])
                for item, key in zip(window.shot["details"], keys["details"]):
                    a = window.frame(item["start"]) - window.head
                    boxes, crops = cache.read_json(key), read_frames(key)
                    frames[a:a + crops.shape[0]] = detail.stitch(frames[a:a + crops.shape[0]], crops, boxes, item["follow"] == "face")
                writer.write(frames)
                wave, at = self.shot_audio(window)
                lo, hi = round(at * AUDIO_RATE / FPS), round((window.start + window.keep) * AUDIO_RATE / FPS)
                waveform[:, lo:hi] = 0.0
                if wave is not None:
                    count = min(hi - lo, wave.shape[-1])
                    waveform[:, lo:lo + count] = wave[:, :count]
                self.event(window.index, "complete")
            writer.finish(waveform)
        except BaseException:
            writer.abort()
            if os.path.exists(temporary):
                os.remove(temporary)
            raise
        os.replace(temporary, path)
        return InputImpl.VideoFromFile(path), {"filename": name, "subfolder": subfolder, "type": "output"}

    def run(self, filename_prefix):
        self.plan()
        self.encode()
        self.sample_shots()
        self.refine_shots()
        self.decode_shots()
        self.detail_shots()
        return self.assemble(filename_prefix)
