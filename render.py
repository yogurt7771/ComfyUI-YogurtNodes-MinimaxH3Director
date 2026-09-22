"""One long-video sampler for every H3 plan: checkpointed segments, latent continuation and native second pass."""

import nodes
import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_custom_sampler import RandomNoise
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
from server import PromptServer

from .cache import digest, tensor_digest, video_digest, upstream_signature, selected_segments, read_checkpoint, write_checkpoint
from .continuation import carry_latent_tail, install_drift_control
from .planning import H3_FPS, canvas
from .references import combined_references, reference_digest

_CATEGORY = "Yogurt/MiniMax H3 Director"
_INTERNAL_CATEGORY = "Yogurt/MiniMax H3 Director/Internal"
Plan = io.Custom("YOGURT_H3_PLAN")
RefineSettings = io.Custom("YOGURT_H3_REFINE")
Carry = io.Custom("YOGURT_H3_CARRY")


def make_plan(aspect, megapixels, segments, target_frames, audio, continue_audio, ref_image_size,
              images=(), reference_media=None, source=None, refine=None, **fields):
    """Assemble the sampler contract. JSON-safe fields sit at the top level; tensors, media and models in ``inputs``."""
    width, height = canvas(*aspect, megapixels)
    inputs = dict(images=list(images), reference_media=reference_media, source=source)
    refine_view = None
    if refine:
        refine_width, refine_height = canvas(*aspect, refine["megapixels"])
        refine_view = dict(width=refine_width, height=refine_height, upscale_method=refine["upscale_method"],
                           resample=refine["resample"], steps=refine["steps"], denoise=refine["denoise"])
        inputs.update(refine_model=refine["model"], upscale_model=refine["upscale_model"])
    return dict(fields, type="yogurt_h3_plan", version=4, fps=H3_FPS, width=width, height=height, refine=refine_view,
                segments=segments, target_frames=int(target_frames), audio=audio, continue_audio=bool(continue_audio),
                ref_image_size=ref_image_size, inputs=inputs)


def plan_view(plan):
    return {k: v for k, v in plan.items() if k != "inputs"}


def _require_plan(value):
    if not isinstance(value, dict) or value.get("type") != "yogurt_h3_plan":
        raise ValueError("计划必须来自 Yogurt H3 长视频分段导演台或故事分段规划节点。")
    if not value.get("segments"):
        raise ValueError("计划没有分段，请重新运行规划节点。")
    return value


def _carries(segment):
    return bool(segment["overlap_frames"]) and not segment.get("guide_reset")


def _carry(images, latent, refined):
    carry = dict(images=images[-1:].clone(), low=latent, latent=refined)
    carry["digest"] = digest([tensor_digest(item["samples"].unbind()[0]) if item else None for item in (latent, refined)])
    return carry


def _av(video, audio):
    return {"samples": NestedTensor((video, audio))}


def segment_event(owner, number, state, key=None):
    if hasattr(PromptServer, "instance"):
        PromptServer.instance.send_sync("yogurt_h3_segment", {"node": owner, "segment": number, "state": state, "key": key})


class YogurtH3FreshNoise(RandomNoise):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3FreshNoise", category=_INTERNAL_CATEGORY, is_dev_only=True,
                         inputs=[io.Int.Input("noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF)],
                         outputs=[io.Noise.Output()])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Disk checkpoints own reuse; a cache miss must not reuse an in-memory sampled window.
        return float("nan")


class YogurtH3RefineSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3RefineSettings", display_name="Yogurt H3 精修与放大", category=_CATEGORY,
            description="接到规划节点的 refine：首采解码后放大，重新编码并低去噪二采。画面比例跟随计划。",
            inputs=[io.Float.Input("megapixels", default=1.03, min=0.05, max=16.0, step=0.01,
                                   tooltip="二采输出的百万像素数，宽高按计划的画面比例换算并对齐到 32。"),
                    io.Combo.Input("upscale_method", options=["lanczos", "model", "nvidia_rtx_vsr"], default="lanczos"),
                    io.Boolean.Input("resample", default=True, tooltip="放大后重新编码并进行低去噪采样；关闭则只放大。"),
                    io.Int.Input("steps", default=8, min=1),
                    io.Float.Input("denoise", default=0.25, min=0, max=1, step=0.01),
                    io.Model.Input("model", optional=True, tooltip="可选另一套兼容 H3 模型；不连接则复用主模型。"),
                    io.UpscaleModel.Input("upscale_model", optional=True)],
            outputs=[RefineSettings.Output()])

    @classmethod
    def execute(cls, megapixels, upscale_method, resample, steps, denoise, model=None, upscale_model=None):
        if upscale_method == "model" and upscale_model is None:
            raise ValueError("model 放大方式需要连接 UpscaleModelLoader。")
        if upscale_method == "nvidia_rtx_vsr" and "ImageResizeKJv2" not in nodes.NODE_CLASS_MAPPINGS:
            raise ValueError("RTX VSR 需要 KJNodes 的 ImageResizeKJv2 节点和兼容的 nvidia-vfx 环境；也可选择 lanczos 或 model。")
        return io.NodeOutput(dict(megapixels=megapixels, upscale_method=upscale_method, resample=resample, steps=steps,
                                  denoise=denoise, model=model, upscale_model=upscale_model))


class YogurtH3EncodeSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3EncodeSegment", category=_INTERNAL_CATEGORY, is_dev_only=True,
            inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"), Plan.Input("plan"),
                    io.Int.Input("index"), io.Int.Input("width"), io.Int.Input("height"),
                    io.Image.Input("source_frames", optional=True), io.Audio.Input("source_audio", optional=True)],
            outputs=[io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, clip, vae, audio_vae, plan, index, width, height, source_frames=None, source_audio=None):
        segment = plan["segments"][index]
        inputs = plan["inputs"]
        length = segment["generation_frames"]
        if segment["mode"] == "r2v":
            refs = combined_references([inputs["images"][i] for i in segment["images"]], source_frames, source_audio,
                                       inputs["reference_media"])
            result = MiniMaxH3ReferenceToVideo.execute(clip=clip, vae=vae, audio_vae=audio_vae, prompt=segment["prompt"],
                                                       width=width, height=height, length=length,
                                                       ref_image_size=plan["ref_image_size"], **refs)
        else:
            frame = lambda name: inputs["images"][segment[name]] if name in segment else None
            result = MiniMaxH3ImageToVideo.execute(clip, vae, segment["prompt"], width, height, length,
                                                   first_frame=frame("first_frame"), last_frame=frame("last_frame"))
        return io.NodeOutput(*result.result)


class YogurtH3Continuation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3Continuation", category=_INTERNAL_CATEGORY, is_dev_only=True,
            inputs=[io.Conditioning.Input("positive"), io.Latent.Input("target_latent"), io.Model.Input("model"),
                    io.Sigmas.Input("sigmas"), io.Int.Input("overlap_frames"), io.Boolean.Input("continue_audio"),
                    io.Latent.Input("previous_latent"), io.Image.Input("previous_images"), io.Vae.Input("vae")],
            outputs=[io.Conditioning.Output(), io.Latent.Output(), io.Model.Output()])

    @classmethod
    def execute(cls, positive, target_latent, model, sigmas, overlap_frames, continue_audio, previous_latent, previous_images, vae):
        if overlap_frames == 1:
            # A touching seam anchors only the previous final frame.
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=target_latent, frame_idx=0,
                                                 vae=vae, image=previous_images[-1:]).result[0]
            return io.NodeOutput(positive, target_latent, model)
        latent, prefix_tokens = carry_latent_tail(target_latent, previous_latent, overlap_frames, continue_audio)
        return io.NodeOutput(positive, latent, install_drift_control(model, latent, sigmas, prefix_tokens))


class YogurtH3SaveSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3SaveSegment", category=_INTERNAL_CATEGORY, is_dev_only=True,
            inputs=[io.Image.Input("images"), io.String.Input("key"), io.String.Input("owner"), io.Int.Input("number"),
                    io.Int.Input("content_frames"), io.Boolean.Input("save"), io.Latent.Input("latent", optional=True),
                    io.Latent.Input("refined", optional=True), io.Audio.Input("audio", optional=True)],
            outputs=[io.Image.Output(), io.Audio.Output(), Carry.Output()])

    @classmethod
    def execute(cls, images, key, owner, number, content_frames, save, latent=None, refined=None, audio=None):
        # Commit only complete generation windows. Failures never replace a valid checkpoint.
        if len(images) < content_frames:
            raise ValueError(f"第 {number} 段只生成了 {len(images)} 帧，少于有效窗口 {content_frames} 帧。")
        images = images[:content_frames]
        if save:
            tensors = {"images": images}
            for name, item in (("low", latent), ("latent", refined)):
                if item is not None:
                    tensors[f"{name}_video"], tensors[f"{name}_audio"] = item["samples"].unbind()
            if audio is not None:
                tensors["audio"] = audio["waveform"]
                tensors["audio_rate"] = torch.tensor([int(audio["sample_rate"])])
            write_checkpoint(key, tensors)
        segment_event(owner, number, "complete", key if save else None)
        return io.NodeOutput(images, audio, _carry(images, latent, refined))


def _cached(checkpoint):
    latents = [_av(checkpoint[f"{name}_video"], checkpoint[f"{name}_audio"]) if f"{name}_video" in checkpoint else None
               for name in ("low", "latent")]
    audio = {"waveform": checkpoint["audio"], "sample_rate": int(checkpoint["audio_rate"][0])} if "audio" in checkpoint else None
    return checkpoint["images"], audio, _carry(checkpoint["images"], *latents)


class YogurtH3CachedSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3CachedSegment", category=_INTERNAL_CATEGORY, is_dev_only=True, enable_expand=True,
            inputs=[io.Model.Input("model"), io.Clip.Input("clip"), io.Vae.Input("vae"), io.Vae.Input("audio_vae"),
                    Plan.Input("plan"), io.Int.Input("index"), io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                    io.Int.Input("seed"), io.String.Input("base_key"), io.String.Input("cache_mode"), io.String.Input("owner"),
                    io.Boolean.Input("force_rerun"), Carry.Input("previous", optional=True),
                    io.Audio.Input("master_audio", optional=True)],
            outputs=[io.Image.Output(), io.Audio.Output(), Carry.Output()])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, plan, index, sampler, sigmas, seed, base_key, cache_mode, owner,
                force_rerun, previous=None, master_audio=None):
        segment = plan["segments"][index]
        number = index + 1
        carry = previous if previous is not None and _carries(segment) else None
        seed = (int(seed) + int(segment.get("seed_offset", 0))) % 2**64
        key = digest([base_key, segment, seed, carry["digest"] if carry else None])
        if cache_mode == "reuse" and not force_rerun:
            checkpoint = read_checkpoint(key)
            if checkpoint is not None:
                segment_event(owner, number, "cached", key)
                return io.NodeOutput(*_cached(checkpoint))
        segment_event(owner, number, "running")
        refine = plan["refine"]
        width, height = (refine["width"], refine["height"]) if refine else (plan["width"], plan["height"])
        graph = GraphBuilder()
        saved = dict(key=key, owner=owner, number=number, content_frames=segment["content_frames"], save=cache_mode != "off")
        extracted = None
        if segment.get("source"):
            extracted = graph.node("YogurtH3ExtractCharacterReplaceSegment", id="extract", video=plan["inputs"]["source"],
                start_frame=segment["start_frame"], content_frames=segment["content_frames"],
                generation_frames=segment["generation_frames"], master_audio=master_audio)
        if segment.get("bypass"):
            images = graph.node("ImageScale", image=extracted.out(0), upscale_method="lanczos", width=width, height=height,
                                crop="disabled").out(0)
            done = graph.node("YogurtH3SaveSegment", images=images, **saved)
            return io.NodeOutput(done.out(0), done.out(1), done.out(2), expand=graph.finalize())

        def encode(w, h):
            encoded = graph.node("YogurtH3EncodeSegment", clip=clip, vae=vae, audio_vae=audio_vae, plan=plan, index=index,
                                 width=w, height=h)
            if extracted is not None:
                encoded.set_input("source_frames", extracted.out(0))
                encoded.set_input("source_audio", extracted.out(1))
            return encoded.out(0), encoded.out(1)

        def sample(positive, latent, stage_model, stage_sigmas, previous_latent):
            if carry is not None:
                continued = graph.node("YogurtH3Continuation", positive=positive, target_latent=latent, model=stage_model,
                    sigmas=stage_sigmas, overlap_frames=segment["overlap_frames"], continue_audio=plan["continue_audio"],
                    previous_latent=previous_latent, previous_images=carry["images"], vae=vae)
                positive, latent, stage_model = continued.out(0), continued.out(1), continued.out(2)
            guider = graph.node("BasicGuider", model=stage_model, conditioning=positive)
            return graph.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0), sampler=sampler,
                              sigmas=stage_sigmas, latent_image=latent).out(0)

        noise = graph.node("YogurtH3FreshNoise", noise_seed=seed)
        low = sample(*encode(plan["width"], plan["height"]), model, sigmas, carry and carry["low"])
        images = graph.node("VAEDecode", samples=low, vae=vae).out(0)
        final = low
        refined = None
        if refine:
            if refine["upscale_method"] == "nvidia_rtx_vsr":
                images = graph.node("ImageResizeKJv2", image=images, width=width, height=height,
                    upscale_method="nvidia_rtx_vsr", keep_proportion="stretch", pad_color="0, 0, 0",
                    crop_position="center", divisible_by=32, device="gpu").out(0)
            if refine["upscale_method"] == "model":
                images = graph.node("ImageUpscaleWithModel", upscale_model=plan["inputs"]["upscale_model"], image=images).out(0)
            images = graph.node("ImageScale", image=images, upscale_method="lanczos", width=width, height=height,
                                crop="disabled").out(0)
            if refine["resample"] and refine["denoise"] > 0:
                refine_model = plan["inputs"]["refine_model"] or model
                schedule = graph.node("BasicScheduler", model=refine_model, scheduler="simple",
                                      steps=refine["steps"], denoise=refine["denoise"]).out(0)
                positive, _ = encode(width, height)
                video_latent = graph.node("VAEEncode", pixels=images, vae=vae)
                audio_latent = graph.node("LTXVSeparateAVLatent", av_latent=low)
                latent = graph.node("LTXVConcatAVLatent", video_latent=video_latent.out(0), audio_latent=audio_latent.out(1)).out(0)
                refined = final = sample(positive, latent, refine_model, schedule, carry and carry["latent"])
                images = graph.node("VAEDecode", samples=refined, vae=vae).out(0)
        done = graph.node("YogurtH3SaveSegment", images=images, latent=low, **saved)
        if refined is not None:
            done.set_input("refined", refined)
        if plan["audio"] == "generate":
            done.set_input("audio", graph.node("VAEDecodeAudio", samples=final, vae=audio_vae).out(0))
        return io.NodeOutput(done.out(0), done.out(1), done.out(2), expand=graph.finalize())


class YogurtH3JoinSegments(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3JoinSegments", category=_INTERNAL_CATEGORY, is_dev_only=True,
            inputs=[io.Image.Input("images"), io.Int.Input("overlap_frames"), io.Boolean.Input("soft_audio"),
                    io.Audio.Input("audio", optional=True), io.Image.Input("merged_images", optional=True),
                    io.Audio.Input("merged_audio", optional=True)],
            outputs=[io.Image.Output(), io.Audio.Output()])

    @classmethod
    def execute(cls, images, overlap_frames, soft_audio, audio=None, merged_images=None, merged_audio=None):
        if merged_images is None:
            return io.NodeOutput(images, audio)
        # The previous segment owns the repeated frames. With carried audio the
        # incoming segment owns the seam sound instead, so the soft release survives.
        if images.shape[0] <= overlap_frames:
            raise ValueError(f"本段只有 {images.shape[0]} 帧，无法去除 {overlap_frames} 帧重叠。")
        merged = torch.cat((merged_images, images[overlap_frames:]), dim=0)
        if audio is None:
            return io.NodeOutput(merged, None)
        rate = int(audio["sample_rate"])
        seam = round(overlap_frames * rate / H3_FPS)
        previous = merged_audio["waveform"][..., :merged_audio["waveform"].shape[-1] - seam] if soft_audio else merged_audio["waveform"]
        current = audio["waveform"] if soft_audio else audio["waveform"][..., seam:]
        return io.NodeOutput(merged, {"waveform": torch.cat((previous, current.to(previous)), dim=-1), "sample_rate": rate})


class YogurtH3LongVideoSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3LongVideoSampler",
            display_name="Yogurt H3 长视频采样",
            category=_CATEGORY,
            description="按计划逐段生成：上一段 AV latent 尾部直接续接到下一段开头，可选放大二采，去除重叠后合成一条视频。"
                        "角色替换计划恢复源音轨，故事计划使用生成音频。",
            enable_expand=True,
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                Plan.Input("plan", display_name="计划"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Combo.Input("cache_mode", options=["reuse", "refresh", "off"], default="reuse", optional=True),
                io.String.Input("rerun_segments", default="", optional=True, tooltip="从 1 开始，如 2,4-6；其他段读取匹配的缓存。"),
                io.String.Input("cache_revision", default="", optional=True, tooltip="手动变更自定义模型补丁后可修改此值以失效缓存。"),
            ],
            outputs=[
                io.Video.Output(display_name="输出视频"),
                io.Image.Output(display_name="输出帧"),
                io.Audio.Output(display_name="输出音频"),
                io.String.Output(display_name="运行报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, plan, sampler, sigmas, seed, cache_mode="reuse", rerun_segments="",
                cache_revision=""):
        plan = _require_plan(plan)
        hidden = cls.hidden
        owner = str(hidden.unique_id or "")
        if not hidden.prompt:
            cache_mode = "off"
        inputs = plan["inputs"]
        base_key = ""
        if cache_mode != "off":
            base_key = digest(["yogurt-h3-v4", upstream_signature(hidden.prompt, owner), tensor_digest(sigmas), seed,
                               plan_view(plan), [tensor_digest(image) for image in inputs["images"]],
                               reference_digest(inputs["reference_media"]),
                               video_digest(inputs["source"]) if inputs["source"] is not None else "", cache_revision])
        segments = plan["segments"]
        rerun = selected_segments(rerun_segments, len(segments))
        graph = GraphBuilder()
        master_audio = None
        if plan["audio"] == "source":
            master_audio = graph.node("YogurtH3OriginalAudio", id="original_audio", video=inputs["source"],
                                      target_frames=plan["target_frames"]).out(0)
        merged_images = merged_audio = previous = None
        for index, segment in enumerate(segments):
            number = index + 1
            current = graph.node("YogurtH3CachedSegment", id=f"segment_{number}", model=model, clip=clip, vae=vae,
                audio_vae=audio_vae, plan=plan, index=index, sampler=sampler, sigmas=sigmas, seed=seed, base_key=base_key,
                cache_mode=cache_mode, owner=owner, force_rerun=number in rerun)
            if previous is not None and _carries(segment):
                current.set_input("previous", previous)
            if master_audio is not None:
                current.set_input("master_audio", master_audio)
            soft_audio = plan["continue_audio"] and _carries(segment) and segment["overlap_frames"] >= 5
            joined = graph.node("YogurtH3JoinSegments", id=f"join_{number}", images=current.out(0),
                                overlap_frames=segment["overlap_frames"], soft_audio=soft_audio)
            if master_audio is None:
                joined.set_input("audio", current.out(1))
            if merged_images is not None:
                joined.set_input("merged_images", merged_images)
                if master_audio is None:
                    joined.set_input("merged_audio", merged_audio)
            merged_images, merged_audio = joined.out(0), joined.out(1)
            previous = current.out(2)

        trimmed = graph.node("YogurtH3TrimCharacterReplaceOutput", id="trim_exact_output", images=merged_images,
                             audio=master_audio if master_audio is not None else merged_audio,
                             target_frames=plan["target_frames"])
        video = graph.node("CreateVideo", id="create_output_video", images=trimmed.out(0), audio=trimmed.out(1), fps=float(H3_FPS))
        refine = plan["refine"]
        overlaps = [segment["overlap_frames"] for segment in segments]
        report = (f"{len(segments)} 段，{plan['width']}×{plan['height']}"
                  + (f" → {refine['width']}×{refine['height']}（{refine['upscale_method']}"
                     + (f"，二采 {refine['steps']} 步 denoise {refine['denoise']}）" if refine["resample"] and refine["denoise"] > 0 else "，仅放大）")
                     if refine else "")
                  + f"；段间重叠 {overlaps} 帧，上一段 AV latent 尾部直接续接下一段开头（1 帧时用末帧 AddGuide）；"
                  + f"生成 {sum(s['generation_frames'] for s in segments)} 帧，输出 {plan['target_frames']} 帧"
                  + f"（{plan['target_frames'] / H3_FPS:.2f}s @ {H3_FPS}fps）；"
                  + ("恢复源音轨。" if plan["audio"] == "source" else
                     "音频随 latent 续接并半余弦释放。" if plan["continue_audio"] else "音频各段独立生成。"))
        return io.NodeOutput(video.out(0), trimmed.out(0), trimmed.out(1), report, expand=graph.finalize())
