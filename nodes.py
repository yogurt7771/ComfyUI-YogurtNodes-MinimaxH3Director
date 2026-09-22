"""Finite long-video character replacement for MiniMax H3."""

from __future__ import annotations

import math
import json

import torch
from comfy_api.latest import io

from .planning import H3_FPS, canvas, timeline_segments, segment_frame_limit, normalize_overlap
from .media import original_audio, slice_audio
from .timeline import source_stamp, detect_cuts
from .references import ReferenceMedia
from .render import Plan, RefineSettings, make_plan, plan_view, _CATEGORY, _INTERNAL_CATEGORY


ROLE_ANALYSIS_SYSTEM_PROMPT = """You are a precise visual identity analyst for MiniMax H3 reference-video editing. Analyze only the supplied reference image. Do not identify or name a character, franchise, celebrity, actor, or adaptation. Do not use outside knowledge. Return exactly one compact English paragraph in plain text, with no heading, bullets, markdown, labels, or quotation marks. Describe only stable visible identity attributes that must remain consistent during video generation: rendering medium and stylization, apparent gender and age range, face shape and facial proportions, eyes, eyebrows, nose, mouth, skin tone, hairstyle and hair accessories, body proportions, garment construction and layers, colors, materials, patterns, footwear, and permanent accessories. If the image is a multi-view character sheet, merge the views into one consistent description. Ignore pose, camera, background, layout, annotations, text, weapons, handheld props, temporary lighting, and effects. Never invent hidden details and do not describe an action."""

TARGET_PERFORMER = "<Subject 2> is the original main performer in <Video 1>"

ROLE_ANALYSIS_USER_PROMPT = """Describe the single target character shown in this image for identity-preserving character replacement. The description will be inserted after: '<Subject 1> is the single character shown in <Picture 1>.'"""


def _resize_frames(frames: torch.Tensor, output_frames: int, source_fps: float, start_frame=0) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("源视频分段没有可解码的画面。")
    source_fps = float(source_fps)
    if source_fps <= 0:
        source_fps = H3_FPS
    source_start = math.floor(start_frame * source_fps / H3_FPS)
    indices = [min(len(frames) - 1, math.floor((start_frame + i) * source_fps / H3_FPS) - source_start)
               for i in range(output_frames)]
    return frames[indices].contiguous()


class YogurtH3RoleAnalysisInstructions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3RoleAnalysisInstructions",
            display_name="Yogurt H3 角色图识别指令",
            category=_CATEGORY,
            description="为 OpenRouter Image Understand 生成只分析角色参考图的指令。",
            inputs=[],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="推荐模型"),
            ],
        )

    @classmethod
    def execute(cls):
        return io.NodeOutput(
            ROLE_ANALYSIS_SYSTEM_PROMPT,
            ROLE_ANALYSIS_USER_PROMPT,
            "google/gemini-3.8-flash",
        )


class YogurtH3BuildCharacterReplacePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3BuildCharacterReplacePrompt",
            display_name="Yogurt H3 角色替换提示词",
            category=_CATEGORY,
            description="把角色图识别结果组装为 MiniMax H3 Ref2VA 六段式英文提示词。",
            inputs=[
                io.String.Input("identity_description", multiline=True, force_input=True),
                io.String.Input(
                    "extra_direction",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="可选的全局编辑要求，不要在这里描述角色身份。",
                ),
            ],
            outputs=[io.String.Output(display_name="H3 prompt")],
        )

    @classmethod
    def execute(cls, identity_description, extra_direction=""):
        identity = str(identity_description or "").strip()
        if not identity:
            raise ValueError("角色图识别结果为空。请连接 OpenRouter 输出，或手工填写角色外观描述。")
        direction = str(extra_direction or "").strip()
        direction_text = f" Additional user direction: {direction}" if direction else ""
        prompt = f"""subject_definitions:
<Subject 1> is the single target character in <Picture 1>, the only source of its visible identity: {identity}
{TARGET_PERFORMER}, whose exact motion, choreography, timing, gaze, screen position, scale, interactions, and occlusions are reused while the visible identity, body appearance, hair, and clothing are replaced by <Subject 1>.
<Video 1> is the source video for the target video edit and defines the complete timeline, framing, camera movement, lens perspective, scene, lighting, props, effects, and secondary people.
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

summary:
[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. Only <Subject 2> is replaced by <Subject 1>, whose complete appearance comes from <Picture 1>; the performance, camera, scene, every other person, and <Audio 1> are kept.{direction_text}

retention_analysis:
<Subject 1> (appears in [Shot 1]): attribute_transfer - every stable visible facial, hairstyle, body-proportion, clothing, color, material, pattern, footwear, accessory, and rendering-style attribute of <Picture 1> is transferred to the performer.
<Subject 2> (appears in [Shot 1]): partially_preserved - the exact motion, choreography, timing, poses, gaze, expressions, position, scale, interactions, and occlusions are retained while the original face, body appearance, hair, and clothing are replaced.
<Video 1> (source video edit): fully_preserved - duration, frame sequence, shot boundaries, camera motion, framing, perspective, background, lighting, props, effects, and every non-target person are retained.
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

detailed_description:
The target video keeps the visual style, color grading, lighting, and lens character of <Video 1>.
[Shot 1] The shot follows <Video 1> from its first frame to its last frame and replaces only <Subject 2>'s visible identity and outfit with <Subject 1>. Every step, body turn, jump, pose, hand gesture, head movement, gaze change, expression beat, screen position, scale change, occlusion, interaction, camera path, lens perspective, framing, cut, set, background, lighting direction, shadow, prop, effect, and secondary person stays at the exact source timing of <Video 1>. <Subject 1> matches the source perspective, illumination, contact shadows, floor contact, depth, and occlusion while keeping the complete identity and rendering style of <Picture 1>, without blending in the original performer's face or clothing. No other person is replaced, and no object, text, symbol, energy, or effect absent from <Video 1> is added.{direction_text}

overall_soundscape:
The ambience and physical sounds of <Audio 1> are copied unchanged throughout the target video.

non_diegetic_music:
Any audience-only score in <Audio 1> is copied unchanged; no new music is added."""
        return io.NodeOutput(prompt)


def timeline_plan(video, segment_seconds, megapixels, overlap_frames=22, scene_detection=False, scene_threshold=0.28,
                  timeline_json="{}"):
    """Source timing and window layout; shared by the planner and the pre-run timeline preview."""
    duration = float(video.get_duration())
    source_fps = float(video.get_frame_rate())
    target_frames = max(1, round(duration * H3_FPS))
    timeline = json.loads(timeline_json or "{}")
    stamp = source_stamp(video) if timeline.get("source_stamp") or scene_detection else ""
    if timeline.get("source_stamp") and timeline["source_stamp"] != stamp:
        raise ValueError("源视频已更换，时间线编辑不再匹配。请在导演台点击‘重建时间线’。")
    cuts = list(timeline.get("cuts", []))
    if scene_detection:
        cuts.extend(detect_cuts(video, scene_threshold))
    segments = timeline_segments(target_frames, int(segment_seconds), overlap_frames, cuts, timeline.get("segments", []))
    legal_limit = segment_frame_limit(int(segment_seconds))
    width, height = canvas(*video.get_dimensions(), megapixels)
    return dict(fps=H3_FPS, source_fps=source_fps, source_duration=duration, target_frames=target_frames,
                segment_seconds=int(segment_seconds), requested_segment_frames=int(segment_seconds) * H3_FPS,
                legal_segment_frames=legal_limit, requested_overlap_frames=int(overlap_frames),
                overlap_frames=normalize_overlap(overlap_frames, legal_limit), width=width, height=height,
                segments=segments, cuts=sorted(set(cuts)), source_stamp=stamp)


class YogurtH3CharacterReplacePlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3CharacterReplacePlanner",
            display_name="Yogurt H3 长视频分段导演台",
            category=_CATEGORY,
            description="按视频真实时长自动规划有限段，把每段映射到 H3 的 17k+5 帧网格，并打包角色图、提示词和参考素材给长视频采样。",
            inputs=[
                io.Video.Input("video"),
                io.Image.Input("role_image"),
                io.String.Input("prompt", multiline=True, force_input=True),
                io.Int.Input(
                    "segment_seconds",
                    display_name="每段秒数",
                    min=1,
                    max=150,
                    force_input=True,
                    tooltip="连接一个独立 Primitive Int；默认工作流使用 6 秒。",
                ),
                io.Float.Input("megapixels", default=0.41, min=0.05, max=16.0, step=0.01,
                               tooltip="首采画面的百万像素数；画面比例跟随源视频，宽高对齐到 32。0.41 在 16:9 下为 864×480。"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Int.Input("overlap_frames", display_name="重叠帧数", default=22, min=0, max=3591,
                             optional=True, tooltip="0 关闭；1 单帧引导；5/22/39/56… 用上一段 latent 尾部续接。自动向下对齐，必须小于单段长度。"),
                io.Boolean.Input("scene_detection", default=False, optional=True),
                io.Float.Input("scene_threshold", default=0.28, min=0, max=1, step=0.01, optional=True),
                io.String.Input("timeline_json", default="{}", multiline=True, optional=True,
                                tooltip="由时间线编辑器保存；cuts 为 24fps 绝对帧号，segments 按 start_frame 匹配。"),
                io.Autogrow.Input("additional_roles", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("role"), prefix="role_", min=0)),
                ReferenceMedia.Input("reference_media", optional=True),
                RefineSettings.Input("refine", optional=True),
            ],
            outputs=[
                Plan.Output(display_name="计划"),
                io.Int.Output(display_name="段数"),
                io.Int.Output(display_name="目标帧数"),
                io.String.Output(display_name="计划报告"),
            ],
        )

    @classmethod
    def execute(cls, video, role_image, prompt, segment_seconds, megapixels, ref_image_size="match", overlap_frames=22,
                scene_detection=False, scene_threshold=0.28, timeline_json="{}", additional_roles=None,
                reference_media=None, refine=None):
        timeline = timeline_plan(video, segment_seconds, megapixels, overlap_frames, scene_detection, scene_threshold, timeline_json)
        roles = [role_image] + list((additional_roles or {}).values())
        segments = []
        for segment in timeline.pop("segments"):
            role_index = int(segment.get("role_index", 0))
            if not 0 <= role_index < len(roles):
                raise ValueError(f"第 {segment['index'] + 1} 段角色索引 {role_index} 未连接参考图。")
            segment_prompt = str(segment.get("prompt") or prompt)
            target = str(segment.get("target_description", "")).strip()
            if target and TARGET_PERFORMER in segment_prompt:
                segment_prompt = segment_prompt.replace(
                    TARGET_PERFORMER, f"<Subject 2> is the person in <Video 1> matching this visible description: {target}")
            elif target:
                segment_prompt += f"\nOnly the person matching this visible description in <Video 1> is replaced: {target}. Every other person is preserved."
            segments.append(dict(segment, prompt=segment_prompt, mode="r2v", images=[role_index], source=True))
        fields = {k: v for k, v in timeline.items() if k not in ("fps", "width", "height", "target_frames")}
        plan = make_plan(video.get_dimensions(), megapixels, segments, timeline["target_frames"], "source", False,
                         ref_image_size, roles, reference_media, video, refine, **fields)
        lengths = [int(segment["content_frames"]) for segment in segments]
        generated = [int(segment["generation_frames"]) for segment in segments]
        report = (
            f"源视频 {timeline['source_duration']:.3f}s @ {timeline['source_fps']:.3f}fps；按 {H3_FPS}fps 输出 {plan['target_frames']} 帧，"
            f"首采 {plan['width']}×{plan['height']}。"
            f"请求每段 {int(segment_seconds)}s/{timeline['requested_segment_frames']} 帧，H3 合法整段为 "
            f"{timeline['legal_segment_frames']} 帧/{timeline['legal_segment_frames'] / H3_FPS:.3f}s。共 {len(segments)} 段；"
            f"重叠 {timeline['overlap_frames']} 帧；源窗口帧={lengths}；生成帧={generated}；"
            f"每段先去重叠和尾部补帧，最终 {plan['target_frames']} 帧。"
        )
        return io.NodeOutput(plan, len(segments), plan["target_frames"], report, ui={"yogurt_h3_plan": [plan_view(plan)]})


class YogurtH3ExtractCharacterReplaceSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3ExtractCharacterReplaceSegment",
            display_name="Yogurt H3 提取源视频分段（内部）",
            category=_INTERNAL_CATEGORY,
            is_dev_only=True,
            inputs=[
                io.Video.Input("video"),
                io.Int.Input("start_frame", force_input=True),
                io.Int.Input("content_frames", force_input=True),
                io.Int.Input("generation_frames", force_input=True),
                io.Audio.Input("master_audio", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="24fps padded frames"),
                io.Audio.Output(display_name="conditioning audio"),
                io.Audio.Output(display_name="original audio"),
            ],
        )

    @classmethod
    def execute(cls, video, start_frame, content_frames, generation_frames, master_audio=None):
        start_frame = int(start_frame)
        content_frames = int(content_frames)
        generation_frames = int(generation_frames)
        source_fps = float(video.get_frame_rate())
        decode_start = math.floor(start_frame * source_fps / H3_FPS) / source_fps
        decode_end = min(float(video.get_duration()), (start_frame + content_frames) / H3_FPS)
        segment = video.as_trimmed(
            decode_start,
            decode_end - decode_start,
            strict_duration=False,
        )
        if segment is None:
            raise ValueError(f"无法读取第 {start_frame} 帧开始的视频分段。")
        components = segment.get_components()
        frames = _resize_frames(components.images, content_frames, source_fps, start_frame)
        if generation_frames > content_frames:
            padding = frames[-1:].repeat(generation_frames - content_frames, 1, 1, 1)
            frames = torch.cat((frames, padding), dim=0)
        audio = master_audio if master_audio is not None else original_audio(video, round(video.get_duration() * H3_FPS))
        conditioning_audio = slice_audio(audio, start_frame, generation_frames)
        source_audio = slice_audio(audio, start_frame, content_frames)
        return io.NodeOutput(frames, conditioning_audio, source_audio)


class YogurtH3OriginalAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3OriginalAudio", category=_INTERNAL_CATEGORY, is_dev_only=True,
            inputs=[io.Video.Input("video"), io.Int.Input("target_frames", force_input=True)],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, video, target_frames):
        return io.NodeOutput(original_audio(video, target_frames))


class YogurtH3TrimCharacterReplaceOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3TrimCharacterReplaceOutput",
            display_name="Yogurt H3 精确裁剪输出（内部）",
            category=_INTERNAL_CATEGORY,
            is_dev_only=True,
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio"),
                io.Int.Input("target_frames", force_input=True),
            ],
            outputs=[io.Image.Output(), io.Audio.Output()],
        )

    @classmethod
    def execute(cls, images, audio, target_frames):
        target_frames = int(target_frames)
        if images.shape[0] < target_frames:
            raise ValueError(
                f"生成结果只有 {images.shape[0]} 帧，无法裁剪为目标 {target_frames} 帧。"
            )
        output_images = images[:target_frames].clone()
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        target_samples = round(target_frames / H3_FPS * sample_rate)
        output_audio = dict(audio)
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - int(waveform.shape[-1])))
        output_audio["waveform"] = waveform[..., :target_samples].clone()
        return io.NodeOutput(output_images, output_audio)
