"""Character cards + story -> segment prompts -> a long-video plan for the shared H3 sampler."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from .planning import normalize_overlap
from .references import ReferenceMedia
from .render import Plan, RefineSettings, make_plan
from .rewriter import constrained_json, textgen_inputs
from .story_planning import (
    ASPECT_RATIOS,
    MAX_CARDS,
    MAX_SEGMENTS,
    MODES,
    build_cards,
    build_plan,
    build_rewrite_request,
    extract_json,
    join_segments,
    normalize_llm_plan,
    prompt_fields,
    STANDALONE_FIELDS,
    apply_standalone,
    standalone_request,
)

_CATEGORY = "Yogurt/MiniMax H3 Director/Story"


def _sampler_plan(story, cards, reference_media, refine, megapixels, ref_image_size, continue_audio):
    """Map a story prompt plan onto the shared sampler contract."""
    mode = story["mode"]
    needed = {"I2VA": 1, "L2VA": 1, "FL2VA": 2}.get(mode, 0)
    if len(cards) < needed:
        raise ValueError(f"{mode} 模式需要至少 {needed} 张角色卡图片。")
    length = int(story["length_frames"])
    overlap = normalize_overlap(int(story["overlap_frames"]), length)
    items = story["segments"][:MAX_SEGMENTS]
    first = {"I2VA": 0, "FL2VA": 0}.get(mode)
    last = {"L2VA": 0, "FL2VA": 1}.get(mode)
    segments = []
    for index, item in enumerate(items):
        segment = dict(index=index, prompt=item["prompt"],
                       content_frames=length, generation_frames=length, overlap_frames=overlap if index else 0,
                       mode="r2v" if mode == "Ref2VA" else "i2v", images=list(range(len(cards))) if mode == "Ref2VA" else [])
        if index == 0 and first is not None:
            segment["first_frame"] = first
        if index == len(items) - 1 and last is not None:
            segment["last_frame"] = last
        segments.append(segment)
    aspect = tuple(map(int, story["aspect_ratio"].split(":")))
    plan = make_plan(aspect, megapixels, segments, len(items) * length - (len(items) - 1) * overlap, "generate",
                     continue_audio, ref_image_size, cards, reference_media if mode == "Ref2VA" else None,
                     refine=refine, story=story)
    plan["inputs"]["story"] = dict(cards=cards, reference_media=reference_media, refine=refine, megapixels=megapixels,
                                   ref_image_size=ref_image_size, continue_audio=continue_audio)
    return plan


def _outputs(plan):
    story = plan["story"]
    return (plan, join_segments(story), len(plan["segments"]), int(story["length_frames"]),
            float(story["segment_duration_seconds"]))


class YogurtH3AutoStoryPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        cards = []
        for i in range(1, MAX_CARDS + 1):
            cards += [io.String.Input(f"card_{i}_name", default="", optional=True),
                      io.String.Input(f"card_{i}_notes", default="", multiline=True, optional=True)]
        return io.Schema(
            node_id="YogurtH3AutoStoryPlan",
            display_name="Yogurt H3 故事分段规划",
            category=_CATEGORY,
            description="角色卡 + 剧情 + 段数 → 长视频计划、分段提示词和给 LLM 的改写请求。不加载任何模型。",
            inputs=[
                io.String.Input("story", multiline=True, default="两名角色在雨夜巷口相遇，短暂对峙后一起走进灯光里。"),
                io.Combo.Input("mode", options=MODES, default="Ref2VA"),
                io.Int.Input("segment_count", default=2, min=1, max=MAX_SEGMENTS),
                io.Float.Input("seconds_per_segment", default=8.0, min=4.0, max=15.0, step=0.1),
                io.Int.Input("overlap_frames", default=39, min=0, max=96,
                             tooltip="段间重叠帧，向下对齐到 1 或 5/22/39/56/73/90。"),
                io.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, default="16:9",
                               tooltip="画面比例：写入 LLM 改写请求，并与 megapixels 一起决定输出分辨率。"),
                io.Float.Input("megapixels", default=1.03, min=0.05, max=16.0, step=0.01,
                               tooltip="首采画面的百万像素数；宽高按画面比例换算并对齐到 32。1.03 在 16:9 下为 1344×768。"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Boolean.Input("continue_audio", default=True, tooltip="重叠区音频 latent 也跨段续接，接缝处用半余弦释放。"),
                io.String.Input("global_soundscape", multiline=True, default=""),
                io.String.Input("global_music", multiline=True, default=""),
                *cards,
                io.Autogrow.Input("cards", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("card", tooltip="角色卡图片，顺序对应 <Picture 1>、<Picture 2>…"),
                    prefix="card_", min=0, max=MAX_CARDS)),
                ReferenceMedia.Input("reference_media", optional=True, tooltip="仅 Ref2VA 使用，排在角色卡之后。"),
                RefineSettings.Input("refine", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="llm_prompt"),
                Plan.Output(display_name="plan"),
                io.String.Output(display_name="all_segment_prompts"),
                io.Int.Output(display_name="segment_count"),
                io.Int.Output(display_name="length_frames"),
                io.Float.Output(display_name="seconds_per_segment"),
            ],
        )

    @classmethod
    def execute(cls, story, mode, segment_count, seconds_per_segment, overlap_frames, aspect_ratio, megapixels,
                ref_image_size, continue_audio, global_soundscape, global_music, cards=None,
                reference_media=None, refine=None, **kwargs):
        names = build_cards([(kwargs.get(f"card_{i}_name", ""), kwargs.get(f"card_{i}_notes", "")) for i in range(1, MAX_CARDS + 1)])
        story_plan = build_plan(story, segment_count, seconds_per_segment, overlap_frames, mode, aspect_ratio,
                                global_soundscape, global_music, names)
        request = build_rewrite_request(story_plan, story, mode, aspect_ratio, global_soundscape, global_music, names)
        story_plan["request"] = request
        cards = cards or {}
        images = [cards[name] for name in sorted(cards, key=lambda name: int(name.rsplit("_", 1)[1])) if cards[name] is not None]
        plan = _sampler_plan(story_plan, images, reference_media, refine, megapixels, ref_image_size, continue_audio)
        return io.NodeOutput(request, *_outputs(plan))


class YogurtH3AutoStoryParse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3AutoStoryParse",
            display_name="Yogurt H3 解析 LLM 分段",
            category=_CATEGORY,
            description="用 LLM 返回的分段 JSON 替换计划中的提示词（段数与时长不变）；解析失败时沿用模板规划。",
            inputs=[
                Plan.Input("plan"),
                io.String.Input("llm_text", multiline=True, default="", optional=True),
            ],
            outputs=[
                Plan.Output(display_name="plan"),
                io.String.Output(display_name="all_segment_prompts"),
                io.Int.Output(display_name="segment_count"),
                io.Int.Output(display_name="length_frames"),
                io.Float.Output(display_name="seconds_per_segment"),
                io.String.Output(display_name="parse_report"),
            ],
        )

    @classmethod
    def execute(cls, plan, llm_text=""):
        if "story" not in plan:
            raise ValueError("plan 必须来自 Yogurt H3 故事分段规划。")
        parsed = extract_json(llm_text)
        if parsed:
            story, notes = normalize_llm_plan(parsed, plan["story"])
            plan = _sampler_plan(story, **plan["inputs"]["story"])
            report = f"used_llm=true segments={len(plan['segments'])}" + "".join(f"; {note}" for note in notes)
        else:
            report = "used_llm=false fallback_template=true"
            if llm_text.strip():
                report += " (LLM text present but JSON parse failed)"
        return io.NodeOutput(*_outputs(plan), report)


class YogurtH3StoryRewrite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3StoryRewrite",
            display_name="Yogurt H3 故事分段改写",
            category=_CATEGORY,
            description="用本地文本模型（Qwen 系列）按官方指南改写计划中每段的提示词；JSON 约束解码保证每段每个段落都生成，"
                        "段数与时长不变。旁路本节点则直接使用模板提示词。",
            inputs=[io.Clip.Input("clip"), Plan.Input("plan"),
                    io.Image.Input("image", optional=True, tooltip="可接“角色卡叠图”，让模型看到角色外观。"),
                    io.Int.Input("max_length", default=8192, min=64, max=32768),
                    *textgen_inputs("sampling_mode"),
                    io.Boolean.Input("standalone_check", default=True,
                                     tooltip="改写后让模型逐段检查：找出只有看过其他段才讲得通的说法，并改成直接描述。")],
            outputs=[
                Plan.Output(display_name="plan"),
                io.String.Output(display_name="all_segment_prompts"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, clip, plan, max_length, sampling_mode, standalone_check=True, image=None):
        if "story" not in plan:
            raise ValueError("plan 必须来自 Yogurt H3 故事分段规划。")
        story = plan["story"]
        fields = prompt_fields(story["mode"])
        count = len(story["segments"])
        values, closed = constrained_json(clip, story["request"], [fields] * count, max_length, sampling_mode,
                                          array_key="segments", image=image)
        rewritten = {"segments": [{name: values.get((i, name), "") for name in fields} for i in range(count)]}
        story, notes = normalize_llm_plan(rewritten, story)
        if standalone_check:
            segments = []
            for i, segment in enumerate(story["segments"]):
                found, _ = constrained_json(clip, standalone_request(segment["prompt"]), [STANDALONE_FIELDS], 1024,
                                            {"sampling_mode": "off"})
                prompt, changed = apply_standalone(segment["prompt"], found.get((0, "phrases"), ""),
                                                   found.get((0, "replacements"), ""))
                if changed:
                    notes.append(f"segment {i + 1} made standalone: {', '.join(changed)}")
                segments.append(dict(segment, prompt=prompt))
            story = dict(story, segments=segments)
        plan = _sampler_plan(story, **plan["inputs"]["story"])
        if not closed:
            notes.append(f"max_length={max_length} 用完，后面的段落未写完；请调大 max_length")
        return io.NodeOutput(plan, join_segments(story), "; ".join(notes) or "ok")


class YogurtH3AutoStoryPick(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3AutoStoryPick",
            display_name="Yogurt H3 选取分段提示词",
            category=_CATEGORY,
            description="取出计划中某一段的完整 H3 提示词，用于预览或单段生成。",
            inputs=[
                Plan.Input("plan"),
                io.Int.Input("segment_index", default=1, min=1, max=MAX_SEGMENTS),
            ],
            outputs=[
                io.String.Output(display_name="segment_prompt"),
                io.Int.Output(display_name="segment_index"),
                io.Int.Output(display_name="segment_count"),
            ],
        )

    @classmethod
    def execute(cls, plan, segment_index):
        segments = plan["segments"]
        idx = max(1, min(int(segment_index), len(segments)))
        return io.NodeOutput(segments[idx - 1]["prompt"], idx, len(segments))


class YogurtH3AutoStoryStackImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3AutoStoryStackImages",
            display_name="Yogurt H3 角色卡叠图",
            category=_CATEGORY,
            description="把角色卡图片叠成一个 batch，给 Generate Text 看图改写。",
            inputs=[io.Image.Input(f"image_{i}", optional=True) for i in range(1, MAX_CARDS + 1)],
            outputs=[io.Image.Output(display_name="images"), io.Int.Output(display_name="count")],
        )

    @classmethod
    def execute(cls, **kwargs):
        frames = [kwargs[f"image_{i}"] for i in range(1, MAX_CARDS + 1) if kwargs.get(f"image_{i}") is not None]
        if not frames:
            return io.NodeOutput(torch.zeros((1, 64, 64, 3)), 0)
        h = min(t.shape[1] for t in frames)
        w = min(t.shape[2] for t in frames)
        return io.NodeOutput(torch.cat([t[:1, :h, :w, :3] for t in frames], dim=0), len(frames))


NODES = [
    YogurtH3AutoStoryPlan,
    YogurtH3AutoStoryParse,
    YogurtH3StoryRewrite,
    YogurtH3AutoStoryPick,
    YogurtH3AutoStoryStackImages,
]
