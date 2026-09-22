"""Yogurt MiniMax H3 long-video nodes: character replacement and multi-segment stories on one sampler."""

from .nodes import (
    YogurtH3BuildCharacterReplacePrompt,
    YogurtH3CharacterReplacePlanner,
    YogurtH3ExtractCharacterReplaceSegment,
    YogurtH3RoleAnalysisInstructions,
    YogurtH3TrimCharacterReplaceOutput,
    YogurtH3OriginalAudio,
)
from .render import (
    YogurtH3CachedSegment,
    YogurtH3Continuation,
    YogurtH3EncodeSegment,
    YogurtH3FreshNoise,
    YogurtH3JoinSegments,
    YogurtH3LongVideoSampler,
    YogurtH3RefineSettings,
    YogurtH3SaveSegment,
)
from .rewriter import YogurtH3PromptRewriter
from .timeline import register_routes
from .references import YogurtH3ReferenceMedia, YogurtH3DirectorConditioning
from .story import NODES as STORY_NODES

WEB_DIRECTORY = "./web"
register_routes()


NODE_CLASS_MAPPINGS = {
    "YogurtH3FreshNoise": YogurtH3FreshNoise,
    "YogurtH3ReferenceMedia": YogurtH3ReferenceMedia,
    "YogurtH3DirectorConditioning": YogurtH3DirectorConditioning,
    "YogurtH3LongVideoSampler": YogurtH3LongVideoSampler,
    "YogurtH3CachedSegment": YogurtH3CachedSegment,
    "YogurtH3EncodeSegment": YogurtH3EncodeSegment,
    "YogurtH3Continuation": YogurtH3Continuation,
    "YogurtH3SaveSegment": YogurtH3SaveSegment,
    "YogurtH3JoinSegments": YogurtH3JoinSegments,
    "YogurtH3RefineSettings": YogurtH3RefineSettings,
    "YogurtH3PromptRewriter": YogurtH3PromptRewriter,
    "YogurtH3RoleAnalysisInstructions": YogurtH3RoleAnalysisInstructions,
    "YogurtH3BuildCharacterReplacePrompt": YogurtH3BuildCharacterReplacePrompt,
    "YogurtH3CharacterReplacePlanner": YogurtH3CharacterReplacePlanner,
    "YogurtH3ExtractCharacterReplaceSegment": YogurtH3ExtractCharacterReplaceSegment,
    "YogurtH3TrimCharacterReplaceOutput": YogurtH3TrimCharacterReplaceOutput,
    "YogurtH3OriginalAudio": YogurtH3OriginalAudio,
}
NODE_CLASS_MAPPINGS.update({node.GET_SCHEMA().node_id: node for node in STORY_NODES})

NODE_DISPLAY_NAME_MAPPINGS = {
    "YogurtH3ReferenceMedia": "Yogurt H3 多模态参考组",
    "YogurtH3DirectorConditioning": "Yogurt H3 通用导演条件",
    "YogurtH3LongVideoSampler": "Yogurt H3 长视频采样",
    "YogurtH3RefineSettings": "Yogurt H3 精修与放大",
    "YogurtH3PromptRewriter": "Yogurt H3 提示词改写",
    "YogurtH3RoleAnalysisInstructions": "Yogurt H3 角色图识别指令",
    "YogurtH3BuildCharacterReplacePrompt": "Yogurt H3 角色替换提示词",
    "YogurtH3CharacterReplacePlanner": "Yogurt H3 长视频分段导演台",
    "YogurtH3ExtractCharacterReplaceSegment": "Yogurt H3 提取源视频分段（内部）",
    "YogurtH3TrimCharacterReplaceOutput": "Yogurt H3 精确裁剪输出（内部）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
