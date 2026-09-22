"""Native H3 conditioning modes and reusable reference-media groups."""

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

from .planning import align_h3_frames
from .cache import digest, tensor_digest

ReferenceMedia = io.Custom("YOGURT_H3_REFERENCE_MEDIA")
_CATEGORY = "Yogurt/MiniMax H3 Director"


def reference_digest(media):
    if not media:
        return ""
    result = {}
    for kind, values in media.items():
        result[kind] = []
        for name, value in values.items():
            if kind in ("audios", "video_audios"):
                result[kind].append([name, value["sample_rate"], tensor_digest(value["waveform"])])
            else:
                result[kind].append([name, tensor_digest(value)])
    return digest(result)


def combined_references(images=(), source_frames=None, source_audio=None, media=None):
    media = media or {}
    images = list(images) + list(media.get("images", {}).values())
    videos = ([source_frames] if source_frames is not None else []) + list(media.get("videos", {}).values())
    paired = dict(media.get("video_audios", {}))
    # Pair audio by the numbered socket, not by the count of connected audio sockets.
    video_audio = {"ref_video_audio_0": source_audio} if source_audio is not None else {}
    offset = int(source_frames is not None)
    video_names = list(media.get("videos", {}))
    for name, audio in paired.items():
        number = name.removeprefix("video_audio_")
        video_name = "video_" + number
        if video_name not in video_names:
            raise ValueError(f"配对音轨 {name} 缺少对应的 {video_name} 视频。")
        video_audio[f"ref_video_audio_{offset+video_names.index(video_name)}"] = audio
    return dict(ref_images={f"ref_image_{i}": value for i, value in enumerate(images)},
                ref_videos={f"ref_video_{i}": value for i, value in enumerate(videos)},
                ref_video_audios=video_audio,
                ref_audios={f"ref_audio_{i}": value for i, value in enumerate(media.get("audios", {}).values())})


class YogurtH3ReferenceMedia(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3ReferenceMedia", display_name="Yogurt H3 多模态参考组", category=_CATEGORY,
            description="附加图片、24fps 视频帧及音轨。角色替换中 Picture1/Video1 保留给主角色和当前源片；附加参考从 2 开始。",
            inputs=[io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"), prefix="image_", min=0)),
                    io.Autogrow.Input("videos", optional=True, template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("video", tooltip="通过 GetVideoComponents 获取；预先转换为 24fps。"), prefix="video_", min=0)),
                    io.Autogrow.Input("video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("video_audio"), prefix="video_audio_", min=0)),
                    io.Autogrow.Input("audios", optional=True, template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"), prefix="audio_", min=0))],
            outputs=[ReferenceMedia.Output()])

    @classmethod
    def execute(cls, images=None, videos=None, video_audios=None, audios=None):
        return io.NodeOutput(dict(images=images or {}, videos=videos or {}, video_audios=video_audios or {}, audios=audios or {}))


class YogurtH3DirectorConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="YogurtH3DirectorConditioning", display_name="Yogurt H3 通用导演条件", category=_CATEGORY,
            description="原生 T2V、I2V、首尾帧或多模态 R2V 条件；连接标准 BasicGuider 和 SamplerCustomAdvanced。",
            inputs=[io.Clip.Input("clip"), io.Vae.Input("vae"),
                    io.Combo.Input("mode", options=["t2v", "i2v", "first_last", "r2v"], default="r2v"),
                    io.String.Input("prompt", multiline=True), io.Int.Input("duration_seconds", default=6, min=1, max=150),
                    io.Int.Input("width", default=864, min=32, step=32), io.Int.Input("height", default=480, min=32, step=32),
                    io.Image.Input("first_frame", optional=True), io.Image.Input("last_frame", optional=True),
                    io.Vae.Input("audio_vae", optional=True), ReferenceMedia.Input("reference_media", optional=True)],
            outputs=[io.Conditioning.Output(), io.Latent.Output(), io.Int.Output(display_name="生成帧数")])

    @classmethod
    def execute(cls, clip, vae, mode, prompt, duration_seconds, width, height,
                first_frame=None, last_frame=None, audio_vae=None, reference_media=None):
        length = align_h3_frames(duration_seconds*24)
        if mode == "r2v":
            result = MiniMaxH3ReferenceToVideo.execute(clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
                width=width, height=height, length=length, **combined_references(media=reference_media))
        else:
            if mode in ("i2v", "first_last") and first_frame is None:
                raise ValueError("I2V/首尾帧模式需要连接 first_frame。")
            if mode == "first_last" and last_frame is None:
                raise ValueError("首尾帧模式需要连接 last_frame。")
            result = MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length,
                first_frame=first_frame if mode != "t2v" else None,
                last_frame=last_frame if mode == "first_last" else None)
        return io.NodeOutput(result[0], result[1], length)
