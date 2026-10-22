"""Yogurt H3 导演台: one node from the visual timeline to a finished MiniMax H3 video."""

import json
import os

from PIL import Image

import comfy.samplers
import folder_paths
from comfy_api.latest import io

from . import latent_upscale
from .cache import digest
from .media import Media
from .project import DEFAULT_PROJECT, OVERLAPS, REFINE_UPSCALERS, SETTINGS, load_project, plan_windows
from .render import Models, Render, send_event
from .rewrite import Rewriter, rewrite_project


def _media(project):
    """Every media item the project references: uploads and connected sockets."""
    shot_media = []
    for shot in project["shots"]:
        refs = shot["refs"]
        shot_media += [k["media"] for k in shot["keyframes"]] + [m["media"] for m in refs["images"]] + refs["audios"]
        shot_media += [v["media"] for v in refs["videos"]] + [layer["mask"] for layer in shot["repaints"]]
        if shot["source"]:
            shot_media.append(shot["source"]["media"])
    return [m for m in shot_media + [project["soundtrack"]] if m is not None]


def _upscaler_inputs(prefix, name):
    """How a 二采 pass lifts its input to its canvas, and the latent upscaler weights it uses."""
    return [
        io.Combo.Input(f"{prefix}_upscaler", options=list(REFINE_UPSCALERS), default=SETTINGS[f"{prefix}_upscaler"],
                       tooltip=f"{name}放大画面的方式。\n"
                               "h3_latent：H3 latent 放大模型直接放大视频 latent，不经过 VAE（快；源视频局部重绘镜头仍走像素）。\n"
                               "lanczos：VAE 解码 → Lanczos 放大 → VAE 编码。\n"
                               "model：VAE 解码 → 接入的 upscale_model 放大 → VAE 编码。"),
        io.Combo.Input(f"{prefix}_latent_model", options=latent_upscale.choices(), default=SETTINGS[f"{prefix}_latent_model"],
                       tooltip=f"{name}用 h3_latent 时的放大权重（models/latent_upscale_models）。默认权重缺失时，首次使用会从 Hugging Face "
                               "下载约 690 MB（LBH-123-AI/Minimax_h3_latent_Upscaler，校验 SHA-256；可用 HF_ENDPOINT 换镜像）。"),
    ]


def _input_previews(project, media, node_id):
    """Small pictures of the connected image / video / mask sockets the project uses, saved to temp for the editor."""
    folder = os.path.join(folder_paths.get_temp_directory(), "yogurt_h3", "inputs")
    out = {}
    for item in _media(project):
        key = item.get("input")
        if key is None or key in out or key.startswith("audio_") or key not in media.inputs:
            continue
        if key.startswith("video_"):
            width, height = media.video_size(item)
            scale = min(1.0, 320 / max(width, height))
            frame = media.video(item, 0, 1, max(2, round(width * scale)), max(2, round(height * scale)), "disabled")
        elif key.startswith("mask_"):
            frame = media.mask(item)[:1, ..., None].expand(-1, -1, -1, 3)
        else:
            frame = media.image(item)
        picture = Image.fromarray((frame[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy())
        picture.thumbnail((320, 320))
        os.makedirs(folder, exist_ok=True)
        name = f"{node_id}_{key}.png"
        picture.save(os.path.join(folder, name))
        out[key] = {"filename": name, "subfolder": "yogurt_h3/inputs", "type": "temp"}
    return out


class YogurtH3Director(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3Director",
            display_name="Yogurt H3 导演台",
            category="Yogurt/MiniMax H3",
            description="可视化时间线导演台：文生、首尾帧、参考生成、源视频改写、局部重绘、二采放大与局部精修，直接输出带声音的成片。",
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 fl2va 模型：文生、首尾帧和局部重绘。"),
                io.Clip.Input("clip", tooltip="CLIPLoader，type 选 minimax（Qwen3-VL）。"),
                io.Vae.Input("vae", tooltip="MiniMax H3 视频 VAE。"),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 音频 VAE。"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="全片种子；每个镜头另有种子偏移。固定种子才能复用已生成的镜头。"),
                io.Int.Input("steps", default=25, min=1, max=200, tooltip="首采与局部重绘步数；接了 sigmas 时忽略。"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1, round=0.01, tooltip="H3 为蒸馏模型，通常保持 1。"),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple"),
                io.String.Input("filename_prefix", default="yogurt_h3/director"),
                io.String.Input("project", multiline=True, default=json.dumps(DEFAULT_PROJECT, ensure_ascii=False),
                                tooltip="导演台工程，由节点里的时间线编辑器写入。"),
                io.Int.Input("width", default=SETTINGS["width"], min=64, max=4096, step=32, tooltip="成片宽度（32 的倍数）；开二采时首采按各遍的放大倍数缩小。"),
                io.Int.Input("height", default=SETTINGS["height"], min=64, max=4096, step=32, tooltip="成片高度（32 的倍数）；开二采时首采按各遍的放大倍数缩小。"),
                io.Combo.Input("overlap", options=[str(v) for v in OVERLAPS], default=str(SETTINGS["overlap"]),
                               tooltip="续接重叠帧：勾选“接着上一镜”的镜头从上一镜最后这么多帧接着生成（22 帧约 0.92 秒）；0 关闭续接。"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default=SETTINGS["ref_image_size"],
                               tooltip="参考图尺寸：match 等比缩到成片的像素量（只缩小不放大）；max 短边最大 2048（更清晰，参考 token 更多，更慢）。"),
                io.String.Input("negative_prompt", default=SETTINGS["negative_prompt"], tooltip="负面提示词，cfg ≠ 1 时生效。"),
                io.Boolean.Input("two_pass", default=SETTINGS["two_pass"],
                                 tooltip="两段式二采：首采在小画幅上只跑调度表前面的步数，放大 two_pass_scale 倍后跑最后 two_pass_steps 步"
                                         "（总步数仍是 steps，快）。可与低噪重绘同时开，两段式在前。"),
                io.Float.Input("two_pass_scale", default=SETTINGS["two_pass_scale"], min=1.0, max=4.0, step=0.125,
                               tooltip="两段式的放大倍数：首采画幅 = 两段式画幅 ÷ two_pass_scale。两段式画幅是成片尺寸，同开低噪重绘时再除以 rerender_scale。"),
                io.Int.Input("two_pass_steps", default=SETTINGS["two_pass_steps"], min=1, max=200,
                             tooltip="调度表最后几步在两段式画幅上跑（例如 steps 8、这里 2 = 首采 6 步 + 放大后 2 步）。"),
                *_upscaler_inputs("two_pass", "两段式"),
                io.Boolean.Input("rerender", default=SETTINGS["rerender"],
                                 tooltip="低噪重绘：放大 rerender_scale 倍到成片尺寸，从 rerender_denoise 的噪声再采 rerender_steps 步，音轨锁定。\n"
                                         "单开时接在完整跑完的首采后；和两段式同开时接在两段式之后。"),
                io.Float.Input("rerender_scale", default=SETTINGS["rerender_scale"], min=1.0, max=4.0, step=0.125,
                               tooltip="低噪重绘的放大倍数：它的输入画幅 = width × height ÷ rerender_scale；1 = 不放大，在成片尺寸上重采。"),
                io.Int.Input("rerender_steps", default=SETTINGS["rerender_steps"], min=1, max=200, tooltip="低噪重绘的步数。"),
                io.Float.Input("rerender_denoise", default=SETTINGS["rerender_denoise"], min=0.05, max=1.0, step=0.05,
                               tooltip="低噪重绘从多大的噪声开始（1 = 纯噪声），常用 0.25–0.4。"),
                *_upscaler_inputs("rerender", "低噪重绘"),
                io.Boolean.Input("detail", default=SETTINGS["detail"],
                                 tooltip="局部精修总开关：关掉时跳过所有精修区域。区域在导演台“局部精修”页框选。"),
                io.Int.Input("detail_steps", default=SETTINGS["detail_steps"], min=1, max=200, tooltip="局部精修步数。"),
                io.Model.Input("ref_model", optional=True,
                               tooltip="MiniMax H3 ref2va 模型：带参考素材或改写源视频的镜头使用它；不接则用 model。"),
                io.Model.Input("refine_model", optional=True, tooltip="二采与局部精修使用的模型；不接则沿用镜头的模型。"),
                Rewriter.Input("rewriter", optional=True, tooltip="Yogurt H3 改写模型：工程里打开“自动改写”后，按剧情和分段情况写每一镜的 H3 提示词。"),
                io.ModelPatch.Input("inpaint_patch", optional=True,
                                    tooltip="MiniMax H3 Fun ControlNet（ModelPatchLoader）：局部重绘时让遮罩区参考周围画面；不接则只用 latent 遮罩。"),
                io.UpscaleModel.Input("upscale_model", optional=True, tooltip="两段式或低噪重绘的放大方式选 model 时使用。"),
                io.Sampler.Input("sampler", optional=True, tooltip="自定义采样器（如 Turbo Sampler），替换 sampler_name。"),
                io.Sigmas.Input("sigmas", optional=True, tooltip="首采噪声表，替换 steps/scheduler；局部重绘按强度截取尾部。"),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("image", tooltip="编辑器里可选作关键帧、参考图或参考视频（帧序列按 24fps）。"),
                    prefix="image_", min=0, max=16)),
                io.Autogrow.Input("masks", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Mask.Input("mask", tooltip="编辑器里可选作局部重绘遮罩；多帧遮罩逐帧对应。"),
                    prefix="mask_", min=0, max=16)),
                io.Autogrow.Input("videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Video.Input("video", tooltip="编辑器里可选作源视频、参考视频，或取其原声作参考音频 / 配乐。"),
                    prefix="video_", min=0, max=16)),
                io.Autogrow.Input("audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("audio", tooltip="编辑器里可选作参考音频或配乐音轨。"),
                    prefix="audio_", min=0, max=16)),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, project, **kwargs):
        # Re-run when an uploaded file is replaced in place under the same name.
        try:
            files = [m for m in _media(load_project(project)) if "input" not in m]
        except ValueError:
            return project
        media = Media()
        return digest([project, [media.identity(m) for m in files]])

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, seed, steps, cfg, sampler_name, scheduler, filename_prefix, project,
                ref_model=None, refine_model=None, rewriter=None, inpaint_patch=None, upscale_model=None, sampler=None,
                sigmas=None, images=None, masks=None, videos=None, audios=None, **settings) -> io.NodeOutput:
        spec = load_project(project, settings)
        media = Media(images, masks, videos, audios)
        prompts, report = {}, []
        if spec["rewrite"]["enabled"]:
            if rewriter is None:
                raise ValueError("工程打开了自动改写提示词，请把“Yogurt H3 改写模型”连接到导演台的 rewriter。")
            node_id = cls.hidden.unique_id
            windows = plan_windows(spec)
            spec, prompts, report = rewrite_project(spec, rewriter, media,
                                                    lambda index, stage, prompt=None: send_event(node_id, windows[index].shot["id"], stage, prompt))
        models = Models(model, clip, vae, audio_vae, ref_model, refine_model, inpaint_patch, upscale_model,
                        sampler or comfy.samplers.sampler_object(sampler_name), sigmas, seed, steps, cfg, scheduler)
        render = Render(cls.hidden.unique_id, cls.hidden.prompt, spec, models, media,
                        [seed, steps, cfg, sampler_name, scheduler], report)
        video, saved = render.run(filename_prefix)
        timeline = [{"id": w.shot["id"], "start": w.start, "frames": w.keep, "head": w.head, "gen": w.gen} for w in render.windows]
        return io.NodeOutput(video, ui={"images": [saved], "animated": (True,),
                                        "yogurt_h3": [{"timeline": timeline, "report": render.report, "video": saved, "prompts": prompts,
                                                       "inputs": _input_previews(spec, media, cls.hidden.unique_id),
                                                       "size": [render.out_w, render.out_h]}]})
