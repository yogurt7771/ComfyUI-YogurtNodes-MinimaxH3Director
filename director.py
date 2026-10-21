"""Yogurt H3 导演台: one node from the visual timeline to a finished MiniMax H3 video."""

import json

import comfy.samplers
from comfy_api.latest import io

from .cache import digest
from .media import Media
from .project import DEFAULT_PROJECT, load_project, plan_windows
from .render import Models, Render, send_event
from .rewrite import Rewriter, rewrite_project


def _files(project):
    shot_media = []
    for shot in project["shots"]:
        refs = shot["refs"]
        shot_media += [k["media"] for k in shot["keyframes"]] + refs["images"] + refs["audios"]
        shot_media += [v["media"] for v in refs["videos"]] + [layer["mask"] for layer in shot["repaints"]]
        if shot["source"]:
            shot_media.append(shot["source"]["media"])
    return [m for m in shot_media + [project["soundtrack"]] if m is not None and "input" not in m]


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
                io.Model.Input("ref_model", optional=True,
                               tooltip="MiniMax H3 ref2va 模型：带参考素材或改写源视频的镜头使用它；不接则用 model。"),
                io.Model.Input("refine_model", optional=True, tooltip="二采与局部精修使用的模型；不接则沿用镜头的模型。"),
                Rewriter.Input("rewriter", optional=True, tooltip="Yogurt H3 改写模型：工程里打开“自动改写”后，按剧情和分段情况写每一镜的 H3 提示词。"),
                io.ModelPatch.Input("inpaint_patch", optional=True,
                                    tooltip="MiniMax H3 Fun ControlNet（ModelPatchLoader）：局部重绘时让遮罩区参考周围画面；不接则只用 latent 遮罩。"),
                io.UpscaleModel.Input("upscale_model", optional=True, tooltip="二采选择“放大模型”时使用。"),
                io.Sampler.Input("sampler", optional=True, tooltip="自定义采样器（如 Turbo Sampler），替换 sampler_name。"),
                io.Sigmas.Input("sigmas", optional=True, tooltip="首采噪声表，替换 steps/scheduler；局部重绘按强度截取尾部。"),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("image", tooltip="编辑器里可选作关键帧、参考图或参考视频（帧序列按 24fps）。"),
                    prefix="image_", min=0, max=16)),
                io.Autogrow.Input("masks", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Mask.Input("mask", tooltip="编辑器里可选作局部重绘遮罩；多帧遮罩逐帧对应。"),
                    prefix="mask_", min=0, max=16)),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, project, **kwargs):
        # Re-run when an uploaded file is replaced in place under the same name.
        try:
            files = _files(load_project(project))
        except ValueError:
            return project
        media = Media()
        return digest([project, [media.identity(m) for m in files]])

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, seed, steps, cfg, sampler_name, scheduler, filename_prefix, project,
                ref_model=None, refine_model=None, rewriter=None, inpaint_patch=None, upscale_model=None, sampler=None,
                sigmas=None, images=None, masks=None) -> io.NodeOutput:
        spec = load_project(project)
        media = Media(images, masks)
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
                                                       "size": [render.out_w, render.out_h]}]})
