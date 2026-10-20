"""Story -> per-segment H3 prompts, written by a deepagents agent with MiniMax's h3-prompt-writing skill.

The agent receives the director timeline as it will be generated (segment lengths, continuations,
keyframe positions, reference labels, sound) plus the global prompt, and returns one complete H3 prompt per
segment. Prompts are cached per segment: a segment is rewritten only when its own inputs, the global prompt or,
for a continued segment, the previous segment's prompt change; the other segments stay as they are.
"""

import base64
import copy
from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

import comfy.model_management
from comfy_api.latest import io

from . import cache
from .cache import digest
from .conditioning import video_size
from .project import FPS, plan_windows

VERSION = "rewrite-1"
SKILL = Path(__file__).resolve().parent / "h3-prompt-writing"
PROVIDERS = ("openai", "anthropic", "ollama")
Rewriter = io.Custom("YOGURT_H3_REWRITER")

SYSTEM_PROMPT = """You are the screenwriter of a film made of several segments generated one by one with MiniMax H3. \
Each segment is a separate H3 generation of the length given; the film is the segments played in order.

Use the `h3-prompt-writing` skill. Read its SKILL.md, then the guide for each segment's mode \
(references/base-en.txt for T2VA, I2VA, FL2VA and L2VA; references/ref-en.txt for Ref2VA), and write every \
prompt in exactly the final structure that guide specifies.

The global prompt describes the whole film: story, characters, style, setting. It is background, not text \
to copy into a segment. Plan before writing: give each segment one beat of it in order, sized to the segment \
length, so the film tells the story once without repeating or skipping events. A segment prompt directs only \
that segment and takes priority there. A segment marked as not using the global prompt is written from its own \
segment prompt and timeline facts alone. Segments marked fixed are already written; keep the film consistent \
with them.

H3 never sees the other segments' prompts. Restate the visual style, every character's appearance and \
clothing, the setting and the lighting in each prompt, worded consistently across segments.

Each prompt's timeline covers exactly its segment length. Inside a segment, [Shot N] marks camera cuts. \
A continued segment opens inside the previous segment's last moment (same picture, camera and sound): \
describe that state first without a cut, then continue the action.

Use exactly the labels listed for a segment (<Picture N>, <Video N>, <Audio N>) and no others. Keep dialogue \
in the language the user wrote it in, using the guide's dialogue markup.

When the prompts are finished, call `submit_prompts` with one entry per requested segment id."""


class SegmentPrompt(BaseModel):
    id: str = Field(description="Segment id exactly as given in the task")
    prompt: str = Field(description="The complete H3 prompt for this segment, in the guide's final structure")


def rewriter_config(provider, model, base_url="", api_key="", temperature=0.7, max_tokens=8192, vision=True):
    if provider not in PROVIDERS:
        raise ValueError(f"改写模型 provider 只能是 {'/'.join(PROVIDERS)}。")
    if not str(model).strip():
        raise ValueError("改写模型需要填写模型名称。")
    return {"provider": provider, "model": str(model).strip(), "base_url": str(base_url or "").strip(),
            "api_key": str(api_key or "").strip(), "temperature": float(temperature), "max_tokens": int(max_tokens), "vision": bool(vision)}


class YogurtH3RewriteModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3RewriteModel",
            display_name="Yogurt H3 改写模型",
            category="Yogurt/MiniMax H3",
            description="导演台提示词改写使用的大语言模型；连接到导演台的 rewriter，在工程里打开“自动改写”。",
            inputs=[
                io.Combo.Input("provider", options=list(PROVIDERS), default="openai",
                               tooltip="openai 也用于各类 OpenAI 兼容接口（填 base_url）；anthropic；ollama 为本地模型。"),
                io.String.Input("model", default="gpt-4.1", tooltip="模型名称，例如 gpt-4.1、claude-sonnet-4-5、qwen3:32b。需要支持工具调用。"),
                io.String.Input("base_url", default="", tooltip="接口地址；留空使用环境变量 OPENAI_BASE_URL 或官方地址，Ollama 默认 http://localhost:11434。"),
                io.String.Input("api_key", default="", tooltip="留空读取环境变量 OPENAI_API_KEY / ANTHROPIC_API_KEY；填写会保存在工作流文件里。"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Int.Input("max_tokens", default=8192, min=256, max=200000,
                             tooltip="每次回复的最大输出长度；不设上限时部分服务（如 OpenRouter）会按模型上限预扣额度。"),
                io.Boolean.Input("vision", default=True, tooltip="把首尾帧、参考图和源视频画面发给模型看；模型不支持图片时关闭。"),
            ],
            outputs=[Rewriter.Output("rewriter")],
        )

    @classmethod
    def execute(cls, provider, model, base_url, api_key, temperature, max_tokens, vision) -> io.NodeOutput:
        return io.NodeOutput(rewriter_config(provider, model, base_url, api_key, temperature, max_tokens, vision))


def chat_model(config):
    from langchain.chat_models import init_chat_model

    options = {"temperature": config["temperature"]}
    options["num_predict" if config["provider"] == "ollama" else "max_tokens"] = config["max_tokens"]
    if config["base_url"]:
        options["base_url"] = config["base_url"]
    if config["api_key"]:
        options["api_key"] = config["api_key"]
    if config["provider"] == "openai":
        # OpenAI-compatible servers rarely implement the Responses API.
        options["use_responses_api"] = False
    return init_chat_model(f"{config['provider']}:{config['model']}", **options)


def skill_files():
    return {f"/skills/h3-prompt-writing/{path.relative_to(SKILL).as_posix()}": path.read_text(encoding="utf-8")
            for path in sorted(SKILL.rglob("*")) if path.suffix in (".md", ".txt") and path.is_file()}


def rewritable(shot):
    return shot["rewrite"] and not (shot["kind"] == "source" and not shot["source"]["regen"])


def _seconds(frames):
    return f"{frames / FPS:.2f}"


def labels(window):
    """(label, description, media) in the order MiniMaxH3ReferenceToVideo / ImageToVideo present them."""
    shot = window.shot
    refs = shot["refs"]
    if refs["images"] or refs["videos"] or refs["audios"] or (shot["source"] and shot["source"]["regen"]):
        shared = shot["global_refs"]

        def scope(kind, i):
            return f"global reference {kind}, the same in every segment" if i < shared[kind + "s"] else f"reference {kind}"

        out = [(f"<Picture {i + 1}>", scope("image", i), m) for i, m in enumerate(refs["images"])]
        videos = ([("the source video this segment rewrites; keep its motion, framing and timing", shot["source"]["media"], False)]
                  if shot["source"] and shot["source"]["regen"] else [])
        videos += [(scope("video", i), v["media"], v["audio"]) for i, v in enumerate(refs["videos"])]
        audio = 0
        for k, (text, item, with_audio) in enumerate(videos, 1):
            if with_audio:
                audio += 1
                out.append((f"<Audio {audio}>", f"soundtrack of <Video {k}>", None))
            out.append((f"<Video {k}>", text, item))
        for i in range(len(refs["audios"])):
            audio += 1
            out.append((f"<Audio {audio}>", scope("audio", i), None))
        return "Ref2VA", out
    frames = {window.frame(k["at"]): k["media"] for k in shot["keyframes"]}
    first, last = frames.get(0), frames.get(window.gen - 1)
    out = []
    if first is not None:
        out.append(("<Picture 1>", "the first frame, aligned at 0.00 s", first))
    if last is not None:
        out.append((f"<Picture {len(out) + 1}>", f"the last frame, aligned at {_seconds(window.gen - 1)} s", last))
    mode = "FL2VA" if first is not None and last is not None else "I2VA" if first is not None else "L2VA" if last is not None else "T2VA"
    return mode, out


def brief(window):
    shot = window.shot
    mode, items = labels(window)
    lines = [f"## Segment {window.index + 1} — id `{shot['id']}`",
             f"- Length: {_seconds(window.gen)} s ({window.gen} frames at 24 fps); write the timeline for exactly 0.00–{_seconds(window.gen)} s.",
             f"- Mode: {mode}."]
    if window.carry:
        lines.append(f"- Continued: 0.00–{_seconds(window.head)} s is the end of segment {window.index} as already generated "
                     f"(same picture, camera and sound). Open [Shot 1] in that moment, then continue; no new scene.")
    elif window.index:
        lines.append("- Hard cut from the previous segment.")
    for label, text, _ in items:
        lines.append(f"- {label}: {text}.")
    pinned = {window.frame(k["at"]) for k in shot["keyframes"]}
    for index in sorted(i for i in pinned if mode == "Ref2VA" or i not in (0, window.gen - 1)):
        lines.append(f"- An unlabeled guide image pins the picture at {_seconds(index)} s; describe that moment consistently with it.")
    sound = {"generate": "generated by H3 with the picture; write overall_soundscape and non_diegetic_music.",
             "soundtrack": "locked to a prerecorded film soundtrack that H3 cannot change; match on-screen action and lip movement to it and keep the sound fields brief.",
             "source": "locked to the source video's original audio; keep the sound fields brief.",
             "silent": "silent; write `overall_soundscape: Silence.` and `non_diegetic_music: None.`"}[shot["audio"]]
    lines.append(f"- Sound: {sound}")
    if shot["inherit_prompt"]:
        lines.append(f"- Segment prompt: {shot['prompt'].strip() or '(none — derive this segment from the global prompt)'}")
    else:
        lines.append("- Does not use the global prompt: write this segment only from its segment prompt and the facts above.")
        lines.append(f"- Segment prompt: {shot['prompt'].strip() or '(none)'}")
    return "\n".join(lines)


def _image_block(image, caption):
    pixels = (image[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
    picture = Image.fromarray(pixels)
    picture.thumbnail((640, 640))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG", quality=85)
    return [{"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}]


def attachments(window, media):
    """Images the model can look at: labelled keyframes / references, guide frames and the first frame of videos."""
    shot = window.shot
    blocks = []
    mode, items = labels(window)
    for label, text, item in items:
        if item is None or ("input" in item and item["input"] not in media.inputs):
            continue
        caption = f"Segment {window.index + 1} {label} ({text}):"
        if label.startswith("<Video"):
            width, height = video_size(media, item)
            scale = 512 / max(width, height)
            image = media.video(item, 0, 1, max(2, round(width * scale)), max(2, round(height * scale)), "disabled")
        else:
            image = media.image(item)
        blocks += _image_block(image, caption)
    for k in shot["keyframes"]:
        index = window.frame(k["at"])
        if mode == "Ref2VA" or index not in (0, window.gen - 1):
            if "input" in k["media"] and k["media"]["input"] not in media.inputs:
                continue
            blocks += _image_block(media.image(k["media"]), f"Segment {window.index + 1} guide image at {_seconds(index)} s:")
    return blocks


def _identity(media, item):
    # Graph inputs are referenced by name: their tensors exist only while the graph executes.
    return ["input", item["input"]] if "input" in item else media.identity(item)


def _segment_key(window, project, config, skill, media, previous):
    shot = window.shot
    items = [k["media"] for k in shot["keyframes"]] + shot["refs"]["images"] + shot["refs"]["audios"] + [v["media"] for v in shot["refs"]["videos"]]
    if shot["source"]:
        items.append(shot["source"]["media"])
    model = {k: v for k, v in config.items() if k != "api_key"}
    shared = [project["story"], project["rewrite"]["notes"]] if shot["inherit_prompt"] else None
    return digest([VERSION, model, skill, shared, project["rewrite"]["seed"],
                   project["width"], project["height"], {**shot, "repaints": None, "details": None},
                   [_identity(media, m) for m in items], window.head, window.gen, window.carry, previous])


def rewrite_project(project, config, media, event=None):
    """(project with rewritten prompts, {shot id: prompt}, report lines). Only uncached segments call the model."""
    windows = plan_windows(project)
    skill = skill_files()
    skill_id = digest(skill)
    prompts, targets = {}, []
    for window in windows:
        shot = window.shot
        if not rewritable(shot):
            continue
        previous = windows[window.index - 1].shot if window.carry else None
        if previous is not None and previous["id"] in {w.shot["id"] for w in targets}:
            targets.append(window)
            continue
        before = prompts.get(previous["id"], previous["prompt"]) if previous is not None else None
        key = _segment_key(window, project, config, skill_id, media, before)
        if cache.exists(key, ".json"):
            prompts[shot["id"]] = cache.read_json(key)["prompt"]
        else:
            targets.append(window)
    report = []
    if targets:
        for window in targets:
            if event:
                event(window.index, "rewriting")
        written = _write(project, windows, targets, prompts, config, media, skill)
        prompts.update(written)
        for window in targets:
            previous = windows[window.index - 1].shot if window.carry else None
            before = prompts.get(previous["id"], previous["prompt"]) if previous is not None else None
            cache.write_json(_segment_key(window, project, config, skill_id, media, before), {"prompt": prompts[window.shot["id"]]})
        report.append(f"改写：第 {'、'.join(str(w.index + 1) for w in targets)} 镜（{config['provider']}:{config['model']}）")
    cached = [w.index + 1 for w in windows if w.shot["id"] in prompts and w not in targets]
    if cached:
        report.append(f"改写读取缓存：第 {'、'.join(map(str, cached))} 镜")
    rewritten = copy.deepcopy(project)
    for shot in rewritten["shots"]:
        if shot["id"] in prompts:
            shot["prompt"] = prompts[shot["id"]]
    return rewritten, prompts, report


def _message(project, windows, targets, prompts, config, media):
    from langchain_core.messages import HumanMessage

    target_ids = {w.shot["id"] for w in targets}
    total = sum(w.keep for w in windows)
    text = [f"# Global prompt (the whole film)\n{project['story'].strip() or '(not given — build the film from the segment prompts)'}"]
    if project["rewrite"]["notes"].strip():
        text.append(f"# Global requirements\n{project['rewrite']['notes'].strip()}")
    text.append(f"# Timeline\n{len(windows)} segments, {_seconds(total)} s of film, canvas {project['width']}x{project['height']}.")
    for window in windows:
        shot = window.shot
        if shot["id"] in target_ids:
            text.append(brief(window))
        else:
            fixed = prompts.get(shot["id"], shot["prompt"]).strip() or "(no prompt)"
            text.append(f"## Segment {window.index + 1} — id `{shot['id']}` (fixed, {_seconds(window.gen)} s; do not rewrite)\n{fixed}")
    ids = ", ".join(f"`{w.shot['id']}`" for w in targets)
    text.append(f"# Task\nWrite the prompts for segments {ids}.")
    content = [{"type": "text", "text": "\n\n".join(text)}]
    if config["vision"]:
        for window in targets:
            content += attachments(window, media)
    return HumanMessage(content=content)


def submit_tool(target_ids, written):
    """The agent hands in prompts through an ordinary tool: forced structured-output tool calls are refused by
    reasoning models such as Claude Opus, and the reply lists what is still missing so the agent can finish."""
    from langchain_core.tools import tool

    @tool
    def submit_prompts(segments: list[SegmentPrompt]) -> str:
        """Submit finished H3 prompts, one entry per segment id. Can be called again; a later prompt replaces an earlier one."""
        for item in segments:
            item = SegmentPrompt.model_validate(item) if isinstance(item, dict) else item
            if item.id not in target_ids:
                return f"Unknown segment id `{item.id}`. Requested ids: {', '.join(target_ids)}. Nothing was saved from this call."
            written[item.id] = item.prompt.strip()
        missing = [i for i in target_ids if not written.get(i)]
        return f"Saved. Still missing: {', '.join(missing)}." if missing else "All prompts saved. You are done."

    return submit_prompts


def _write(project, windows, targets, prompts, config, media, skill):
    try:
        from deepagents import create_deep_agent
        from deepagents.backends.utils import create_file_data
    except ImportError as error:
        raise RuntimeError("提示词改写需要 deepagents：在 ComfyUI 的 Python 环境中 pip install deepagents。") from error
    target_ids = [w.shot["id"] for w in targets]
    written = {}
    agent = create_deep_agent(model=chat_model(config), tools=[submit_tool(target_ids, written)], system_prompt=SYSTEM_PROMPT,
                              skills=["/skills/"], name="yogurt-h3-screenwriter")
    inputs = {"messages": [_message(project, windows, targets, prompts, config, media)],
              "files": {path: create_file_data(text) for path, text in skill.items()}}
    for _ in agent.stream(inputs, config={"recursion_limit": 200}, stream_mode="values"):
        comfy.model_management.throw_exception_if_processing_interrupted()
    missing = [w.index + 1 for w in targets if not written.get(w.shot["id"])]
    if missing:
        raise RuntimeError(f"改写模型没有交回第 {'、'.join(map(str, missing))} 镜的提示词。")
    return {i: written[i] for i in target_ids}
