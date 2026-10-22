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

import comfy.model_management
from comfy_api.latest import io

from . import cache
from .cache import digest
from .project import FPS, plan_windows

VERSION = "rewrite-2"
SKILL = Path(__file__).resolve().parent / "h3-prompt-writing"
PROVIDERS = ("openai", "anthropic", "ollama")
REMINDERS = 2
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
in the language the user wrote it in, using the guide's dialogue markup. Some media are withheld from you for \
privacy: write about them only what the user's description says, and never guess how they look.

Work in two steps. First write /plan.md: one beat per requested segment in timeline order, with the \
characters, wardrobe, setting and sound that carry across segments. Then write the requested segments one at \
a time in timeline order: finish one segment's prompt, submit it with `submit_prompt`, and only then start the \
next, continuing from the prompts already written. Prompts are saved only through `submit_prompt`, never from \
your text replies. Keep calling tools until `submit_prompt` reports that every segment is saved: a reply without \
a tool call ends your session."""


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


TRANSLATE = """Rewrite the user's description of what to select in a video as the short English noun phrase an \
open-vocabulary segmentation model (SAM 3) expects, e.g. "girl in red raincoat". Keep commas between separate things \
and any ":N" count after a thing. Reply with the phrase only."""


def english_targets(texts, config):
    """{text: English phrase} for SAM 3 targets written in another language; one cached model call per text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    model = {k: v for k, v in config.items() if k != "api_key"}
    out, chat = {}, None
    for text in texts:
        key = digest(["sam3-target-1", model, text])
        if cache.exists(key, ".json"):
            out[text] = cache.read_json(key)["english"]
            continue
        chat = chat or chat_model(config)
        reply = str(chat.invoke([SystemMessage(TRANSLATE), HumanMessage(text)]).text).strip()
        out[text] = reply.splitlines()[0].strip().strip('"') if reply else text
        cache.write_json(key, {"english": out[text]})
    return out


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
    """(label, description, media, entry) in the order MiniMaxH3ReferenceToVideo / ImageToVideo present them;
    ``entry`` is the project item carrying the user's privacy choice (None for audio)."""
    shot = window.shot
    refs = shot["refs"]
    if refs["images"] or refs["videos"] or refs["audios"] or (shot["source"] and shot["source"]["regen"]):
        shared = shot["global_refs"]

        def scope(kind, i):
            return f"global reference {kind}, the same in every segment" if i < shared[kind + "s"] else f"reference {kind}"

        out = [(f"<Picture {i + 1}>", scope("image", i), m["media"], m) for i, m in enumerate(refs["images"])]
        videos = ([("the source video this segment rewrites; keep its motion, framing and timing", shot["source"], False)]
                  if shot["source"] and shot["source"]["regen"] else [])
        videos += [(scope("video", i), v, v["audio"]) for i, v in enumerate(refs["videos"])]
        audio = 0
        for k, (text, entry, with_audio) in enumerate(videos, 1):
            if with_audio:
                audio += 1
                out.append((f"<Audio {audio}>", f"soundtrack of <Video {k}>", None, None))
            out.append((f"<Video {k}>", text, entry["media"], entry))
        for i in range(len(refs["audios"])):
            audio += 1
            out.append((f"<Audio {audio}>", scope("audio", i), None, None))
        return "Ref2VA", out
    frames = {window.frame(k["at"]): k for k in shot["keyframes"]}
    first, last = frames.get(0), frames.get(window.gen - 1)
    out = []
    if first is not None:
        out.append(("<Picture 1>", "the first frame, aligned at 0.00 s", first["media"], first))
    if last is not None:
        out.append((f"<Picture {len(out) + 1}>", f"the last frame, aligned at {_seconds(window.gen - 1)} s", last["media"], last))
    mode = "FL2VA" if first is not None and last is not None else "I2VA" if first is not None else "L2VA" if last is not None else "T2VA"
    return mode, out


def _withheld(entry):
    """What the brief adds about media the user keeps out of the model's context, or a note they wrote for it."""
    note = entry["description"].strip() if entry else ""
    if entry is None or entry["share"]:
        return f"; the user notes: {note}" if note else ""
    if note:
        return f"; not shown to you for privacy, the user describes it as: {note}"
    return "; not shown to you for privacy and not described: refer to it only by its label and do not invent how it looks"


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
    for label, text, _, entry in items:
        lines.append(f"- {label}: {text}{_withheld(entry)}.")
    pinned = {window.frame(k["at"]): k for k in shot["keyframes"]}
    for index in sorted(i for i in pinned if mode == "Ref2VA" or i not in (0, window.gen - 1)):
        lines.append(f"- An unlabeled guide image pins the picture at {_seconds(index)} s; describe that moment consistently with it"
                     f"{_withheld(pinned[index])}.")
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


def _video_frame(media, item, start):
    width, height = media.video_size(item)
    scale = 512 / max(width, height)
    return media.video(item, start, 1, max(2, round(width * scale)), max(2, round(height * scale)), "disabled")


def _visible(media, item):
    return item is not None and ("input" not in item or item["input"] in media.inputs)


def global_attachments(project, targets, media):
    """Global references once for the whole request, instead of once per segment that inherits them."""
    if not any(w.shot["inherit_refs"] for w in targets):
        return []
    refs, blocks = project["refs"], []
    for i, entry in enumerate(refs["images"]):
        item = entry["media"]
        if entry["share"] and _visible(media, item):
            blocks += _image_block(media.image(item), f"Global reference image {i + 1} (<Picture {i + 1}> in every segment that uses global references):")
    for i, video in enumerate(refs["videos"]):
        if video["share"] and _visible(media, video["media"]):
            blocks += _image_block(_video_frame(media, video["media"], video["start"]),
                                   f"Global reference video {i + 1}, first frame (its <Video N> label is listed in each segment):")
    return blocks


def attachments(window, media):
    """The segment's own images: keyframes, its references, guide frames and the opening frame of its videos."""
    shot = window.shot
    blocks = []
    mode, items = labels(window)
    starts = {id(v["media"]): v["start"] for v in shot["refs"]["videos"]}
    if shot["source"]:
        starts[id(shot["source"]["media"])] = shot["source"]["start"]
    for label, text, item, entry in items:
        if not _visible(media, item) or not entry["share"] or text.startswith("global reference"):
            continue
        caption = f"Segment {window.index + 1} {label} ({text}):"
        if label.startswith("<Video"):
            image = _video_frame(media, item, starts[id(item)])
        else:
            image = media.image(item)
        blocks += _image_block(image, caption)
    for k in shot["keyframes"]:
        index = window.frame(k["at"])
        if mode == "Ref2VA" or index not in (0, window.gen - 1):
            if not k["share"] or "input" in k["media"] and k["media"]["input"] not in media.inputs:
                continue
            blocks += _image_block(media.image(k["media"]), f"Segment {window.index + 1} guide image at {_seconds(index)} s:")
    return blocks


def _identity(media, item):
    # Graph inputs are referenced by name: their tensors exist only while the graph executes.
    return ["input", item["input"]] if "input" in item else media.identity(item)


def _segment_key(window, project, config, skill, media, previous):
    shot = window.shot
    items = [k["media"] for k in shot["keyframes"]] + [m["media"] for m in shot["refs"]["images"]] + shot["refs"]["audios"] + [v["media"] for v in shot["refs"]["videos"]]
    if shot["source"]:
        items.append(shot["source"]["media"])
    model = {k: v for k, v in config.items() if k != "api_key"}
    shared = [project["story"], project["rewrite"]["notes"]] if shot["inherit_prompt"] else None
    return digest([VERSION, model, skill, shared, project["rewrite"]["seed"],
                   project["width"], project["height"], {**shot, "repaints": None, "details": None},
                   [_identity(media, m) for m in items], window.head, window.gen, window.carry, previous])


def rewrite_project(project, config, media, event=None):
    """(project with rewritten prompts, {shot id: prompt}, report lines). Only uncached segments call the model.

    ``event(index, stage, prompt=None)`` reports progress; every cached or newly written prompt is reported
    as stage "rewritten" as soon as it is known, before rendering starts."""
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
            if event:
                event(window.index, "rewritten", prompts[shot["id"]])
        else:
            targets.append(window)
    report = []
    if targets:
        written = _write(project, windows, targets, prompts, config, media, skill, event)
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
    text.append(f"# Task\nWrite the prompts for segments {ids}, one at a time in this order.")
    content = [{"type": "text", "text": "\n\n".join(text)}]
    if config["vision"]:
        content += global_attachments(project, targets, media)
        for window in targets:
            content += attachments(window, media)
    return HumanMessage(content=content)


def submit_tool(targets, written, event=None):
    """The agent hands in one segment per call, so each segment is written with the earlier ones already in its
    context. Calls batched in one message run concurrently, so the order is guided by the replies, not enforced.
    An ordinary tool also avoids the forced tool choice that reasoning models such as Claude Opus refuse."""
    from langchain_core.tools import tool

    order = [w.shot["id"] for w in targets]
    number = {w.shot["id"]: w.index + 1 for w in targets}

    @tool
    def submit_prompt(id: str, prompt: str) -> str:
        """Submit the finished H3 prompt of one segment. Submit segments one at a time in timeline order;
        submitting a segment again replaces its prompt."""
        if id not in number:
            return f"Unknown segment id `{id}`. Requested ids in order: {', '.join(order)}."
        if not prompt.strip():
            return "The prompt is empty; submit the complete H3 prompt."
        written[id] = prompt.strip()
        if event:
            event(number[id] - 1, "rewritten", written[id])
        pending = [i for i in order if i not in written]
        if not pending:
            return f"All {len(order)} segments saved. You are done."
        if event:
            event(number[pending[0]] - 1, "rewriting")
        return f"Saved segment {number[id]} ({len(order) - len(pending)}/{len(order)}). Next: segment {number[pending[0]]} (id `{pending[0]}`)."

    return submit_prompt


def _write(project, windows, targets, prompts, config, media, skill, event=None):
    try:
        from deepagents import create_deep_agent
        from deepagents.backends.utils import create_file_data
    except ImportError as error:
        raise RuntimeError("提示词改写需要 deepagents：在 ComfyUI 的 Python 环境中 pip install deepagents。") from error
    from langchain_core.messages import AIMessage, HumanMessage

    written = {}
    if event:
        event(targets[0].index, "rewriting")
    agent = create_deep_agent(model=chat_model(config), tools=[submit_tool(targets, written, event)], system_prompt=SYSTEM_PROMPT,
                              skills=["/skills/"], name="yogurt-h3-screenwriter")
    messages = [_message(project, windows, targets, prompts, config, media)]
    files = {path: create_file_data(text) for path, text in skill.items()}
    for attempt in range(REMINDERS + 1):
        for state in agent.stream({"messages": messages, "files": files}, config={"recursion_limit": 100 + 10 * len(targets)}, stream_mode="values"):
            comfy.model_management.throw_exception_if_processing_interrupted()
        missing = [w for w in targets if not written.get(w.shot["id"])]
        if not missing:
            return {w.shot["id"]: written[w.shot["id"]] for w in targets}
        if attempt < REMINDERS:
            # Any reply without a tool call ends the run; models often stop after the plan or paste a prompt as text.
            ids = ", ".join(f"`{w.shot['id']}`" for w in missing)
            messages = state["messages"] + [HumanMessage(
                content=f"Not submitted yet: segments {ids}. Prompts are saved only through `submit_prompt`; a text reply is not saved "
                        f"and ends the session. Call `submit_prompt` now for segment {missing[0].index + 1} (id `{missing[0].shot['id']}`), "
                        f"then the rest in order.")]
            files = state["files"]
    reply = next((" ".join(m.text.split()) for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.text.strip()), "")
    raise RuntimeError(f"改写模型没有交回第 {'、'.join(str(w.index + 1) for w in missing)} 镜的提示词（已提醒 {REMINDERS} 次）。"
                       + (f"模型最后的回复：{reply[:300]}" if reply else "模型没有文字回复，请确认它支持工具调用。"))
