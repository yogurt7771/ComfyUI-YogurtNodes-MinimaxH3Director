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
import re

from PIL import Image

import comfy.model_management
from comfy_api.latest import io

from . import cache
from .agents import AGENTS, Rewriter, answers, run_agent
from .cache import digest
from .project import FPS, plan_windows, swap_prompt

VERSION = "rewrite-3"
SWAP_VERSION = "swap-4"
SWAP_FRAMES = 4
ABSENT = "NONE"
ABSENT_NOTE = "画面里没有要换的对象，这一镜保留原画。"
SKILL = Path(__file__).resolve().parent / "h3-prompt-writing"
PROVIDERS = ("openai", "anthropic", "ollama")
REMINDERS = 2

SCREENWRITER = """You are the screenwriter of a film made of several segments generated one by one with MiniMax H3. \
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

A video's frames are shown once a second with their times. H3 follows your words over the video, so write the \
timeline from those frames: who is on screen, who enters and leaves, every move and camera change at its time. \
Never describe a state as lasting past what the frames show. If a video's frames are not shown to you, say only \
that the action follows that video.

Use exactly the labels listed for a segment (<Picture N>, <Video N>, <Audio N>) and no others. Keep dialogue \
in the language the user wrote it in, using the guide's dialogue markup. Some media are withheld from you for \
privacy: write about them only what the user's description says, and never guess how they look."""

SYSTEM_PROMPT = SCREENWRITER + """

Work in two steps. First write /plan.md: one beat per requested segment in timeline order, with the \
characters, wardrobe, setting and sound that carry across segments. Then write the requested segments one at \
a time in timeline order: finish one segment's prompt, submit it with `submit_prompt`, and only then start the \
next, continuing from the prompts already written. Prompts are saved only through `submit_prompt`, never from \
your text replies. Keep calling tools until `submit_prompt` reports that every segment is saved: a reply without \
a tool call ends your session."""

AGENT_PROMPT = SCREENWRITER + """

The skill is in skills/h3-prompt-writing under the current folder. Images mentioned below are files in the current \
folder: open them to see them. You can read files but not change any, and nobody can answer questions, so decide \
for yourself and finish.

Plan first: one beat per requested segment in timeline order, with the characters, wardrobe, setting and sound that \
carry across segments. Then write the requested segments in timeline order, continuing from the prompts already \
written. Your final message must hold every requested segment in exactly this form, N being the segment's number \
in the timeline:

===== SEGMENT N =====
the complete H3 prompt
===== END SEGMENT N ====="""


SWAP_WRITER = """One subject of a video is being replaced with MiniMax H3. H3 takes the motion, camera and scene from \
the video by itself, and a fixed text tells it to. For each segment you fill in only two short fields, from that \
segment's frames and the reference pictures:

WHAT is the subject to replace as it looks in this segment: clothing, hair or position, enough to tell it from anyone \
else on screen. Example of the form: `girl in a red dress`.
INTO is who the new subject is and how they look: a noun first, then face, hair and clothing, read off the reference \
picture. Word it the same in every segment. Example of the form: `silver-haired elf with a green cloak and tall boots`.

It is one subject through the whole film. If the user names it, keep to it, fitting their words to what the frames \
show. If they do not, choose the person on screen most like the reference picture in kind (gender, age, build), and if \
that does not settle it, the most prominent one. When a segment's frames do not show that subject, write \
`WHAT: NONE` for that segment instead of picking someone else: the segment then stays as it is. Keep the user's choice \
of what it becomes. Never write actions, camera, scene or timing. Both fields are in lowercase English, with no leading \
article and at most 15 words. With one reference picture INTO has no label; with several, start it with the label of \
the one to use. Some media are withheld from you for privacy: write about them only what the user's description says.

Answer in exactly this form, N being the segment's number:

===== SEGMENT N =====
WHAT: ...
INTO: ...
===== END SEGMENT N ====="""


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
    model = {k: v for k, v in config.items() if k != "api_key"}
    out, chat = {}, None
    for text in texts:
        key = digest(["sam3-target-1", model, text])
        if cache.exists(key, ".json"):
            out[text] = cache.read_json(key)["english"]
            continue
        if config["provider"] in AGENTS:
            reply = answers(run_agent(config, f"{TRANSLATE}\n\nAnswer in your final message, exactly like this:\n===== SEGMENT 1 =====\nthe phrase\n"
                                              f"===== END SEGMENT 1 =====\n\nThe description:\n{text}")).get(1, "")
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

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


def _video_blocks(media, item, start, count, caption, moments=None):
    """What a video shows over the ``count`` frames a segment takes from it: ``moments``, by default one frame a second
    and the last, each with its time."""
    width, height = media.video_size(item)
    scale = 512 / max(width, height)
    frames = media.video(item, start, count, max(2, round(width * scale)), max(2, round(height * scale)), "disabled")
    blocks = []
    for at in sorted(moments or {*range(0, count, FPS), count - 1}):
        blocks += _image_block(frames[at:at + 1], f"{caption} at {_seconds(at)} s:")
    return blocks


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
            blocks += _video_blocks(media, video["media"], video["start"], max(w.gen for w in targets),
                                    f"Global reference video {i + 1} (its <Video N> label is listed in each segment)")
    return blocks


def attachments(window, media):
    """The segment's own images: keyframes, its references, guide frames and its videos over the segment."""
    shot = window.shot
    blocks = []
    mode, items = labels(window)
    for label, text, item, entry in items:
        if not _visible(media, item) or not entry["share"] or text.startswith("global reference"):
            continue
        caption = f"Segment {window.index + 1} {label} ({text})"
        if entry is shot["source"]:
            # A carried head is the previous segment's stretch of the same video: 0.00 s of this segment's timeline.
            blocks += _video_blocks(media, item, entry["start"] - window.head, window.head + window.keep, caption)
        elif label.startswith("<Video"):
            blocks += _video_blocks(media, item, entry["start"], window.gen, caption)
        else:
            blocks += _image_block(media.image(item), caption + ":")
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


def _replaced(shot):
    return shot["rewrite"] and shot["source"] is not None and (shot["source"]["regen"] or shot["source"]["reference"])


def _swap_key(window, project, config, media):
    source, model = window.shot["source"], {k: v for k, v in config.items() if k != "api_key"}
    swap, rewrite = project["swap"], project["rewrite"]
    # What the model could see is part of the key: the editor's 只改写 has no connected sockets, a render does.
    pictures = [[_identity(media, m["media"]), m["share"] and _visible(media, m["media"]), m["description"]] for m in window.shot["refs"]["images"]]
    return digest([SWAP_VERSION, model, swap["what"], swap["into"], swap["track"], rewrite["notes"], rewrite["seed"], pictures, window.shot["frames"],
                   _identity(media, source["media"]), source["start"], source["share"] and _visible(media, source["media"]), source["description"]])


def _swap_content(project, targets, media):
    swap, notes = project["swap"], project["rewrite"]["notes"].strip()
    text = [f"The user says the subject to replace is: {swap['what'].strip() or '(not said)'}",
            f"The user says it becomes: {swap['into'].strip() or '(not said)'}",
            *(["WHAT is also what a segmentation model (SAM 3) selects it by, so keep it to a noun phrase of at most 6 words."] if swap["track"] else [])]
    if notes:
        text.append(f"The user also asks: {notes}")
    text += [f"<Picture {i}>{_withheld(m)}." for i, m in enumerate(project["refs"]["images"], 1) if _withheld(m)]
    blocks = global_attachments(project, targets, media)
    for window in targets:
        shot, number = window.shot, window.index + 1
        source, count = shot["source"], shot["frames"]
        seen = source["share"] and _visible(media, source["media"])
        text.append(f"## Segment {number} — id `{shot['id']}`\n{_seconds(count)} s of the video{_withheld(source)}"
                    f"{'' if seen or not source['share'] else '; its frames cannot be shown here: rely on what the user says'}.")
        shared = shot["global_refs"]["images"]
        for i, entry in enumerate(shot["refs"]["images"][shared:], shared + 1):
            text.append(f"<Picture {i}> (segment {number} only){_withheld(entry)}.")
            if entry["share"] and _visible(media, entry["media"]):
                blocks += _image_block(media.image(entry["media"]), f"Segment {number} <Picture {i}>:")
        if seen:
            blocks += _video_blocks(media, source["media"], source["start"], count, f"Segment {number}",
                                    {(count - 1) * k // (SWAP_FRAMES - 1) for k in range(SWAP_FRAMES)})
    return [{"type": "text", "text": "\n\n".join(text)}] + blocks


def _swap_fields(answer):
    fields = dict(re.findall(r"^\s*(WHAT|INTO)\s*[:：]\s*(.+?)\s*$", answer, re.M))
    return {"what": fields.get("WHAT", ""), "into": fields.get("INTO", "")}


def _write_swaps(project, targets, config, media):
    written, missing, reply = {}, targets, ""
    for _ in range(REMINDERS + 1):
        content = _swap_content(project, missing, media)
        if config["provider"] in AGENTS:
            text, images = _agent_brief(content)
            reply = run_agent(config, f"{SWAP_WRITER}\n\nImages mentioned below are files in the current folder: open them to see them.\n\n{text}", images)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            reply = str(chat_model(config).invoke([SystemMessage(SWAP_WRITER), HumanMessage(content=content)]).text)
        found = answers(reply)
        for window in missing:
            fields = _swap_fields(found.get(window.index + 1, ""))
            if fields["what"] and (fields["into"] or fields["what"] == ABSENT):
                written[window.shot["id"]] = fields
        missing = [w for w in missing if w.shot["id"] not in written]
        if not missing:
            return written
    raise RuntimeError(f"{config['provider']} 没有交回第 {'、'.join(str(w.index + 1) for w in missing)} 镜的替换描述（已重试 {REMINDERS} 次）。"
                       + (f"它最后的回复：{reply.strip()[-300:]}" if reply.strip() else "它没有任何输出。"))


def _keep_source(shot):
    """A shot without the subject stays as the video has it, instead of H3 replacing whoever else is on screen."""
    shot["source"].update(regen=False, reference=False)
    shot["prompt"] = ""
    shot["repaints"] = [layer for layer in shot["repaints"] if layer["id"] != f"{shot['id']}-swap"]


def _rewrite_swaps(project, config, media, event):
    """替换模式: the model only names what each 替换 shot replaces with what, from a few of its frames and the reference
    pictures. The rest stays the measured template, since a long description overrides the source video."""
    if not config["vision"]:
        return project, {}, ["改写模型没开 vision，看不到画面：替换用“把什么”和“换成”里写的内容。"]
    windows = [w for w in plan_windows(project) if _replaced(w.shot)]
    fields, targets = {}, []
    for window in windows:
        key = _swap_key(window, project, config, media)
        if cache.exists(key, ".json"):
            fields[window.shot["id"]] = cache.read_json(key)
        else:
            targets.append(window)
    report = []
    if targets:
        if event:
            event(targets[0].index, "rewriting")
        written = _write_swaps(project, targets, config, media)
        for window in targets:
            cache.write_json(_swap_key(window, project, config, media), written[window.shot["id"]])
        fields.update(written)
        report.append(f"改写：第 {'、'.join(str(w.index + 1) for w in targets)} 镜（{config['provider']}:{config['model'] or '默认模型'}）")
    cached = [w.index + 1 for w in windows if w not in targets]
    if cached:
        report.append(f"改写读取缓存：第 {'、'.join(map(str, cached))} 镜")
    rewritten, prompts = copy.deepcopy(project), {}
    for window in windows:
        shot = next(s for s in rewritten["shots"] if s["id"] == window.shot["id"])
        what, into = fields[shot["id"]]["what"], fields[shot["id"]]["into"]
        if what == ABSENT:
            _keep_source(shot)
            prompts[shot["id"]] = ABSENT_NOTE
            report.append(f"第 {window.index + 1} 镜画面里没有要换的对象，保留原画")
        else:
            prompts[shot["id"]] = shot["prompt"] = swap_prompt(what, into, bool(project["refs"]["images"]), project["swap"]["track"])
            if project["swap"]["track"]:
                shot["repaints"][0]["target"]["text"] = what
        if event:
            event(window.index, "rewritten", prompts[shot["id"]])
    return rewritten, prompts, report


def rewrite_project(project, config, media, event=None):
    """(project with rewritten prompts, {shot id: prompt}, report lines). Only uncached segments call the model.

    ``event(index, stage, prompt=None)`` reports progress; every cached or newly written prompt is reported
    as stage "rewritten" as soon as it is known, before rendering starts."""
    if project["swap"]["enabled"]:
        return _rewrite_swaps(project, config, media, event)
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
        report.append(f"改写：第 {'、'.join(str(w.index + 1) for w in targets)} 镜（{config['provider']}:{config['model'] or '默认模型'}）")
    cached = [w.index + 1 for w in windows if w.shot["id"] in prompts and w not in targets]
    if cached:
        report.append(f"改写读取缓存：第 {'、'.join(map(str, cached))} 镜")
    rewritten = copy.deepcopy(project)
    for shot in rewritten["shots"]:
        if shot["id"] in prompts:
            shot["prompt"] = prompts[shot["id"]]
    return rewritten, prompts, report


def _content(project, windows, targets, prompts, config, media):
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
    return content


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


def _agent_brief(content):
    """A model message as text plus image files, since an agent CLI takes its images from disk."""
    parts, images = [], {}
    for block in content:
        if block["type"] == "text":
            parts.append(block["text"])
        else:
            name = f"image-{len(images) + 1}.jpg"
            images[name] = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
            parts.append(f"[Image file: {name}]")
    return "\n\n".join(parts), images


def _write_by_agent(project, windows, targets, prompts, config, media, skill, event=None):
    """One run writes every requested segment; a run that leaves some out is followed by one for those, with the
    finished ones marked fixed. Prompts reach the editor when a run ends: the CLIs only hand over a final message."""
    guide = {path.lstrip("/"): text.encode("utf-8") for path, text in skill.items()}
    written, missing, reply = {}, targets, ""
    if event:
        event(targets[0].index, "rewriting")
    for _ in range(REMINDERS + 1):
        text, images = _agent_brief(_content(project, windows, missing, {**prompts, **written}, config, media))
        reply = run_agent(config, f"{AGENT_PROMPT}\n\n{text}", {**guide, **images})
        found = answers(reply)
        for window in missing:
            if found.get(window.index + 1):
                written[window.shot["id"]] = found[window.index + 1]
                if event:
                    event(window.index, "rewritten", written[window.shot["id"]])
        missing = [w for w in missing if w.shot["id"] not in written]
        if not missing:
            return written
    raise RuntimeError(f"{config['provider']} 没有交回第 {'、'.join(str(w.index + 1) for w in missing)} 镜的提示词（已重试 {REMINDERS} 次）。"
                       + (f"它最后的回复：{reply.strip()[-300:]}" if reply.strip() else "它没有任何输出。"))


def _write(project, windows, targets, prompts, config, media, skill, event=None):
    if config["provider"] in AGENTS:
        return _write_by_agent(project, windows, targets, prompts, config, media, skill, event)
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
    messages = [HumanMessage(content=_content(project, windows, targets, prompts, config, media))]
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
