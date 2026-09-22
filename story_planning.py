"""Story segment prompts in the official MiniMax H3 prompt structure.

The official h3-prompt-writing guides (MiniMax H3 Community License) are read at runtime from
``h3-prompt-writing/references``; they are not part of this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .planning import H3_FPS, align_h3_frames, normalize_overlap


MODES = ["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
MAX_CARDS = 9
MAX_SEGMENTS = 12
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"]
SEGMENT_SEPARATOR = "\n\n==========\n\n"
GUIDE_DIR = Path(__file__).resolve().parent / "h3-prompt-writing" / "references"
GUIDE_SOURCE = "https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing"


def official_guide(mode: str) -> str:
    """Official prompt guide text for a mode; Ref2VA builds on the base guide."""
    names = ["base-en.txt"] + (["ref-en.txt"] if mode == "Ref2VA" else [])
    paths = [GUIDE_DIR / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少官方 H3 提示词指南 {path}。请把 {GUIDE_SOURCE} 整个文件夹放到插件目录下的 h3-prompt-writing。")
    return "\n\n".join(path.read_text(encoding="utf-8").strip() for path in paths)


def _clean(s: Any) -> str:
    return (s or "").strip()


def secs_to_frames(seconds: float) -> int:
    return align_h3_frames(round(float(seconds) * H3_FPS))


def build_cards(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    cards = []
    for name, notes in pairs:
        name, notes = _clean(name), _clean(notes)
        if name or notes:
            cards.append((name or f"Character {len(cards) + 1}", notes))
    return cards


def keyframes(mode: str, index: int, total: int) -> tuple[bool, bool]:
    """Whether segment ``index`` (1-based) gets the story's first and/or last keyframe."""
    return mode in ("I2VA", "FL2VA") and index == 1, mode in ("L2VA", "FL2VA") and index == total


def alignment_line(first: bool, last: bool, frames: int, final_shot: int | str = 1) -> str:
    """Official keyframe instruction; the last-frame mark is the effective duration."""
    end = f"{frames / H3_FPS:.2f}"
    if first and last:
        return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
                f"mark of the target video; Picture 2 (from Shot {final_shot}) aligns with the {end}-second mark of the target video.")
    if first:
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if last:
        return (f"How the reference pictures align with the target video — <Picture 1> (from [Shot {final_shot}]) aligns with the "
                f"{end}-second mark of the target video.")
    return ""


BASE_SECTIONS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
REF_SECTIONS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:",
                "non_diegetic_music:")
_HEADER = re.compile(r"^[ \t>#*_\-]*(" + "|".join(dict.fromkeys(name[:-1] for name in BASE_SECTIONS + REF_SECTIONS))
                     + r")[ \t*_]*[:：][ \t*_]*(.*)$", re.I | re.M)
_ALIGNMENT = re.compile(r"^\s*(How the reference pictures align with the target video|For the target video, at 0\.00 seconds)[^\n]*\n*", re.M)


def prompt_fields(mode: str) -> list[str]:
    return [name[:-1] for name in (REF_SECTIONS if mode == "Ref2VA" else BASE_SECTIONS)]


def field_guide(mode: str, subjects: str = "") -> str:
    """What goes into each JSON string field; the keyframe line and section headers are added by code."""
    shots = ("[Shot 1] has no timestamp; each later shot starts with [Shot N] At MM:SS.mmm, the shot cuts to ... "
             "Speakers get stable IDs (S1), (S2); dialogue is <d>[Language] exact original words</d>. Dialogue, lyrics and "
             "on-screen text written in the request are copied verbatim in their original language, never translated.")
    if mode == "Ref2VA":
        rows = [("subject_definitions", "one line per reference label, e.g. <Subject 1> is ... in <Picture 1>, with ...; "
                 "reference pictures define subjects, not frames" + (f". Use exactly these subjects:\n{subjects}" if subjects else "")),
                ("summary", "one paragraph starting with [reference generation]; there is no source video to continue or edit"),
                ("retention_analysis", "one line per label, e.g. <Subject 1> (appears in [Shot 1]): fully_preserved - ..."),
                ("detailed_description", f"one style sentence, then the shots. {shots}"),
                ("overall_soundscape", "1-4 sentences of ambience and physical sounds"),
                ("non_diegetic_music", "1-3 sentences of audience-only music, or N/A")]
    else:
        rows = [("integrated_multimodal_description", f"starts with [Shot 1] <style>, <opening composition>. {shots}"),
                ("overall_soundscape", "1-4 sentences of ambience and physical sounds"),
                ("non_diegetic_music", "1-3 sentences of audience-only music, or N/A")]
    return "\n".join(f'- "{name}": {text}' for name, text in rows) + (
        "\nDo not write section headers or the keyframe alignment line; they are added automatically. "
        "Inside JSON strings write line breaks as \\n.")


# Text every value of a field starts with; the local decoder writes it itself.
FIELD_PREFIX = {"integrated_multimodal_description": "[Shot 1] "}


def _field_text(name: str, value: Any) -> str:
    """A field's own content: models sometimes repeat the keyframe line, headers, or later sections inside it."""
    text = _ALIGNMENT.sub("", _clean(value)).strip()
    own = next((m for m in _HEADER.finditer(text) if m.group(1).lower() == name), None)
    if own:
        text = text[own.start(2):]
    other = next((m for m in _HEADER.finditer(text) if m.group(1).lower() != name), None)
    return _clean(text[:other.start()] if other else text)


def render_fields(mode: str, values: dict[str, str], first: bool, last: bool, frames: int) -> str:
    """Official prompt text from field values: headers, order and keyframe line come from code."""
    sep = "\n" if mode == "Ref2VA" else " "
    body = "\n\n".join(f"{name}:{sep}{_field_text(name, values.get(name))}" for name in prompt_fields(mode))
    return conform(body, mode, first, last, frames)[0]


def conform(prompt: str, mode: str, first: bool, last: bool, frames: int) -> tuple[str, list[str]]:
    """Normalize the guide's mechanical rules in model output and report missing sections.

    The keyframe line is replaced with the one this mode needs, the first shot loses any
    timestamp, and sections are put back in the official order.
    """
    body = _ALIGNMENT.sub("", prompt).strip()
    # Small or fine-tuned models decorate headers ("**Summary:**", "### summary：").
    body = _HEADER.sub(lambda m: m.group(1).lower() + ":" + (" " if m.group(2).strip() else "") + m.group(2).lstrip(), body)
    # Each segment is its own video: shots count from 1 in order of appearance and the first has no timestamp.
    order = {n: str(k) for k, n in enumerate(dict.fromkeys(re.findall(r"\[Shot (\d+)\]", body)), 1)}
    body = re.sub(r"\[Shot (\d+)\]", lambda m: f"[Shot {order[m.group(1)]}]", body)
    body = re.sub(r"\[Shot 1\]\s*At \d\d:\d\d(?:\.\d+)?,\s*", "[Shot 1] ", body)
    sections = REF_SECTIONS if mode == "Ref2VA" else BASE_SECTIONS
    starts = {name: m.start() for name in sections if (m := re.search(rf"^{name}", body, re.M))}
    missing = [name for name in sections if name not in starts]
    if not missing:
        bounds = sorted(starts.values()) + [len(body)]
        chunks = {name: body[start:bounds[bounds.index(start) + 1]].strip() for name, start in starts.items()}
        body = "\n\n".join(chunks[name] for name in sections)
    shots = [int(n) for n in re.findall(r"\[Shot (\d+)\]", body)]
    align = "" if mode == "Ref2VA" else alignment_line(first, last, frames, max(shots, default=1))
    return (f"{align}\n\n{body}" if align else body), missing


def _people(cards, labelled):
    if not cards:
        return "the leads of the story"
    return ", ".join(f"<Subject {i}>, {name}" if labelled else name for i, (name, _) in enumerate(cards, 1))


def _sounds(sound, music):
    sound = _clean(sound) or "Ambience that matches the location continues underneath, with footsteps and fabric movement whenever the characters move."
    return sound, _clean(music) or "N/A"


_RULE = re.compile(r"^.{0,4}(不要|禁止|别|保留|请勿|务必|do not|don't|keep|never)", re.I)


def story_beats(story: str, count: int) -> list[str]:
    """Split the story into ``count`` consecutive beats of similar length: sentences first, then clauses.

    Sentences that state a global rule ("不要换脸", "Keep dialogue in Chinese") are not plot and join no beat.
    """
    sentences = [x.strip() for x in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]*", story) if x.strip()]
    plot = "".join(x for x in sentences if not _RULE.match(x)) or story
    parts = [x.strip() for x in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]*", plot) if x.strip()]
    if len(parts) < count:
        parts = [x.strip() for x in re.findall(r"[^。！？!?；;，,\n]+[。！？!?；;，,]*", plot) if x.strip()] or [plot]
    if len(parts) <= count:
        return [parts[i * len(parts) // count].strip("，, ") for i in range(count)]
    lengths = [0]
    for part in parts:
        lengths.append(lengths[-1] + len(part))
    beats, start = [], 0
    for k in range(count):
        remaining = count - k
        end = len(parts)
        if remaining > 1:
            target = lengths[start] + (lengths[-1] - lengths[start]) / remaining
            end = start + 1
            while end < len(parts) - (remaining - 1) and abs(lengths[end + 1] - target) <= abs(lengths[end] - target):
                end += 1
        beats.append("".join(parts[start:end]).strip("，, "))
        start = end
    return beats


_ARC = ["setup", "rising action", "turning point", "climax", "falling action", "resolution"]


def story_stage(index: int, total: int) -> str:
    """Narrative stage of segment ``index`` (1-based) when a short story is spread over ``total`` segments."""
    return _ARC[0] if total == 1 else _ARC[round((index - 1) * (len(_ARC) - 1) / (total - 1))]


def _segment_prompt(mode, index, total, frames, story, cards, sound, music, stage="", opening_state=""):
    first, last = keyframes(mode, index, total)
    opening = "a medium-wide shot establishes the location and the characters"
    if first:
        opening = "the shot begins from <Picture 1>, preserving its subjects, clothing, composition, and lighting"
    if opening_state and opening_state != story:
        opening = f"the shot opens on the closing moment of this: {opening_state}"
    ending = ""
    if last:
        ending = (" Toward the end the characters, objects, and camera settle into the exact poses, composition, and lighting "
                  f"established by {'Picture 2' if first else '<Picture 1>'}.")
    ref = mode == "Ref2VA"
    moment = f"the {stage} of the story" if stage else "this moment of the story"
    action = (f"{_people(cards, ref)} play out {moment}: {story} "
              "The camera pushes in with small amplitude at slow speed as the action develops." + ending)
    sound, music = _sounds(sound, music)
    if not ref:
        body = (f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, {opening}. {action}\n\n"
                f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")
        align = alignment_line(first, last, frames)
        return f"{align}\n\n{body}" if align else body
    definitions = "\n".join(f"<Subject {i}> is {name} in <Picture {i}>{', ' + notes if notes else ''}."
                            for i, (name, notes) in enumerate(cards, 1)) or "No reference subjects are defined."
    retention = "\n".join(f"<Subject {i}> (appears in [Shot 1]): fully_preserved - the face, hair, body proportions, and outfit "
                          f"from <Picture {i}> are retained." for i in range(1, len(cards) + 1)) or "No referenced content."
    return (f"subject_definitions:\n{definitions}\n\n"
            f"summary:\n[reference generation] The target video shows {_people(cards, True)} in one continuous shot: {story}\n\n"
            f"retention_analysis:\n{retention}\n\n"
            f"detailed_description:\nThe target video is live-action and cinematic.\n[Shot 1] In the opening, {opening}. {action}\n\n"
            f"overall_soundscape:\n{sound}\n\nnon_diegetic_music:\n{music}")


def build_plan(story: str, segment_count: int, seconds: float, overlap_frames: int, mode: str, aspect: str, sound: str,
               music: str, cards: list[tuple[str, str]]) -> dict[str, Any]:
    story = _clean(story) or "A continuous scene with clear subject action and diegetic sound."
    segment_count = max(1, min(int(segment_count), MAX_SEGMENTS))
    seconds = min(15.0, max(4.0, float(seconds)))
    frames = secs_to_frames(seconds)
    overlap_frames = normalize_overlap(max(0, min(int(overlap_frames), 96)), frames)
    beats = story_beats(story, segment_count)
    # A story shorter than the segment count repeats beats; the stage keeps each segment's own place in the arc.
    stages = [story_stage(i, segment_count) if len(set(beats)) < segment_count else "" for i in range(1, segment_count + 1)]
    return {
        "segment_count": segment_count,
        "segment_duration_seconds": seconds,
        "overlap_frames": overlap_frames,
        "mode": mode,
        "aspect_ratio": aspect,
        "beats": beats,
        "segments": [
            {"id": i, "prompt": _segment_prompt(mode, i, segment_count, frames, beats[i - 1], cards, sound, music,
                                                stages[i - 1], beats[i - 2] if i > 1 else "")}
            for i in range(1, segment_count + 1)
        ],
        "length_frames": frames,
    }


def _segment_rules(plan: dict[str, Any]) -> str:
    frames, total = plan["length_frames"], plan["segment_count"]
    lines = []
    for index in range(1, total + 1):
        first, last = keyframes(plan["mode"], index, total)
        rule = (f"- Segment {index}: {frames / H3_FPS:.2f} seconds on its own timeline, 00:00.000 to 00:{frames / H3_FPS:06.3f}"
                + (f" (it does not start at 00:{(index - 1) * frames / H3_FPS:06.3f})." if index > 1 else "."))
        if first:
            rule += " It opens on the first keyframe <Picture 1>."
        if last:
            rule += f" It ends on the last keyframe {'Picture 2' if first else '<Picture 1>'}."
        if index > 1:
            rule += f" Its [Shot 1] opens on the same visual state that ends segment {index - 1}, described again in full."
        if len(set(plan["beats"])) < total:
            rule += (f" Arc stage: {story_stage(index, total)}; invent new concrete events for this stage that come after "
                     f"segment {index - 1}'s events" if index > 1 else f" Arc stage: {story_stage(index, total)}; invent concrete opening events")
            rule += f"; it must show: {plan['beats'][index - 1]}"
        else:
            rule += f" It must show: {plan['beats'][index - 1]}"
        lines.append(rule)
    return "\n".join(lines)


def build_rewrite_request(plan: dict[str, Any], story: str, mode: str, aspect: str, sound: str, music: str,
                          cards: list[tuple[str, str]]) -> str:
    subjects = "\n".join(f"<Subject {i}> is {name} in <Picture {i}>{', ' + notes if notes else ''}"
                         for i, (name, notes) in enumerate(cards, 1))
    if mode == "Ref2VA":
        card_lines = "\n".join(f"<Picture {i}>: {name}{' — ' + notes if notes else ''}" for i, (name, notes) in enumerate(cards, 1))
    else:
        card_lines = "\n".join(f"{name}{' — ' + notes if notes else ''}" for name, notes in cards)
    return (
        "Follow the official MiniMax H3 prompt writing guide below exactly: field names, section order, labels, "
        "keyframe instructions, and timing notation.\n\n"
        f"=== OFFICIAL GUIDE ===\n{official_guide(mode)}\n=== END OF GUIDE ===\n\n"
        f"Rewrite the story below into {plan['segment_count']} consecutive {mode} prompts, one per generated segment. "
        "Each prompt is a complete H3 prompt in the structure the guide gives for this mode. The video model reads each "
        "prompt alone and never sees the others, so every prompt describes only what is on screen and audible: characters "
        "with their full appearance, location, lighting, action, and sound. Nothing in a prompt may depend on footage "
        "outside it: no references to other segments or to events, styles, or people that this prompt itself has not "
        "shown yet (referring back to something shown earlier in the same prompt is fine). Keep each character's "
        "appearance and wardrobe consistent across prompts. Every event of the story must be shown in some segment. "
        + ("Split the story into consecutive beats, one beat per segment (a suggested split is listed below; adjust the "
           "boundaries only if the story needs it)" if len(set(plan["beats"])) == plan["segment_count"] else
           f"The story is shorter than {plan['segment_count']} segments: expand it into {plan['segment_count']} distinct "
           "consecutive beats along the arc stages listed below, inventing concrete actions, reactions, and dialogue that fit it")
        + ". Never repeat an earlier segment's events, and match each segment's timeline to its duration.\n\n"
        f"Segment requirements:\n{_segment_rules(plan)}\n\n"
        f"Frame: {aspect}.\n"
        f"Characters{' (reference images in order)' if mode == 'Ref2VA' else ''}:\n{card_lines or '(none named; invent consistent leads)'}\n"
        f"Story:\n{_clean(story)}\n"
        f"Preferred soundscape: {_clean(sound) or '(choose one that fits)'}\n"
        f"Preferred audience-only music: {_clean(music) or '(none; use N/A)'}\n\n"
        f"{json_request(mode, plan['segment_count'], subjects)}"
    )


def json_request(mode: str, count: int | None = None, subjects: str = "") -> str:
    """Output contract shared by the local constrained decoder and external LLMs."""
    fields = ", ".join(f'"{name}": "..."' for name in prompt_fields(mode))
    shape = f'{{"segments": [{{{fields}}}, ...]}} with exactly {count} segment objects' if count else f"{{{fields}}}"
    return f"Return only one JSON object, with no markdown fences or commentary: {shape}. Field contents:\n{field_guide(mode, subjects)}"


def extract_json(text: str) -> dict[str, Any] | None:
    text = _clean(text)
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "segments" in obj:
            return obj
    return None


# The video model sees one segment at a time, so a prompt must not point at another one.
# Unambiguous meta references; anything subtler is left to the semantic standalone check.
_CROSS_REFERENCE = re.compile(r"\b(previous|prior|last|next) (segment|clip)\b|\bsegment \d", re.I)
STANDALONE_FIELDS = ["phrases", "replacements"]
_SEPARATOR = " || "


def standalone_request(prompt: str) -> str:
    """Ask for phrases that only make sense with footage outside this prompt, plus direct replacements."""
    return (
        "You check one MiniMax H3 video prompt. The video model reads it completely alone: it is the only information "
        "about this clip. Find the phrases that only make sense to a reader who has seen other clips or footage outside "
        "this prompt, for example 'maintains its warm style' when no style was stated earlier in this prompt, 'the same "
        "girl' when no girl appeared earlier in this prompt, or 'walks back again' when no earlier walk is described "
        "here. Referring back to something described earlier inside this prompt is fine and must not be listed. For each "
        "phrase, write a replacement that states the content directly.\n\n"
        f"=== PROMPT ===\n{prompt}\n=== END ===\n\n"
        'Return only one JSON object: {"phrases": "...", "replacements": "..."}. Copy each phrase exactly as it appears; '
        f"separate items with '{_SEPARATOR.strip()}' in the same order in both fields; leave both empty when there are none."
    )


def _protected(prompt: str) -> list[tuple[int, int]]:
    """Spans of subject_definitions and retention_analysis: they cite reference labels by design."""
    heads = [(m.start(), m.group(1).lower()) for m in _HEADER.finditer(prompt)] + [(len(prompt), "")]
    return [(start, end) for (start, name), (end, _) in zip(heads, heads[1:]) if name in ("subject_definitions", "retention_analysis")]


def apply_standalone(prompt: str, phrases: str, replacements: str) -> tuple[str, list[str]]:
    """Replace each reported phrase found verbatim outside the reference sections; returns the prompt and the phrases replaced."""
    changed = []
    for phrase, replacement in zip(phrases.split(_SEPARATOR.strip()), replacements.split(_SEPARATOR.strip())):
        phrase, replacement = phrase.strip(), replacement.strip()
        spans = _protected(prompt)
        at = next((m.start() for m in re.finditer(re.escape(phrase), prompt)
                   if not any(start <= m.start() < end for start, end in spans)), None) if phrase else None
        if at is not None and replacement and replacement != phrase:
            prompt = prompt[:at] + replacement + prompt[at + len(phrase):]
            changed.append(phrase)
    return prompt, changed


_TIME = re.compile(r"\b([Aa]t) (\d\d):(\d\d(?:\.\d+)?)")


def local_times(prompt: str, index: int, frames: int) -> tuple[str, bool]:
    """Keep cut times inside segment ``index``'s own timeline.

    Times written on the whole story's timeline are shifted back by the segment offset; any other
    time past the segment's end rescales all times into it.
    """
    duration = frames / H3_FPS
    times = [int(m) * 60 + float(sec) for _, m, sec in _TIME.findall(prompt)]
    if not times or max(times) <= duration:
        return prompt, False
    offset = (index - 1) * duration
    shift, scale = (offset, 1.0) if index > 1 and offset <= min(times) and max(times) <= offset + duration else (0.0, 0.9 * duration / max(times))

    def fix(match):
        t = (int(match.group(2)) * 60 + float(match.group(3)) - shift) * scale
        return f"{match.group(1)} {int(t // 60):02d}:{t % 60:06.3f}"

    return _TIME.sub(fix, prompt), True


def normalize_llm_plan(obj: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Take only the rewritten prompts; timing and segment count stay with the plan.

    Keyframe lines are rewritten to the mode's official line, and a segment missing a required
    section keeps its template prompt.
    """
    rewritten = obj.get("segments") or []
    mode, total, frames = fallback["mode"], len(fallback["segments"]), fallback["length_frames"]
    segments, notes = [], []
    for i, item in enumerate(fallback["segments"]):
        value = rewritten[i] if i < len(rewritten) else ""
        first, last = keyframes(mode, i + 1, total)
        if isinstance(value, dict) and any(_clean(value.get(name)) for name in prompt_fields(mode)):
            raw = render_fields(mode, value, first, last, frames)
        else:
            raw = _clean(value.get("prompt", "") if isinstance(value, dict) else str(value))
        prompt, missing = conform(raw, mode, first, last, frames)
        if not raw or missing:
            notes.append(f"segment {i + 1} kept template" + (f" (missing {', '.join(missing)})" if raw else " (empty)"))
            prompt = item["prompt"]
        if _CROSS_REFERENCE.search(prompt):
            notes.append(f"segment {i + 1} refers to another segment")
        if any(prompt == earlier["prompt"] for earlier in segments):
            notes.append(f"segment {i + 1} repeats an earlier segment")
        prompt, moved = local_times(prompt, i + 1, frames)
        if moved:
            notes.append(f"segment {i + 1} times moved into its own {frames / H3_FPS:.2f}s")
        segments.append({"id": i + 1, "prompt": prompt})
    return dict(fallback, segments=segments), notes


def join_segments(plan: dict[str, Any]) -> str:
    return SEGMENT_SEPARATOR.join(s["prompt"] for s in plan["segments"])
