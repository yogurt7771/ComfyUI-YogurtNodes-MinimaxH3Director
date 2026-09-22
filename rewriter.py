"""Rewrite requests into MiniMax H3 prompts with a local text-generation model and JSON-constrained decoding.

The JSON around each string field is written by code; inside a string, tokens that would break the
string are masked out. The model therefore always yields every field, and the official prompt
layout is rendered from those fields.
"""

import json

import comfy.model_prefetch
import torch
from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerate

from .planning import H3_FPS
from .story_planning import FIELD_PREFIX, MODES, alignment_line, json_request, official_guide, prompt_fields, render_fields, secs_to_frames

_CATEGORY = "Yogurt/MiniMax H3 Director"
_KEYFRAMES = {"I2VA": (True, False), "FL2VA": (True, True), "L2VA": (False, True)}
_ESCAPES = b'n"\\/t'
# Byte tables per tokenizer vocabulary, built once and valid for the tokenizer's lifetime.
_TABLES = {}


def _scan(data, escaped):
    """(escape state, None) for plain string content, (False, i) when an unescaped quote at i closes the
    string, or None when ``data`` cannot appear inside a JSON string."""
    for i, byte in enumerate(data):
        if escaped:
            if byte not in _ESCAPES:
                return None
            escaped = False
        elif byte == 0x5C:
            escaped = True
        elif byte == 0x22:
            return False, i
        elif byte < 0x20:
            return None
    return escaped, None


def _tables(tokenizer):
    """Per escape state: {token: next state} for string content, and {token: quote offset} for tokens that close it."""
    key = id(tokenizer)
    if key not in _TABLES:
        decoder = tokenizer.byte_decoder
        raw = {i: bytes(decoder[c] for c in token) for token, i in tokenizer.encoder.items()}
        content, closing = ({}, {}), ({}, {})
        for i, data in raw.items():
            for state in (False, True):
                scanned = _scan(data, state) if data else None
                if scanned is not None:
                    after, quote = scanned
                    if quote is None:
                        content[state][i] = after
                    else:
                        closing[state][i] = quote
        _TABLES[key] = raw, content, closing
    return _TABLES[key]


def json_pieces(objects, array_key=None):
    """Literal byte runs around string fields; each run after a field starts with that field's closing quote.

    A field listed in FIELD_PREFIX opens with its fixed prefix as part of the literal.
    """
    pieces = []
    text = (f'{{"{array_key}": [' if array_key else "") + "{"
    for index, keys in enumerate(objects):
        for position, name in enumerate(keys):
            if position:
                text = '", '
            elif index:
                text = '"}, {'
            pieces += [(text + f'"{name}": "' + FIELD_PREFIX.get(name, "")).encode(), (index, name)]
    pieces.append(('"}' + ("]}" if array_key else "")).encode())
    return pieces


class JsonFields:
    """Per-step token rule: force the pending literal, or keep sampling inside a JSON string."""

    def __init__(self, tokenizer, pieces, stop_token):
        self.tokenizer = tokenizer
        self.raw, self.content, self.closing = _tables(tokenizer)
        self.pieces = pieces
        self.stop_token = stop_token
        self.index = 0
        self.pending = pieces[0]
        self.current = bytearray()
        self.escaped = False
        self.values = {}
        self.masks = {}

    @property
    def done(self):
        return self.index >= len(self.pieces)

    def _closes(self, token):
        """Quote offset when ``token`` ends the current string and starts the next literal, else None."""
        quote = self.closing[self.escaped].get(token)
        if quote is not None and self.pieces[self.index + 1].startswith(self.raw[token][quote:]):
            return quote
        return None

    def blocked(self, size):
        """CPU mask of the tokens this step must not produce; a forced token leaves only itself open."""
        if self.done or self.pending:
            forced = self.stop_token if self.done else self.tokenizer.encode(self.pending.decode(), add_special_tokens=False)[0]
            blocked = torch.ones(size, dtype=torch.bool)
            blocked[forced] = False
            return blocked
        # Cached masks stay on the CPU: the decode loop recycles device memory allocated inside a step.
        key = (self.escaped, size)
        if key not in self.masks:
            mask = torch.ones(size, dtype=torch.bool)
            mask[[i for i in self.content[self.escaped] if i < size]] = False
            self.masks[key] = mask
        blocked = self.masks[key].clone()
        blocked[[i for i in self.closing[self.escaped] if i < size and self._closes(i) is not None]] = False
        return blocked

    def advance(self, token):
        if self.done:
            return
        data = self.raw[token]
        if self.pending:
            self.pending = self.pending[len(data):]
            if not self.pending:
                self.index += 1
            return
        quote = self._closes(token)
        if quote is None:
            self.current += data
            self.escaped = self.content[self.escaped][token]
            return
        self.current += data[:quote]
        self._close()
        self.index += 1
        self.pending = self.pieces[self.index][len(data) - quote:]
        if not self.pending:
            self.index += 1

    def _close(self):
        text = bytes(self.current).decode("utf-8", errors="replace")
        self.values[self.pieces[self.index]] = json.loads(f'"{text}"')
        self.current = bytearray()
        self.escaped = False

    def finish(self):
        """Keep a string cut off by max_length; report whether the JSON closed."""
        if not self.done and not self.pending and self.current:
            if self.escaped:
                self.current = self.current[:-1]
            self._close()
        return self.done


def sampling_args(sampling_mode):
    return dict(do_sample=sampling_mode.get("sampling_mode") == "on", temperature=sampling_mode.get("temperature", 1.0),
                top_k=sampling_mode.get("top_k", 50), top_p=sampling_mode.get("top_p", 1.0), min_p=sampling_mode.get("min_p", 0.0),
                seed=sampling_mode.get("seed", None), repetition_penalty=sampling_mode.get("repetition_penalty", 1.0),
                presence_penalty=sampling_mode.get("presence_penalty", 0.0))


def constrained_json(clip, request, objects, max_length, sampling_mode, array_key=None, image=None, video=None, audio=None):
    """Generate string fields for ``objects`` (lists of keys); returns ({(object, key): text}, closed)."""
    model = getattr(clip.cond_stage_model, clip.cond_stage_model.clip)
    transformer = getattr(model, "transformer", None)
    tokenizer = getattr(getattr(clip.tokenizer, clip.tokenizer.clip), "tokenizer", None)
    if not hasattr(transformer, "sample_token") or not hasattr(tokenizer, "byte_decoder"):
        raise ValueError("JSON 约束解码需要 Qwen 系列等使用字节级 BPE 的文本生成模型。")
    rule = JsonFields(tokenizer, json_pieces(objects, array_key), transformer.model.config.stop_tokens[0])
    original = transformer.sample_token

    def sample_token(logits, *args, **kwargs):
        # A forced token goes through the model's own sampler with a one-token mask, so every step runs the same ops.
        blocked = rule.blocked(logits.shape[-1]).to(logits.device)
        token = original(logits.masked_fill(blocked, float("-inf")), *args, **kwargs)
        rule.advance(int(token.reshape(-1)[0]))
        return token

    tokens = clip.tokenize(request, image=image, video=video, audio=audio, min_length=1, thinking=False)
    # ComfyUI's generate loop exposes no logits hook; wrap this instance's sampler for the call only.
    transformer.sample_token = sample_token
    try:
        clip.generate(tokens, max_length=max_length, mtp=False, **sampling_args(sampling_mode))
    finally:
        del transformer.sample_token
        # Decode graphs are captured against this call's KV cache and ComfyUI drops them only after a node
        # finishes; a node that generates more than once must drop them between calls.
        comfy.model_prefetch.cleanup_prefetch_queues()
    if rule.index == 0 and rule.pending == rule.pieces[0]:
        raise ValueError("该文本模型的生成流程没有经过 sample_token，无法做 JSON 约束解码。")
    closed = rule.finish()
    return {key: FIELD_PREFIX.get(key[1], "") + text for key, text in rule.values.items()}, closed


def rewrite_request(request, mode, duration_seconds, references=""):
    frames = secs_to_frames(duration_seconds)
    first, last = _KEYFRAMES.get(mode, (False, False))
    rules = [f"Mode: {mode}.", f"Target duration: {frames / H3_FPS:.2f} seconds; the timeline must end within it."]
    if first or last:
        rules.append(f"Keyframes (the alignment line is added automatically): {alignment_line(first, last, frames, 'N')}")
    rules.append(f"Reference assets, in order: {references.strip() or '(none)'}")
    return (
        "Follow the official MiniMax H3 prompt writing guide below exactly: labels, shot and timing notation, "
        "speaker IDs, and dialogue blocks.\n\n"
        f"=== OFFICIAL GUIDE ===\n{official_guide(mode)}\n=== END OF GUIDE ===\n\n"
        "Rewrite the user request into one H3 prompt.\n" + "\n".join(rules) + "\n\n"
        f"User request:\n{request.strip()}\n\n{json_request(mode)}"
    )


def textgen_inputs(*names):
    by_id = {item.id: item for item in TextGenerate.define_schema().inputs}
    return [by_id[name] for name in names]


class YogurtH3PromptRewriter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YogurtH3PromptRewriter",
            display_name="Yogurt H3 提示词改写",
            category=_CATEGORY,
            description="按 MiniMax 官方 h3-prompt-writing 指南，用本地文本模型（Qwen 系列）把粗略需求改写成一段 H3 提示词。"
                        "JSON 约束解码保证每个段落都生成，段落标题、顺序和关键帧对齐句由节点写入。",
            search_aliases=["prompt enhance", "LLM", "MiniMax H3"],
            inputs=[*textgen_inputs("clip", "prompt"),
                    io.Combo.Input("mode", options=MODES, default="Ref2VA"),
                    io.Float.Input("duration_seconds", default=8.0, min=4.0, max=15.0, step=0.1),
                    io.String.Input("references", multiline=True, default="", optional=True,
                                    tooltip="每行一个参考素材，按顺序对应 <Picture 1>/<Video 1>/<Audio 1>，例如：Picture 1: 女主角正面照。"),
                    *textgen_inputs("image", "video", "audio"),
                    io.Int.Input("max_length", default=2048, min=64, max=32768),
                    *textgen_inputs("sampling_mode")],
            outputs=[io.String.Output(display_name="H3 prompt"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, clip, prompt, mode, duration_seconds, max_length, sampling_mode, references="", image=None,
                video=None, audio=None):
        fields = prompt_fields(mode)
        values, closed = constrained_json(clip, rewrite_request(prompt, mode, duration_seconds, references), [fields],
                                          max_length, sampling_mode, image=image, video=video, audio=audio)
        first, last = _KEYFRAMES.get(mode, (False, False))
        text = render_fields(mode, {name: values.get((0, name), "") for name in fields}, first, last,
                             secs_to_frames(duration_seconds))
        empty = [name for name in fields if not values.get((0, name), "").strip()]
        report = "ok" if closed and not empty else (
            f"max_length={max_length} 用完，{', '.join(empty) or '最后一段'} 未写完；请调大 max_length。")
        return io.NodeOutput(text, report)
