import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import video_latent_t

from test_continuity_runtime import PLUGIN, PLUGIN_PATH

PLUGIN_PATH_MISSING = PLUGIN_PATH / "missing-guide"

STORY = sys.modules[PLUGIN.__name__ + ".story"]
REWRITER = sys.modules[PLUGIN.__name__ + ".rewriter"]
QWEN = __import__("comfy.text_encoders.qwen35", fromlist=["tokenizer"])
NODES = sys.modules[PLUGIN.__name__ + ".nodes"]
RENDER = sys.modules[PLUGIN.__name__ + ".render"]
CONTINUATION = sys.modules[PLUGIN.__name__ + ".continuation"]


def av_latent(frames, value=0.0, audio_value=0.0):
    t = video_latent_t(frames)
    audio_t = round(frames / 24 * 40)
    return {"samples": NestedTensor((torch.full((1, 24, t, 4, 6), value), torch.full((1, 32, 2, audio_t), audio_value)))}


def story_plan(**overrides):
    values = dict(story="雨夜巷口", mode="Ref2VA", segment_count=3, seconds_per_segment=8.0, overlap_frames=48,
                  aspect_ratio="16:9", megapixels=1.03, ref_image_size="match", continue_audio=True,
                  global_soundscape="", global_music="", card_1_name="A", cards={"card_0": torch.zeros(1, 8, 8, 3)})
    values.update(overrides)
    return STORY.YogurtH3AutoStoryPlan.execute(**values).result[1]


class FakeTransformer:
    """Greedy stand-in for ComfyUI's llama generate loop: it wants to write ``target`` byte for byte."""

    class model:
        class config:
            stop_tokens = [248044, 248046]

    def sample_token(self, logits, *args, **kwargs):
        return torch.argmax(logits, dim=-1, keepdim=True)


class FakeTextClip:
    tokenizer_holder = None

    def __init__(self, target):
        if FakeTextClip.tokenizer_holder is None:
            FakeTextClip.tokenizer_holder = QWEN.tokenizer("qwen35_4b")()
        self.tokenizer = FakeTextClip.tokenizer_holder
        self.hf = getattr(self.tokenizer, self.tokenizer.clip).tokenizer
        self.transformer = FakeTransformer()
        self.cond_stage_model = SimpleNamespace(clip="inner", inner=SimpleNamespace(transformer=self.transformer))
        self.target = target.encode()
        self.requests = []

    def tokenize(self, prompt, **kwargs):
        self.requests.append(prompt)
        return prompt

    def generate(self, tokens, max_length, **kwargs):
        written = b""
        raw = _TABLES_FOR(self.hf)
        for _ in range(max_length):
            rest = self.target[len(written):] if self.target.startswith(written) else b"x"
            wanted = self.hf.encode(rest.decode(errors="ignore") or "x", add_special_tokens=False)[0]
            logits = torch.zeros(1, 248320)
            logits[0, wanted] = 10.0
            token = int(self.transformer.sample_token(logits, 1.0, 0, 1.0, 0.0, 1.0, [], None, do_sample=False)[0, 0])
            if token in FakeTransformer.model.config.stop_tokens:
                break
            written += raw[token]
        return []


def _TABLES_FOR(hf):
    return REWRITER._tables(hf)[0]


class StoryPlanTests(unittest.TestCase):
    def test_overlap_is_aligned_before_prompts_are_written(self):
        plan = story_plan()
        self.assertEqual(plan["story"]["length_frames"], 192)
        self.assertEqual([s["overlap_frames"] for s in plan["segments"]], [0, 39, 39])
        for segment in plan["segments"]:
            self.assertNotRegex(segment["prompt"], r"(?i)previous|left off|same characters|segment \d")
        for segment in plan["segments"]:
            self.assertNotIn("overlap", segment["prompt"])
            self.assertNotIn("1.62", segment["prompt"])
        self.assertEqual(plan["target_frames"], 3 * 192 - 2 * 39)
        self.assertEqual(plan["audio"], "generate")

    def test_story_is_split_into_distinct_consecutive_beats(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        story = "两名角色按角色卡外貌出场。雨夜巷口短暂对峙，随后一起走向远处的暖色灯牌。不要换脸、不要换服装。对白保留中文。"
        self.assertEqual(planning.story_beats(story, 3), ["两名角色按角色卡外貌出场。", "雨夜巷口短暂对峙", "随后一起走向远处的暖色灯牌。"])
        self.assertEqual(planning.story_beats("两人保持距离。一起离开。", 2), ["两人保持距离。", "一起离开。"])
        beats = planning.story_beats("甲进门。乙抬头。两人握手。一起离开。", 3)
        self.assertEqual(len(set(beats)), 3)
        self.assertEqual("".join(beats), "甲进门。乙抬头。两人握手。一起离开。")
        self.assertEqual(planning.story_beats("雨夜巷口短暂对峙，随后一起走向灯牌", 2), ["雨夜巷口短暂对峙", "随后一起走向灯牌"])
        self.assertEqual(planning.story_beats("对峙", 2), ["对峙", "对峙"])

    def test_template_segments_each_play_their_own_beat(self):
        plan = story_plan(story="两人在巷口相遇。他们短暂对峙。最后一起走向灯牌。")
        prompts = [s["prompt"] for s in plan["segments"]]
        for prompt, beat in zip(prompts, ("两人在巷口相遇。", "他们短暂对峙。", "最后一起走向灯牌。")):
            self.assertIn(beat, prompt)
        self.assertNotIn("他们短暂对峙", prompts[0])
        self.assertIn("It must show: 他们短暂对峙。", plan["story"]["request"])

    def test_short_story_is_spread_over_the_arc(self):
        plan = story_plan(story="宇航员在月球上发现了一朵花。", segment_count=6)
        request = plan["story"]["request"]
        self.assertIn("The story is shorter than 6 segments: expand it into 6 distinct consecutive beats", request)
        self.assertIn("- Segment 1: 8.00 seconds on its own timeline, 00:00.000 to 00:08.000. Arc stage: setup; invent concrete opening events", request)
        self.assertIn("Arc stage: resolution; invent new concrete events for this stage that come after segment 5's events", request)
        prompts = [s["prompt"] for s in plan["segments"]]
        self.assertEqual(len(set(prompts)), 6)
        self.assertIn("play out the climax of the story", prompts[3])

    def test_later_template_segment_opens_on_the_previous_beat(self):
        plan = story_plan(story="两人在巷口相遇。他们短暂对峙。最后一起走向灯牌。")
        self.assertIn("the shot opens on the closing moment of this: 两人在巷口相遇。", plan["segments"][1]["prompt"])
        self.assertNotIn("closing moment", plan["segments"][0]["prompt"])
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        _, notes = planning.normalize_llm_plan({"segments": [{"prompt": plan["segments"][0]["prompt"] + " It picks up from the previous segment."}]},
                                               plan["story"])
        self.assertIn("segment 1 refers to another segment", notes)

    def test_resolution_follows_aspect_ratio_and_megapixels(self):
        plan = story_plan()
        self.assertEqual((plan["width"], plan["height"]), (1344, 768))
        plan = story_plan(aspect_ratio="9:16", megapixels=0.41)
        self.assertEqual((plan["width"], plan["height"]), (480, 864))

    def test_parse_replaces_prompts_but_keeps_segment_timing(self):
        fallback = story_plan()
        self.assertIn("used_llm=false", STORY.YogurtH3AutoStoryParse.execute(fallback, "not json").result[-1])
        good = ("subject_definitions:\n<Subject 1> is A in <Picture 1>.\n\nsummary:\n[reference generation] x\n\n"
                "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - x\n\ndetailed_description:\n"
                "The target video is cinematic.\n[Shot 1] x\n\noverall_soundscape:\nrain\n\nnon_diegetic_music:\nN/A")
        wrong_mode = "integrated_multimodal_description: [Shot 1] x\n\noverall_soundscape: rain\n\nnon_diegetic_music: N/A"
        llm = json.dumps({"segments": [{"prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> "
                                                  "(from [Shot 1]) is fully referenced.\n\n" + good}, wrong_mode]})
        result = STORY.YogurtH3AutoStoryParse.execute(fallback, "<think>plan</think>```json\n" + llm + "\n```").result
        prompts = [s["prompt"] for s in result[0]["segments"]]
        self.assertEqual(prompts[0], good)
        self.assertEqual(prompts[1:], [s["prompt"] for s in fallback["segments"][1:]])
        self.assertIn("segment 2 kept template (missing subject_definitions:", result[-1])
        self.assertIn("segment 3 kept template (empty)", result[-1])
        self.assertEqual(result[2], 3)
        self.assertEqual(result[0]["target_frames"], fallback["target_frames"])
        self.assertIs(result[0]["inputs"]["images"][0], fallback["inputs"]["images"][0])

    def test_ref2va_template_uses_official_six_sections(self):
        prompt = story_plan()["segments"][0]["prompt"]
        sections = ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                    "overall_soundscape:", "non_diegetic_music:"]
        self.assertEqual([prompt.index(name) for name in sections], sorted(prompt.index(name) for name in sections))
        self.assertIn("<Subject 1> is A in <Picture 1>.", prompt)
        self.assertIn("[reference generation]", prompt)
        self.assertIn("<Subject 1> (appears in [Shot 1]): fully_preserved", prompt)
        self.assertIn("The target video is live-action and cinematic.\n[Shot 1]", prompt)
        for invented in ("aspect:", "story_spine", "--- segment", "mode:"):
            self.assertNotIn(invented, prompt)

    def test_keyframe_modes_use_official_alignment_lines(self):
        plan = story_plan(mode="FL2VA", segment_count=1, cards={"card_0": torch.zeros(1, 8, 8, 3), "card_1": torch.zeros(1, 8, 8, 3)})
        self.assertTrue(plan["segments"][0]["prompt"].startswith(
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
            "mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.\n\n"
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic,"))
        plan = story_plan(mode="L2VA")
        self.assertTrue(plan["segments"][0]["prompt"].startswith("integrated_multimodal_description:"))
        self.assertTrue(plan["segments"][2]["prompt"].startswith(
            "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 8.00-second mark"))
        self.assertIn("non_diegetic_music: N/A", plan["segments"][2]["prompt"])

    def test_llm_request_carries_official_guide_and_segment_rules(self):
        request = STORY.YogurtH3AutoStoryPlan.execute(**dict(
            story="雨夜巷口", mode="I2VA", segment_count=2, seconds_per_segment=8.0, overlap_frames=39, aspect_ratio="9:16",
            megapixels=1.03, ref_image_size="match", continue_audio=True, global_soundscape="", global_music="",
            card_1_name="A", cards={"card_0": torch.zeros(1, 8, 8, 3)})).result[0]
        self.assertIn("# Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)", request)
        self.assertNotIn("# Full-Reference Mode Rewrite Output Format Guide", request)
        self.assertIn("- Segment 1: 8.00 seconds on its own timeline, 00:00.000 to 00:08.000. It opens on the first keyframe <Picture 1>.", request)
        self.assertIn("- Segment 2: 8.00 seconds on its own timeline, 00:00.000 to 00:08.000 (it does not start at 00:08.000). Its [Shot 1] opens on the same visual state that ends segment 1, described again in full.", request)
        self.assertIn("The video model reads each prompt alone and never sees the others", request)
        self.assertNotIn("restage", request)
        self.assertIn("Frame: 9:16.", request)

    def test_frame_modes_require_cards_and_anchor_first_segment(self):
        with self.assertRaises(ValueError):
            story_plan(mode="FL2VA")
        plan = story_plan(mode="I2VA", overlap_frames=0)
        self.assertEqual([s["mode"] for s in plan["segments"]], ["i2v"] * 3)
        self.assertEqual(plan["segments"][0]["first_frame"], 0)
        self.assertNotIn("first_frame", plan["segments"][1])

    def test_missing_official_guide_fails_clearly(self):
        with patch.object(sys.modules[PLUGIN.__name__ + ".story_planning"], "GUIDE_DIR", PLUGIN_PATH_MISSING):
            with self.assertRaises(FileNotFoundError):
                story_plan()

    def test_refine_canvas_keeps_story_aspect(self):
        refine = RENDER.YogurtH3RefineSettings.execute(2.0, "lanczos", True, 8, 0.25)[0]
        plan = story_plan(megapixels=0.41, refine=refine)
        self.assertEqual((plan["width"], plan["height"]), (864, 480))
        self.assertEqual((plan["refine"]["width"], plan["refine"]["height"]), (1888, 1056))


REF_FIELDS = dict(subject_definitions="<Subject 1> is A in <Picture 1>.\n<Subject 2> is B in <Picture 2>.",
                  summary="[reference generation] A and B argue.",
                  retention_analysis="<Subject 1> (appears in [Shot 1]): fully_preserved - hair.",
                  detailed_description='The target video is cinematic.\n[Shot 1] A (S1) says: <d>[Chinese] 你好。</d> A sign reads "营业中".',
                  overall_soundscape="Rain.", non_diegetic_music="N/A")


class RewriterTests(unittest.TestCase):
    def test_request_embeds_mode_guide_duration_and_json_contract(self):
        request = REWRITER.rewrite_request("she opens the umbrella", "L2VA", 6.0, "Picture 1: last frame")
        self.assertIn("# Video Prompt Writing Guide", request)
        self.assertIn("Target duration: 6.58 seconds", request)
        self.assertIn("aligns with the 6.58-second mark of the target video.", request)
        self.assertIn("Reference assets, in order: Picture 1: last frame", request)
        self.assertIn('{"integrated_multimodal_description": "...", "overall_soundscape": "...", "non_diegetic_music": "..."}', request)
        self.assertIn("# Full-Reference Mode Rewrite Output Format Guide", REWRITER.rewrite_request("x", "Ref2VA", 8.0))

    def test_masks_keep_json_strings_valid(self):
        hf = FakeTextClip("").hf
        raw, content, closing = REWRITER._tables(hf)
        ids = {b: i for i, b in raw.items() if b in (b"\n", b"\\n", b'\\"', b'"', b'",', b"\xe6\x88\x91", b"\\")}
        self.assertNotIn(ids[b"\n"], content[False])
        self.assertNotIn(ids[b'"'], content[False])
        self.assertEqual({content[False][ids[b]] for b in (b"\\n", b'\\"', b"\xe6\x88\x91")}, {False})
        self.assertIs(content[False][ids[b"\\"]], True)
        self.assertEqual((closing[False][ids[b'"']], closing[False][ids[b'",']]), (0, 0))
        self.assertNotIn(ids[b'"'], closing[True])
        self.assertIs(content[True][ids[b'"']], False)
        self.assertEqual(REWRITER.json_pieces([["a", "b"], ["a", "b"]], "segments"),
                         [b'{"segments": [{"a": "', (0, "a"), b'", "b": "', (0, "b"), b'"}, {"a": "', (1, "a"),
                          b'", "b": "', (1, "b"), b'"}]}'])
        self.assertEqual(REWRITER.json_pieces([["a"]]), [b'{"a": "', (0, "a"), b'"}'])
        self.assertEqual(REWRITER.json_pieces([["integrated_multimodal_description"]])[0],
                         b'{"integrated_multimodal_description": "[Shot 1] ')

    def test_constrained_decoding_returns_every_field_and_renders_official_layout(self):
        clip = FakeTextClip(json.dumps(REF_FIELDS, ensure_ascii=False))
        text, report = REWRITER.YogurtH3PromptRewriter.execute(clip, "rain", "Ref2VA", 8.0, 2048, {"sampling_mode": "off"}).result
        self.assertEqual(report, "ok")
        self.assertEqual(text, "\n\n".join(f"{name}:\n{value}" for name, value in REF_FIELDS.items()))
        self.assertIn("User request:\nrain", clip.requests[0])
        self.assertNotIn("sample_token", vars(clip.transformer))

    def test_markdown_and_prose_cannot_escape_the_json(self):
        clip = FakeTextClip("```json\n{\"integrated_multimodal_description\": \"x\"}```")
        text, report = REWRITER.YogurtH3PromptRewriter.execute(clip, "rain", "I2VA", 8.0, 64, {"sampling_mode": "off"}).result
        self.assertTrue(text.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) "
                                        "is fully referenced.\n\nintegrated_multimodal_description: "))
        self.assertIn("overall_soundscape:", text)
        self.assertIn("max_length=64", report)

    def test_standalone_check_replaces_reported_phrases_only(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        prompt = "The target video maintains its warm style. [Shot 1] The same girl waits. [Shot 2] At 00:04.000, the same girl smiles."
        found = json.dumps({"phrases": "maintains its warm style || The same girl waits || not in prompt",
                            "replacements": "uses a warm style || A girl in a yellow raincoat waits || x"})
        values, closed = REWRITER.constrained_json(FakeTextClip(found), planning.standalone_request(prompt),
                                                   [planning.STANDALONE_FIELDS], 1024, {"sampling_mode": "off"})
        self.assertTrue(closed)
        fixed, changed = planning.apply_standalone(prompt, values[(0, "phrases")], values[(0, "replacements")])
        self.assertEqual(fixed, "The target video uses a warm style. [Shot 1] A girl in a yellow raincoat waits. "
                                "[Shot 2] At 00:04.000, the same girl smiles.")
        self.assertEqual(changed, ["maintains its warm style", "The same girl waits"])
        self.assertIn("Referring back to something described earlier inside this prompt is fine", planning.standalone_request(prompt))
        ref = ("subject_definitions:\n<Subject 1> is A in <Picture 1>, with red hair.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): "
               "fully_preserved - red hair.\n\ndetailed_description:\nThe target video maintains its style. A has red hair.")
        fixed, changed = planning.apply_standalone(ref, "red hair || maintains its style || A has red hair", "blue hair || uses a warm style || A has red hair")
        self.assertEqual(changed, ["red hair", "maintains its style"])
        self.assertIn("with red hair.", fixed)
        self.assertIn("fully_preserved - red hair.", fixed)
        self.assertIn("The target video uses a warm style. A has blue hair.", fixed)

    def test_story_rewrite_fills_every_segment(self):
        plan = story_plan()
        segments = [dict(REF_FIELDS, summary=f"[reference generation] Beat {i}.") for i in range(3)]
        target = json.dumps({"segments": segments}, ensure_ascii=False)
        new_plan, prompts, report = STORY.YogurtH3StoryRewrite.execute(FakeTextClip(target), plan, 8192, {"sampling_mode": "off"},
                                                                       standalone_check=False).result
        self.assertEqual(report, "ok")
        self.assertEqual([s["prompt"] for s in new_plan["segments"]], [s["prompt"] for s in new_plan["story"]["segments"]])
        self.assertTrue(all(s["prompt"].startswith("subject_definitions:\n<Subject 1> is A") for s in new_plan["segments"]))
        self.assertEqual(new_plan["target_frames"], plan["target_frames"])
        self.assertIn('"segments": [{"subject_definitions": "...", "summary": "..."', plan["story"]["request"])

    def test_fields_drop_repeated_alignment_headers_and_sections(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        align = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        value = f"{align}\n\nintegrated_multimodal_description: [Shot 1] a man stands.\n\noverall_soundscape: x.\n\nnon_diegetic_music: y."
        text = planning.render_fields("I2VA", {"integrated_multimodal_description": value, "overall_soundscape": "fabric.",
                                               "non_diegetic_music": "cello."}, True, False, 158)
        self.assertEqual(text, f"{align}\n\nintegrated_multimodal_description: [Shot 1] a man stands.\n\n"
                               "overall_soundscape: fabric.\n\nnon_diegetic_music: cello.")

    def test_shots_restart_per_segment(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        text = ("subject_definitions:\nx\n\nsummary:\nx\n\nretention_analysis:\n<Subject 1> (appears in [Shot 2], [Shot 3]): kept\n\n"
                "detailed_description:\nStyle.\n[Shot 2] At 00:08.000, a.\n[Shot 3] At 00:12.500, b.\n\noverall_soundscape:\nr\n\n"
                "non_diegetic_music:\nN/A")
        fixed, missing = planning.conform(text, "Ref2VA", False, False, 192)
        self.assertEqual(missing, [])
        self.assertIn("(appears in [Shot 1], [Shot 2])", fixed)
        self.assertIn("Style.\n[Shot 1] a.\n[Shot 2] At 00:12.500, b.", fixed)
        plan = story_plan()
        story, notes = planning.normalize_llm_plan({"segments": [{"prompt": fixed}]}, plan["story"])
        self.assertIn("segment 1 times moved into its own 8.00s", notes)
        self.assertIn("[Shot 2] At 00:07.200, b.", story["segments"][0]["prompt"])
        shifted, moved = planning.local_times("[Shot 1] a, the camera cuts at 00:09.000. [Shot 2] At 00:12.000, b.", 2, 192)
        self.assertTrue(moved)
        self.assertEqual(shifted, "[Shot 1] a, the camera cuts at 00:01.000. [Shot 2] At 00:04.000, b.")
        self.assertEqual(planning.local_times("[Shot 2] At 00:03.000, b.", 2, 192), ("[Shot 2] At 00:03.000, b.", False))

    def test_decorated_headers_are_normalized(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        decorated = ("**Subject_Definitions:**\n<Subject 1> is A in <Picture 1>.\n\n### summary：\n[reference generation] x\n\n"
                     "- retention_analysis:\nr\n\n**detailed_description**:\nThe target video is cinematic.\n[Shot 1] x\n\n"
                     "overall_soundscape:\nrain\n\n**non_diegetic_music:** N/A")
        text, missing = planning.conform(decorated, "Ref2VA", False, False, 192)
        self.assertEqual(missing, [])
        self.assertTrue(text.startswith("subject_definitions:\n<Subject 1> is A in <Picture 1>.\n\nsummary:\n[reference generation]"))
        self.assertTrue(text.endswith("non_diegetic_music: N/A"))

    def test_conform_replaces_keyframe_line_for_the_mode(self):
        planning = sys.modules[PLUGIN.__name__ + ".story_planning"]
        stray = ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 0.00-second "
                 "mark of the target video; <Picture 2> (from [Shot 1]) aligns with the 10.12-second mark.\n\nsubject_definitions:\nx")
        self.assertEqual(planning.conform(stray, "Ref2VA", False, False, 243)[0], "subject_definitions:\nx")
        ordered, missing = planning.conform("overall_soundscape: rain\n\nnon_diegetic_music: N/A\n\n"
                                            "integrated_multimodal_description: [Shot 1] At 00:00.000, the shot opens", "T2VA",
                                            False, False, 192)
        self.assertEqual((ordered, missing), ("integrated_multimodal_description: [Shot 1] the shot opens\n\n"
                                              "overall_soundscape: rain\n\nnon_diegetic_music: N/A", []))
        text, _ = planning.conform("integrated_multimodal_description: [Shot 1] a [Shot 2] At 00:03.000, b", "L2VA", False, True, 192)
        self.assertTrue(text.startswith("How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) "
                                        "aligns with the 8.00-second mark of the target video.\n\nintegrated_multimodal_description:"))


class CharacterReplacePromptTests(unittest.TestCase):
    def test_prompt_follows_official_video_editing_format(self):
        prompt = NODES.YogurtH3BuildCharacterReplacePrompt.execute("short red hair").result[0]
        self.assertIn("[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>.", prompt)
        self.assertIn("<Audio 1>: fully_copy", prompt)
        self.assertIn("<Subject 1> (appears in [Shot 1]): attribute_transfer", prompt)
        self.assertNotIn("00:00.000", prompt)
        self.assertIn("of <Video 1>.\n[Shot 1] The shot follows <Video 1>", prompt)
        self.assertIn(NODES.TARGET_PERFORMER, prompt)


class ContinuationTests(unittest.TestCase):
    def test_carry_copies_tail_and_masks_prefix(self):
        previous = av_latent(192, value=1.0, audio_value=2.0)
        latent, tokens = CONTINUATION.carry_latent_tail(av_latent(192), previous, 39, True)
        video, audio = latent["samples"].unbind()
        video_mask, audio_mask = latent["noise_mask"].unbind()
        self.assertEqual(tokens, video_latent_t(39))
        self.assertTrue(torch.equal(video[:, :, :tokens], torch.ones_like(video[:, :, :tokens])))
        self.assertTrue((video[:, :, tokens:] == 0).all())
        self.assertTrue((video_mask[:, :, :tokens] == 0).all() and (video_mask[:, :, tokens:] == 1).all())
        audio_tokens = round(39 * 40 / 24)
        self.assertTrue((audio[..., :audio_tokens] == 2).all() and (audio[..., audio_tokens:] == 0).all())
        ramp = audio_mask[0, 0, 0, :audio_tokens]
        self.assertTrue((ramp[:-8] == 0).all())
        self.assertAlmostEqual(float(ramp[-1]), 1.0, places=6)
        self.assertTrue((ramp[-8:].diff() > 0).all())

    def test_carry_without_audio_leaves_audio_generated(self):
        latent, _ = CONTINUATION.carry_latent_tail(av_latent(192), av_latent(192, 1.0, 2.0), 22, False)
        self.assertTrue((latent["samples"].unbind()[1] == 0).all())
        self.assertTrue((latent["noise_mask"].unbind()[1] == 1).all())

    def test_carry_rejects_overlap_as_long_as_segment(self):
        with self.assertRaises(ValueError):
            CONTINUATION.carry_latent_tail(av_latent(39), av_latent(39), 39, True)

    def test_drift_mask_follows_schedule(self):
        latent, tokens = CONTINUATION.carry_latent_tail(av_latent(192), av_latent(192), 39, True)
        video, audio = latent["samples"].unbind()
        sigmas = torch.tensor([1.0, 0.8, 0.4, 0.0])
        state = CONTINUATION.DriftControlState(video.shape, audio.shape, sigmas, tokens)
        packed = torch.cat((torch.ones(1, 1, video.numel()), torch.full((1, 1, audio.numel()), 0.5)), dim=-1)
        state.denoise_mask_function(torch.tensor([0.8]), packed)
        prefix = state.video_mask[0, 0, :tokens, 0, 0]
        self.assertAlmostEqual(float(prefix[0]), 0.5, places=2)
        self.assertEqual(float(prefix[-1]), 0.0)
        self.assertTrue((state.video_mask[0, 0, tokens:] == 1).all())
        self.assertTrue((state.audio_mask == 0.5).all())


class JoinTests(unittest.TestCase):
    def audio(self, frames, value):
        return {"waveform": torch.full((1, 2, frames * 2000), float(value)), "sample_rate": 48000}

    def test_hard_join_drops_incoming_head(self):
        images, audio = RENDER.YogurtH3JoinSegments.execute(
            torch.ones(56, 2, 2, 3), 39, False, self.audio(56, 1), torch.zeros(56, 2, 2, 3), self.audio(56, 0)).result
        self.assertEqual(images.shape[0], 73)
        self.assertEqual(audio["waveform"].shape[-1], 73 * 2000)
        self.assertEqual(float(audio["waveform"][0, 0, 56 * 2000 - 1]), 0.0)

    def test_soft_join_keeps_incoming_seam_audio(self):
        _, audio = RENDER.YogurtH3JoinSegments.execute(
            torch.ones(56, 2, 2, 3), 39, True, self.audio(56, 1), torch.zeros(56, 2, 2, 3), self.audio(56, 0)).result
        self.assertEqual(audio["waveform"].shape[-1], 73 * 2000)
        self.assertEqual(float(audio["waveform"][0, 0, 17 * 2000]), 1.0)


class StorySamplerTests(unittest.TestCase):
    def expand(self, plan):
        with patch.object(RENDER.YogurtH3LongVideoSampler, "hidden", SimpleNamespace(prompt=None, unique_id="test")):
            output = RENDER.YogurtH3LongVideoSampler.execute(model="MODEL", clip="CLIP", vae="VAE", audio_vae="AUDIO_VAE",
                plan=plan, sampler="SAMPLER", sigmas=torch.tensor([1.0, 0.5, 0.0]), seed=7)
        return output.expand, output.result[3]

    def test_story_plan_chains_segments_and_uses_generated_audio(self):
        graph, report = self.expand(story_plan())
        segments = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == "YogurtH3CachedSegment"]
        self.assertEqual(len(segments), 3)
        self.assertNotIn("previous", segments[0][1])
        self.assertEqual(segments[2][1]["previous"], [segments[1][0], 2])
        self.assertNotIn("YogurtH3OriginalAudio", [node["class_type"] for node in graph.values()])
        joins = [node["inputs"] for node in graph.values() if node["class_type"] == "YogurtH3JoinSegments"]
        self.assertEqual([join["soft_audio"] for join in joins], [False, True, True])
        self.assertTrue(all("audio" in join for join in joins))
        self.assertIn("输出 498 帧", report)
        self.assertIn("1344×768", report)


if __name__ == "__main__":
    unittest.main()
