import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from support import load, write_video

director = load()
rewrite = director.rewrite
import comfy.model_management  # noqa: E402
import folder_paths  # noqa: E402
import psutil  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedModel(BaseChatModel):
    """Plays back AIMessages and records every prompt the agent sends."""

    replies: list = []
    seen: list = []
    tools: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.tools = [getattr(t, "name", None) or t.get("name") or t.get("function", {}).get("name") for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def submit(shot_id, prompt):
    return AIMessage(content="", tool_calls=[{"name": "submit_prompt", "id": f"out-{shot_id}", "args": {"id": shot_id, "prompt": prompt}}])


def answers(prompts):
    return [submit(shot_id, prompt) for shot_id, prompt in prompts.items()] + [DONE]


def tool_reply(messages):
    return [m for m in messages if isinstance(m, ToolMessage)][-1].content


def brightness(block):
    data = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
    return np.asarray(Image.open(BytesIO(data))).mean()


DONE = AIMessage(content="Done.")


READ_SKILL = AIMessage(content="", tool_calls=[{"name": "read_file", "id": "r1", "args": {"file_path": "/skills/h3-prompt-writing/SKILL.md"}}])
CONFIG = rewrite.rewriter_config("openai", "scripted-model")
SIZE = {"width": 128, "height": 64}


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "input").mkdir()
        (root / "output").mkdir()
        self.saved_dirs = folder_paths.get_output_directory(), folder_paths.get_input_directory()
        folder_paths.set_output_directory(str(root / "output"))
        folder_paths.set_input_directory(str(root / "input"))
        Image.fromarray(np.full((64, 128, 3), 120, dtype=np.uint8)).save(root / "input" / "first.png")
        self.model = ScriptedModel()
        self.saved_model = rewrite.chat_model
        rewrite.chat_model = lambda config: self.model

    def tearDown(self):
        rewrite.chat_model = self.saved_model
        folder_paths.set_output_directory(self.saved_dirs[0])
        folder_paths.set_input_directory(self.saved_dirs[1])
        self.tmp.cleanup()

    def project(self, **changes):
        shots = [{"id": "a", "frames": 48, "prompt": "", "continue": False,
                  "keyframes": [{"at": 0, "media": {"filename": "first.png", "subfolder": "", "type": "input"}}]},
                 {"id": "b", "frames": 48, "prompt": "她回头微笑"},
                 {"id": "c", "frames": 48, "prompt": "", "continue": False}]
        data = {"story": "雨夜街头，一个穿红雨衣的女孩走进咖啡馆。",
                "rewrite": {"enabled": True}, "shots": shots}
        for shot_id, fields in changes.items():
            next(s for s in shots if s["id"] == shot_id).update(fields)
        return director.project.load_project(json.dumps(data), SIZE)

    def run_rewrite(self, project, event=None):
        return rewrite.rewrite_project(project, CONFIG, director.media.Media(), event)

    def test_agent_reads_the_skill_and_writes_segment_by_segment(self):
        self.model.replies = [READ_SKILL] + answers({i: f"prompt {i}" for i in "abc"})
        events = []
        project, prompts, report = self.run_rewrite(self.project(), lambda *event: events.append(event))
        self.assertEqual(prompts, {"a": "prompt a", "b": "prompt b", "c": "prompt c"})
        self.assertEqual([s["prompt"] for s in project["shots"]], ["prompt a", "prompt b", "prompt c"])
        # Each prompt reaches the editor as soon as it is submitted.
        self.assertEqual(events, [(0, "rewriting"), (0, "rewritten", "prompt a"), (1, "rewriting"), (1, "rewritten", "prompt b"),
                                  (2, "rewriting"), (2, "rewritten", "prompt c")])
        first, second, third, fourth, last = self.model.seen
        system = first[0].content if isinstance(first[0].content, str) else json.dumps(first[0].content, ensure_ascii=False)
        self.assertIn("h3-prompt-writing", system)
        human = first[1].content
        text = human[0]["text"]
        self.assertIn("雨夜街头", text)
        self.assertIn("Mode: I2VA", text)
        self.assertIn("<Picture 1>: the first frame, aligned at 0.00 s", text)
        self.assertIn("Continued: 0.00–0.92 s is the end of segment 1", text)
        self.assertIn("Segment prompt: 她回头微笑", text)
        self.assertTrue(any(block.get("type") == "image_url" for block in human[1:]))
        skill = next(m for m in second if isinstance(m, ToolMessage))
        self.assertIn("H3 Prompt Writing", skill.content)
        self.assertIn("submit_prompt", self.model.tools)
        self.assertEqual(tool_reply(third), "Saved segment 1 (1/3). Next: segment 2 (id `b`).")
        # Segment 2 is written with segment 1's prompt already in the conversation.
        self.assertTrue(any(isinstance(m, AIMessage) and m.tool_calls and m.tool_calls[0]["args"].get("prompt") == "prompt a" for m in third))
        self.assertEqual(tool_reply(fourth), "Saved segment 2 (2/3). Next: segment 3 (id `c`).")
        self.assertEqual(tool_reply(last), "All 3 segments saved. You are done.")
        self.assertTrue(report[0].startswith("改写：第 1、2、3 镜"))

    def test_only_changed_segments_are_rewritten(self):
        self.model.replies = answers({i: f"v1 {i}" for i in "abc"})
        self.run_rewrite(self.project())
        events = []
        _, prompts, report = self.run_rewrite(self.project(), lambda *event: events.append(event))
        self.assertEqual(prompts["b"], "v1 b")
        self.assertEqual(report, ["改写读取缓存：第 1、2、3 镜"])
        self.assertEqual(events, [(0, "rewritten", "v1 a"), (1, "rewritten", "v1 b"), (2, "rewritten", "v1 c")])

        self.model.replies = answers({"c": "v2 c"})
        _, prompts, _ = self.run_rewrite(self.project(c={"prompt": "咖啡馆里看窗外"}))
        self.assertEqual((prompts["a"], prompts["c"]), ("v1 a", "v2 c"))
        task = self.model.seen[-1][1].content[0]["text"].split("# Task")[1]
        self.assertIn("`c`", task)
        self.assertNotIn("`a`", task)

        # A continued segment follows its predecessor's new prompt.
        self.model.replies = answers({"a": "v3 a", "b": "v3 b"})
        _, prompts, _ = self.run_rewrite(self.project(a={"prompt": "她撑着伞"}, c={"prompt": "咖啡馆里看窗外"}))
        self.assertEqual((prompts["a"], prompts["b"], prompts["c"]), ("v3 a", "v3 b", "v2 c"))

    def test_locked_shot_is_context_only(self):
        self.model.replies = answers({"a": "pa", "b": "pb"})
        project, prompts, _ = self.run_rewrite(self.project(c={"rewrite": False, "prompt": "Handwritten cafe prompt."}))
        self.assertNotIn("c", prompts)
        self.assertEqual(project["shots"][2]["prompt"], "Handwritten cafe prompt.")
        self.assertIn("(fixed, 2.33 s; do not rewrite)\nHandwritten cafe prompt.", self.model.seen[0][1].content[0]["text"])

    def test_global_prompt_only_drives_inheriting_segments(self):
        hero = {"filename": "first.png", "subfolder": "", "type": "input"}
        project = self.project(c={"inherit_prompt": False, "prompt": "空镜：雨夜的霓虹招牌"})
        project = director.project.load_project(json.dumps({**{k: v for k, v in project.items() if k != "shots"},
                                                             "refs": {"images": [hero]},
                                                             "shots": [{**s, "refs": {"images": [], "videos": [], "audios": []}} for s in project["shots"]]}), SIZE)
        self.model.replies = answers({i: f"v1 {i}" for i in "abc"})
        self.run_rewrite(project)
        human = self.model.seen[0][1].content
        text = human[0]["text"]
        segment_c = text.split("## Segment 3")[1]
        self.assertIn("Does not use the global prompt", segment_c)
        self.assertIn("<Picture 1>: global reference image, the same in every segment", text.split("## Segment 2")[1])
        # The global image is attached once, not once per inheriting segment.
        captions = [block["text"] for block in human[1:] if block["type"] == "text"]
        self.assertEqual(sum("Global reference image 1" in c for c in captions), 1)
        self.assertFalse(any(c.startswith("Segment") and "global reference" in c for c in captions))

        # Editing the global prompt rewrites the inheriting segments only.
        project["story"] = "清晨的海边，一个穿红雨衣的女孩在散步。"
        self.model.replies = answers({"a": "v2 a", "b": "v2 b"})
        _, prompts, _ = self.run_rewrite(project)
        self.assertEqual(prompts, {"a": "v2 a", "b": "v2 b", "c": "v1 c"})

    def test_batched_and_repeated_submissions(self):
        # Tool calls of one message run concurrently; all of them must land.
        batch = AIMessage(content="", tool_calls=[{"name": "submit_prompt", "id": f"p{i}", "args": {"id": i, "prompt": f"p{i}"}} for i in "ab"])
        self.model.replies = [submit("c", "pc"), batch, submit("a", "pa fixed"), DONE]
        _, prompts, _ = self.run_rewrite(self.project())
        self.assertEqual(prompts, {"a": "pa fixed", "b": "pb", "c": "pc"})
        self.assertEqual(tool_reply(self.model.seen[1]), "Saved segment 3 (1/3). Next: segment 1 (id `a`).")

    def test_private_media_stay_out_of_the_model_context(self):
        hero = {"filename": "first.png", "subfolder": "", "type": "input"}
        project = director.project.load_project(json.dumps({
            "story": "雨夜街头。", "rewrite": {"enabled": True},
            "refs": {"images": [{"media": hero, "share": False, "description": "穿红雨衣的短发女孩"}]},
            "shots": [{"id": "a", "frames": 48, "prompt": "", "refs": {"images": [{"media": hero}, {"media": hero, "share": False}]},
                       "keyframes": [{"at": 20, "media": hero, "share": False, "description": "她回头的瞬间"}]}]}), SIZE)
        self.model.replies = answers({"a": "pa"})
        self.run_rewrite(project)
        human = self.model.seen[0][1].content
        text = human[0]["text"]
        self.assertIn("<Picture 1>: global reference image, the same in every segment; not shown to you for privacy, "
                      "the user describes it as: 穿红雨衣的短发女孩.", text)
        self.assertIn("<Picture 3>: reference image; not shown to you for privacy and not described", text)
        self.assertIn("guide image pins the picture at 0.83 s; describe that moment consistently with it; not shown to you for privacy, "
                      "the user describes it as: 她回头的瞬间.", text)
        # Only the shared reference image goes out: not the private global one, the private own one or the private guide.
        captions = [block["text"] for block in human[1:] if block["type"] == "text"]
        self.assertEqual((captions, sum(block["type"] == "image_url" for block in human)), (["Segment 1 <Picture 2> (reference image):"], 1))

    def test_source_video_is_shown_from_its_start(self):
        write_video(Path(self.tmp.name) / "input" / "clip.mp4", 48, 128, 64)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 36, "regen": True}
        project = director.project.load_project(json.dumps({"rewrite": {"enabled": True},
                                                            "shots": [{"id": "s", "kind": "source", "frames": 12, "prompt": "换成白天", "source": source}]}), SIZE)
        self.model.replies = answers({"s": "day"})
        self.run_rewrite(project)
        human = self.model.seen[0][1].content
        self.assertIn("<Video 1>: the source video this segment rewrites", human[0]["text"])
        self.assertAlmostEqual(brightness(human[2]), 255 * 36 / 47, delta=4)

    def test_source_video_is_shown_over_the_whole_segment(self):
        write_video(Path(self.tmp.name) / "input" / "clip.mp4", 120, 128, 64)
        media = {"filename": "clip.mp4", "subfolder": "", "type": "input"}
        project = director.project.load_project(json.dumps({"rewrite": {"enabled": True}, "shots": [
            {"id": "a", "kind": "source", "frames": 39, "prompt": "换人", "source": {"media": media, "start": 0, "regen": True}, "continue": False},
            {"id": "b", "kind": "source", "frames": 30, "prompt": "换人", "source": {"media": media, "start": 39, "regen": True}, "continue": True}]}), SIZE)
        self.model.replies = answers({"a": "pa", "b": "pb"})
        self.run_rewrite(project)
        blocks = self.model.seen[0][1].content
        shown = [(blocks[i]["text"], brightness(blocks[i + 1])) for i in range(1, len(blocks) - 1) if blocks[i + 1]["type"] == "image_url"]
        # The continued segment's timeline starts on its carried head, 22 frames before its own start.
        expected = {"a": {0.0: 0, 1.0: 24, 1.58: 38}, "b": {0.0: 17, 1.0: 41, 2.0: 65, 2.12: 68}}
        for number, key in ((1, "a"), (2, "b")):
            frames = [(text, value) for text, value in shown if text.startswith(f"Segment {number} <Video 1>")]
            self.assertEqual([text.rsplit(" at ", 1)[1] for text, _ in frames], [f"{at:.2f} s:" for at in expected[key]])
            for (_, value), at in zip(frames, expected[key].values()):
                self.assertAlmostEqual(value, 255 * at / 119, delta=4)

    def swap_project(self, **swap):
        write_video(Path(self.tmp.name) / "input" / "clip.mp4", 120, 128, 64)
        media = {"filename": "clip.mp4", "subfolder": "", "type": "input"}
        return director.project.load_project(json.dumps({
            "rewrite": {"enabled": True}, "refs": {"images": [{"media": {"filename": "first.png", "subfolder": "", "type": "input"}}]}, "swap": {"enabled": True, **swap},
            "shots": [{"id": "a", "kind": "source", "frames": 40, "source": {"media": media, "start": 0, "regen": True}, "continue": False},
                      {"id": "b", "kind": "source", "frames": 40, "source": {"media": media, "start": 40, "regen": True}, "continue": False},
                      {"id": "g", "frames": 24, "prompt": "生成的镜头"},
                      {"id": "k", "kind": "source", "frames": 40, "source": {"media": media, "start": 80}}]}), SIZE)

    SWAP_REPLY = ("===== SEGMENT 1 =====\nWHAT: 穿白衬衫的男人\nINTO: 老武士，黑袍\n===== END SEGMENT 1 =====\n"
                  "===== SEGMENT 2 =====\nWHAT：背对镜头的男人\nINTO: 老武士，黑袍\n===== END SEGMENT 2 =====\n")

    def test_swap_writes_only_what_becomes_what_from_a_few_frames(self):
        self.model.replies = [AIMessage(content=self.SWAP_REPLY)]
        events = []
        project, prompts, report = self.run_rewrite(self.swap_project(what="那个男人"), lambda *event: events.append(event))
        template = director.project.swap_prompt
        self.assertEqual(prompts, {"a": template("穿白衬衫的男人", "老武士，黑袍", True, False), "b": template("背对镜头的男人", "老武士，黑袍", True, False)})
        self.assertEqual([s["prompt"] for s in project["shots"]], [prompts["a"], prompts["b"], "生成的镜头", ""])
        self.assertEqual(events[0], (0, "rewriting"))
        self.assertEqual(events[1:], [(0, "rewritten", prompts["a"]), (1, "rewritten", prompts["b"])])
        self.assertIn("第 1、2 镜", report[0])
        system, human = self.model.seen[0]
        self.assertIn("WHAT is the subject", system.content)
        self.assertIn("The user says the subject to replace is: 那个男人", human.content[0]["text"])
        self.assertNotIn("生成的镜头", human.content[0]["text"])
        captions = [block["text"] for block in human.content[1:] if block["type"] == "text"]
        self.assertTrue(captions[0].startswith("Global reference image 1"))
        self.assertEqual([c for c in captions if c.startswith("Segment 2")], [f"Segment 2 at {at} s:" for at in ("0.00", "0.54", "1.08", "1.62")])
        shown = dict(zip(captions, [brightness(block) for block in human.content[1:] if block["type"] == "image_url"]))
        self.assertAlmostEqual(shown["Segment 2 at 0.00 s:"], 255 * 40 / 119, delta=4)

    def test_swap_answers_come_from_the_cache(self):
        self.model.replies = [AIMessage(content=self.SWAP_REPLY)]
        project = self.swap_project()
        self.run_rewrite(project)
        _, prompts, report = self.run_rewrite(project)
        self.assertEqual(len(self.model.seen), 1)
        self.assertIn("老武士", prompts["b"])
        self.assertEqual(report, ["改写读取缓存：第 1、2 镜"])

    def test_tracked_swap_tracks_what_the_model_names(self):
        reply = self.SWAP_REPLY.replace("穿白衬衫的男人", "man in white shirt").replace("背对镜头的男人", "man from behind")
        self.model.replies = [AIMessage(content=reply)]
        project, prompts, _ = self.run_rewrite(self.swap_project(track=True))
        self.assertEqual([s["repaints"][0]["target"]["text"] for s in project["shots"][:2]], ["man in white shirt", "man from behind"])
        self.assertEqual(prompts["a"], director.project.swap_prompt("man in white shirt", "老武士，黑袍", True, True))
        self.assertNotIn("man in white shirt", prompts["a"])
        self.assertIn("segmentation model", self.model.seen[0][1].content[0]["text"])

    def test_swap_is_asked_again_for_what_it_left_out(self):
        first, second = self.SWAP_REPLY.split("===== SEGMENT 2")
        self.model.replies = [AIMessage(content=first), AIMessage(content="===== SEGMENT 2" + second)]
        _, prompts, _ = self.run_rewrite(self.swap_project())
        self.assertEqual(set(prompts), {"a", "b"})
        self.assertNotIn("## Segment 1", self.model.seen[1][1].content[0]["text"])

    def test_a_shot_without_the_subject_keeps_the_source(self):
        reply = self.SWAP_REPLY.replace("WHAT：背对镜头的男人\nINTO: 老武士，黑袍", "WHAT: NONE")
        for track, layers in ((False, 0), (True, 0)):
            self.model.replies = [AIMessage(content=reply)]
            project, prompts, report = self.run_rewrite(self.swap_project(track=track, objects=track + 1))
            kept = project["shots"][1]
            self.assertEqual((kept["source"]["regen"], kept["source"]["reference"], kept["prompt"], len(kept["repaints"])), (False, False, "", layers))
            self.assertEqual(prompts["b"], rewrite.ABSENT_NOTE)
            self.assertIn("第 2 镜画面里没有要换的对象，保留原画", report)
            self.assertIn("老武士", prompts["a"])

    def test_swap_brief_lists_a_picture_only_when_it_has_something_to_say(self):
        self.model.replies = [AIMessage(content=self.SWAP_REPLY)]
        self.run_rewrite(self.swap_project())
        self.assertNotIn("<Picture 1>.", self.model.seen[0][1].content[0]["text"])

    def test_swap_without_vision_keeps_the_fields(self):
        project = self.swap_project(what="男人", into="武士")
        written, prompts, report = rewrite.rewrite_project(project, {**CONFIG, "vision": False}, director.media.Media())
        self.assertEqual(self.model.seen, [])
        self.assertEqual(prompts, {})
        self.assertEqual(written["shots"][0]["prompt"], project["shots"][0]["prompt"])
        self.assertIn("vision", report[0])

    def test_agent_is_reminded_when_it_stops_early(self):
        plan = AIMessage(content="", tool_calls=[{"name": "write_file", "id": "w1", "args": {"file_path": "/plan.md", "content": "1. walk 2. turn 3. cafe"}}])
        read = AIMessage(content="", tool_calls=[{"name": "read_file", "id": "r2", "args": {"file_path": "/plan.md"}}])
        self.model.replies = [plan, AIMessage(content="The plan is ready. Next I will write segment 1."), read] + answers({i: f"p{i}" for i in "abc"})
        _, prompts, _ = self.run_rewrite(self.project())
        self.assertEqual(prompts, {"a": "pa", "b": "pb", "c": "pc"})
        reminder = self.model.seen[2][-1]
        self.assertIsInstance(reminder, HumanMessage)
        self.assertIn("segment 1 (id `a`)", reminder.content)
        # The conversation and the plan file carry over into the reminded run.
        self.assertIn("The plan is ready. Next I will write segment 1.", [m.content for m in self.model.seen[2]])
        self.assertIn("walk", tool_reply(self.model.seen[3]))

    def test_unsubmitted_segment_is_an_error(self):
        self.model.replies = [submit("a", "pa"), AIMessage(content="Done."), AIMessage(content="Done."), AIMessage(content="I have finished the film.")]
        with self.assertRaisesRegex(RuntimeError, "没有交回第 2、3 镜.*I have finished the film."):
            self.run_rewrite(self.project())


FAKE_AGENT = r"""
import json, os, re, sys, time
from pathlib import Path

home = Path(__file__).parent
brief = Path("brief.md").read_text(encoding="utf-8")
files = sorted(p.relative_to(".").as_posix() for p in Path(".").rglob("*") if p.is_file())
calls = json.loads((home / "calls.json").read_text(encoding="utf-8")) if (home / "calls.json").exists() else []
calls.append({"argv": sys.argv[1:], "brief": brief, "files": files})
(home / "calls.json").write_text(json.dumps(calls), encoding="utf-8")
mode = (home / "mode.txt").read_text(encoding="utf-8").strip() if (home / "mode.txt").exists() else ""


def say(text):
    sys.stdout.buffer.write(text.encode("utf-8"))


if mode == "login":
    say("Not logged in")
    sys.exit(1)
if mode == "hang":
    (home / "pid.txt").write_text(str(os.getpid()))
    time.sleep(60)
if mode == "silent":
    say("I would rather chat.")
    sys.exit(0)
if "The description:" in brief:
    say("I will translate.===== SEGMENT 1 =====\ngirl in red\n===== END SEGMENT 1 =====\n")
    sys.exit(0)
wanted = [int(n) for n in re.findall(r"^## Segment (\d+) — id `[^`]+`$", brief, re.M)]
if mode == "partial" and len(calls) == 1:
    wanted = wanted[:1]
if "WHAT is the subject" in brief:
    say("".join(f"===== SEGMENT {n} =====\nWHAT: 男人 {n}\nINTO: 武士\n===== END SEGMENT {n} =====\n" for n in wanted))
    sys.exit(0)
say("Narration before the answer." + "".join(f"===== SEGMENT {n} =====\n雨夜 prompt {n}\n===== END SEGMENT {n} =====\n" for n in wanted))
"""


class AgentTests(unittest.TestCase):
    """Real run_agent against a fake CLI behind the same kind of launcher the real ones install."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "input").mkdir()
        (self.root / "output").mkdir()
        self.saved_dirs = folder_paths.get_output_directory(), folder_paths.get_input_directory()
        folder_paths.set_output_directory(str(self.root / "output"))
        folder_paths.set_input_directory(str(self.root / "input"))
        Image.fromarray(np.full((64, 128, 3), 120, dtype=np.uint8)).save(self.root / "input" / "first.png")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "fake.py").write_text(FAKE_AGENT, encoding="utf-8")

    def tearDown(self):
        folder_paths.set_output_directory(self.saved_dirs[0])
        folder_paths.set_input_directory(self.saved_dirs[1])
        self.tmp.cleanup()

    def cli(self, name, folder=None):
        folder = folder or self.bin
        fake = self.bin / "fake.py"
        if os.name == "nt":
            path = folder / f"{name}.cmd"
            path.write_text(f'@"{sys.executable}" "{fake}" %*\n', encoding="utf-8")
        else:
            path = folder / name
            path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n', encoding="utf-8")
            path.chmod(0o755)
        return path

    def config(self, name="codex", **options):
        return director.agents.agent_config(name, str(self.cli(name)), **options)

    def calls(self):
        return json.loads((self.bin / "calls.json").read_text(encoding="utf-8"))

    def project(self):
        shots = [{"id": "a", "frames": 48, "prompt": "", "continue": False,
                  "keyframes": [{"at": 0, "media": {"filename": "first.png", "subfolder": "", "type": "input"}}]},
                 {"id": "b", "frames": 48, "prompt": "她回头微笑"},
                 {"id": "c", "frames": 48, "prompt": "", "continue": False}]
        return director.project.load_project(json.dumps({"story": "雨夜街头 & 咖啡馆", "rewrite": {"enabled": True}, "shots": shots}), SIZE)

    def run_rewrite(self, config, event=None):
        return rewrite.rewrite_project(self.project(), config, director.media.Media(), event)

    def test_the_agent_answers_every_segment_in_one_read_only_run(self):
        events = []
        project, prompts, report = self.run_rewrite(self.config(), lambda *event: events.append(event))
        self.assertEqual(prompts, {"a": "雨夜 prompt 1", "b": "雨夜 prompt 2", "c": "雨夜 prompt 3"})
        self.assertEqual([s["prompt"] for s in project["shots"]], list(prompts.values()))
        self.assertEqual(events, [(0, "rewriting"), (0, "rewritten", "雨夜 prompt 1"), (1, "rewritten", "雨夜 prompt 2"), (2, "rewritten", "雨夜 prompt 3")])
        self.assertIn("codex:默认模型", report[0])
        call = self.calls()[0]
        self.assertEqual(call["argv"][call["argv"].index("-s") + 1], "read-only")
        # The project travels in a file: a .cmd launcher would hand command-line text to cmd.exe.
        self.assertNotIn("雨夜", " ".join(call["argv"]))
        self.assertIn("雨夜街头 & 咖啡馆", call["brief"])
        self.assertIn("[Image file: image-1.jpg]", call["brief"])
        self.assertIn("image-1.jpg", call["files"])
        self.assertIn("skills/h3-prompt-writing/SKILL.md", call["files"])
        self.assertIn("skills/h3-prompt-writing/references/ref-en.txt", call["files"])

    def test_finished_segments_come_from_the_cache(self):
        config = self.config()
        self.run_rewrite(config)
        _, prompts, report = self.run_rewrite(config)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(prompts["c"], "雨夜 prompt 3")
        self.assertIn("改写读取缓存", report[0])

    def test_swap_fields_come_back_from_the_cli(self):
        write_video(self.root / "input" / "clip.mp4", 48, 128, 64)
        project = director.project.load_project(json.dumps({"rewrite": {"enabled": True}, "swap": {"enabled": True}, "shots": [
            {"id": "a", "kind": "source", "frames": 40, "source": {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "regen": True}}]}), SIZE)
        _, prompts, _ = rewrite.rewrite_project(project, self.config(), director.media.Media())
        self.assertIn("<Subject 1> is the 武士, whose position, pose, movement and timing come from the 男人 1 in <Video 1>.", prompts["a"])
        call = self.calls()[0]
        self.assertNotIn("skills/h3-prompt-writing/SKILL.md", call["files"])
        self.assertEqual(len([f for f in call["files"] if f.endswith(".jpg")]), 4)

    def test_vision_off_sends_no_images(self):
        self.run_rewrite(self.config(vision=False))
        call = self.calls()[0]
        self.assertFalse([f for f in call["files"] if f.endswith(".jpg")])
        self.assertNotIn("[Image file:", call["brief"])

    def test_segments_left_out_are_asked_for_again(self):
        (self.bin / "mode.txt").write_text("partial", encoding="utf-8")
        _, prompts, _ = self.run_rewrite(self.config())
        self.assertEqual(set(prompts), {"a", "b", "c"})
        first, second = self.calls()
        self.assertIn("id `a` (fixed", second["brief"])
        self.assertIn("雨夜 prompt 1", second["brief"])
        self.assertIn("`b`, `c`", second["brief"])

    def test_an_agent_that_never_answers_fails_with_its_reply(self):
        (self.bin / "mode.txt").write_text("silent", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "I would rather chat"):
            self.run_rewrite(self.config())
        self.assertEqual(len(self.calls()), rewrite.REMINDERS + 1)

    def test_a_failed_run_reports_what_the_cli_said(self):
        (self.bin / "mode.txt").write_text("login", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "codex 运行失败.*Not logged in"):
            self.run_rewrite(self.config())
        self.assertEqual(len(self.calls()), 1)

    def test_interrupt_stops_the_whole_process_tree(self):
        (self.bin / "mode.txt").write_text("hang", encoding="utf-8")
        found = []

        def interrupt():
            while not (self.bin / "pid.txt").exists():
                time.sleep(0.1)
            found.append(psutil.Process(int((self.bin / "pid.txt").read_text())))
            comfy.model_management.interrupt_current_processing()

        thread = threading.Thread(target=interrupt)
        thread.start()
        started = time.time()
        with self.assertRaises(comfy.model_management.InterruptProcessingException):
            director.agents.run_agent(self.config(), "brief")
        thread.join()
        self.assertLess(time.time() - started, 30)
        _, alive = psutil.wait_procs(found, timeout=5)
        self.assertEqual(alive, [])

    def test_each_cli_gets_only_read_access_and_its_own_flags(self):
        for name, flags in (("codex", ["-m", "some-model", "-c", "model_reasoning_effort=high"]),
                            ("claude", ["--tools", "Read", "--model", "some-model", "--effort", "high"]),
                            ("grok", ["--tools", "Read", "-m", "some-model", "--reasoning-effort", "high"])):
            (self.bin / "calls.json").unlink(missing_ok=True)
            director.agents.run_agent(self.config(name, model="some-model", effort="high"), "brief")
            argv = self.calls()[0]["argv"]
            for flag, value in zip(flags[::2], flags[1::2]):
                self.assertEqual(argv[argv.index(flag) + 1], value, name)
            (self.bin / "calls.json").unlink()
            director.agents.run_agent(self.config(name), "brief")
            argv = self.calls()[0]["argv"]
            self.assertFalse({"-m", "--model", "--effort", "--reasoning-effort", "-c"} & set(argv), name)

    def test_targets_are_translated_through_the_cli(self):
        self.assertEqual(rewrite.english_targets(["穿红衣的女孩"], self.config()), {"穿红衣的女孩": "girl in red"})
        self.assertIn("穿红衣的女孩", self.calls()[0]["brief"])

    def test_a_workflow_can_only_run_the_cli_itself(self):
        agent_config = director.agents.agent_config
        with self.assertRaisesRegex(ValueError, "必须是 codex"):
            agent_config("codex", str(self.cli("notes")))
        with self.assertRaisesRegex(ValueError, "必须是 codex"):
            agent_config("codex", str(self.cli("codex", self.root / "input")))
        with self.assertRaisesRegex(FileNotFoundError, "找不到 codex"):
            agent_config("codex", str(self.bin / "nothing" / "codex"))
        for model in ("a&calc", "--dangerous", "a b", 'a"b', "a%PATH%"):
            with self.assertRaisesRegex(ValueError, "模型名称不合法"):
                agent_config("codex", str(self.cli("codex")), model=model)
        with self.assertRaisesRegex(ValueError, "effort"):
            agent_config("codex", str(self.cli("codex")), effort="max&calc")
        with self.assertRaisesRegex(ValueError, "不支持"):
            agent_config("notepad", "notepad")


if __name__ == "__main__":
    unittest.main()
