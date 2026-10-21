import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from support import load, write_video

director = load()
rewrite = director.rewrite
import folder_paths  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
