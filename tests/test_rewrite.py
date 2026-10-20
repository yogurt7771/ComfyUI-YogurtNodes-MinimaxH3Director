import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from support import load

director = load()
rewrite = director.rewrite
import folder_paths  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
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


def answer(segments):
    return AIMessage(content="", tool_calls=[{"name": "submit_prompts", "id": "out", "args": {"segments": segments}}])


DONE = AIMessage(content="Done.")


READ_SKILL = AIMessage(content="", tool_calls=[{"name": "read_file", "id": "r1", "args": {"file_path": "/skills/h3-prompt-writing/SKILL.md"}}])
CONFIG = rewrite.rewriter_config("openai", "scripted-model")


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
        data = {"width": 128, "height": 64, "story": "雨夜街头，一个穿红雨衣的女孩走进咖啡馆。",
                "rewrite": {"enabled": True}, "shots": shots}
        for shot_id, fields in changes.items():
            next(s for s in shots if s["id"] == shot_id).update(fields)
        return director.project.load_project(json.dumps(data))

    def run_rewrite(self, project):
        return rewrite.rewrite_project(project, CONFIG, director.media.Media())

    def test_agent_reads_the_skill_and_gets_the_timeline(self):
        self.model.replies = [READ_SKILL, answer([{"id": i, "prompt": f"prompt {i}"} for i in "abc"]), DONE]
        project, prompts, report = self.run_rewrite(self.project())
        self.assertEqual(prompts, {"a": "prompt a", "b": "prompt b", "c": "prompt c"})
        self.assertEqual([s["prompt"] for s in project["shots"]], ["prompt a", "prompt b", "prompt c"])
        first, second, third = self.model.seen
        system = first[0].content if isinstance(first[0].content, str) else json.dumps(first[0].content, ensure_ascii=False)
        self.assertIn("h3-prompt-writing", system)
        human = first[1].content
        text = human[0]["text"]
        self.assertIn("雨夜街头", text)
        self.assertIn("Mode: I2VA", text)
        self.assertIn("<Picture 1>: the first frame, aligned at 0.00 s", text)
        self.assertIn("Continued: 0.00–0.92 s is the end of segment 1", text)
        self.assertIn("User notes: 她回头微笑", text)
        self.assertTrue(any(block.get("type") == "image_url" for block in human[1:]))
        skill = next(m for m in second if isinstance(m, ToolMessage))
        self.assertIn("H3 Prompt Writing", skill.content)
        self.assertIn("submit_prompts", self.model.tools)
        self.assertEqual(next(m for m in third if isinstance(m, ToolMessage) and m.name == "submit_prompts").content, "All prompts saved. You are done.")
        self.assertTrue(report[0].startswith("改写：第 1、2、3 镜"))

    def test_only_changed_segments_are_rewritten(self):
        self.model.replies = [answer([{"id": i, "prompt": f"v1 {i}"} for i in "abc"]), DONE]
        self.run_rewrite(self.project())
        _, prompts, report = self.run_rewrite(self.project())
        self.assertEqual(prompts["b"], "v1 b")
        self.assertEqual(report, ["改写读取缓存：第 1、2、3 镜"])

        self.model.replies = [answer([{"id": "c", "prompt": "v2 c"}]), DONE]
        _, prompts, _ = self.run_rewrite(self.project(c={"prompt": "咖啡馆里看窗外"}))
        self.assertEqual((prompts["a"], prompts["c"]), ("v1 a", "v2 c"))
        task = self.model.seen[-1][1].content[0]["text"].split("# Task")[1]
        self.assertIn("`c`", task)
        self.assertNotIn("`a`", task)

        # A continued segment follows its predecessor's new prompt.
        self.model.replies = [answer([{"id": "a", "prompt": "v3 a"}, {"id": "b", "prompt": "v3 b"}]), DONE]
        _, prompts, _ = self.run_rewrite(self.project(a={"prompt": "她撑着伞"}, c={"prompt": "咖啡馆里看窗外"}))
        self.assertEqual((prompts["a"], prompts["b"], prompts["c"]), ("v3 a", "v3 b", "v2 c"))

    def test_locked_shot_is_context_only(self):
        self.model.replies = [answer([{"id": "a", "prompt": "pa"}, {"id": "b", "prompt": "pb"}]), DONE]
        project, prompts, _ = self.run_rewrite(self.project(c={"rewrite": False, "prompt": "Handwritten cafe prompt."}))
        self.assertNotIn("c", prompts)
        self.assertEqual(project["shots"][2]["prompt"], "Handwritten cafe prompt.")
        self.assertIn("(fixed, 2.33 s; do not rewrite)\nHandwritten cafe prompt.", self.model.seen[0][1].content[0]["text"])

    def test_agent_is_told_what_is_missing_and_can_finish(self):
        self.model.replies = [answer([{"id": "a", "prompt": "pa"}]), answer([{"id": "b", "prompt": "pb"}, {"id": "c", "prompt": "pc"}]), DONE]
        _, prompts, _ = self.run_rewrite(self.project())
        self.assertEqual(prompts, {"a": "pa", "b": "pb", "c": "pc"})
        feedback = next(m for m in self.model.seen[1] if isinstance(m, ToolMessage))
        self.assertEqual(feedback.content, "Saved. Still missing: b, c.")

    def test_unsubmitted_segment_is_an_error(self):
        self.model.replies = [answer([{"id": "a", "prompt": "pa"}]), DONE]
        with self.assertRaisesRegex(RuntimeError, "没有交回第 2、3 镜"):
            self.run_rewrite(self.project())


if __name__ == "__main__":
    unittest.main()
