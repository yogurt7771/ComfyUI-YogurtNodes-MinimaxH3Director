import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
from PIL import Image
import torch

from support import FakeSampler, load, models, write_video

director = load()
render = director.render
import folder_paths  # noqa: E402


def png(path, value, size=(128, 64), left_half=False):
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8) if left_half else np.full((size[1], size[0], 3), value, dtype=np.uint8)
    if left_half:
        pixels[:, : size[0] // 2] = 255
    Image.fromarray(pixels).save(path)
    return {"filename": Path(path).name, "subfolder": "", "type": "input"}


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "input").mkdir()
        (self.root / "output").mkdir()
        self.saved_dirs = folder_paths.get_output_directory(), folder_paths.get_input_directory()
        folder_paths.set_output_directory(str(self.root / "output"))
        folder_paths.set_input_directory(str(self.root / "input"))
        self.fake = FakeSampler()
        self.saved = render.sampling.sample, render.sampling.schedule, render.install_drift_control
        render.sampling.sample, render.sampling.schedule = self.fake.sample, self.fake.schedule
        self.drift = []
        render.install_drift_control = lambda model, latent, sigmas, tokens: self.drift.append(tokens) or model

    def tearDown(self):
        render.sampling.sample, render.sampling.schedule, render.install_drift_control = self.saved
        folder_paths.set_output_directory(self.saved_dirs[0])
        folder_paths.set_input_directory(self.saved_dirs[1])
        self.tmp.cleanup()

    def run_project(self, data, **kwargs):
        spec = director.project.load_project(json.dumps({"width": 128, "height": 64, **data}))
        job = render.Render("7", {}, spec, models(director, **kwargs), director.media.Media(), [11, 6, 1.0, "euler", "simple"])
        self.fake.calls.clear()
        video, saved = job.run("test/out")
        return job, Path(folder_paths.get_output_directory()) / saved["subfolder"] / saved["filename"]

    def probe(self, path):
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        with av.open(str(path)) as container:
            samples = sum(frame.samples for frame in container.decode(audio=0))
        return frames, samples / 48000

    def test_generated_film_with_keyframes_references_and_continuation(self):
        end = png(self.root / "input" / "end.png", 200)
        hero = png(self.root / "input" / "hero.png", 90)
        shots = [{"id": "a", "frames": 40, "prompt": "开场"},
                 {"id": "b", "frames": 40, "prompt": "续接", "keyframes": [{"at": -1, "media": end}]},
                 {"id": "c", "frames": 30, "prompt": "参考", "continue": False, "refs": {"images": [hero]}}]
        job, path = self.run_project({"shots": shots}, ref_model="ref_model")
        a, b, c = self.fake.calls
        self.assertEqual([a["prompt"], b["prompt"], c["prompt"]], ["开场", "续接", "参考"])
        self.assertEqual(b["keyframes"], [job.windows[1].gen - 1])
        self.assertEqual(b["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertEqual(self.drift, [7])
        self.assertEqual((c["model"], c["refs"]), ("ref_model", ["image"]))
        self.assertEqual(a["model"], "model")
        clip_calls = job.m.clip.calls
        self.assertEqual([call["images"] for call in clip_calls[:2]], [0, 1])
        self.assertEqual(clip_calls[2]["refs"], ["image"])
        total = sum(w.keep for w in job.windows)
        frames, seconds = self.probe(path)
        self.assertEqual(frames, total)
        self.assertAlmostEqual(seconds, total / 24, delta=0.05)

    def test_cache_reuses_unchanged_shots_and_reruns_dependents(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"},
                 {"id": "c", "frames": 30, "prompt": "三", "continue": False}]
        self.run_project({"shots": shots})
        self.assertEqual(len(self.fake.calls), 3)
        self.assertEqual(len({call["seed"] for call in self.fake.calls}), 3, "equal-length shots must not share noise")
        self.run_project({"shots": shots})
        self.assertEqual(self.fake.calls, [])
        shots[0]["prompt"] = "一改"
        self.run_project({"shots": shots})
        self.assertEqual([call["prompt"] for call in self.fake.calls], ["一改", "二"])
        shots[2]["seed"] = 3
        self.run_project({"shots": shots})
        self.assertEqual([call["prompt"] for call in self.fake.calls], ["三"])

    def test_source_repaint_keeps_pixels_outside_the_mask(self):
        write_video(self.root / "input" / "clip.mp4", 48, 128, 64, audio_seconds=2)
        mask = png(self.root / "input" / "mask.png", 0, left_half=True)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 6}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "改左边", "source": source,
                  "repaints": [{"mask": mask, "strength": 0.6, "grow": 0, "feather": 0}]}]
        job, path = self.run_project({"shots": shots})
        (call,) = self.fake.calls
        self.assertEqual((call["steps"], call["start"]), (6, 0.6))
        self.assertEqual(call["video_mask"][0, 0, 0, 0].tolist(), [1.0] * 4 + [0.0] * 4)
        self.assertEqual(call["audio_mask"].abs().sum().item(), 0.0)
        frames = director.output.read_frames(job.keys[0]["frames"])
        expected = job.source_frames(job.windows[0], 128, 64)[:30]
        self.assertTrue(torch.allclose(frames[:, :, 64:], expected[:, :, 64:], atol=1 / 255))
        self.assertFalse(torch.allclose(frames[:, :, :64], expected[:, :, :64], atol=0.02))
        self.assertAlmostEqual(self.probe(path)[1], 30 / 24, delta=0.05)

    def test_refine_upscales_every_shot_and_carries_the_refined_tail(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        job, path = self.run_project({"shots": shots, "refine": {"enabled": True, "scale": 2.0, "steps": 4, "denoise": 0.5}})
        base, refine_a, refine_b = self.fake.calls[0], self.fake.calls[2], self.fake.calls[3]
        self.assertEqual(base["shape"][-2:], (4, 8))
        self.assertEqual((refine_a["shape"][-2:], refine_a["steps"], refine_a["start"]), ((8, 16), 4, 0.5))
        self.assertEqual(refine_b["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertEqual(refine_b["audio_mask"].abs().sum().item(), 0.0)
        with av.open(str(path)) as container:
            self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (256, 128))

    def test_detail_region_is_resampled_and_stitched(self):
        shots = [{"id": "a", "frames": 30, "prompt": "人物", "details": [{"box": [0.25, 0.25, 0.5, 0.5], "denoise": 0.5,
                                                                      "megapixels": 0.1, "prompt": "脸部细节"}]}]
        job, _ = self.run_project({"shots": shots, "detail": {"steps": 4}})
        base, crop = self.fake.calls
        self.assertEqual(crop["prompt"], "脸部细节")
        self.assertEqual((crop["steps"], crop["start"]), (4, 0.5))
        self.assertEqual(crop["audio_mask"].abs().sum().item(), 0.0)
        self.assertTrue(director.cache.exists(job.keys[0]["details"][0], ".mkv"))
        self.run_project({"shots": shots, "detail": {"steps": 4}})
        self.assertEqual(self.fake.calls, [])

    def test_soundtrack_locks_the_audio_and_is_kept_in_the_output(self):
        write_video(self.root / "input" / "music.mp4", 96, 64, 64, audio_seconds=4)
        track = {"filename": "music.mp4", "subfolder": "", "type": "input"}
        shots = [{"id": "a", "frames": 40, "prompt": "一", "audio": "soundtrack"},
                 {"id": "b", "frames": 40, "prompt": "二", "audio": "soundtrack"}]
        job, path = self.run_project({"shots": shots, "soundtrack": track})
        self.assertTrue(all(call["audio_mask"].abs().sum().item() == 0.0 for call in self.fake.calls))
        with av.open(str(path)) as container:
            audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        self.assertGreater(np.abs(audio[:, 48000:96000]).mean(), 0.05)


if __name__ == "__main__":
    unittest.main()
