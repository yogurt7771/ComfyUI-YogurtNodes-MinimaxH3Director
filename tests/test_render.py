import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import av
import numpy as np
from PIL import Image
import torch

from support import FakeAudioVAE, FakeClip, FakeSampler, FakeVAE, load, models, tiny_upscaler, write_video

director = load()
render = director.render
import folder_paths  # noqa: E402
from comfy_api.latest import InputImpl  # noqa: E402


def square_mask(path, box, size=(128, 64)):
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    pixels[y0:y1, x0:x1] = 255
    Image.fromarray(pixels).save(path)
    return {"filename": Path(path).name, "subfolder": "", "type": "input"}


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
        self.saved_dirs = folder_paths.get_output_directory(), folder_paths.get_input_directory(), folder_paths.get_temp_directory()
        folder_paths.set_output_directory(str(self.root / "output"))
        folder_paths.set_input_directory(str(self.root / "input"))
        folder_paths.set_temp_directory(str(self.root / "temp"))
        # SAM 3 stand-in: "left ..." marks the left half, "nothing" finds nothing, anything else a 20 px square.
        self.tracked, self.saved_track = [], render.segment.track_mask

        def track(model, clip, frames, target):
            self.tracked.append((frames.clone(), target))
            masks = torch.zeros(frames.shape[:3])
            if target["text"].startswith("left"):
                masks[:, :, : frames.shape[2] // 2] = 1.0
            elif target["text"] != "nothing":
                masks[:, 20:40, 40:60] = 1.0
            return masks, int(masks.any())
        render.segment.track_mask = track
        self.fake = FakeSampler()
        self.saved = render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise, render.install_drift_control
        render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise = self.fake.sample, self.fake.schedule, self.fake.trajectory_noise
        self.drift = []
        render.install_drift_control = lambda model, latent, sigmas, tokens: self.drift.append(tokens) or model

    def tearDown(self):
        render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise, render.install_drift_control = self.saved
        render.segment.track_mask = self.saved_track
        folder_paths.set_output_directory(self.saved_dirs[0])
        folder_paths.set_input_directory(self.saved_dirs[1])
        folder_paths.set_temp_directory(self.saved_dirs[2])
        self.tmp.cleanup()

    def run_project(self, data, settings=None, media=None, **kwargs):
        # The real latent upscaler stays out of unit tests; the h3_latent tests bring a tiny one.
        spec = director.project.load_project(json.dumps(data), {"width": 128, "height": 64, "two_pass_upscaler": "lanczos", "rerender_upscaler": "lanczos", **(settings or {})})
        job = render.Render("7", {}, spec, models(director, **kwargs), media or director.media.Media(), [11, 6, 1.0, "euler", "simple"])
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

    def test_global_reference_turns_inheriting_shots_into_reference_shots(self):
        hero = png(self.root / "input" / "hero.png", 90)
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二", "continue": False, "inherit_refs": False}]
        self.run_project({"shots": shots, "refs": {"images": [hero]}}, ref_model="ref_model")
        a, b = self.fake.calls
        self.assertEqual((a["model"], a["refs"]), ("ref_model", ["image"]))
        self.assertEqual((b["model"], b["refs"]), ("model", []))

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
        # The grid pads 30 frames to 39; the layer runs to the shot's end, so the frames after it regenerate too, and
        # they hold the shot's last source frame instead of reading on into the clip.
        self.assertEqual(call["video_mask"][0, 0, :, 0, 0].tolist(), [1.0] * call["video_mask"].shape[2])
        padded = job.source_frames(job.windows[0], 128, 64)
        self.assertTrue(torch.equal(padded[30:], padded[29:30].expand(9, -1, -1, -1)))
        self.assertEqual(call["audio_mask"].abs().sum().item(), 0.0)
        frames = director.output.read_frames(job.keys[0]["frames"])
        expected = job.source_frames(job.windows[0], 128, 64)[:30]
        self.assertTrue(torch.allclose(frames[:, :, 64:], expected[:, :, 64:], atol=1 / 255))
        self.assertFalse(torch.allclose(frames[:, :, :64], expected[:, :, :64], atol=0.02))
        self.assertAlmostEqual(self.probe(path)[1], 30 / 24, delta=0.05)

    def test_continued_source_edit_opens_on_the_previous_result(self):
        write_video(self.root / "input" / "clip.mp4", 96, 128, 64, audio_seconds=4)
        mask = png(self.root / "input" / "mask.png", 0, left_half=True)
        clip = {"filename": "clip.mp4", "subfolder": "", "type": "input"}
        layer = lambda: {"mask": mask, "strength": 1.0, "grow": 0, "feather": 0}
        shots = [{"id": "a", "kind": "source", "frames": 39, "prompt": "一", "source": {"media": clip, "start": 6}, "repaints": [layer()]},
                 {"id": "b", "kind": "source", "frames": 34, "prompt": "二", "continue": True, "source": {"media": clip, "start": 45},
                  "repaints": [layer()]}]
        job, path = self.run_project({"shots": shots})
        a, b = job.windows
        self.assertEqual((b.carry, b.head, a.keep), (True, 22, 39))
        # b's window opens on the 22 source frames a ended on.
        self.assertTrue(torch.equal(job.source_frames(b, 128, 64)[:22], job.source_frames(a, 128, 64)[17:39]))
        # Its repaint keeps those tokens, and they are a's result, not the source.
        tokens = render.video_latent_t(22)
        first, second = (director.cache.read_tensors(keys["final"])["video"] for keys in job.keys)
        self.assertTrue(torch.equal(second[:, :, :tokens], first[:, :, -tokens:]))
        self.assertEqual(self.fake.calls[1]["video_mask"][0, 0, :tokens].abs().sum().item(), 0.0)
        frames, seconds = self.probe(path)
        self.assertEqual(frames, 73)
        self.assertAlmostEqual(seconds, 73 / 24, delta=0.05)

    def test_a_cover_shows_the_generated_frame_it_repeats(self):
        write_video(self.root / "input" / "clip.mp4", 48, 128, 64)
        clip = {"filename": "clip.mp4", "subfolder": "", "type": "input"}
        shots = [{"id": "c", "kind": "source", "frames": 2, "source": {"media": clip, "start": 0}, "still": {"shot": "s", "frame": 20}},
                 {"id": "s", "kind": "source", "frames": 30, "prompt": "改写", "source": {"media": clip, "start": 2, "regen": True}}]
        job, path = self.run_project({"shots": shots}, ref_model="ref_model")
        with av.open(str(path)) as container:
            film = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]).astype(float)
        # Both cover frames are frame 20 of the rewrite (the film's frame 22), not the clip's dark first frames.
        self.assertLess(np.abs(film[:2] - film[22]).mean(), 2.0)
        self.assertGreater(np.abs(film[0] - director.output.read_frames(job.keys[0]["frames"])[0].numpy() * 255).mean(), 20.0)

    def test_continued_source_rewrite_carries_the_tail(self):
        write_video(self.root / "input" / "clip.mp4", 96, 128, 64)
        clip = {"filename": "clip.mp4", "subfolder": "", "type": "input"}
        shots = [{"id": "a", "kind": "source", "frames": 39, "prompt": "一", "source": {"media": clip, "start": 0, "regen": True}},
                 {"id": "b", "kind": "source", "frames": 34, "prompt": "二", "continue": True, "source": {"media": clip, "start": 39, "regen": True}}]
        job, _ = self.run_project({"shots": shots}, ref_model="ref_model")
        self.assertEqual(self.drift, [render.video_latent_t(22)])
        self.assertEqual([call["model"] for call in self.fake.calls], ["ref_model", "ref_model"])
        # The reference video of b starts on a's last 22 frames, in step with the carried head.
        self.assertTrue(torch.equal(job.source_frames(job.windows[1], 128, 64)[:22], job.source_frames(job.windows[0], 128, 64)[17:39]))

    def test_source_rewrite_samples_its_own_sound_and_keeps_the_source_audio(self):
        # Locked to the source's audio, H3 follows the sound and drifts off the video it rewrites.
        write_video(self.root / "input" / "clip.mp4", 48, 256, 128, audio_seconds=2)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 6, "regen": True}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "改写", "source": source}]
        job, path = self.run_project({"shots": shots}, {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2},
                                     ref_model="ref_model")
        base, refine = self.fake.calls
        self.assertEqual((base["audio_mask"].min().item(), refine["audio_mask"].min().item()), (1.0, 1.0))
        self.assertTrue(base["last"])
        self.assertIsNotNone(refine["audio_noise"])
        with av.open(str(path)) as container:
            audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        self.assertGreater(np.abs(audio[:, 4800:43200]).mean(), 0.1)

    def test_two_pass_refine_splits_one_schedule(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        job, path = self.run_project({"shots": shots}, {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2})
        base_a, base_b, refine_a, refine_b = self.fake.calls
        # 6 steps: the draft runs the first 4 at half size and lands on 0, the last 2 run at the output size.
        self.assertEqual((base_a["shape"][-2:], base_a["sigmas"]), ((4, 8), [1.0, 0.8333, 0.6667, 0.5, 0.0]))
        self.assertEqual((refine_a["shape"][-2:], refine_a["sigmas"]), ((8, 16), [0.3333, 0.1667, 0.0]))
        # Generated sound finishes in the full-size pass too; the continued shot keeps its carried head.
        self.assertEqual(refine_a["audio_mask"].min().item(), 1.0)
        self.assertEqual(refine_b["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertEqual((refine_b["audio_mask"][..., 0].item(), refine_b["audio_mask"][..., -1].item()), (0.0, 1.0))
        with av.open(str(path)) as container:
            self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (256, 128))

    def test_two_pass_source_edit_finishes_inside_its_masks_from_its_own_schedule(self):
        write_video(self.root / "input" / "clip.mp4", 48, 256, 128, audio_seconds=2)
        mask = png(self.root / "input" / "mask.png", 0, size=(256, 128), left_half=True)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 6}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "改左边", "source": source,
                  "repaints": [{"mask": mask, "strength": 0.6, "grow": 0, "feather": 0}]}]
        job, path = self.run_project({"shots": shots}, {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2})
        repaint, refine = self.fake.calls
        self.assertEqual((repaint["shape"][-2:], repaint["start"]), ((4, 8), 0.6))
        self.assertEqual((refine["shape"][-2:], refine["sigmas"]), ((8, 16), [0.2, 0.1, 0.0]))
        self.assertEqual(refine["video_mask"][0, 0, 0, 0].tolist(), [1.0] * 8 + [0.0] * 8)
        self.assertEqual(refine["audio_mask"].abs().sum().item(), 0.0)
        frames = director.output.read_frames(job.keys[0]["frames"])
        expected = job.source_frames(job.windows[0], 256, 128)[:30]
        self.assertTrue(torch.allclose(frames[:, :, 128:], expected[:, :, 128:], atol=1 / 255))

    def test_two_pass_lifts_in_latent_space_and_keeps_the_audio_path(self):
        folder = self.root / "latent_upscale_models"
        folder.mkdir()
        tiny_upscaler(folder / "tiny.safetensors")
        saved = folder_paths.folder_names_and_paths.get("latent_upscale_models")
        folder_paths.folder_names_and_paths["latent_upscale_models"] = ([str(folder)], {".safetensors"})
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        settings = {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2,
                    "two_pass_upscaler": "h3_latent", "two_pass_latent_model": "tiny.safetensors"}
        try:
            job, _ = self.run_project({"shots": shots}, settings)
            base_a, base_b, refine_a, refine_b = self.fake.calls
            # No VAE round trip: the learned upscaler takes the draft latent straight to the output grid.
            self.assertEqual(job.m.vae.encoded, [])
            self.assertEqual(refine_a["shape"][-2:], (8, 16))
            # The draft reports its last step; the full-size pass restarts the audio on the noise it was on there
            # (the stand-in tags it with that step's sigma, 0.5 in [1, .83, .67, .5, 0]).
            self.assertTrue(base_a["last"] and torch.all(refine_a["audio_noise"] == 0.5))
            # A cached draft keeps that noise for a later run that only changes the upscaler.
            job, _ = self.run_project({"shots": shots}, {**settings, "two_pass_upscaler": "lanczos"})
            refine_a, refine_b = self.fake.calls
            self.assertTrue(torch.all(refine_a["audio_noise"] == 0.5))
            self.assertEqual(job.m.vae.encoded[0][1:3], (128, 256))
        finally:
            folder_paths.folder_names_and_paths["latent_upscale_models"] = saved

    def test_two_pass_references_keep_the_output_resolution(self):
        hero = png(self.root / "input" / "hero.png", 200, size=(256, 128))
        shots = [{"id": "a", "frames": 30, "prompt": "一", "refs": {"images": [hero]}}]
        job, _ = self.run_project({"shots": shots}, {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2})
        base = self.fake.calls[0]
        self.assertEqual((base["shape"][-2:], base["refs"]), ((4, 8), ["image"]))
        # The reference is encoded at the 256x128 output size, not squeezed to the 128x64 draft.
        self.assertEqual(job.m.vae.encoded[0], (1, 128, 256, 3))

    def test_fit_modes_place_images_on_the_canvas(self):
        image = torch.zeros(1, 64, 64, 3)
        image[..., 0] = 1.0
        crop, pad, stretch = (director.conditioning.place(image, 128, 64, mode) for mode in ("crop", "pad", "stretch"))
        self.assertEqual([tuple(x.shape) for x in (crop, pad, stretch)], [(1, 64, 128, 3)] * 3)
        # pad centres the square between black bars; crop and stretch fill the canvas.
        self.assertEqual((float(pad[0, 32, 0, 0]), float(pad[0, 32, 64, 0])), (0.0, 1.0))
        self.assertTrue(bool((crop[..., 0] > 0.99).all() and (stretch[..., 0] > 0.99).all()))

    def test_reference_fit_and_strength_reach_the_conditioning(self):
        hero = png(self.root / "input" / "hero.png", 200, size=(64, 64))
        shot = {"id": "a", "frames": 30, "prompt": "一", "ref_strength": 0.5,
                "refs": {"images": [{"media": hero, "fit": "crop"}, {"media": hero}]}}
        job, _ = self.run_project({"shots": [shot]})
        # crop takes the 128x64 canvas aspect at the image's pixel budget; auto keeps the square.
        self.assertEqual(job.m.vae.encoded[:2], [(1, 32, 96, 3), (1, 64, 64, 3)])
        self.assertAlmostEqual(self.fake.calls[0]["cond_strength"], 0.999 * 0.5, places=6)
        self.run_project({"shots": [shot]})
        self.assertEqual(self.fake.calls, [])
        shot["refs"]["images"][0]["fit"] = "pad"
        self.run_project({"shots": [shot]})
        self.assertEqual(len(self.fake.calls), 1)

    def test_two_pass_shot_without_refine_runs_its_whole_schedule_on_the_draft(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一", "refine": False}]
        self.run_project({"shots": shots}, {"two_pass": True, "width": 256, "height": 128, "two_pass_scale": 2.0, "two_pass_steps": 2})
        (base,) = self.fake.calls
        self.assertEqual((base["shape"][-2:], base["steps"]), ((4, 8), 6))

    def test_rerender_resamples_every_shot_and_carries_the_refined_tail(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        job, path = self.run_project({"shots": shots}, {"rerender": True, "width": 256, "height": 128, "rerender_scale": 2.0,
                                                        "rerender_steps": 4, "rerender_denoise": 0.5})
        base, refine_a, refine_b = self.fake.calls[0], self.fake.calls[2], self.fake.calls[3]
        self.assertEqual(base["shape"][-2:], (4, 8))
        self.assertEqual((refine_a["shape"][-2:], refine_a["steps"], refine_a["start"]), ((8, 16), 4, 0.5))
        self.assertEqual(refine_b["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertEqual(refine_b["audio_mask"].abs().sum().item(), 0.0)
        with av.open(str(path)) as container:
            self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (256, 128))

    def test_two_pass_and_rerender_run_one_after_the_other(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        job, path = self.run_project({"shots": shots}, {"two_pass": True, "two_pass_steps": 2, "rerender": True, "rerender_steps": 4,
                                                        "rerender_denoise": 0.5, "width": 256, "height": 128, "two_pass_scale": 2.0})
        # Pass by pass: every shot's two-pass, then every shot's rerender.
        base_a, base_b, tail_a, tail_b, again_a, again_b = self.fake.calls
        self.assertEqual((base_a["shape"][-2:], base_a["sigmas"]), ((4, 8), [1.0, 0.8333, 0.6667, 0.5, 0.0]))
        self.assertEqual((tail_a["shape"][-2:], tail_a["sigmas"]), ((8, 16), [0.3333, 0.1667, 0.0]))
        # The rerender starts from what the tail finished, with its sound locked and noise of its own.
        self.assertEqual((again_a["shape"][-2:], again_a["steps"], again_a["start"]), ((8, 16), 4, 0.5))
        self.assertEqual(again_a["audio_mask"].abs().sum().item(), 0.0)
        self.assertNotEqual(again_a["seed"], tail_a["seed"])
        # Both passes of the continued shot keep the head the first shot ends on, one drift anchor each.
        for call in (tail_b, again_b):
            self.assertEqual(call["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertEqual(len(self.drift), 3)
        self.assertEqual(len([line for line in job.report if "两段式二采 128×64 → 256×128" in line or "低噪重绘 256×128" in line]), 4)
        with av.open(str(path)) as container:
            self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (256, 128))

    def test_rerender_at_scale_one_resamples_the_draft_in_place(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}]
        # A missing latent upscaler would fail the run if the draft were lifted.
        job, _ = self.run_project({"shots": shots}, {"rerender": True, "rerender_scale": 1.0, "rerender_upscaler": "h3_latent",
                                                     "rerender_latent_model": "missing.safetensors"})
        base, again = self.fake.calls
        self.assertEqual((base["shape"][-2:], again["shape"][-2:], again["start"]), ((4, 8), (4, 8), 0.35))
        self.assertEqual(job.m.vae.encoded, [])

    def test_rerender_lora_patches_only_the_rerender_model_once_per_run(self):
        loras = self.root / "loras"
        loras.mkdir()
        import safetensors.torch
        safetensors.torch.save_file({"lora.weight": torch.zeros(1)}, str(loras / "regen.safetensors"))
        saved_folder, saved_load = folder_paths.folder_names_and_paths.get("loras"), render.comfy.sd.load_lora_for_models
        folder_paths.folder_names_and_paths["loras"] = ([str(loras)], {".safetensors"})
        patched = []
        render.comfy.sd.load_lora_for_models = lambda model, clip, lora, strength, strength_clip: patched.append((model, strength)) or (f"{model}+lora", None)
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        settings = {"two_pass": True, "two_pass_steps": 2, "rerender": True, "rerender_steps": 2, "rerender_denoise": 0.71,
                    "rerender_lora": "regen.safetensors", "rerender_lora_strength": 0.8, "width": 256, "height": 128}
        try:
            job, _ = self.run_project({"shots": shots}, settings)
            base_a, base_b, tail_a, tail_b, again_a, again_b = self.fake.calls
            self.assertEqual([c["model"] for c in (base_a, tail_a, again_a, again_b)], ["model", "model", "model+lora", "model+lora"])
            self.assertEqual(patched, [("model", 0.8)])
            self.assertTrue(any("低噪重绘" in line and "LoRA regen.safetensors" in line for line in job.report))
            # A LoRA name is a file name: one that is not in models/loras fails at the load, whatever the combo offered.
            with self.assertRaises(FileNotFoundError):
                self.run_project({"shots": shots}, {**settings, "rerender_lora": "missing.safetensors"})
        finally:
            folder_paths.folder_names_and_paths["loras"] = saved_folder
            render.comfy.sd.load_lora_for_models = saved_load

    @unittest.skipUnless(director.superres.VideoSuperRes is not None and torch.cuda.is_available(), "needs nvidia-vfx and an NVIDIA GPU")
    def test_rtx_vsr_lifts_the_film_from_a_smaller_h3_canvas(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}]
        settings = {"rerender": True, "rerender_scale": 2.0, "rerender_steps": 2, "rtx_vsr": True, "rtx_vsr_scale": 2.0,
                    "width": 512, "height": 256}
        job, path = self.run_project({"shots": shots}, settings)
        base, again = self.fake.calls
        # 512x256 film: RTX lifts it from 256x128, where the rerender ends; the draft is 128x64.
        self.assertEqual((base["shape"][-2:], again["shape"][-2:]), ((4, 8), (8, 16)))
        self.assertEqual((job.out_w, job.out_h), (256, 128))
        self.assertIn("成片超分：RTX ULTRA 256×128 → 512×256", job.report)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            self.assertEqual((stream.width, stream.height, stream.frames), (512, 256, 30))
            frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
        # The fake decoder paints flat colour; super resolution keeps it.
        expected = director.output.read_frames(job.keys[0]["frames"])[0].mean((0, 1)) * 255
        self.assertTrue(np.allclose(frame.reshape(-1, 3).mean(0), expected.numpy(), atol=3))

    def test_each_pass_lifts_by_its_own_scale_and_caches_on_its_own(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        settings = {"two_pass": True, "two_pass_scale": 2.0, "two_pass_steps": 2, "rerender": True, "rerender_scale": 2.0,
                    "rerender_steps": 4, "rerender_denoise": 0.5, "width": 512, "height": 256}
        job, path = self.run_project({"shots": shots}, settings)
        # Pass by pass: every shot's two-pass, then every shot's rerender.
        base_a, base_b, tail_a, tail_b, again_a, again_b = self.fake.calls
        # 512x256 output: the rerender starts from 256x128, two-pass from the 128x64 draft.
        self.assertEqual([c["shape"][-2:] for c in (base_a, tail_a, again_a)], [(4, 8), (8, 16), (16, 32)])
        self.assertEqual((tail_a["sigmas"], again_a["steps"], again_a["start"]), ([0.3333, 0.1667, 0.0], 4, 0.5))
        # The continued shot carries the first shot's result of the same pass, at that pass's size.
        for call in (tail_b, again_b):
            self.assertEqual(call["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        self.assertTrue(any("两段式二采 128×64 → 256×128" in line for line in job.report))
        self.assertTrue(any("低噪重绘 256×128 → 512×256" in line for line in job.report))
        with av.open(str(path)) as container:
            self.assertEqual((container.streams.video[0].width, container.streams.video[0].height), (512, 256))
        # Changing only the rerender re-runs only the rerender, from the cached two-pass results.
        self.run_project({"shots": shots}, {**settings, "rerender_denoise": 0.6})
        self.assertEqual([(c["start"], c["shape"][-2:]) for c in self.fake.calls], [(0.6, (16, 32))] * 2)

    def test_base_cache_key_follows_the_sampler_settings_not_the_passes(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一"}]
        settings = {"rerender": True, "rerender_scale": 2.0, "width": 256, "height": 128}
        self.run_project({"shots": shots}, settings)
        self.run_project({"shots": shots}, {**settings, "rerender_denoise": 0.6})
        self.assertEqual([c["start"] for c in self.fake.calls], [0.6])
        spec = director.project.load_project(json.dumps({"shots": shots}), settings)
        envs = [render.Render("7", {}, spec, models(director), director.media.Media(), [11, steps, 1.0, "euler", "simple"]).env["base"]
                for steps in (6, 8)]
        self.assertNotEqual(*envs)

    def test_passthrough_source_is_encoded_on_every_pass_canvas_for_the_shot_after_it(self):
        write_video(self.root / "input" / "clip.mp4", 48, 512, 256, audio_seconds=2)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 0}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "", "source": source},
                 {"id": "b", "frames": 30, "prompt": "接着", "continue": True}]
        job, _ = self.run_project({"shots": shots}, {"two_pass": True, "two_pass_scale": 2.0, "rerender": True, "rerender_scale": 2.0,
                                                     "width": 512, "height": 256})
        base, tail, again = self.fake.calls
        self.assertEqual([c["shape"][-2:] for c in (base, tail, again)], [(4, 8), (8, 16), (16, 32)])
        for call in (base, tail, again):
            self.assertEqual(call["video_mask"][0, 0, :7].flatten().tolist(), [0.0] * 7)
        sizes = {shape[1:3] for shape in job.m.vae.encoded}
        self.assertTrue({(64, 128), (128, 256), (256, 512)} <= sizes)

    def test_detail_region_is_resampled_and_stitched(self):
        shots = [{"id": "a", "frames": 30, "prompt": "人物", "details": [{"box": [0.25, 0.25, 0.5, 0.5], "denoise": 0.5,
                                                                      "megapixels": 0.1, "prompt": "脸部细节"}]}]
        job, _ = self.run_project({"shots": shots}, {"detail_steps": 4})
        base, crop = self.fake.calls
        self.assertEqual(crop["prompt"], "脸部细节")
        self.assertEqual((crop["steps"], crop["start"]), (4, 0.5))
        self.assertEqual(crop["audio_mask"].abs().sum().item(), 0.0)
        self.assertTrue(director.cache.exists(job.keys[0]["details"][0], ".mkv"))
        self.run_project({"shots": shots}, {"detail_steps": 4})
        self.assertEqual(self.fake.calls, [])

    def test_text_mask_of_a_source_edit_is_tracked_once_on_the_source_frames(self):
        write_video(self.root / "input" / "clip.mp4", 48, 128, 64, audio_seconds=2)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 6}
        layer = {"id": "r", "source": "text", "target": "left person", "threshold": 0.4, "strength": 0.6, "grow": 0, "feather": 0}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "改左边", "source": source, "repaints": [layer]}]
        job, _ = self.run_project({"shots": shots})
        (call,) = self.fake.calls
        self.assertEqual(call["video_mask"][0, 0, 0, 0].tolist(), [1.0] * 4 + [0.0] * 4)
        ((frames, target),) = self.tracked
        self.assertEqual((tuple(frames.shape), target["threshold"]), ((30, 64, 128, 3), 0.4))
        self.assertTrue(torch.allclose(frames, job.source_frames(job.windows[0], 128, 64)[:30]))
        self.assertIn("第 1 镜重绘 1：识别“left person”，1 个目标", job.report)
        out = director.output.read_frames(job.keys[0]["frames"])
        self.assertTrue(torch.allclose(out[:, :, 64:], job.source_frames(job.windows[0], 128, 64)[:30, :, 64:], atol=1 / 255))
        # Cached: neither SAM 3 nor H3 runs again; a new threshold re-tracks and repaints.
        self.run_project({"shots": shots})
        self.assertEqual((len(self.tracked), self.fake.calls), (1, []))
        layer["threshold"] = 0.5
        self.run_project({"shots": shots})
        self.assertEqual((len(self.tracked), len(self.fake.calls)), (2, 1))

    def test_text_mask_of_a_generated_shot_tracks_the_frames_it_repaints(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一", "repaints": [{"source": "text", "target": "left girl", "start": 6, "end": 17, "grow": 0}]}]
        self.run_project({"shots": shots})
        base, repaint = self.fake.calls
        ((frames, _),) = self.tracked
        # Frames 6..17 of the decoded base, which the stand-in sampler filled with one colour per seed.
        self.assertEqual(tuple(frames.shape), (12, 64, 128, 3))
        self.assertTrue(torch.allclose(frames, torch.full_like(frames, (base["seed"] % 89 + 5) / 100)))
        self.assertGreater(repaint["video_mask"][..., :4].sum().item(), 0)
        self.assertEqual(repaint["video_mask"][..., 4:].abs().sum().item(), 0.0)

    def test_crop_layer_regenerates_and_pastes_only_inside_its_mask(self):
        mask = square_mask(self.root / "input" / "mask.png", (40, 20, 60, 40))
        layer = {"id": "c", "mask": mask, "crop": True, "strength": 0.5, "megapixels": 0.1, "grow": 0, "feather": 0, "prompt": "细节"}
        shots = [{"id": "a", "frames": 30, "prompt": "人物", "repaints": [layer]}]
        job, _ = self.run_project({"shots": shots}, {"detail_steps": 4})
        base, crop = self.fake.calls
        self.assertEqual((crop["prompt"], crop["steps"], crop["start"]), ("细节", 4, 0.5))
        # The crop keeps a margin around the square: its corner tokens stay, the square is regenerated.
        self.assertEqual(crop["video_mask"].amax().item(), 1.0)
        self.assertEqual(crop["video_mask"][..., 0, 0].abs().sum().item(), 0.0)
        key = job.keys[0]["details"][0]
        weights = director.cache.read_tensors(key)["mask"].float() / 255
        middle = weights[:, weights.shape[1] // 2, weights.shape[2] // 2]
        self.assertEqual((weights[:, 0, 0].max().item(), middle.min().item()), (0.0, 1.0))
        self.assertTrue(any("重绘 1 放大贴回" in line for line in job.report))

    def test_stitch_blends_through_the_given_weights(self):
        frames, crops = torch.rand(2, 32, 32, 3), torch.rand(2, 32, 32, 3)
        weights = torch.zeros(2, 32, 32)
        weights[:, :, :16] = 1.0
        before = frames.clone()
        out = director.detail.stitch(frames, crops, [[0.0, 0.0, 32.0, 32.0]] * 2, False, weights)
        self.assertTrue(torch.equal(out[:, :, 20:], before[:, :, 20:]))
        self.assertFalse(torch.allclose(out[:, :, :12], before[:, :, :12]))

    def test_empty_text_mask_skips_its_crop_layer_once(self):
        shots = [{"id": "a", "frames": 30, "prompt": "一", "repaints": [{"source": "text", "target": "nothing", "crop": True}]}]
        job, _ = self.run_project({"shots": shots})
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("第 1 镜：重绘 1 的遮罩是空的，跳过", job.report)
        self.run_project({"shots": shots})
        self.assertEqual((self.fake.calls, len(self.tracked)), ([], 1))

    def test_legacy_detail_region_is_a_box_crop_layer(self):
        shots = [{"id": "a", "frames": 30, "prompt": "人物", "details": [{"box": [0.25, 0.25, 0.5, 0.5], "denoise": 0.5, "megapixels": 0.1}]}]
        job, _ = self.run_project({"shots": shots}, {"detail_steps": 4})
        (item,) = job.windows[0].shot["details"]
        self.assertEqual((item["source"], item["strength"], item["crop"]), ("box", 0.5, True))
        # A box crop is resampled whole and pasted with its feathered rectangle, so no mask is stored.
        self.assertIsNone(self.fake.calls[1]["video_mask"].min().item() < 1.0 or None)
        self.assertFalse(director.cache.exists(job.keys[0]["details"][0]))

    def test_mask_preview_dims_outside_the_tracked_mask_without_sampling(self):
        write_video(self.root / "input" / "clip.mp4", 48, 128, 64, audio_seconds=2)
        source = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "start": 30}
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "", "source": source,
                  "repaints": [{"id": "r", "source": "text", "target": "left person", "start": 0, "end": 11, "grow": 0}]},
                 {"id": "g", "frames": 30, "prompt": "一", "continue": False, "repaints": [{"id": "q", "source": "text", "target": "cup"}]}]
        spec = director.project.load_project(json.dumps({"shots": shots}), {"width": 128, "height": 64})
        job = render.Render("7", {}, spec, models(director), director.media.Media(), [11, 6, 1.0, "euler", "simple"])
        info = job.preview("r")
        self.assertEqual((info["frames"], info["coverage"]), (12, 0.5))
        image = Path(folder_paths.get_temp_directory()) / info["image"]["subfolder"] / info["image"]["filename"]
        with Image.open(image) as picture:
            luma = np.asarray(picture.convert("L"), dtype=np.float32) / 255
            self.assertEqual((picture.is_animated, picture.size), (True, (128, 64)))
        # The tracked left half keeps its brightness, the rest is dimmed to a quarter.
        self.assertAlmostEqual(luma[:, 68:].mean() / luma[:, :60].mean(), 0.25, delta=0.05)
        self.assertEqual(self.fake.calls, [])
        # A generated shot's mask is found on its own render, which does not exist yet.
        with self.assertRaisesRegex(ValueError, "先运行一次"):
            job.preview("q")

    def test_node_preview_evaluates_only_sam3_vae_and_rewriter(self):
        node = director.director.YogurtH3Director
        shots = [{"id": "a", "frames": 30, "prompt": "", "repaints": [{"id": "r", "source": "text", "target": "cup"}]}]
        inputs = dict(model=None, clip=None, vae=None, audio_vae=None, rewriter=None, sam3=None, sam3_clip=None, seed=1)
        preview = json.dumps({"shots": shots, "mask_preview": {"layer": "r"}})
        self.assertEqual(node.check_lazy_status(preview, **inputs), ["vae", "rewriter", "sam3", "sam3_clip"])
        self.assertEqual(node.check_lazy_status(json.dumps({"shots": shots}), **{**inputs, "model": "loaded"}),
                         ["clip", "vae", "audio_vae", "rewriter", "sam3", "sam3_clip"])
        # Without a text target the SAM 3 checkpoint is never loaded.
        shots[0]["repaints"][0]["source"] = "box"
        self.assertEqual(node.check_lazy_status(json.dumps({"shots": shots}), **inputs), ["model", "clip", "vae", "audio_vae", "rewriter"])
        # 读取连线素材 needs no model at all.
        self.assertEqual(node.check_lazy_status(json.dumps({"shots": shots, "inputs_preview": True}), **inputs), [])

    def test_a_run_announces_its_phases_and_their_shots(self):
        sent, saved = [], render.send_plan
        render.send_plan = lambda node_id, phases: sent.append(phases)
        shots = [{"id": "a", "frames": 30, "prompt": "一"}, {"id": "b", "frames": 30, "prompt": "二"}]
        settings = {"rerender": True, "rerender_steps": 2, "width": 128, "height": 64, "rerender_scale": 2.0}
        try:
            self.run_project({"shots": shots}, settings)
            self.assertEqual(sent[-1], [["encoding", ["a", "b"]], ["sampling", ["a", "b"]], ["rerender", ["a", "b"]], ["decoding", ["a", "b"]],
                                        ["detailing", []], ["assembling", ["a", "b"]]])
            # Only the edited shot is worked on again; the panel shows the other one as cached.
            shots[1]["prompt"] = "改"
            self.run_project({"shots": shots}, settings)
            self.assertEqual(sent[-1][:4], [["encoding", ["b"]], ["sampling", ["b"]], ["rerender", ["b"]], ["decoding", ["b"]]])
        finally:
            render.send_plan = saved

    def test_foreign_targets_are_translated_once_by_the_rewrite_model(self):
        replies = []

        class Chat:
            def invoke(self, messages):
                replies.append(messages[-1].content)
                return SimpleNamespace(text="girl in red raincoat\n")
        saved, director.rewrite.chat_model = director.rewrite.chat_model, lambda config: Chat()
        try:
            shots = [{"id": "a", "frames": 30, "prompt": "", "repaints": [{"source": "text", "target": "红雨衣女孩"}, {"source": "text", "target": "cup"}]}]
            config = director.rewrite.rewriter_config("openai", "m", api_key="secret")
            for _ in range(2):
                spec = director.project.load_project(json.dumps({"shots": shots}), {"width": 128, "height": 64})
                english = director.director._translate(spec, config, [])
            self.assertEqual((english, replies), ({"红雨衣女孩": "girl in red raincoat"}, ["红雨衣女孩"]))
            self.assertEqual([layer["target"].get("english") for layer in spec["shots"][0]["repaints"]], ["girl in red raincoat", None])
            report = []
            director.director._translate(spec, None, report)
            self.assertIn("不是英文", report[0])
        finally:
            director.rewrite.chat_model = saved

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

    def test_connected_source_video_and_soundtrack(self):
        write_video(self.root / "clip.mp4", 48, 128, 64)
        media = director.media.Media(videos={"video_0": InputImpl.VideoFromFile(str(self.root / "clip.mp4"))},
                                     audios={"audio_0": {"waveform": torch.full((1, 1, 32000 * 3), 0.2), "sample_rate": 32000}})
        shots = [{"id": "s", "kind": "source", "frames": 30, "prompt": "", "audio": "soundtrack", "source": {"media": {"input": "video_0"}, "start": 6}}]
        job, path = self.run_project({"shots": shots, "soundtrack": {"input": "audio_0"}}, media=media)
        self.assertEqual(self.fake.calls, [])
        frames = director.output.read_frames(job.keys[0]["frames"])
        self.assertEqual([round(v * 47) for v in frames[:, 0, 0, 0].tolist()], list(range(6, 36)))
        with av.open(str(path)) as container:
            audio = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        self.assertAlmostEqual(float(np.abs(audio[:, 4800:43200]).mean()), 0.2, delta=0.02)

    def test_reference_video_starts_at_its_in_point(self):
        write_video(self.root / "input" / "ref.mp4", 96, 128, 64)
        ref = {"media": {"filename": "ref.mp4", "subfolder": "", "type": "input"}, "start": 30, "audio": True}
        spec = director.project.load_project(json.dumps({"shots": [{"id": "a", "frames": 30, "prompt": "", "refs": {"videos": [ref]}}]}),
                                             {"width": 128, "height": 64})
        window = director.project.plan_windows(spec)[0]
        items, _ = director.conditioning.references(director.media.Media(), window, spec, None, FakeVAE(), FakeAudioVAE())
        video = next(item for item in items if item["type"] == "video")
        self.assertEqual([round(v * 95) for v in video["data"][:3, 0, 0, 0].tolist()], [30, 42, 54])

    def test_node_reads_setting_widgets_and_video_sockets(self):
        write_video(self.root / "clip.mp4", 48, 128, 64)
        node = director.director.YogurtH3Director
        project = json.dumps({"shots": [{"id": "s", "kind": "source", "frames": 30, "prompt": "", "source": {"media": {"input": "video_0"}, "start": 6}}]})
        saved, node.hidden = node.hidden, SimpleNamespace(unique_id="9", prompt={})
        try:
            out = node.execute(model="model", clip=FakeClip(), vae=FakeVAE(), audio_vae=FakeAudioVAE(), seed=1, steps=4, cfg=1.0, sampler_name="euler",
                               scheduler="simple", filename_prefix="test/node", project=project,
                               videos={"video_0": InputImpl.VideoFromFile(str(self.root / "clip.mp4"))}, audios={"audio_0": None},
                               width=96, height=64, overlap="22", ref_image_size="match", negative_prompt="", two_pass=False, two_pass_scale=2.0, two_pass_steps=2,
                               two_pass_upscaler="lanczos", two_pass_latent_model="missing.safetensors", rerender=False, rerender_scale=1.0,
                               rerender_steps=2, rerender_denoise=0.35, rerender_upscaler="lanczos", rerender_latent_model="missing.safetensors",
                               detail=True, detail_steps=8)
        finally:
            node.hidden = saved
        info = out.ui["yogurt_h3"][0]
        self.assertEqual((info["size"], info["timeline"][0]["frames"]), ([96, 64], 30))


if __name__ == "__main__":
    unittest.main()
