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
        self.saved = render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise, render.install_drift_control
        render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise = self.fake.sample, self.fake.schedule, self.fake.trajectory_noise
        self.drift = []
        render.install_drift_control = lambda model, latent, sigmas, tokens: self.drift.append(tokens) or model

    def tearDown(self):
        render.sampling.sample, render.sampling.schedule, render.sampling.trajectory_noise, render.install_drift_control = self.saved
        folder_paths.set_output_directory(self.saved_dirs[0])
        folder_paths.set_input_directory(self.saved_dirs[1])
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
        self.assertEqual(call["audio_mask"].abs().sum().item(), 0.0)
        frames = director.output.read_frames(job.keys[0]["frames"])
        expected = job.source_frames(job.windows[0], 128, 64)[:30]
        self.assertTrue(torch.allclose(frames[:, :, 64:], expected[:, :, 64:], atol=1 / 255))
        self.assertFalse(torch.allclose(frames[:, :, :64], expected[:, :, :64], atol=0.02))
        self.assertAlmostEqual(self.probe(path)[1], 30 / 24, delta=0.05)

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
        tiny_upscaler(director, folder / "tiny.safetensors")
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
