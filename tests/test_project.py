import json
import unittest

import torch

from support import load

director = load()
project = director.project
sampling = director.sampling


def build(*shots, **extra):
    return project.load_project(json.dumps({"width": 128, "height": 64, "shots": list(shots), **extra}))


class WindowTests(unittest.TestCase):
    def test_single_shot_crops_grid_padding(self):
        (window,) = project.plan_windows(build({"frames": 100}))
        self.assertEqual((window.head, window.gen, window.keep, window.start), (0, 107, 100, 0))

    def test_continued_shots_keep_the_whole_window_before_a_carry(self):
        windows = project.plan_windows(build({"frames": 100}, {"frames": 100}, {"frames": 50, "continue": False}))
        self.assertEqual([(w.head, w.gen, w.keep, w.start) for w in windows],
                         [(0, 107, 107, 0), (22, 124, 100, 107), (0, 56, 50, 207)])
        self.assertEqual([w.carry for w in windows], [False, True, False])

    def test_end_keyframe_pins_the_last_generated_frame(self):
        shot = {"frames": 100, "keyframes": [{"at": 0, "media": {"filename": "a.png"}}, {"at": -1, "media": {"filename": "b.png"}}]}
        (window,) = project.plan_windows(build(shot))
        self.assertEqual(window.keep, window.gen)
        self.assertEqual([window.frame(k["at"]) for k in window.shot["keyframes"]], [0, window.gen - 1])

    def test_keyframes_of_a_continued_shot_start_after_the_carried_frames(self):
        windows = project.plan_windows(build({"frames": 60}, {"frames": 60, "keyframes": [{"at": 0, "media": {"filename": "a.png"}}]}))
        self.assertEqual(windows[1].frame(0), 22)
        self.assertEqual(windows[1].span(0, -1), (22, 22 + windows[1].keep))

    def test_too_short_previous_shot_falls_back_to_a_cut(self):
        windows = project.plan_windows(build({"frames": 5}, {"frames": 60}))
        self.assertFalse(windows[1].carry)
        self.assertEqual(windows[1].head, 0)

    def test_source_shots_never_carry(self):
        source = {"kind": "source", "frames": 40, "source": {"media": {"filename": "v.mp4"}}}
        windows = project.plan_windows(build({"frames": 40}, source))
        self.assertFalse(windows[1].carry)
        self.assertEqual(windows[1].shot["audio"], "source")

    def test_overlap_snaps_down_to_the_latent_grid(self):
        self.assertEqual(build({"frames": 10}, overlap=30)["overlap"], 22)
        self.assertEqual(build({"frames": 10}, overlap=4)["overlap"], 0)

    def test_validation_messages(self):
        with self.assertRaisesRegex(ValueError, "还没有镜头"):
            project.load_project("{}")
        with self.assertRaisesRegex(ValueError, "配乐音轨"):
            build({"frames": 10, "audio": "soundtrack"})
        with self.assertRaisesRegex(ValueError, "源视频"):
            build({"kind": "source", "frames": 10})
        with self.assertRaisesRegex(ValueError, "素材目录"):
            build({"frames": 10, "keyframes": [{"at": 0, "media": {"filename": "a.png", "type": "../x"}}]})


class MaskTests(unittest.TestCase):
    def test_token_mask_follows_the_h3_temporal_grid(self):
        pixels = torch.zeros((22, 64, 128))
        pixels[5, 10, 40] = 1.0  # frame 5 is the second frame of token 2 (frames 5-8)
        mask = sampling.token_mask(pixels, 7)
        self.assertEqual(tuple(mask.shape), (1, 1, 7, 4, 8))
        self.assertEqual(mask[0, 0, :, 0, 2].tolist(), [0, 0, 1, 0, 0, 0, 0])

    def test_window_mask_holds_a_still_and_pads_a_sequence(self):
        still = sampling.window_mask(torch.ones((1, 8, 8)), 10, 2, 5, 16, 16)
        self.assertEqual(still.sum(dim=(1, 2)).tolist(), [0, 0, 256, 256, 256, 0, 0, 0, 0, 0])
        sequence = torch.stack([torch.full((8, 8), v) for v in (0.25, 0.5)])
        placed = sampling.window_mask(sequence, 6, 1, 5, 8, 8)
        self.assertEqual(placed[:, 0, 0].tolist(), [0, 0.25, 0.5, 0.5, 0.5, 0])

    def test_partial_passes_start_at_the_requested_noise_level(self):
        import comfy.model_sampling

        class Flow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        flow = Flow()
        flow.set_parameters(shift=12.0, multiplier=1000)
        model = type("Model", (), {"get_model_object": lambda self, name: flow})()
        sigmas = sampling.schedule(model, 6, "simple", 0.35)
        self.assertEqual(sigmas.shape[-1], 7)
        self.assertLessEqual(float(sigmas[0]), 0.35 + 1e-6)
        self.assertGreater(float(sigmas[0]), 0.3)
        self.assertAlmostEqual(float(sampling.schedule(model, 6, "simple", 1.0)[0]), 1.0, places=3)

    def test_custom_sigmas_keep_the_denoise_tail(self):
        custom = torch.linspace(1, 0, 9)
        self.assertEqual(sampling.schedule(None, 0, "simple", 0.5, custom).tolist(), custom[-5:].tolist())


if __name__ == "__main__":
    unittest.main()
