import json
import unittest

import torch

from support import load

director = load()
project = director.project
sampling = director.sampling


def build(*shots, settings=None, **extra):
    return project.load_project(json.dumps({"shots": list(shots), **extra}), {"width": 128, "height": 64, **(settings or {})})


class PromptTests(unittest.TestCase):
    SOURCE = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}}
    LAYER = {"source": "box", "box": [0.1, 0.1, 0.5, 0.5]}

    def prompts(self, *shots, rewrite=False):
        loaded = build(*shots, story="全局", rewrite={"enabled": rewrite})
        return [shot["prompt"] for shot in loaded["shots"]]

    def test_a_shot_without_a_prompt_uses_the_global_one(self):
        self.assertEqual(self.prompts({"prompt": ""}, {"prompt": "  "}, {"prompt": "本镜"}), ["全局", "全局", "本镜"])

    def test_a_source_rewrite_uses_it_too(self):
        self.assertEqual(self.prompts({"kind": "source", "source": {**self.SOURCE, "regen": True}}), ["全局"])

    def test_a_shot_that_does_not_inherit_keeps_its_empty_prompt(self):
        self.assertEqual(self.prompts({"inherit_prompt": False}), [""])

    def test_the_rewrite_model_writes_from_the_global_prompt_itself(self):
        self.assertEqual(self.prompts({}, {"rewrite": False}, rewrite=True), ["", "全局"])

    def test_a_kept_source_takes_it_only_for_its_repaint_layers(self):
        kept = {"kind": "source", "source": self.SOURCE}
        self.assertEqual(self.prompts(kept, {**kept, "repaints": [self.LAYER]}, rewrite=True), ["", "全局"])


class SwapTests(unittest.TestCase):
    SOURCE = {"media": {"filename": "clip.mp4", "subfolder": "", "type": "input"}, "regen": True, "strength": 0.5}
    PICTURE = {"images": [{"filename": "who.png"}]}

    def swap(self, *shots, refs=None, **swap):
        return build(*shots, story="全局", refs=refs or self.PICTURE, rewrite={"enabled": True},
                     swap={"enabled": True, "what": "穿白衬衫的男人", **swap})

    def test_a_replaced_shot_takes_the_template_at_full_strength(self):
        loaded = self.swap({"id": "a", "kind": "source", "prompt": "旧提示词", "source": self.SOURCE})
        shot = loaded["shots"][0]
        self.assertEqual(shot["prompt"], project.swap_prompt("穿白衬衫的男人", "", True, False))
        self.assertTrue(shot["prompt"].startswith("subject_definitions:\n<Subject 1> is the person in <Picture 1>, whose appearance comes from <Picture 1> and "))
        self.assertIn("takes the place of the 穿白衬衫的男人;", shot["prompt"])
        for section in ("summary:\n[video editing]", "retention_analysis:\n", "attribute_transfer", "detailed_description:\n", "overall_soundscape:\n", "non_diegetic_music:\nN/A"):
            self.assertIn(section, shot["prompt"])
        self.assertEqual((shot["source"]["regen"], shot["source"]["strength"], shot["repaints"]), (True, 1.0, []))

    def test_it_switches_the_global_prompt_off(self):
        loaded = self.swap({"prompt": ""}, {"prompt": "本镜"})
        self.assertEqual([shot["prompt"] for shot in loaded["shots"]], ["", "本镜"])

    def test_tracking_with_blank_fields_waits_for_the_rewrite_model(self):
        source = {"kind": "source", "source": self.SOURCE}
        self.assertEqual(self.swap(source, what="", track=True)["shots"][0]["repaints"][0]["target"]["text"], "person")
        unwritten = build(source, refs=self.PICTURE, swap={"enabled": True, "what": "", "track": True})["shots"][0]
        self.assertEqual(unwritten["repaints"], [])

    def test_a_kept_shot_stays_as_it_is(self):
        shot = self.swap({"kind": "source", "source": {**self.SOURCE, "regen": False}})["shots"][0]
        self.assertEqual((shot["prompt"], shot["source"]["regen"], shot["source"]["reference"], shot["repaints"]), ("", False, False, []))

    def test_tracked_keeps_the_pixels_and_shows_the_source_as_a_video(self):
        shot = self.swap({"id": "a", "kind": "source", "source": self.SOURCE, "repaints": [{"source": "box", "box": [0, 0, 1, 1]}]}, what="man in white shirt", track=True, objects=2)["shots"][0]
        self.assertEqual((shot["source"]["regen"], shot["source"]["reference"], shot["ref_strength"]), (False, True, 0.7))
        first, second = shot["repaints"]
        self.assertEqual((first["id"], first["target"], first["grow"], first["number"], second["number"]),
                         ("a-swap", {"text": "man in white shirt", "threshold": 0.5, "objects": 2}, 24, 1, 2))
        self.assertNotIn("man in white shirt", shot["prompt"])
        self.assertIn("<Video 1>", shot["prompt"])

    def test_tracked_without_a_picture_still_samples_with_the_reference_model(self):
        shot = self.swap({"kind": "source", "source": self.SOURCE}, what="man", into="a robot", track=True, refs={"images": []})["shots"][0]
        self.assertTrue(director.render.Render.uses_refs(shot))

    def test_nothing_to_replace_with_leaves_no_prompt(self):
        source = {"kind": "source", "source": self.SOURCE}
        self.assertEqual(self.swap(source, refs={"images": []})["shots"][0]["prompt"], "")
        self.assertEqual(self.swap(source, what="  ")["shots"][0]["prompt"], "")
        self.assertIn("穿红裙的女人", self.swap(source, into="穿红裙的女人", refs={"images": []})["shots"][0]["prompt"])

    def test_words_for_the_new_look_point_at_the_reference_picture(self):
        self.assertIn("<Subject 1> is the samurai in <Picture 1>,", project.swap_prompt("man", "samurai", True, False))
        for typed in ("<Picture 2> samurai", "<Picture 2> 里的samurai"):
            tagged = project.swap_prompt("man", typed, True, False)
            self.assertIn("<Subject 1> is the samurai in <Picture 2>, whose appearance comes from <Picture 2> and ", tagged)
            self.assertIn("clothing of <Picture 2> are transferred", tagged)
            self.assertNotIn("<Picture 1>", tagged)
        words = project.swap_prompt("man", "samurai", False, False)
        self.assertNotIn("<Picture", words)
        self.assertIn("<Subject 1> is the samurai, whose position, pose", words)
        self.assertIn("this look is transferred", words)

    def test_off_changes_nothing(self):
        loaded = build({"kind": "source", "source": self.SOURCE}, story="全局", swap={"enabled": False, "what": "x", "into": "y"})
        self.assertEqual((loaded["shots"][0]["prompt"], loaded["shots"][0]["source"]["strength"]), ("全局", 0.5))


class LayerTests(unittest.TestCase):
    def layers(self, *items):
        shot = build({"frames": 30, "repaints": list(items)})["shots"][0]
        return shot["repaints"] + shot["details"]

    def test_mask_sources(self):
        painted = {"filename": "m.png", "subfolder": "", "type": "input"}
        layers = self.layers({"mask": painted}, {"source": "input", "input": "mask_1", "mask": painted},
                             {"mask": {"input": "mask_0", "file": painted}}, {"source": "text", "target": " girl, umbrella:2 ", "objects": 99},
                             {"source": "text", "target": "  "}, {"source": "face", "box": [0.2, 0.2, 0.3, 0.4]}, {"source": "box"})
        self.assertEqual([(layer["source"], layer["number"]) for layer in layers], [("paint", 1), ("input", 2), ("input", 3), ("text", 4), ("face", 6)])
        self.assertEqual([layer["mask"] for layer in layers[:3]], [painted, {"input": "mask_1"}, {"input": "mask_0"}])
        self.assertEqual(layers[3]["target"], {"text": "girl, umbrella:2", "threshold": 0.5, "objects": 64})
        self.assertEqual(layers[4]["box"], [0.2, 0.2, 0.3, 0.4])

    def test_crop_switch_moves_a_layer_after_the_draft(self):
        shot = build({"frames": 30, "repaints": [{"source": "box", "box": [0, 0, 1, 1], "crop": True, "strength": 0.3},
                                                 {"source": "box", "box": [0, 0, 1, 1]}]})["shots"][0]
        self.assertEqual([(layer["number"], layer["strength"]) for layer in shot["details"]], [(1, 0.3)])
        self.assertEqual([layer["number"] for layer in shot["repaints"]], [2])

    def test_mask_preview_request(self):
        self.assertEqual(build({"frames": 30}, mask_preview={"layer": 7})["preview"], {"layer": "7"})
        self.assertIsNone(build({"frames": 30})["preview"])


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

    def test_a_source_shot_after_a_generated_one_starts_fresh(self):
        source = {"kind": "source", "frames": 40, "continue": True, "source": {"media": {"filename": "v.mp4"}}}
        windows = project.plan_windows(build({"frames": 40}, source))
        self.assertFalse(windows[1].carry)
        self.assertEqual(windows[1].shot["audio"], "source")

    def test_source_shots_carry_only_where_the_same_video_left_off(self):
        def source(start, frames):
            return {"kind": "source", "frames": frames, "continue": True, "source": {"media": {"filename": "v.mp4"}, "start": start}}
        windows = project.plan_windows(build(source(0, 39), source(39, 34), source(90, 40)))
        self.assertEqual([w.carry for w in windows], [False, True, False])
        self.assertEqual([(w.head, w.gen, w.keep) for w in windows[:2]], [(0, 39, 39), (22, 56, 34)])
        # The previous window must end on its own last frame: 40 frames pad to 56, so the next shot joins without carrying.
        windows = project.plan_windows(build(source(0, 40), source(40, 30)))
        self.assertFalse(windows[1].carry)
        # Source shots continue only when asked.
        windows = project.plan_windows(build(source(0, 39), {"kind": "source", "frames": 30, "source": {"media": {"filename": "v.mp4"}, "start": 39}}))
        self.assertFalse(windows[1].carry)

    def test_global_references_come_first_and_can_be_switched_off(self):
        image = lambda name: {"filename": name}
        data = build({"frames": 40, "refs": {"images": [image("own.png")]}},
                     {"frames": 40, "inherit_refs": False, "refs": {"images": [image("solo.png")]}},
                     {"frames": 40},
                     refs={"images": [image("hero.png"), image("cafe.png")], "audios": [image("voice.wav")]})
        a, b, c = data["shots"]
        self.assertEqual([m["media"]["filename"] for m in a["refs"]["images"]], ["hero.png", "cafe.png", "own.png"])
        self.assertEqual(a["global_refs"], {"images": 2, "videos": 0, "audios": 1})
        self.assertEqual([m["media"]["filename"] for m in b["refs"]["images"]], ["solo.png"])
        self.assertEqual(b["global_refs"], {"images": 0, "videos": 0, "audios": 0})
        self.assertEqual(len(c["refs"]["audios"]), 1)
        self.assertTrue(a["inherit_prompt"])

    def test_fit_and_reference_strength_are_parsed(self):
        data = build({"frames": 40, "ref_strength": 0.05, "keyframes": [{"at": 0, "media": {"filename": "a.png"}, "fit": "pad"}],
                      "refs": {"images": [{"media": {"filename": "b.png"}, "fit": "crop"}, {"media": {"filename": "c.png"}, "fit": "zoom"}]}})
        shot = data["shots"][0]
        self.assertEqual([m["fit"] for m in shot["refs"]["images"]], ["crop", "auto"])
        self.assertEqual((shot["keyframes"][0]["fit"], shot["ref_strength"]), ("pad", 0.1))

    def test_references_beyond_the_ui_limit_are_kept(self):
        shot = build({"frames": 40, "refs": {"images": [{"filename": f"{i}.png"} for i in range(11)]}})["shots"][0]
        self.assertEqual(len(shot["refs"]["images"]), 11)

    def test_detail_switch_drops_every_region(self):
        layers = {"frames": 30, "details": [{"box": [0.1, 0.1, 0.3, 0.3]}],
                  "repaints": [{"source": "box", "box": [0, 0, 0.5, 0.5], "crop": True}, {"source": "box", "box": [0, 0, 0.5, 0.5]}]}
        self.assertEqual(len(build(layers)["shots"][0]["details"]), 2)
        self.assertEqual(len(build(layers, settings={"detail": False})["shots"][0]["repaints"]), 1)
        shot = {"frames": 30, "details": [{"box": [0.1, 0.1, 0.3, 0.3]}]}
        self.assertEqual(len(build(shot)["shots"][0]["details"]), 1)
        self.assertEqual(build(shot, settings={"detail": False})["shots"][0]["details"], [])

    def test_a_cover_shows_a_frame_of_a_shot_that_still_exists(self):
        source = {"media": {"filename": "v.mp4"}, "start": 0}
        cover = lambda still: {"id": "c", "kind": "source", "frames": 2, "source": source, "still": still}
        shot = {"id": "s", "kind": "source", "frames": 40, "source": {**source, "start": 2}}
        stills = [build(cover(still), shot)["shots"][0]["still"]
                  for still in ({"shot": "s", "frame": 30}, {"shot": "s", "frame": 99}, {"shot": "gone", "frame": 3}, {"shot": "c", "frame": 0})]
        self.assertEqual(stills, [{"shot": "s", "frame": 30}, {"shot": "s", "frame": 39}, None, None])
        self.assertIsNone(build({"id": "g", "frames": 10, "still": {"shot": "s"}}, shot)["shots"][0]["still"])

    def test_overlap_snaps_down_to_the_latent_grid(self):
        self.assertEqual(build({"frames": 10}, settings={"overlap": "30"})["overlap"], 22)
        self.assertEqual(build({"frames": 10}, settings={"overlap": "4"})["overlap"], 0)

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

    def test_regenerate_lora_settings_reach_the_official_two_step_grid(self):
        import comfy.model_sampling

        class Flow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        flow = Flow()
        flow.set_parameters(shift=6.0, multiplier=1000)
        model = type("Model", (), {"get_model_object": lambda self, name: flow})()
        # Open-H3-Regenerate's 2-step grid is 0.7, 0.494, 0; rerender_denoise 0.71 / 2 steps at shift 6 lands next to it.
        self.assertEqual([round(float(s), 2) for s in sampling.schedule(model, 2, "simple", 0.71)], [0.71, 0.5, 0.0])

    def test_rerender_lora_and_superres_settings(self):
        spec = build({"id": "a"}, settings={"rerender": True, "rerender_lora": "none", "rtx_vsr": True, "rtx_vsr_scale": 9,
                                            "rtx_vsr_quality": "BOGUS"})
        self.assertEqual((spec["refine"]["rerender"]["lora"], spec["superres"]), (None, {"scale": 4.0, "quality": "ULTRA"}))
        spec = build({"id": "a"}, settings={"rerender": True, "rerender_lora": "MinimaxH3\\regen.safetensors"})
        self.assertEqual((spec["refine"]["rerender"]["lora"], spec["superres"]), ("MinimaxH3\\regen.safetensors", None))

    def test_width_and_height_are_the_draft_each_pass_lifts(self):
        spec = build({"id": "a"}, settings={"width": 608, "height": 352, "two_pass": True, "two_pass_scale": 1.5, "rerender": True,
                                            "rerender_scale": 1.5, "rtx_vsr": True, "rtx_vsr_scale": 2.0})
        # On the 32 px grid with Python's round: 912 x 528 lands on 896 x 512.
        self.assertEqual((spec["canvas"], spec["film"]), ([(608, 352), (896, 512), (1344, 768)], (2688, 1536)))
        spec = build({"id": "a"}, settings={"width": 1344, "height": 768})
        self.assertEqual((spec["canvas"], spec["film"]), ([(1344, 768)], (1344, 768)))

    def test_custom_sigmas_keep_the_denoise_tail(self):
        custom = torch.linspace(1, 0, 9)
        self.assertEqual(sampling.schedule(None, 0, "simple", 0.5, custom).tolist(), custom[-5:].tolist())


if __name__ == "__main__":
    unittest.main()
