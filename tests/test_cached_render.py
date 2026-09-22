import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_continuity_runtime import PLUGIN, torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent


RENDER = importlib.import_module(PLUGIN.__name__ + ".render")
CACHE = importlib.import_module(PLUGIN.__name__ + ".cache")


def av_latent(value):
    return {"samples": NestedTensor((torch.full((1, 24, 42, 2, 2), float(value)), torch.full((1, 32, 2, 235), float(value))))}


def replace_plan(refine=None, **segment):
    segments = [dict(index=0, start_frame=0, content_frames=141, generation_frames=141, overlap_frames=0),
                dict(index=1, start_frame=119, content_frames=141, generation_frames=141, overlap_frames=22, **segment)]
    segments = [dict(item, prompt="role prompt", mode="r2v", images=[0], source=True) for item in segments]
    return RENDER.make_plan((32, 32), 0.001024, segments, 260, "source", False, "match",
                            [torch.full((1, 32, 32, 3), 0.8)], None, "source video", refine)


class CachedRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.output_patch = patch.object(CACHE.folder_paths, "get_output_directory", return_value=self.temporary.name)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.event_patch = patch.object(RENDER, "segment_event")
        self.event_patch.start()
        self.addCleanup(self.event_patch.stop)
        self.previous = RENDER._carry(torch.full((141, 32, 32, 3), 0.2), av_latent(0.2), None)
        self.arguments = dict(
            model="model", clip="clip", vae="vae", audio_vae="audio_vae", plan=replace_plan(), index=1,
            sampler="sampler", sigmas=torch.tensor([1.0, 0.0]), seed=10, base_key=CACHE.digest("test job"),
            cache_mode="reuse", owner="parent", force_rerun=False, previous=self.previous, master_audio="master audio",
        )

    def nodes(self, result, class_type):
        return [node["inputs"] for node in result.expand.values() if node["class_type"] == class_type]

    def cache_first_result(self):
        result = RENDER.YogurtH3CachedSegment.execute(**self.arguments)
        save = self.nodes(result, "YogurtH3SaveSegment")[0]
        kwargs = {key: value for key, value in save.items() if key not in ("images", "latent")}
        RENDER.YogurtH3SaveSegment.execute(images=torch.arange(158.0).reshape(158, 1, 1, 1), latent=av_latent(0.5), **kwargs)
        return save["key"]

    def test_resume_reuses_complete_window_and_latent_without_resampling(self):
        self.cache_first_result()
        result = RENDER.YogurtH3CachedSegment.execute(**self.arguments)
        self.assertIsNone(result.expand)
        images, audio, carry = result.result
        torch.testing.assert_close(images.flatten(), torch.arange(141.0))
        self.assertIsNone(audio)
        torch.testing.assert_close(carry["low"]["samples"].unbind()[0], av_latent(0.5)["samples"].unbind()[0])
        self.assertEqual(carry["images"].flatten().tolist(), [140.0])

    def test_rerun_and_changed_upstream_latent_do_not_reuse_old_window(self):
        self.cache_first_result()
        self.assertTrue(RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, force_rerun=True)).expand)
        changed = RENDER._carry(self.previous["images"], av_latent(0.3), None)
        self.assertTrue(RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, previous=changed)).expand)

    def test_incomplete_generation_cannot_overwrite_valid_checkpoint(self):
        key = self.cache_first_result()
        original = CACHE.read_checkpoint(key)["images"]
        with self.assertRaises(ValueError):
            RENDER.YogurtH3SaveSegment.execute(images=torch.zeros(100, 1, 1, 1), key=key, owner="parent", number=2,
                                               content_frames=141, save=True, latent=av_latent(0))
        torch.testing.assert_close(CACHE.read_checkpoint(key)["images"], original, rtol=0, atol=0)

    def test_continued_segment_carries_previous_latent_tail(self):
        result = RENDER.YogurtH3CachedSegment.execute(**self.arguments)
        continuation = self.nodes(result, "YogurtH3Continuation")
        self.assertEqual(len(continuation), 1)
        self.assertIs(continuation[0]["previous_latent"], self.previous["low"])
        self.assertEqual(continuation[0]["overlap_frames"], 22)
        self.assertFalse(continuation[0]["continue_audio"])
        self.assertFalse(self.nodes(result, "VAEDecodeAudio"))

    def test_refinement_resamples_at_plan_aspect_and_carries_refined_tail(self):
        refine = RENDER.YogurtH3RefineSettings.execute(0.004096, "lanczos", True, 8, 0.25)[0]
        previous = RENDER._carry(self.previous["images"], av_latent(0.2), av_latent(0.7))
        result = RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, plan=replace_plan(refine), previous=previous))
        encoders = self.nodes(result, "YogurtH3EncodeSegment")
        self.assertEqual([(e["width"], e["height"]) for e in encoders], [(32, 32), (64, 64)])
        continuation = self.nodes(result, "YogurtH3Continuation")
        self.assertEqual([c["previous_latent"] for c in continuation], [previous["low"], previous["latent"]])
        self.assertEqual(len(self.nodes(result, "SamplerCustomAdvanced")), 2)
        self.assertEqual(self.nodes(result, "BasicScheduler")[0]["denoise"], 0.25)
        self.assertIn("refined", self.nodes(result, "YogurtH3SaveSegment")[0])

    def test_native_av_helpers_retain_h3_audio_during_high_resolution_refine(self):
        video = torch.ones(1, 24, 42, 2, 2)
        audio = torch.full((1, 32, 2, 235), 0.5)
        original = {"samples": NestedTensor((video, audio))}
        separated = LTXVSeparateAVLatent.execute(original)
        high_resolution = {"samples": torch.ones(1, 24, 42, 4, 4)}
        merged = LTXVConcatAVLatent.execute(high_resolution, separated[1])[0]
        self.assertEqual(tuple(merged["samples"].tensors[0].shape), (1, 24, 42, 4, 4))
        torch.testing.assert_close(merged["samples"].tensors[1], audio, atol=0, rtol=0)

    def test_role_change_does_not_carry_previous_identity(self):
        result = RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, plan=replace_plan(guide_reset=True)))
        self.assertFalse(self.nodes(result, "YogurtH3Continuation"))
        self.assertEqual(self.nodes(result, "YogurtH3SaveSegment")[0]["content_frames"], 141)

    def test_bypass_uses_source_without_running_model_and_preserves_duration(self):
        result = RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, plan=replace_plan(bypass=True, guide_reset=True)))
        self.assertFalse(self.nodes(result, "SamplerCustomAdvanced"))
        self.assertEqual(self.nodes(result, "YogurtH3SaveSegment")[0]["content_frames"], 141)

    def test_rtx_upscale_uses_existing_kj_node_contract(self):
        with patch.dict(RENDER.nodes.NODE_CLASS_MAPPINGS, {"ImageResizeKJv2": object()}):
            refine = RENDER.YogurtH3RefineSettings.execute(0.94208, "nvidia_rtx_vsr", False, 8, 0.25)[0]
        result = RENDER.YogurtH3CachedSegment.execute(**dict(self.arguments, plan=replace_plan(refine)))
        rtx = self.nodes(result, "ImageResizeKJv2")[0]
        self.assertEqual((rtx["width"], rtx["height"]), (960, 960))
        self.assertEqual(rtx["upscale_method"], "nvidia_rtx_vsr")
        self.assertEqual(rtx["keep_proportion"], "stretch")
        self.assertEqual(rtx["device"], "gpu")
        self.assertEqual(rtx["divisible_by"], 32)
        self.assertEqual(len(self.nodes(result, "SamplerCustomAdvanced")), 1)

    def test_replacement_plan_restores_source_audio_and_serializes_segments(self):
        with patch.object(RENDER.YogurtH3LongVideoSampler, "hidden", SimpleNamespace(prompt=None, unique_id="test")):
            result = RENDER.YogurtH3LongVideoSampler.execute(model="model", clip="clip", vae="vae", audio_vae="audio_vae",
                plan=replace_plan(), sampler="sampler", sigmas="sigmas", seed=1)
        graph = result.expand
        segments = [(key, node["inputs"]) for key, node in graph.items() if node["class_type"] == "YogurtH3CachedSegment"]
        original = next(key for key, node in graph.items() if node["class_type"] == "YogurtH3OriginalAudio")
        self.assertNotIn("previous", segments[0][1])
        self.assertEqual(segments[1][1]["previous"], [segments[0][0], 2])
        self.assertEqual(segments[1][1]["master_audio"], [original, 0])
        self.assertFalse(any("audio" in inputs for inputs in self.join_inputs(graph)))
        trim = next(node["inputs"] for node in graph.values() if node["class_type"] == "YogurtH3TrimCharacterReplaceOutput")
        self.assertEqual(trim["audio"], [original, 0])
        for node in graph.values():
            if node["class_type"].startswith("YogurtH3"):
                self.assertIn(node["class_type"], PLUGIN.NODE_CLASS_MAPPINGS)

    def join_inputs(self, graph):
        return [node["inputs"] for node in graph.values() if node["class_type"] == "YogurtH3JoinSegments"]


if __name__ == "__main__":
    unittest.main()
