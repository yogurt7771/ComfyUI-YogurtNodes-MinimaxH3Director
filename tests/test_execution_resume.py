import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_continuity_runtime import PLUGIN, torch
from comfy.nested_tensor import NestedTensor
import execution
import nodes


CACHE = importlib.import_module(PLUGIN.__name__ + ".cache")
RENDER = importlib.import_module(PLUGIN.__name__ + ".render")


class Source:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "AUDIO", "SAMPLER", "SIGMAS", "YOGURT_H3_PLAN")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        segment = dict(index=0, start_frame=0, content_frames=22, generation_frames=22, overlap_frames=0,
                       prompt="character", mode="r2v", images=[0], source=True)
        plan = RENDER.make_plan((32, 32), 0.001024, [segment], 22, "source", False, "match",
                                [torch.ones(1, 32, 32, 3)], None, "video")
        return ("model", "clip", "vae", "audio_vae", {"waveform": torch.zeros(1, 1, 32000), "sample_rate": 32000},
                "sampler", torch.tensor([1.0, 0.0]), plan)


class Extract:
    RETURN_TYPES = ("IMAGE", "AUDIO", "AUDIO")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "start_frame": ("INT",), "content_frames": ("INT",),
                             "generation_frames": ("INT",), "master_audio": ("AUDIO",)}}

    def run(self, video, start_frame, content_frames, generation_frames, master_audio):
        return torch.zeros(generation_frames, 1, 1, 1), master_audio, master_audio


class Encode:
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {name: (kind,) for name, kind in (
            ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"), ("plan", "YOGURT_H3_PLAN"), ("index", "INT"),
            ("width", "INT"), ("height", "INT"), ("source_frames", "IMAGE"), ("source_audio", "AUDIO"))}}

    def run(self, **kwargs):
        return "conditioning", {"samples": NestedTensor((torch.zeros(1, 24, 7, 1, 1), torch.zeros(1, 32, 2, 37)))}


class Guider:
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "conditioning": ("CONDITIONING",)}}

    def run(self, model, conditioning):
        return ("guider",)


class Sample:
    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "run"
    calls = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",),
                             "sigmas": ("SIGMAS",), "latent_image": ("LATENT",)}}

    def run(self, noise, guider, sampler, sigmas, latent_image):
        type(self).calls += 1
        latent = {"samples": NestedTensor(tuple(torch.full_like(item, float(type(self).calls))
                                                for item in latent_image["samples"].unbind()))}
        return latent, latent


class Decode:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    def run(self, samples, vae):
        return (torch.full((22, 1, 1, 1), samples["samples"].unbind()[0].mean().item()),)


class Sink:
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    images = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    def run(self, images):
        type(self).images = images.clone()
        return ()


class Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *args):
        pass


class ExecutionResumeTests(unittest.TestCase):
    def test_real_executor_resume_and_repeated_force_rerun(self):
        mappings = dict(PLUGIN.NODE_CLASS_MAPPINGS)
        mappings.update(YogurtTestSource=Source, YogurtTestSink=Sink,
                        YogurtH3ExtractCharacterReplaceSegment=Extract, YogurtH3EncodeSegment=Encode,
                        BasicGuider=Guider, SamplerCustomAdvanced=Sample, VAEDecode=Decode)
        arguments = {name: ["source", index] for index, name in enumerate(
            ("model", "clip", "vae", "audio_vae", "master_audio", "sampler", "sigmas", "plan"))}
        arguments.update(index=0, seed=10, base_key=CACHE.digest("engine test"), cache_mode="reuse", owner="segment",
                         force_rerun=False)
        prompt = {"source": {"class_type": "YogurtTestSource", "inputs": {}},
                  "segment": {"class_type": "YogurtH3CachedSegment", "inputs": arguments},
                  "sink": {"class_type": "YogurtTestSink", "inputs": {"images": ["segment", 0]}}}
        Sample.calls = 0
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch.object(CACHE.folder_paths, "get_output_directory", return_value=directory), patch.dict(nodes.NODE_CLASS_MAPPINGS, mappings), patch.object(RENDER, "segment_event"):
                executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC,
                    cache_args={"ram": 0, "ram_inactive": 0})
                for index, (force, expected_calls) in enumerate(((False, 1), (False, 1), (True, 2), (True, 3), (False, 3))):
                    prompt["segment"]["inputs"]["force_rerun"] = force
                    executor.execute(copy.deepcopy(prompt), f"yogurt-test-{index}", execute_outputs=["sink"])
                    errors = [data for event, data in executor.status_messages if event == "execution_error"]
                    self.assertEqual(errors, [], errors)
                    self.assertTrue(executor.success)
                    self.assertEqual(Sample.calls, expected_calls)
                    self.assertEqual(Sink.images.shape[0], 22)
                    self.assertEqual(Sink.images.mean().item(), expected_calls)


if __name__ == "__main__":
    unittest.main()
