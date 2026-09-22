import copy
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_continuity_runtime import PLUGIN, torch


CACHE = importlib.import_module(PLUGIN.__name__ + ".cache")


class StreamVideo:
    def __init__(self, payload, trim=(0, 0)):
        self.source = io.BytesIO(payload)
        self.trim = trim

    def get_stream_source(self):
        return self.source

    def get_active_trim_window(self):
        return self.trim

    def get_dimensions(self):
        return (32, 32)


class SegmentCacheTests(unittest.TestCase):
    def test_checkpoint_preserves_pixels_across_process_reload_and_replacement(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch.object(CACHE.folder_paths, "get_output_directory", return_value=directory):
                key = CACHE.digest("test segment")
                self.assertIsNone(CACHE.read_checkpoint(key))
                images = torch.linspace(0, 1, 22 * 32 * 32 * 3).reshape(22, 32, 32, 3)
                latent = torch.linspace(-1, 1, 24 * 7 * 2 * 2).reshape(1, 24, 7, 2, 2)
                CACHE.write_checkpoint(key, {"images": images, "low_video": latent})
                fresh_module = importlib.reload(CACHE)
                checkpoint = fresh_module.read_checkpoint(key)
                torch.testing.assert_close(checkpoint["images"], images, rtol=0, atol=0)
                torch.testing.assert_close(checkpoint["low_video"], latent, rtol=0, atol=0)
                replacement = images.flip(0)
                CACHE.write_checkpoint(key, {"images": replacement})
                self.assertEqual(set(CACHE.read_checkpoint(key)), {"images"})
                torch.testing.assert_close(CACHE.read_checkpoint(key)["images"], replacement, rtol=0, atol=0)
                self.assertEqual(len(list(Path(directory).rglob("*.safetensors"))), 1)
                self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    def test_changed_reference_pixels_and_source_trim_invalidate_digest(self):
        image = torch.zeros(1, 32, 32, 3)
        changed = image.clone()
        changed[0, 3, 4, 2] = 0.1
        self.assertNotEqual(CACHE.tensor_digest(image), CACHE.tensor_digest(changed))
        self.assertNotEqual(CACHE.tensor_digest(image), CACHE.tensor_digest(image.double()))
        original = StreamVideo(b"source video")
        self.assertEqual(CACHE.video_digest(original), CACHE.video_digest(original))
        self.assertNotEqual(CACHE.video_digest(original), CACHE.video_digest(StreamVideo(b"source video", (1, 5))))
        self.assertNotEqual(CACHE.video_digest(original), CACHE.video_digest(StreamVideo(b"changed video")))

    def test_model_and_sampling_changes_invalidate_upstream_signature(self):
        prompt = {
            "1": {"class_type": "ModelPatch", "inputs": {"strength": 0.8}},
            "2": {"class_type": "BasicScheduler", "inputs": {"steps": 4, "model": ["1", 0]}},
            "3": {"class_type": "YogurtH3LongVideoSampler", "inputs": {"model": ["1", 0], "sigmas": ["2", 0], "plan": ["6", 0]}},
            "4": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "first"}},
            "5": {"class_type": "YogurtH3RefineSettings", "inputs": {"steps": 8}},
            "6": {"class_type": "YogurtH3AutoStoryParse", "inputs": {"plan": ["7", 0], "llm_text": ["8", 0]}},
            "7": {"class_type": "YogurtH3AutoStoryPlan", "inputs": {"refine": ["5", 0]}},
            "8": {"class_type": "YogurtOpenRouterImageUnderstand", "inputs": {"api_key": "secret"}},
        }
        key = CACHE.upstream_signature(prompt, "3")
        changed = copy.deepcopy(prompt)
        changed["2"]["inputs"]["steps"] = 8
        self.assertNotEqual(key, CACHE.upstream_signature(changed, "3"))
        changed = copy.deepcopy(prompt)
        changed["1"]["inputs"]["strength"] = 0.2
        self.assertNotEqual(key, CACHE.upstream_signature(changed, "3"))
        changed = copy.deepcopy(prompt)
        changed["5"]["inputs"]["steps"] = 4
        self.assertNotEqual(key, CACHE.upstream_signature(changed, "3"))
        changed = copy.deepcopy(prompt)
        changed["4"]["inputs"]["filename_prefix"] = "second"
        changed["8"]["inputs"]["api_key"] = "other"
        self.assertEqual(key, CACHE.upstream_signature(changed, "3"))

    def test_sigma_digest_handles_one_dimensional_tensors(self):
        first = torch.tensor([1.0, 0.7, 0.2, 0.0])
        second = torch.tensor([1.0, 0.6, 0.2, 0.0])
        self.assertEqual(CACHE.tensor_digest(first), CACHE.tensor_digest(first.clone()))
        self.assertNotEqual(CACHE.tensor_digest(first), CACHE.tensor_digest(second))

    def test_selected_segments_support_individuals_ranges_and_chinese_comma(self):
        self.assertEqual(CACHE.selected_segments("2，4-6, 4", 6), {2, 4, 5, 6})
        self.assertEqual(CACHE.selected_segments("", 6), set())
        for value in ("0", "7", "4-2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CACHE.selected_segments(value, 6)


if __name__ == "__main__":
    unittest.main()
