import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from support import load, tiny_upscaler

director = load()
lu = director.latent_upscale
import folder_paths  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402


class LatentUpscaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.saved = folder_paths.folder_names_and_paths.get(lu.FOLDER)
        folder_paths.folder_names_and_paths[lu.FOLDER] = ([str(self.folder)], {".safetensors"})

    def tearDown(self):
        folder_paths.folder_names_and_paths[lu.FOLDER] = self.saved
        self.tmp.cleanup()

    def test_upscale_reaches_the_target_grid_and_keeps_time(self):
        tiny_upscaler(director, self.folder / "tiny.safetensors")
        patcher = lu.load("tiny.safetensors")
        for frames in (6, 40):  # 40 latent frames runs in overlapping temporal chunks
            video = torch.randn(1, 24, frames, 4, 8)
            out = lu.upscale(patcher, video, 8, 16)
            self.assertEqual((tuple(out.shape), out.dtype), ((1, 24, frames, 8, 16), torch.float32))
            self.assertTrue(torch.isfinite(out).all())

    def test_missing_or_foreign_files_fail_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "找不到 latent 放大模型"):
            lu.load("other.safetensors")
        import safetensors.torch
        safetensors.torch.save_file({"weight": torch.zeros(2)}, str(self.folder / "foreign.safetensors"))
        with self.assertRaisesRegex(ValueError, "不是 MiniMax H3 3D latent 放大模型"):
            lu.load("foreign.safetensors")

    def test_download_lands_only_after_the_checksum_matches(self):
        payload = b"h3 latent upscaler" * 4096
        saved_source, saved_open = lu.SOURCE, lu.urllib.request.urlopen
        lu.urllib.request.urlopen = lambda url, timeout: BytesIO(payload)
        try:
            lu.SOURCE = (*saved_source[:3], len(payload), "0" * 64)
            with self.assertRaisesRegex(ValueError, "校验失败"):
                lu._download(lu.DEFAULT)
            self.assertEqual(os.listdir(self.folder), [])
            lu.SOURCE = (*saved_source[:3], len(payload), hashlib.sha256(payload).hexdigest())
            self.assertEqual(Path(lu._download(lu.DEFAULT)).read_bytes(), payload)
        finally:
            lu.SOURCE, lu.urllib.request.urlopen = saved_source, saved_open

    def test_default_file_is_keyed_by_its_pinned_checksum(self):
        self.assertEqual(lu.identity(lu.DEFAULT), lu.SOURCE[4])
        self.assertIn(lu.DEFAULT, lu.choices())


class TrajectoryTests(unittest.TestCase):
    def test_trajectory_noise_recovers_the_noise_the_audio_was_on(self):
        model = SimpleNamespace(get_model_object=lambda name: SimpleNamespace(audio_scale=2.0, noise_scale=1.0))
        clean, noise, sigma = torch.randn(1, 32, 2, 10), torch.randn(1, 32, 2, 10), 0.7
        # The sampler carries audio as audio_scale * audio; its state is sigma * noise + (1 - sigma) * clean.
        state = sigma * noise + (1 - sigma) * 2.0 * clean
        last = {"x": NestedTensor((torch.zeros(1), state)), "x0": NestedTensor((torch.zeros(1), 2.0 * clean))}
        stored = director.sampling.trajectory_noise(model, last, sigma)
        # sample() hands it back as stored * audio_scale / noise_scale: the audio restarts on the same noise.
        self.assertTrue(torch.allclose(stored * 2.0, noise, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
