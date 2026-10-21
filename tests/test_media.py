from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import torch

from support import load, write_video

director = load()
from comfy_api.latest import InputImpl  # noqa: E402

Media = director.media.Media
VIDEO = {"input": "video_0"}


class GraphMediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "clip.mp4"
        write_video(self.path, 48, 64, 32, audio_seconds=2)

    def tearDown(self):
        self.tmp.cleanup()

    def levels(self, frames):
        return [round(v * 47) for v in frames[:, 0, 0, 0].tolist()]

    def test_video_input_is_read_inside_its_trim_window(self):
        # Frame i of the clip has level i; the trim keeps frames 12..35.
        video = InputImpl.VideoFromFile(str(self.path)).as_trimmed(start_time=0.5, duration=1.0)
        media = Media(videos={"video_0": video})
        self.assertEqual(self.levels(media.video(VIDEO, 6, 24, 64, 32)), list(range(18, 36)) + [35] * 6)
        self.assertEqual(media.video_size(VIDEO), (64, 32))
        untrimmed = Media(videos={"video_0": InputImpl.VideoFromFile(str(self.path))})
        self.assertNotEqual(media.identity(VIDEO), untrimmed.identity(VIDEO))

    def test_in_memory_video_can_be_read_repeatedly(self):
        media = Media(videos={"video_0": InputImpl.VideoFromFile(BytesIO(self.path.read_bytes()))})
        self.assertEqual(self.levels(media.video(VIDEO, 0, 3, 64, 32)), [0, 1, 2])
        self.assertEqual(self.levels(media.video(VIDEO, 40, 3, 64, 32)), [40, 41, 42])
        track = media.audio(VIDEO, 0, 24)
        self.assertEqual((track["sample_rate"], track["waveform"].shape), (32000, (1, 2, 32000)))
        self.assertGreater(track["waveform"].abs().max().item(), 0.1)
        same = Media(videos={"video_0": InputImpl.VideoFromFile(BytesIO(self.path.read_bytes()))})
        self.assertEqual(media.identity(VIDEO), same.identity(VIDEO))

    def test_audio_input_is_cut_on_the_timeline(self):
        wave = torch.linspace(-1, 1, 48000).repeat(1, 2, 1)
        media = Media(audios={"audio_0": {"waveform": wave, "sample_rate": 24000}})
        clip = media.audio({"input": "audio_0"}, 24, 12)
        self.assertEqual(clip["sample_rate"], 24000)
        self.assertTrue(torch.equal(clip["waveform"], wave[:, :, 24000:36000]))
        with self.assertRaisesRegex(ValueError, "没有声音"):
            Media(images={"image_0": torch.zeros(1, 8, 8, 3)}).audio({"input": "image_0"}, 0, 1)


if __name__ == "__main__":
    unittest.main()
