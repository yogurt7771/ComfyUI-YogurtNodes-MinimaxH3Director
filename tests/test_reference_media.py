import importlib
import unittest

from test_continuity_runtime import PLUGIN, FakeClip, FakeVideoVAE, torch


REFERENCES = importlib.import_module(PLUGIN.__name__ + ".references")


class ReferenceMediaTests(unittest.TestCase):
    def test_native_modes_create_expected_keyframe_conditions(self):
        image = torch.full((1, 32, 32, 3), 0.5)
        for mode, anchors in (("t2v", []), ("i2v", [0]), ("first_last", [0, 157])):
            with self.subTest(mode=mode):
                result = REFERENCES.YogurtH3DirectorConditioning.execute(
                    clip=FakeClip(), vae=FakeVideoVAE(), mode=mode, prompt="test prompt",
                    duration_seconds=6, width=32, height=32, first_frame=image, last_frame=image,
                )
                self.assertEqual(result[2], 158)
                self.assertTrue(result[1]["samples"].is_nested)
                self.assertEqual([item["resolved_frame_index"] for item in result[0][0][1].get("minimax_keyframes", [])], anchors)

    def test_r2v_media_reaches_native_image_and_video_reference_blocks(self):
        media = REFERENCES.YogurtH3ReferenceMedia.execute(
            images={"image_0": torch.ones(1, 32, 32, 3)},
            videos={"video_0": torch.zeros(22, 32, 32, 3)},
        )[0]
        result = REFERENCES.YogurtH3DirectorConditioning.execute(
            FakeClip(), FakeVideoVAE(), "r2v", "test", 6, 32, 32, reference_media=media
        )
        self.assertEqual([block["kind"] for block in result[0][0][1]["minimax_refs"]], ["image", "video"])

    def test_sparse_numbered_audio_pairs_remain_attached_to_correct_video(self):
        first_audio = {"sample_rate": 32000, "waveform": torch.zeros(1, 1, 50)}
        later_audio = {"sample_rate": 32000, "waveform": torch.ones(1, 1, 50)}
        media = dict(images={"image_8": "extra image"}, videos={"video_2": "extra video2", "video_7": "extra video7"},
                     video_audios={"video_audio_7": later_audio}, audios={})
        result = REFERENCES.combined_references(["role"], "source", first_audio, media)
        self.assertEqual(result["ref_images"], {"ref_image_0": "role", "ref_image_1": "extra image"})
        self.assertEqual(result["ref_videos"], {"ref_video_0": "source", "ref_video_1": "extra video2", "ref_video_2": "extra video7"})
        self.assertEqual(set(result["ref_video_audios"]), {"ref_video_audio_0", "ref_video_audio_2"})
        self.assertIs(result["ref_video_audios"]["ref_video_audio_2"], later_audio)

    def test_changed_audio_pairing_invalidates_reference_cache(self):
        video = torch.zeros(22, 32, 32, 3)
        audio = {"sample_rate": 32000, "waveform": torch.zeros(1, 1, 50)}
        first = dict(videos={"video_0": video, "video_1": video}, video_audios={"video_audio_0": audio})
        second = dict(videos={"video_0": video, "video_1": video}, video_audios={"video_audio_1": audio})
        self.assertNotEqual(REFERENCES.reference_digest(first), REFERENCES.reference_digest(second))


if __name__ == "__main__":
    unittest.main()
