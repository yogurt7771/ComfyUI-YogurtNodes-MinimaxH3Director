import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "planning.py"
SPEC = importlib.util.spec_from_file_location("yogurt_h3_director_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlanningTests(unittest.TestCase):
    def test_six_second_segments_for_current_video(self):
        segments = MODULE.build_segments(740, 6)
        self.assertEqual([item["content_frames"] for item in segments], [141, 141, 141, 141, 141, 35])
        self.assertEqual([item["generation_frames"] for item in segments], [141, 141, 141, 141, 141, 39])

    def test_tiny_tail_is_absorbed_into_previous_segment(self):
        segments = MODULE.build_segments(143, 6)
        self.assertEqual([item["content_frames"] for item in segments], [143])
        self.assertEqual([item["generation_frames"] for item in segments], [158])

    def test_short_video_still_uses_valid_h3_length(self):
        segments = MODULE.build_segments(3, 6)
        self.assertEqual(segments[0]["content_frames"], 3)
        self.assertEqual(segments[0]["generation_frames"], 5)


if __name__ == "__main__":
    unittest.main()
