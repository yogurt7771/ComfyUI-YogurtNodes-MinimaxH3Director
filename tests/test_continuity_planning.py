import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "yogurt_h3_continuity_planning", Path(__file__).resolve().parents[1] / "planning.py"
)
PLANNING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNING)


class ContinuityPlanningTests(unittest.TestCase):
    def assert_exact_coverage(self, target_frames, seconds, requested_overlap):
        segments = PLANNING.build_segments(target_frames, seconds, requested_overlap)
        cursor = 0
        previous_end = 0
        for index, segment in enumerate(segments):
            with self.subTest(target=target_frames, seconds=seconds, overlap=requested_overlap, segment=index):
                start = segment["start_frame"]
                content = segment["content_frames"]
                overlap = segment["overlap_frames"]
                output = segment["output_frames"]
                self.assertEqual(segment["output_start_frame"], cursor)
                self.assertEqual(start + overlap, cursor)
                self.assertEqual(content - overlap, output)
                self.assertGreater(output, 0)
                self.assertGreaterEqual(segment["generation_frames"], content)
                self.assertEqual(segment["generation_frames"] % 17, 5)
                self.assertTrue(overlap in (0, 1) or overlap >= 5 and overlap % 17 == 5)
                if index:
                    self.assertEqual(start + overlap, previous_end)
                    self.assertLess(overlap, content)
                else:
                    self.assertEqual(overlap, 0)
                if index < len(segments) - 1:
                    self.assertEqual(content, PLANNING.segment_frame_limit(seconds))
                cursor += output
                previous_end = start + content
        self.assertEqual(cursor, target_frames)
        self.assertEqual(previous_end, target_frames)

    def test_default_example_windows_revisit_generated_context(self):
        segments = PLANNING.build_segments(740, 6, 22)
        self.assertEqual([s["start_frame"] for s in segments[:3]], [0, 119, 238])
        self.assertEqual([s["overlap_frames"] for s in segments[:3]], [0, 22, 22])
        self.assertEqual([s["output_frames"] for s in segments[:3]], [141, 119, 119])
        self.assert_exact_coverage(740, 6, 22)

    def test_boundaries_and_large_overlaps_never_drop_or_duplicate_frames(self):
        for seconds in (1, 6, 10):
            limit = PLANNING.segment_frame_limit(seconds)
            targets = (1, 3, 5, 22, limit - 1, limit, limit + 1, limit + 4,
                       limit + 5, 2 * limit, 2 * limit + 1, 740)
            for requested_overlap in (0, 1, 3, 5, 21, 22, 39, 56, 10000):
                for target in targets:
                    self.assert_exact_coverage(target, seconds, requested_overlap)

    def test_zero_overlap_preserves_legacy_plan_lengths(self):
        segments = PLANNING.build_segments(740, 6, 0)
        self.assertEqual([s["content_frames"] for s in segments], [141] * 5 + [35])
        self.assertEqual([s["generation_frames"] for s in segments], [141] * 5 + [39])
        self.assertEqual(PLANNING.build_segments(143, 6, 0)[0]["content_frames"], 143)

    def test_scene_boundaries_reset_overlap_and_keep_exact_coverage(self):
        segments = PLANNING.timeline_segments(740, 6, 22, cuts=[200, 2, 200, 738])
        output_ids = []
        for segment in segments:
            output_ids.extend(range(segment["output_start_frame"], segment["output_start_frame"] + segment["output_frames"]))
            if segment["start_frame"] in (0, 2, 200, 738):
                self.assertEqual(segment["overlap_frames"], 0)
        self.assertEqual(output_ids, list(range(740)))

    def test_role_change_and_bypass_reset_identity_guides(self):
        edits = [dict(start_frame=119, role_index=1, prompt="new role"),
                 dict(start_frame=238, role_index=1, bypass=True),
                 dict(start_frame=357, role_index=1)]
        segments = PLANNING.timeline_segments(740, 6, 22, edits=edits)
        self.assertEqual(segments[1]["prompt"], "new role")
        for index in (1, 2, 3):
            self.assertTrue(segments[index]["guide_reset"])

    def test_custom_short_and_long_windows_cover_every_source_frame(self):
        for edits in ({0: 3}, {0: 362}, {119: 5}, {119: 240}, {0: 5, 4: 50}):
            with self.subTest(edits=edits):
                segments = PLANNING.build_segments(740, 6, 22, window_lengths=edits)
                output_ids = []
                for segment in segments:
                    self.assertGreater(segment["output_frames"], 0)
                    if segment is not segments[-1]:
                        self.assertEqual(segment["content_frames"], segment["generation_frames"])
                    self.assertGreaterEqual(segment["generation_frames"], segment["content_frames"])
                    self.assertEqual(segment["generation_frames"] % 17, 5)
                    output_ids.extend(range(segment["output_start_frame"], segment["output_start_frame"] + segment["output_frames"]))
                self.assertEqual(output_ids, list(range(740)))

    def test_per_scene_window_edits_use_absolute_source_frame(self):
        segments = PLANNING.timeline_segments(740, 6, 22, cuts=[200], edits=[dict(start_frame=200, window_frames=240)])
        changed = next(item for item in segments if item["start_frame"] == 200)
        # Carried latent tails need every continued window to end on the 17k+5 lattice.
        self.assertEqual(changed["content_frames"], 226)
        self.assertEqual(changed["overlap_frames"], 0)
        self.assertEqual(sum(item["output_frames"] for item in segments), 740)


if __name__ == "__main__":
    unittest.main()
