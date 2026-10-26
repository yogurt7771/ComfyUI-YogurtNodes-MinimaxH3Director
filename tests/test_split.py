import unittest

import torch

from support import load

director = load()
split = director.split


def scene(count, colour):
    """Flat colour with a white square sliding across it: motion inside one take."""
    frames = torch.tensor(colour).expand(count, 36, 64, 3).clone()
    for i in range(count):
        x = i * 2 % 50
        frames[i, 10:20, x:x + 10] = 1.0
    return frames


class Footage:
    """Media stand-in: ``video`` returns these frames from ``start``."""

    def __init__(self, frames):
        self.frames, self.reads = frames, []

    def video(self, item, start, count, width, height):
        self.reads.append((item, start, count, width, height))
        return self.frames[:count]


def flat(count):
    return [0.0] + [0.01] * (count - 1)


class AnalyzeTests(unittest.TestCase):
    def test_a_colour_change_is_a_cut_motion_is_not_and_a_fade_is_marked(self):
        fade = torch.tensor([0.7, 0.3, 0.2]) * torch.linspace(1, 0, 9)[1:, None, None, None].expand(8, 36, 64, 3)
        change, cuts, possible, fades = split.analyze(torch.cat([scene(40, [0.2, 0.3, 0.6]), scene(40, [0.7, 0.3, 0.2]), fade]))
        self.assertEqual((cuts, possible), ([40], []))
        # The fade and the frame it starts from.
        self.assertTrue(set(range(81, 88)) <= set(fades) <= set(range(79, 88)))
        self.assertEqual(len(change), 88)


class PlanTests(unittest.TestCase):
    def plan(self, count, cuts=(), possible=(), fades=(), shortest=120, longest=240, segments=0):
        result = split.plan(count, flat(count), list(cuts), list(possible), list(fades), shortest, longest, 22, segments)
        if result is not None:
            self.check(count, result, cuts, set(cuts) | set(possible), set(fades), shortest, longest)
        return result

    def check(self, count, result, cuts, takes, fades, shortest, longest):
        """Every hard constraint of the model."""
        self.assertEqual(result[0][0], 0)
        self.assertEqual(sum(n for _, n, _ in result), count)
        bounds = [0, *cuts, count]
        brief = {c for a, c, b in zip(bounds, bounds[1:], bounds[2:]) if c - a < split.BRIEF or b - c < split.BRIEF}
        head = 0
        for k, (start, frames, carried) in enumerate(result):
            self.assertLessEqual(frames, longest, (start, frames))
            # Only a cut next to a brief take may stay inside a segment.
            self.assertFalse([c for c in cuts if start < c < start + frames and c not in brief], (start, frames))
            if k + 1 < len(result):
                self.assertEqual(start + frames, result[k + 1][0])
            if k and carried:
                tail = range(start - 22, start + split.MARGIN)
                self.assertFalse(any(c in takes for c in tail if c > start - 22), f"carried tail at {start} crosses a cut")
                self.assertFalse(any(f in fades for f in tail), f"carried tail at {start} crosses a fade")
                self.assertEqual((head + result[k - 1][1]) % 17, 5, "the carried segment ends on the 17k+5 grid")
                self.assertGreaterEqual(frames, shortest, "a piece of a long take")
            elif k:
                self.assertIn(start, cuts, f"hard join at {start} is not a sure cut")
            head = 22 if carried else 0

    def test_one_take_is_carried_at_grid_points(self):
        result = self.plan(600)
        self.assertGreaterEqual(len(result), 3)
        self.assertTrue(all(carried for _, _, carried in result[1:]))

    def test_joins_avoid_every_kind_of_cut_and_hard_cuts_only_at_sure_ones(self):
        self.assertIsNotNone(self.plan(700, cuts=[250, 470], possible=[380], fades=range(300, 306)))

    def test_every_cut_between_takes_starts_a_segment(self):
        # A 51-frame take is a segment of its own, however short.
        self.assertEqual(self.plan(300, cuts=[127, 178]), [(0, 127, False), (127, 51, False), (178, 122, False)])

    def test_brief_takes_group_and_a_lone_one_opens_the_next_segment(self):
        # 10 + 10 frames is still brief: they open the take after them. 10 + 10 + 10 is a montage of its own.
        self.assertEqual(self.plan(400, cuts=[200, 210, 220, 300]), [(0, 200, False), (200, 100, False), (300, 100, False)])
        self.assertEqual(self.plan(400, cuts=[200, 210, 220, 230, 300]),
                         [(0, 200, False), (200, 30, False), (230, 70, False), (300, 100, False)])
        # A brief take at the end closes the segment before it.
        self.assertEqual(self.plan(410, cuts=[200, 400]), [(0, 200, False), (200, 210, False)])

    def test_the_segment_count_is_honoured(self):
        self.assertEqual(len(self.plan(600, segments=4)), 4)

    def test_short_stretches_stay_whole(self):
        self.assertEqual(self.plan(200), [(0, 200, False)])
        self.assertEqual(self.plan(80), [(0, 80, False)])

    def test_a_take_just_over_the_longest_cannot_be_split(self):
        # One segment is too long, two need 124 + 120 frames: the only gap the model leaves (see split.py).
        self.assertEqual([n for n in range(241, 260) if self.plan(n) is None], [241, 242, 243])


class SplitSourceTests(unittest.TestCase):
    def test_segments_carry_absolute_starts_and_their_own_cuts(self):
        footage = Footage(torch.cat([scene(300, [0.2, 0.3, 0.6]), scene(300, [0.7, 0.3, 0.2])]))
        result = split.split_source(footage, {"filename": "v.mp4"}, 50, 600, 120, 240, 22)
        self.assertEqual(footage.reads, [({"filename": "v.mp4"}, 50, 600, *split.THUMB)])
        segments = result["segments"]
        self.assertEqual(segments[0]["start"], 50)
        self.assertIn(350, [s["start"] for s in segments])
        self.assertEqual(sum(s["frames"] for s in segments), 600)
        for s in segments:
            offset = s["start"] - 50
            self.assertEqual(s["cuts"], [300 - offset] if offset < 300 < offset + s["frames"] else [])
        self.assertEqual(result["note"], "")

    def test_a_stretch_just_over_the_longest_renders_whole(self):
        result = split.split_source(Footage(scene(242, [0.2, 0.3, 0.6])), {"filename": "v.mp4"}, 0, 242, 120, 240, 22)
        self.assertEqual([(s["start"], s["frames"]) for s in result["segments"]], [(0, 242)])
        self.assertIn("放宽", result["note"])

    def opening(self, still):
        """Two frames of ``still`` in front of a blue take (120 frames) and a red one (150)."""
        return torch.cat([still.expand(2, -1, -1, -1), scene(120, [0.2, 0.3, 0.6]), scene(150, [0.7, 0.3, 0.2])])

    def test_a_cover_is_left_out_of_the_segments_and_reported(self):
        frames = self.opening(scene(150, [0.7, 0.3, 0.2])[50:51])
        result = split.split_source(Footage(frames), {"filename": "v.mp4"}, 0, 272, 120, 240, 22)
        cover = result["cover"]
        # The square comes round every 25 frames, so the red take repeats the cover more than once.
        self.assertEqual(cover["frames"], 2)
        self.assertTrue(cover["copy"] >= 2 + 120 and torch.equal(frames[cover["copy"]], frames[0]))
        self.assertEqual(result["segments"][0]["start"], 2)
        self.assertEqual(sum(s["frames"] for s in result["segments"]), 270)

    def test_a_short_opening_take_seen_nowhere_else_is_kept(self):
        result = split.split_source(Footage(self.opening(torch.tensor([0.1, 0.8, 0.1]).expand(1, 36, 64, 3))), {"filename": "v.mp4"},
                                    0, 272, 120, 240, 22)
        self.assertIsNone(result["cover"])
        self.assertEqual(result["segments"][0]["start"], 0)

    def test_cover_detection_can_be_switched_off(self):
        frames = self.opening(scene(150, [0.7, 0.3, 0.2])[50:51])
        result = split.split_source(Footage(frames), {"filename": "v.mp4"}, 0, 272, 120, 240, 22, detect_cover=False)
        self.assertIsNone(result["cover"])
        self.assertEqual(result["segments"][0]["start"], 0)

    def test_only_the_start_of_the_video_can_be_a_cover(self):
        frames = self.opening(scene(150, [0.7, 0.3, 0.2])[50:51])
        self.assertIsNone(split.split_source(Footage(frames), {"filename": "v.mp4"}, 10, 272, 120, 240, 22)["cover"])

    def test_an_impossible_count_is_an_error(self):
        with self.assertRaises(ValueError):
            split.split_source(Footage(scene(200, [0.2, 0.3, 0.6])), {"filename": "v.mp4"}, 0, 200, 120, 240, 22, segments=2)


if __name__ == "__main__":
    unittest.main()
