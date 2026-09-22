import importlib.util
import importlib
import json
from fractions import Fraction
import io as byte_io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PLUGIN_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_PATH.parents[1]))

import comfy.cli_args

comfy.cli_args.args.cpu = True

import torch
import av
import numpy as np
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_minimax_h3 import video_latent_t

SPEC = importlib.util.spec_from_file_location(
    "yogurt_h3_runtime_tests", PLUGIN_PATH / "__init__.py",
    submodule_search_locations=[str(PLUGIN_PATH)],
)
PLUGIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLUGIN
SPEC.loader.exec_module(PLUGIN)
NODES = sys.modules[SPEC.name + ".nodes"]
RENDER = sys.modules[SPEC.name + ".render"]
PLANNING = sys.modules[SPEC.name + ".planning"]
MEDIA = sys.modules[SPEC.name + ".media"]
TIMELINE = importlib.import_module(SPEC.name + ".timeline")


class FakeClip:
    def tokenize(self, prompt, **kwargs):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 1), {}]]


class FakeVideoVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, images):
        self.inputs.append(images.clone())
        length = 1 if len(images) == 1 else video_latent_t(len(images))
        return torch.full((1, 24, length, 2, 2), images.mean().item())


class FakeVideo:
    def get_duration(self):
        return 740 / 24

    def get_frame_rate(self):
        return 30

    def get_dimensions(self):
        return (1920, 1080)


class ContinuityRuntimeTests(unittest.TestCase):
    def plan(self, overlap=22, timeline_json="{}", **kwargs):
        return NODES.YogurtH3CharacterReplacePlanner.execute(FakeVideo(), torch.full((1, 32, 32, 3), 0.8), "prompt", 6, 0.41,
                                                            overlap_frames=overlap, timeline_json=timeline_json, **kwargs)[0]

    def test_replacement_segment_encodes_role_picture_and_source_video(self):
        plan = self.plan()
        vae = FakeVideoVAE()
        result = RENDER.YogurtH3EncodeSegment.execute(
            clip=FakeClip(), vae=vae, audio_vae=None, plan=plan, index=1, width=32, height=32,
            source_frames=torch.full((141, 32, 32, 3), 0.1),
            source_audio={"waveform": torch.zeros(1, 2, 100), "sample_rate": 32000},
        )
        refs = result[0][0][1]["minimax_refs"]
        self.assertEqual([block["kind"] for block in refs], ["image", "video"])
        self.assertAlmostEqual(refs[0]["latent"].mean().item(), vae.inputs[0].mean().item(), places=5)
        self.assertAlmostEqual(refs[1]["latent"].mean().item(), vae.inputs[1].mean().item(), places=5)
        self.assertNotIn("minimax_keyframes", result[0][0][1])
        self.assertTrue(result[1]["samples"].is_nested)

    def test_planner_resolution_follows_source_aspect(self):
        plan = self.plan()
        self.assertEqual((plan["width"], plan["height"]), (864, 480))
        self.assertEqual(plan["audio"], "source")

    def test_joined_windows_reassemble_source_timeline_exactly(self):
        segments = PLANNING.build_segments(740, 6, 22)
        merged = None
        for segment in segments:
            source_ids = torch.arange(segment["start_frame"], segment["start_frame"] + segment["content_frames"], dtype=torch.float32)
            merged = RENDER.YogurtH3JoinSegments.execute(source_ids.reshape(-1, 1, 1, 1), segment["overlap_frames"], False,
                                                         merged_images=merged)[0]
        torch.testing.assert_close(merged.flatten(), torch.arange(740.0))

    def graph(self, overlap):
        plan = self.plan(overlap)
        with patch.object(RENDER.YogurtH3LongVideoSampler, "hidden", SimpleNamespace(prompt=None, unique_id="test")):
            result = RENDER.YogurtH3LongVideoSampler.execute(model="model", clip="clip", vae="vae", audio_vae="audio_vae",
                                                              plan=plan, sampler="sampler", sigmas="sigmas", seed=1)
        return plan, result.expand

    def test_graph_serializes_sampling_via_previous_latent_carry(self):
        plan, graph = self.graph(22)
        segments = [(key, value) for key, value in graph.items() if value["class_type"] == "YogurtH3CachedSegment"]
        self.assertEqual(len(segments), len(plan["segments"]))
        self.assertNotIn("previous", segments[0][1]["inputs"])
        for index in range(1, len(segments)):
            self.assertEqual(segments[index][1]["inputs"]["previous"], [segments[index - 1][0], 2])
            self.assertEqual(plan["segments"][index]["overlap_frames"], 22)

    def test_zero_overlap_graph_has_no_cross_segment_dependency(self):
        _, graph = self.graph(0)
        for node in graph.values():
            if node["class_type"] == "YogurtH3CachedSegment":
                self.assertNotIn("previous", node["inputs"])

    def test_audio_slices_share_absolute_sample_positions(self):
        rate = 44100
        audio = {"waveform": torch.arange(round(740 * rate / 24), dtype=torch.float32).reshape(1, 1, -1), "sample_rate": rate}
        first = MEDIA.slice_audio(audio, 0, 141)
        second = MEDIA.slice_audio(audio, 119, 141)
        overlap_start = round(119 * rate / 24)
        overlap_end = round(141 * rate / 24)
        torch.testing.assert_close(first["waveform"][..., overlap_start:overlap_end], second["waveform"][..., :overlap_end-overlap_start])
        pieces = [MEDIA.slice_audio(audio, segment["output_start_frame"], segment["output_frames"])["waveform"] for segment in PLANNING.build_segments(740, 6, 22)]
        torch.testing.assert_close(torch.cat(pieces, dim=-1), audio["waveform"])

    def test_original_audio_preserves_samples_and_pads_missing_tail(self):
        waveform = torch.arange(100.0).reshape(1, 1, -1)
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=torch.zeros(5, 32, 32, 3), frame_rate=Fraction(24),
            audio={"waveform": waveform, "sample_rate": 2400},
        ))
        result = MEDIA.original_audio(video, 5)
        torch.testing.assert_close(result["waveform"][..., :100], waveform)
        self.assertEqual(result["waveform"].shape[-1], 500)
        self.assertEqual(result["waveform"][..., 100:].count_nonzero().item(), 0)

    def test_resampling_keeps_global_phase_at_overlapping_boundaries(self):
        for fps in (30.0, 30000 / 1001, 25.0):
            all_frames = torch.arange(1100.0).reshape(-1, 1, 1, 1)
            whole = NODES._resize_frames(all_frames, 740, fps, 0)
            for start in (119, 238, 357, 476, 595):
                source_start = int(start * fps / 24)
                part = NODES._resize_frames(all_frames[source_start:], 141, fps, start)
                torch.testing.assert_close(part, whole[start:start+141])

    def test_file_audio_honors_timestamp_delay_and_video_trim(self):
        source = byte_io.BytesIO()
        waveform = np.arange(48000, dtype=np.float32).reshape(1, -1) / 48000
        with av.open(source, "w", format="matroska") as container:
            video_stream = container.add_stream("ffv1", rate=24)
            video_stream.width = video_stream.height = 32
            video_stream.pix_fmt = "bgr0"
            audio_stream = container.add_stream("pcm_f32le", rate=48000, layout="mono")
            for index in range(48):
                frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 24)
                container.mux(video_stream.encode(frame))
            container.mux(video_stream.encode(None))
            frame = av.AudioFrame.from_ndarray(waveform, format="flt", layout="mono")
            frame.sample_rate = 48000
            frame.pts = 6000
            frame.time_base = Fraction(1, 48000)
            container.mux(audio_stream.encode(frame))
            container.mux(audio_stream.encode(None))
        source.seek(0)
        video = InputImpl.VideoFromFile(source)
        full = MEDIA.original_audio(video, 48)["waveform"]
        expected = torch.zeros(1, 1, 96000)
        expected[..., 6000:54000] = torch.from_numpy(waveform)
        torch.testing.assert_close(full, expected, atol=0, rtol=0)
        trimmed = video.as_trimmed(0.25, 1)
        audio = MEDIA.original_audio(trimmed, 24)["waveform"]
        torch.testing.assert_close(audio, expected[..., 12000:60000], atol=0, rtol=0)

    def test_scene_detection_observes_cuts_in_untrimmed_and_trimmed_video(self):
        source = byte_io.BytesIO()
        with av.open(source, "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=24)
            stream.width = stream.height = 32
            stream.pix_fmt = "bgr0"
            for index in range(72):
                value = 255 if 24 <= index < 48 else 0
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 24)
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
        source.seek(0)
        video = InputImpl.VideoFromFile(source)
        self.assertEqual(TIMELINE.detect_cuts(video), [24, 48])
        trimmed = video.as_trimmed(0.5, 1.5)
        self.assertEqual(TIMELINE.detect_cuts(trimmed), [12])

    def test_all_registered_node_schemas_are_valid(self):
        for node_id, node in PLUGIN.NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=node_id):
                schema = node.define_schema()
                self.assertEqual(schema.node_id, node_id)
                inputs = node.INPUT_TYPES()
                self.assertIsInstance(inputs, dict)

    def test_cached_frame_preview_encodes_as_playable_mp4(self):
        frames = torch.linspace(0, 1, 22).reshape(22, 1, 1, 1).expand(22, 32, 32, 3).contiguous()
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, frame_rate=Fraction(24)))
        buffer = byte_io.BytesIO()
        video.save_to(buffer, format="mp4", codec="h264")
        buffer.seek(0)
        with av.open(buffer) as container:
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.name, "h264")
            self.assertEqual(stream.average_rate, Fraction(24))
            self.assertEqual(len(list(container.decode(stream))), 22)

    def test_timeline_edits_reach_plan_role_prompt_target_and_seed(self):
        edits = {"cuts": [200], "segments": [dict(start_frame=200, role_index=1,
                 prompt="edited shot", target_description="person on the left", seed_offset=7)]}
        role = torch.ones(1, 32, 32, 3)
        plan = self.plan(timeline_json=json.dumps(edits), additional_roles={"role_0": role})
        target = next(segment for segment in plan["segments"] if segment["start_frame"] == 200)
        self.assertIs(plan["inputs"]["images"][target["images"][0]], role)
        self.assertTrue(target["prompt"].startswith("edited shot"))
        self.assertIn("person on the left", target["prompt"])
        self.assertEqual(target["seed_offset"], 7)
        self.assertEqual(target["overlap_frames"], 0)
        with self.assertRaises(ValueError):
            self.plan(timeline_json=json.dumps(edits))


if __name__ == "__main__":
    unittest.main()
