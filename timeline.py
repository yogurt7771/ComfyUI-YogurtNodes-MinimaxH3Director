"""Local source inspection and scene-aware timeline preview."""

import asyncio
import io
import os
from fractions import Fraction

import av
import numpy as np
import folder_paths
from aiohttp import web
from comfy_api.latest import InputImpl, Types
from server import PromptServer

from .cache import digest, read_checkpoint


def source_stamp(video):
    source = video.get_stream_source()
    if isinstance(source, str):
        stat = os.stat(source)
        identity = [os.path.normcase(os.path.realpath(source)), stat.st_size, stat.st_mtime_ns]
    else:
        identity = [video.get_duration(), video.get_dimensions()]
    return digest([identity, video.get_active_trim_window()])


def detect_cuts(video, threshold=0.28):
    trim_start, trim_duration = video.get_active_trim_window()
    trim_end = trim_start + (trim_duration if trim_duration > 0 else float(video.get_duration()))
    source = video.get_stream_source()
    if not isinstance(source, str):
        source.seek(0)
    previous = None
    cuts = []
    with av.open(source) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            seconds = float(frame.pts * frame.time_base) - origin
            if seconds < trim_start:
                continue
            if seconds >= trim_end:
                break
            small = frame.reformat(width=64, height=36, format="gray").to_ndarray().astype(np.float32) / 255
            at = round((seconds-trim_start) * 24)
            if previous is not None and np.abs(small-previous).mean() >= threshold and at > 0:
                cuts.append(at)
            previous = small
    return sorted(set(cuts))


def register_routes():
    if not hasattr(PromptServer, "instance"):
        return

    @PromptServer.instance.routes.get("/yogurt_h3/preview/{key}")
    async def preview_segment(request):
        def encode():
            checkpoint = read_checkpoint(request.match_info["key"])
            if checkpoint is None:
                raise web.HTTPNotFound(text="分段缓存不存在，请重新运行对应分段。")
            frames = checkpoint["images"]
            video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, frame_rate=Fraction(24)))
            buffer = io.BytesIO()
            video.save_to(buffer, format="mp4", codec="h264")
            return buffer.getvalue()
        return web.Response(body=await asyncio.to_thread(encode), content_type="video/mp4")

    @PromptServer.instance.routes.post("/yogurt_h3/plan")
    async def preview_plan(request):
        data = await request.json()
        path = os.path.realpath(folder_paths.get_annotated_filepath(data["file"]))
        roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()]
        if not any(os.path.splitdrive(path)[0].lower() == os.path.splitdrive(os.path.realpath(root))[0].lower()
                   and os.path.commonpath([path, os.path.realpath(root)]) == os.path.realpath(root) for root in roots):
            raise web.HTTPBadRequest(text="视频必须位于 ComfyUI 的 input、output 或 temp 目录。")
        if not os.path.isfile(path):
            raise web.HTTPNotFound(text="找不到源视频，请先上传并选择视频。")
        # The same planner drives both preview and execution; no duplicate frame math in JavaScript.
        from .nodes import timeline_plan
        try:
            plan = await asyncio.to_thread(timeline_plan,
                InputImpl.VideoFromFile(path), int(data.get("segment_seconds", 6)), float(data.get("megapixels", 0.41)),
                int(data.get("overlap_frames", 22)), data.get("scene_detection", False),
                float(data.get("scene_threshold", 0.28)), data.get("timeline_json", "{}"))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        plan["source_stamp"] = source_stamp(InputImpl.VideoFromFile(path))
        return web.json_response(plan)
