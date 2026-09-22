"""Audio intervals use absolute sample positions on the source timeline."""

import av
import torch
from comfy_api.latest import InputImpl

from .planning import H3_FPS


def slice_audio(audio, start_frame, frame_count):
    rate = int(audio["sample_rate"])
    start = round(start_frame * rate / H3_FPS)
    end = round((start_frame + frame_count) * rate / H3_FPS)
    waveform = audio["waveform"][..., start:end].clone()
    if waveform.shape[-1] < end - start:
        waveform = torch.nn.functional.pad(waveform, (0, end - start - waveform.shape[-1]))
    return {"waveform": waveform, "sample_rate": rate}


def original_audio(video, target_frames):
    if isinstance(video, InputImpl.VideoFromComponents):
        audio = video.get_components().audio
        if audio is None:
            audio = {"waveform": torch.zeros(1, 2, 0), "sample_rate": 32000}
        return slice_audio(audio, 0, target_frames)

    source = video.get_stream_source()
    if not isinstance(source, str):
        source.seek(0)
    trim_start, _ = video.get_active_trim_window()
    with av.open(source) as container:
        if not container.streams.audio:
            return {"waveform": torch.zeros(1, 2, round(target_frames * 32000 / H3_FPS)), "sample_rate": 32000}
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        layout = stream.codec_context.layout
        wave = torch.zeros(1, len(layout.channels), round(target_frames * rate / H3_FPS))
        video_stream = container.streams.video[0]
        origin = float((video_stream.start_time or 0) * video_stream.time_base) + trim_start
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        cursor = round((float((stream.start_time or 0) * stream.time_base) - origin) * rate)
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                start = round((float(frame.pts * frame.time_base) - origin) * rate) if frame.pts is not None else cursor
                data = torch.from_numpy(frame.to_ndarray())
                end = start + data.shape[-1]
                lo, hi = max(0, start), min(wave.shape[-1], end)
                if hi > lo:
                    wave[0, :, lo:hi] = data[:, lo-start:hi-start]
                cursor = end
            if cursor >= wave.shape[-1]:
                break
    return {"waveform": wave, "sample_rate": rate}
