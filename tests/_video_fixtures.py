"""Videos that reproduce the timing traps of real footage.

`seek_trap_video` is the 240fps capture problem in miniature: the container's
nominal frame rate (240) differs from the average presentation rate (~240.9),
which makes OpenCV's frame seek land one frame early part-way through, exactly
as on the Swing Catalyst captures. Every frame is random noise, so each has a
unique digest.
"""

import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest


def require_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")


def _noise_frames(path, n, size=(96, 64), repeat=1, seed=0):
    """Lossless AVI of n noise frames; each image repeated `repeat` times."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 240, (w, h))
    assert writer.isOpened(), "cv2 cannot write FFV1"
    rng = np.random.default_rng(seed)
    img = None
    for i in range(n):
        if i % repeat == 0:
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(img)
    writer.release()


def _encode(raw, out, setpts, extra=()):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", f"setpts={setpts}",
                    "-fps_mode", "vfr", *extra, str(out)], check=True)
    os.remove(raw)
    return str(out)


def seek_trap_video(tmp_path, n=1000, name="seek_trap.mp4", seed=0):
    require_ffmpeg()
    raw = tmp_path / f"{name}.raw.avi"
    _noise_frames(raw, n, seed=seed)
    return _encode(raw, tmp_path / name, "N/(249.3*TB)",
                   ("-c:v", "mpeg4", "-q:v", "2", "-video_track_timescale", "19928"))


def offset_video(tmp_path, n=120, offset_s=0.25, name="offset.mp4"):
    """Constant 240fps, but the first frame is presented at `offset_s`."""
    require_ffmpeg()
    raw = tmp_path / f"{name}.raw.avi"
    _noise_frames(raw, n, seed=1)
    return _encode(raw, tmp_path / name, f"N/(240*TB)+{offset_s}/TB",
                   ("-c:v", "libx264", "-g", "24", "-pix_fmt", "yuv444p", "-crf", "0",
                    "-video_track_timescale", "24000", "-output_ts_offset", "0"))


def vfr_video(tmp_path, n=200, name="vfr.mp4"):
    """60 frames at 120fps, then the rest at 240fps: genuinely variable."""
    require_ffmpeg()
    raw = tmp_path / f"{name}.raw.avi"
    _noise_frames(raw, n, seed=2)
    setpts = "(N*2-gte(N\\,60)*(N-60))*(1/240)/TB"
    return _encode(raw, tmp_path / name, setpts,
                   ("-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv444p", "-crf", "0",
                    "-video_track_timescale", "24000"))


def repeated_frames_video(tmp_path, n=120, repeat=3, name="repeated.avi"):
    """Each noise image appears `repeat` times in a row (frame-doubled footage)."""
    path = tmp_path / name
    _noise_frames(path, n, repeat=repeat, seed=3)
    return str(path)


def ffprobe_timestamps_ms(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return [float(line.strip().rstrip(",")) * 1000 for line in out.split() if line.strip()]
