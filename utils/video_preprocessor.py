"""
Video preprocessing utilities for time normalization.

Handles VFR (Variable Frame Rate) videos and mixed frame rates by:
1. Extracting per-frame timestamps using ffprobe
2. Transcoding to CFR (Constant Frame Rate) 60fps for analysis
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoInfo:
    """Metadata about a video file."""
    path: str
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    is_vfr: bool
    codec: str


@dataclass
class PreprocessedVideo:
    """Result of video preprocessing."""
    original_path: str
    analysis_path: str  # Path to CFR 60fps copy (or original if already CFR 60fps)
    frame_timestamps_ms: list[float]  # Per-frame timestamps in milliseconds
    original_info: VideoInfo
    is_transcoded: bool


def get_video_info(video_path: str) -> VideoInfo:
    """
    Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with metadata
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,width,height,codec_name",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    # Parse frame rate (format: "num/den")
    def parse_fps(fps_str: str) -> float:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            return float(num) / float(den) if float(den) != 0 else 0
        return float(fps_str)

    r_frame_rate = parse_fps(stream.get("r_frame_rate", "30/1"))
    avg_frame_rate = parse_fps(stream.get("avg_frame_rate", "30/1"))

    # VFR detection: if r_frame_rate and avg_frame_rate differ significantly
    is_vfr = abs(r_frame_rate - avg_frame_rate) > 1.0

    # Get frame count (may need to count if not in metadata)
    nb_frames = stream.get("nb_frames")
    if nb_frames:
        frame_count = int(nb_frames)
    else:
        frame_count = _count_frames(video_path)

    duration = float(stream.get("duration", frame_count / avg_frame_rate))

    return VideoInfo(
        path=video_path,
        fps=avg_frame_rate,
        frame_count=frame_count,
        duration_s=duration,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        is_vfr=is_vfr,
        codec=stream.get("codec_name", "unknown"),
    )


def _count_frames(video_path: str) -> int:
    """Count frames in video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return 0

    data = json.loads(result.stdout)
    return int(data["streams"][0].get("nb_read_frames", 0))


def extract_frame_timestamps(video_path: str) -> list[float]:
    """
    Extract per-frame timestamps using ffprobe.

    Uses best_effort_timestamp_time for accurate VFR handling.

    Args:
        video_path: Path to video file

    Returns:
        List of timestamps in milliseconds for each frame
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe frame extraction failed: {result.stderr}")

    data = json.loads(result.stdout)
    timestamps = []

    for frame in data.get("frames", []):
        ts = frame.get("best_effort_timestamp_time")
        if ts is not None:
            timestamps.append(float(ts) * 1000)  # Convert to ms
        elif timestamps:
            # Fallback: estimate from previous timestamp
            timestamps.append(timestamps[-1] + 16.67)  # ~60fps
        else:
            timestamps.append(0.0)

    return timestamps


def transcode_to_cfr(
    input_path: str,
    output_path: Optional[str] = None,
    target_fps: int = 60,
) -> str:
    """
    Transcode video to constant frame rate.

    Args:
        input_path: Path to input video
        output_path: Path for output (auto-generated if None)
        target_fps: Target frame rate (default 60)

    Returns:
        Path to transcoded video
    """
    if output_path is None:
        # Create temp file with same extension
        input_ext = Path(input_path).suffix
        fd, output_path = tempfile.mkstemp(suffix=f"_cfr{target_fps}{input_ext}")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-vf", f"fps={target_fps}",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",  # No audio needed for analysis
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr}")

    return output_path


def preprocess_video(
    video_path: str,
    target_fps: int = 60,
    force_transcode: bool = False,
    output_dir: Optional[str] = None,
) -> PreprocessedVideo:
    """
    Preprocess video for analysis.

    - Extracts frame timestamps from original
    - Transcodes to CFR 60fps if VFR or different frame rate
    - Returns paths and timing info

    Args:
        video_path: Path to input video
        target_fps: Target frame rate for analysis copy
        force_transcode: Always transcode even if already CFR at target fps
        output_dir: Directory for transcoded files (temp if None)

    Returns:
        PreprocessedVideo with paths and timestamps
    """
    video_path = os.path.abspath(video_path)

    # Get video info
    info = get_video_info(video_path)

    # Extract timestamps from original (before any transcoding)
    timestamps = extract_frame_timestamps(video_path)

    # Decide if transcoding is needed
    needs_transcode = (
        force_transcode
        or info.is_vfr
        or abs(info.fps - target_fps) > 1.0
    )

    if needs_transcode:
        # Generate output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = Path(video_path).stem
            output_path = os.path.join(output_dir, f"{base}_cfr{target_fps}.mp4")
        else:
            output_path = None  # Let transcode_to_cfr create temp file

        analysis_path = transcode_to_cfr(video_path, output_path, target_fps)

        # Get new timestamps for transcoded video
        analysis_timestamps = extract_frame_timestamps(analysis_path)
    else:
        analysis_path = video_path
        analysis_timestamps = timestamps

    return PreprocessedVideo(
        original_path=video_path,
        analysis_path=analysis_path,
        frame_timestamps_ms=analysis_timestamps,
        original_info=info,
        is_transcoded=needs_transcode,
    )


def compute_frame_deltas(timestamps_ms: list[float]) -> list[float]:
    """
    Compute time deltas between consecutive frames.

    Args:
        timestamps_ms: List of timestamps in milliseconds

    Returns:
        List of deltas (first delta is 0)
    """
    if not timestamps_ms:
        return []

    deltas = [0.0]
    for i in range(1, len(timestamps_ms)):
        dt = timestamps_ms[i] - timestamps_ms[i - 1]
        # Clamp to reasonable range (0.1ms to 100ms)
        dt = max(0.1, min(100.0, dt))
        deltas.append(dt)

    return deltas


def cleanup_temp_files(preprocessed: PreprocessedVideo) -> None:
    """Remove temporary transcoded files."""
    if preprocessed.is_transcoded and preprocessed.analysis_path != preprocessed.original_path:
        try:
            os.remove(preprocessed.analysis_path)
        except OSError:
            pass
