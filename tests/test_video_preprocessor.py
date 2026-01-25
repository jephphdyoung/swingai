#!/usr/bin/env python3
"""
Unit tests for video preprocessing utilities.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_preprocessor import (
    VideoInfo,
    PreprocessedVideo,
    compute_frame_deltas,
)


class TestComputeFrameDeltas:
    """Tests for frame delta computation."""

    def test_empty_timestamps(self):
        assert compute_frame_deltas([]) == []

    def test_single_timestamp(self):
        deltas = compute_frame_deltas([0.0])
        assert deltas == [0.0]

    def test_constant_frame_rate_30fps(self):
        # 30fps = ~33.33ms per frame
        timestamps = [0.0, 33.33, 66.67, 100.0]
        deltas = compute_frame_deltas(timestamps)

        assert deltas[0] == 0.0
        assert abs(deltas[1] - 33.33) < 0.1
        assert abs(deltas[2] - 33.34) < 0.1
        assert abs(deltas[3] - 33.33) < 0.1

    def test_constant_frame_rate_60fps(self):
        # 60fps = ~16.67ms per frame
        timestamps = [0.0, 16.67, 33.33, 50.0]
        deltas = compute_frame_deltas(timestamps)

        assert deltas[0] == 0.0
        assert abs(deltas[1] - 16.67) < 0.1

    def test_variable_frame_rate(self):
        # VFR with varying intervals
        timestamps = [0.0, 20.0, 50.0, 60.0, 100.0]
        deltas = compute_frame_deltas(timestamps)

        assert deltas[0] == 0.0
        assert deltas[1] == 20.0
        assert deltas[2] == 30.0
        assert deltas[3] == 10.0
        assert deltas[4] == 40.0

    def test_clamps_small_deltas(self):
        # Very small deltas should be clamped to minimum
        timestamps = [0.0, 0.05, 1.0]  # 0.05ms is too small
        deltas = compute_frame_deltas(timestamps)

        assert deltas[1] >= 0.1  # Should be clamped

    def test_clamps_large_deltas(self):
        # Very large deltas should be clamped
        timestamps = [0.0, 200.0]  # 200ms is too large
        deltas = compute_frame_deltas(timestamps)

        assert deltas[1] <= 100.0  # Should be clamped


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_create_video_info(self):
        info = VideoInfo(
            path="/test/video.mp4",
            fps=60.0,
            frame_count=360,
            duration_s=6.0,
            width=1920,
            height=1080,
            is_vfr=False,
            codec="h264",
        )

        assert info.fps == 60.0
        assert info.frame_count == 360
        assert info.duration_s == 6.0
        assert not info.is_vfr

    def test_vfr_video_info(self):
        info = VideoInfo(
            path="/test/vfr_video.mp4",
            fps=29.97,
            frame_count=300,
            duration_s=10.0,
            width=1280,
            height=720,
            is_vfr=True,
            codec="hevc",
        )

        assert info.is_vfr


class TestPreprocessedVideo:
    """Tests for PreprocessedVideo dataclass."""

    def test_not_transcoded(self):
        info = VideoInfo(
            path="/test/video.mp4",
            fps=60.0,
            frame_count=60,
            duration_s=1.0,
            width=1920,
            height=1080,
            is_vfr=False,
            codec="h264",
        )

        result = PreprocessedVideo(
            original_path="/test/video.mp4",
            analysis_path="/test/video.mp4",  # Same path = no transcode
            frame_timestamps_ms=[i * 16.67 for i in range(60)],
            original_info=info,
            is_transcoded=False,
        )

        assert result.original_path == result.analysis_path
        assert not result.is_transcoded
        assert len(result.frame_timestamps_ms) == 60

    def test_transcoded(self):
        info = VideoInfo(
            path="/test/vfr_video.mp4",
            fps=29.97,
            frame_count=100,
            duration_s=3.33,
            width=1280,
            height=720,
            is_vfr=True,
            codec="hevc",
        )

        result = PreprocessedVideo(
            original_path="/test/vfr_video.mp4",
            analysis_path="/tmp/vfr_video_cfr60.mp4",
            frame_timestamps_ms=[i * 16.67 for i in range(200)],  # 60fps
            original_info=info,
            is_transcoded=True,
        )

        assert result.original_path != result.analysis_path
        assert result.is_transcoded
        # Transcoded to 60fps from ~30fps means ~2x frames
        assert len(result.frame_timestamps_ms) == 200


class TestTimestampConsistency:
    """Tests for timestamp handling consistency."""

    def test_timestamps_are_monotonic(self):
        """Timestamps should always increase."""
        timestamps = [0.0, 16.67, 33.33, 50.0, 66.67]
        deltas = compute_frame_deltas(timestamps)

        # All deltas after first should be positive
        assert all(d > 0 for d in deltas[1:])

    def test_handles_duplicate_timestamps(self):
        """Should handle (and clamp) duplicate timestamps gracefully."""
        timestamps = [0.0, 16.67, 16.67, 33.33]  # Duplicate at index 2
        deltas = compute_frame_deltas(timestamps)

        # Should clamp the 0 delta to minimum
        assert deltas[2] >= 0.1


class TestIntegrationWithPoseFeatures:
    """Integration tests with pose_features module."""

    def test_timestamps_produce_consistent_velocities(self):
        """Verify that time-normalized velocities are frame-rate independent."""
        # Simulate same motion at different frame rates

        # 30fps: 3 frames covering 100ms
        timestamps_30fps = [0.0, 33.33, 66.67, 100.0]

        # 60fps: 6 frames covering 100ms (same motion, more samples)
        timestamps_60fps = [0.0, 16.67, 33.33, 50.0, 66.67, 83.33, 100.0]

        deltas_30 = compute_frame_deltas(timestamps_30fps)
        deltas_60 = compute_frame_deltas(timestamps_60fps)

        # Deltas should reflect actual time intervals
        assert abs(deltas_30[1] - 33.33) < 0.1
        assert abs(deltas_60[1] - 16.67) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
