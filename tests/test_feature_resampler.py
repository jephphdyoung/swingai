#!/usr/bin/env python3
"""
Unit tests for feature resampling utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_resampler import (
    ResampledFeatures,
    align_to_common_rate,
    compute_uniform_timestamps,
    estimate_fps_from_timestamps,
    resample_landmarks,
    resample_to_length,
)


def create_mock_landmarks(n_frames: int, motion_fn=None) -> list:
    """Create mock landmarks with optional motion function."""
    landmarks = []
    for i in range(n_frames):
        if motion_fn:
            x, y = motion_fn(i, n_frames)
        else:
            x, y = 0.5, 0.5

        # 21 landmarks, all same for simplicity
        frame = [(x, y, 0.0)] * 21
        landmarks.append(frame)

    return landmarks


class TestResampleLandmarks:
    """Tests for landmark resampling."""

    def test_empty_input(self):
        result = resample_landmarks([], [])

        assert result.landmarks == []
        assert result.timestamps_ms == []
        assert result.n_frames == 0

    def test_uniform_30fps_to_60fps(self):
        # 30fps input: 100 frames over ~3.3s
        n_frames = 100
        timestamps = [i * 33.33 for i in range(n_frames)]  # 30fps
        landmarks = create_mock_landmarks(n_frames)

        result = resample_landmarks(landmarks, timestamps, target_fps=60.0)

        # Should roughly double the frames
        assert result.n_frames > n_frames
        assert abs(result.n_frames - n_frames * 2) <= 2
        assert result.target_fps == 60.0

    def test_uniform_120fps_to_60fps(self):
        # 120fps input: 200 frames over ~1.67s
        n_frames = 200
        timestamps = [i * 8.33 for i in range(n_frames)]  # 120fps
        landmarks = create_mock_landmarks(n_frames)

        result = resample_landmarks(landmarks, timestamps, target_fps=60.0)

        # Should roughly halve the frames
        assert result.n_frames < n_frames
        assert abs(result.n_frames - n_frames // 2) <= 2

    def test_preserves_motion(self):
        # Linear motion from (0, 0) to (1, 1)
        def motion(i, n):
            t = i / (n - 1)
            return t, t

        n_frames = 50
        timestamps = [i * 20.0 for i in range(n_frames)]  # 50fps
        landmarks = create_mock_landmarks(n_frames, motion)

        result = resample_landmarks(landmarks, timestamps, target_fps=60.0)

        # First frame should be near (0, 0)
        first_lm = result.landmarks[0][0]
        assert abs(first_lm[0]) < 0.05
        assert abs(first_lm[1]) < 0.05

        # Last frame should be near (1, 1)
        last_lm = result.landmarks[-1][0]
        assert abs(last_lm[0] - 1.0) < 0.05
        assert abs(last_lm[1] - 1.0) < 0.05

    def test_timestamps_are_uniform(self):
        n_frames = 100
        # VFR input with varying intervals
        timestamps = [0]
        for i in range(1, n_frames):
            delta = 15 + np.random.uniform(-5, 10)  # 10-25ms intervals
            timestamps.append(timestamps[-1] + delta)

        landmarks = create_mock_landmarks(n_frames)
        result = resample_landmarks(landmarks, timestamps, target_fps=60.0)

        # Check output timestamps are uniform (within 1% tolerance)
        expected_interval = 1000.0 / 60.0  # ~16.67ms
        for i in range(1, len(result.timestamps_ms)):
            delta = result.timestamps_ms[i] - result.timestamps_ms[i-1]
            assert abs(delta - expected_interval) < 0.5  # Within 0.5ms

    def test_timestamps_start_at_zero(self):
        n_frames = 50
        timestamps = [1000 + i * 20.0 for i in range(n_frames)]  # Start at 1000ms
        landmarks = create_mock_landmarks(n_frames)

        result = resample_landmarks(landmarks, timestamps, target_fps=60.0)

        assert result.timestamps_ms[0] == 0.0


class TestResampleToLength:
    """Tests for resampling to specific length."""

    def test_upsample(self):
        landmarks = create_mock_landmarks(50)
        result = resample_to_length(landmarks, 100)

        assert len(result) == 100

    def test_downsample(self):
        landmarks = create_mock_landmarks(100)
        result = resample_to_length(landmarks, 50)

        assert len(result) == 50

    def test_same_length(self):
        landmarks = create_mock_landmarks(50)
        result = resample_to_length(landmarks, 50)

        assert len(result) == 50

    def test_empty_input(self):
        result = resample_to_length([], 50)

        assert result == []

    def test_preserves_endpoints(self):
        def motion(i, n):
            t = i / (n - 1)
            return t, t

        landmarks = create_mock_landmarks(30, motion)
        result = resample_to_length(landmarks, 100)

        # First should be near (0, 0)
        assert abs(result[0][0][0]) < 0.05

        # Last should be near (1, 1)
        assert abs(result[-1][0][0] - 1.0) < 0.05


class TestComputeUniformTimestamps:
    """Tests for uniform timestamp generation."""

    def test_60fps_1_second(self):
        timestamps = compute_uniform_timestamps(1000.0, fps=60.0)

        # 60fps = 16.67ms intervals, so ~60 frames in 1 second
        assert 59 <= len(timestamps) <= 61
        assert timestamps[0] == 0.0
        assert abs(timestamps[-1] - 1000.0) < 20.0  # Within one frame

    def test_30fps(self):
        timestamps = compute_uniform_timestamps(1000.0, fps=30.0)

        # 30fps = ~33.33ms interval
        expected_count = int(1000 / (1000/30)) + 1
        assert len(timestamps) == expected_count

    def test_intervals_are_correct(self):
        timestamps = compute_uniform_timestamps(500.0, fps=60.0)

        expected_interval = 1000.0 / 60.0
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i-1]
            assert abs(delta - expected_interval) < 0.01


class TestEstimateFpsFromTimestamps:
    """Tests for FPS estimation."""

    def test_60fps(self):
        timestamps = [i * 16.67 for i in range(100)]
        fps = estimate_fps_from_timestamps(timestamps)

        assert abs(fps - 60.0) < 1.0

    def test_30fps(self):
        timestamps = [i * 33.33 for i in range(100)]
        fps = estimate_fps_from_timestamps(timestamps)

        assert abs(fps - 30.0) < 1.0

    def test_120fps(self):
        timestamps = [i * 8.33 for i in range(100)]
        fps = estimate_fps_from_timestamps(timestamps)

        assert abs(fps - 120.0) < 2.0

    def test_vfr_uses_median(self):
        # VFR with mostly 60fps but some outliers
        timestamps = [0]
        for i in range(99):
            if i % 10 == 0:
                delta = 50.0  # Outlier
            else:
                delta = 16.67  # 60fps
            timestamps.append(timestamps[-1] + delta)

        fps = estimate_fps_from_timestamps(timestamps)

        # Should be close to 60fps (median ignores outliers)
        assert abs(fps - 60.0) < 5.0

    def test_single_timestamp(self):
        fps = estimate_fps_from_timestamps([0.0])

        assert fps == 60.0  # Default

    def test_empty(self):
        fps = estimate_fps_from_timestamps([])

        assert fps == 60.0  # Default


class TestAlignToCommonRate:
    """Tests for aligning two sequences."""

    def test_aligns_different_fps(self):
        # Sequence 1: 30fps
        n1 = 100
        timestamps1 = [i * 33.33 for i in range(n1)]
        landmarks1 = create_mock_landmarks(n1)

        # Sequence 2: 120fps
        n2 = 200
        timestamps2 = [i * 8.33 for i in range(n2)]
        landmarks2 = create_mock_landmarks(n2)

        result1, result2 = align_to_common_rate(
            landmarks1, timestamps1,
            landmarks2, timestamps2,
            target_fps=60.0
        )

        assert result1.target_fps == 60.0
        assert result2.target_fps == 60.0

    def test_same_fps_similar_length(self):
        # Both at 60fps, same duration
        n = 100
        timestamps = [i * 16.67 for i in range(n)]
        landmarks1 = create_mock_landmarks(n)
        landmarks2 = create_mock_landmarks(n)

        result1, result2 = align_to_common_rate(
            landmarks1, timestamps,
            landmarks2, timestamps,
            target_fps=60.0
        )

        # Should have similar number of frames
        assert abs(result1.n_frames - result2.n_frames) <= 2


class TestResampledFeatures:
    """Tests for ResampledFeatures dataclass."""

    def test_create(self):
        result = ResampledFeatures(
            landmarks=[[(0.5, 0.5, 0.0)] * 21],
            timestamps_ms=[0.0],
            original_duration_ms=1000.0,
            n_frames=1,
            target_fps=60.0,
        )

        assert result.n_frames == 1
        assert result.original_duration_ms == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
