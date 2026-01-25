#!/usr/bin/env python3
"""
Unit tests for swing detection and waggle trimming.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.swing_detector import (
    SwingDetectionParams,
    SwingWindow,
    compute_wrist_speed,
    detect_motion_bursts,
    detect_swing_window,
    estimate_baseline,
    find_address_frame,
    find_main_swing_burst,
    trim_to_swing,
)


def create_mock_landmarks(n_frames: int, wrist_positions: list[tuple[float, float]]) -> list:
    """
    Create mock landmarks with specified wrist positions.

    Args:
        n_frames: Number of frames
        wrist_positions: List of (x, y) tuples for average wrist position per frame

    Returns:
        Mock landmark list (21 landmarks per frame, only wrists vary)
    """
    landmarks = []
    for i in range(n_frames):
        if i < len(wrist_positions):
            wx, wy = wrist_positions[i]
        else:
            wx, wy = wrist_positions[-1]

        # Create 21 landmarks (all same except wrists at indices 5, 6)
        frame = [(0.5, 0.5, 0.0)] * 21
        frame[5] = (wx, wy, 0.0)  # Left wrist
        frame[6] = (wx, wy, 0.0)  # Right wrist
        landmarks.append(frame)

    return landmarks


class TestComputeWristSpeed:
    """Tests for wrist speed computation."""

    def test_stationary_wrist(self):
        positions = [(0.5, 0.5)] * 10
        landmarks = create_mock_landmarks(10, positions)
        speeds = compute_wrist_speed(landmarks)

        assert len(speeds) == 10
        assert speeds[0] == 0.0
        assert all(s == 0.0 for s in speeds[1:])

    def test_constant_velocity(self):
        # Move 0.01 units per frame
        positions = [(0.5 + i * 0.01, 0.5) for i in range(10)]
        landmarks = create_mock_landmarks(10, positions)
        timestamps = [i * 16.67 for i in range(10)]  # 60fps

        speeds = compute_wrist_speed(landmarks, timestamps)

        # Speed should be ~0.6 units/s (0.01 / 0.01667s)
        assert speeds[0] == 0.0
        for s in speeds[1:]:
            assert 0.5 < s < 0.7

    def test_variable_timestamps(self):
        # Move same distance but with variable time
        positions = [(0.5, 0.5), (0.51, 0.5), (0.52, 0.5)]
        landmarks = create_mock_landmarks(3, positions)

        # First interval: 10ms, second: 50ms
        timestamps = [0.0, 10.0, 60.0]
        speeds = compute_wrist_speed(landmarks, timestamps)

        # First move: faster, second: slower
        assert speeds[1] > speeds[2]


class TestEstimateBaseline:
    """Tests for baseline estimation."""

    def test_low_baseline(self):
        # Mostly stationary
        positions = [(0.5 + np.random.normal(0, 0.001), 0.5) for _ in range(100)]
        landmarks = create_mock_landmarks(100, positions)
        speeds = compute_wrist_speed(landmarks)

        median, mad = estimate_baseline(speeds)

        assert median < 0.1  # Low baseline
        assert mad < 0.1

    def test_baseline_duration(self):
        # Use timestamps to control baseline duration
        positions = [(0.5, 0.5)] * 100
        landmarks = create_mock_landmarks(100, positions)
        speeds = compute_wrist_speed(landmarks)

        # 500ms baseline at 60fps = ~30 frames
        timestamps = [i * 16.67 for i in range(100)]
        median, mad = estimate_baseline(speeds, timestamps, duration_ms=500.0)

        assert median == 0.0
        assert mad == 0.001  # Minimum MAD


class TestDetectMotionBursts:
    """Tests for motion burst detection."""

    def test_single_burst(self):
        # Create sequence with one burst of motion
        positions = [(0.5, 0.5)] * 30  # Stationary
        positions += [(0.5 + i * 0.02, 0.5) for i in range(20)]  # Motion
        positions += [(0.9, 0.5)] * 30  # Stationary again

        landmarks = create_mock_landmarks(80, positions)
        speeds = compute_wrist_speed(landmarks)

        median, mad = estimate_baseline(speeds[:30])
        params = SwingDetectionParams()
        bursts = detect_motion_bursts(speeds, median, mad, params)

        assert len(bursts) >= 1
        # Burst should start around frame 30
        start, end, peak = bursts[0]
        assert 25 <= start <= 35

    def test_multiple_bursts(self):
        # Create sequence with multiple bursts (waggle + swing)
        positions = [(0.5, 0.5)] * 20
        positions += [(0.5 + i * 0.01, 0.5) for i in range(10)]  # Small waggle
        positions += [(0.6, 0.5)] * 20
        positions += [(0.6 + i * 0.03, 0.5) for i in range(30)]  # Big swing
        positions += [(1.5, 0.5)] * 20

        landmarks = create_mock_landmarks(100, positions)
        speeds = compute_wrist_speed(landmarks)

        median, mad = estimate_baseline(speeds[:20])
        params = SwingDetectionParams(min_burst_frames=3)
        bursts = detect_motion_bursts(speeds, median, mad, params)

        assert len(bursts) >= 2

    def test_no_burst(self):
        # Completely stationary
        positions = [(0.5, 0.5)] * 50
        landmarks = create_mock_landmarks(50, positions)
        speeds = compute_wrist_speed(landmarks)

        median, mad = estimate_baseline(speeds)
        params = SwingDetectionParams()
        bursts = detect_motion_bursts(speeds, median, mad, params)

        assert len(bursts) == 0


class TestFindMainSwingBurst:
    """Tests for main swing selection."""

    def test_prefers_longer_burst(self):
        # Short burst and long burst
        bursts = [
            (10, 20, 0.5),   # Short, high speed
            (50, 100, 0.4),  # Long, slightly lower speed
        ]

        main = find_main_swing_burst(bursts)

        # Should prefer the longer burst
        assert main[0] == 50

    def test_prefers_higher_speed(self):
        # Similar duration, different speeds
        bursts = [
            (10, 50, 0.3),   # Medium speed
            (60, 100, 0.6),  # High speed
        ]
        timestamps = [i * 16.67 for i in range(150)]

        main = find_main_swing_burst(bursts, timestamps)

        # Should prefer higher speed burst
        assert main[0] == 60

    def test_empty_bursts(self):
        main = find_main_swing_burst([])
        assert main is None


class TestFindAddressFrame:
    """Tests for address (P1) frame detection."""

    def test_finds_stable_frame(self):
        # Create speed profile: low, low, low, HIGH, HIGH, HIGH
        speeds = np.array([0.01, 0.02, 0.01, 0.5, 0.6, 0.5, 0.4])
        takeaway_frame = 3

        address = find_address_frame(speeds, takeaway_frame)

        # Should find one of the low-speed frames
        assert 0 <= address <= 2

    def test_with_timestamps(self):
        speeds = np.array([0.01] * 30 + [0.5] * 20)
        timestamps = [i * 16.67 for i in range(50)]
        takeaway_frame = 30

        params = SwingDetectionParams(pre_window_ms=300.0)
        address = find_address_frame(speeds, takeaway_frame, timestamps, params)

        # Should be within 300ms before takeaway
        assert 10 < address < 30


class TestDetectSwingWindow:
    """Tests for full swing window detection."""

    def test_detects_swing(self):
        # Create realistic swing sequence
        n = 200
        positions = []

        # Setup/waggle (frames 0-50)
        for i in range(50):
            x = 0.5 + np.random.normal(0, 0.005)
            positions.append((x, 0.5))

        # Address (frames 50-60)
        for i in range(10):
            positions.append((0.5, 0.5))

        # Backswing (frames 60-100)
        for i in range(40):
            t = i / 40
            x = 0.5 - 0.2 * t  # Move left
            y = 0.5 - 0.15 * t  # Move up
            positions.append((x, y))

        # Downswing + impact (frames 100-120)
        for i in range(20):
            t = i / 20
            x = 0.3 + 0.4 * t  # Move right fast
            y = 0.35 + 0.15 * t  # Move down
            positions.append((x, y))

        # Follow-through (frames 120-150)
        for i in range(30):
            positions.append((0.7 + i * 0.002, 0.5))

        # Finish (frames 150-200)
        for i in range(50):
            positions.append((0.76, 0.5))

        landmarks = create_mock_landmarks(n, positions)
        timestamps = [i * 16.67 for i in range(n)]

        window = detect_swing_window(landmarks, timestamps)

        assert window is not None
        assert window.start_frame < window.takeaway_frame
        assert window.takeaway_frame < window.impact_frame
        assert window.impact_frame < window.end_frame
        assert window.confidence > 0

    def test_no_swing_in_stationary(self):
        positions = [(0.5, 0.5)] * 100
        landmarks = create_mock_landmarks(100, positions)

        window = detect_swing_window(landmarks)

        assert window is None

    def test_too_few_frames(self):
        positions = [(0.5, 0.5)] * 10
        landmarks = create_mock_landmarks(10, positions)

        window = detect_swing_window(landmarks)

        assert window is None


class TestTrimToSwing:
    """Tests for swing trimming."""

    def test_trims_landmarks(self):
        # Create sequence with swing in middle
        n = 100
        positions = [(0.5, 0.5)] * 30  # Pre-swing
        positions += [(0.5 + i * 0.02, 0.5) for i in range(40)]  # Swing
        positions += [(1.3, 0.5)] * 30  # Post-swing

        landmarks = create_mock_landmarks(n, positions)
        timestamps = [i * 16.67 for i in range(n)]

        trimmed_lm, trimmed_ts, window = trim_to_swing(landmarks, timestamps)

        if window is not None:
            assert len(trimmed_lm) < len(landmarks)
            assert trimmed_ts[0] == 0.0  # Re-zeroed

    def test_no_swing_returns_original(self):
        positions = [(0.5, 0.5)] * 50
        landmarks = create_mock_landmarks(50, positions)
        timestamps = [i * 16.67 for i in range(50)]

        trimmed_lm, trimmed_ts, window = trim_to_swing(landmarks, timestamps)

        assert window is None
        assert trimmed_lm == landmarks


class TestSwingWindow:
    """Tests for SwingWindow dataclass."""

    def test_create_window(self):
        window = SwingWindow(
            start_frame=50,
            end_frame=150,
            takeaway_frame=60,
            impact_frame=110,
            confidence=0.85,
        )

        assert window.start_frame == 50
        assert window.end_frame == 150
        assert window.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
