#!/usr/bin/env python3
"""
Unit tests for DTW alignment and label transfer.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dtw_alignment import (
    AlignmentResult,
    RefinedPositions,
    compute_dtw_alignment,
    detect_p_positions_with_dtw,
    refine_p4_top_of_backswing,
    refine_p6_impact,
    refine_p7_followthrough,
    refine_positions,
    transfer_positions,
)


def create_mock_landmarks(n_frames: int, motion_fn=None) -> list:
    """Create mock landmarks with optional motion function."""
    landmarks = []
    for i in range(n_frames):
        if motion_fn:
            wx, wy = motion_fn(i, n_frames)
        else:
            wx, wy = 0.5, 0.5

        # Create 21 landmarks
        frame = [
            (0.5, 0.3, 0.0),  # 0: nose
            (0.4, 0.4, 0.0),  # 1: left shoulder
            (0.6, 0.4, 0.0),  # 2: right shoulder
            (0.35, 0.5, 0.0),  # 3: left elbow
            (0.65, 0.5, 0.0),  # 4: right elbow
            (wx - 0.1, wy, 0.0),  # 5: left wrist
            (wx + 0.1, wy, 0.0),  # 6: right wrist
            (wx - 0.12, wy + 0.02, 0.0),  # 7: left pinky
            (wx + 0.12, wy + 0.02, 0.0),  # 8: right pinky
            (wx - 0.11, wy + 0.01, 0.0),  # 9: left index
            (wx + 0.11, wy + 0.01, 0.0),  # 10: right index
            (wx - 0.09, wy + 0.01, 0.0),  # 11: left thumb
            (wx + 0.09, wy + 0.01, 0.0),  # 12: right thumb
            (0.425, 0.6, 0.0),  # 13: left hip
            (0.575, 0.6, 0.0),  # 14: right hip
            (0.425, 0.75, 0.0),  # 15: left knee
            (0.575, 0.75, 0.0),  # 16: right knee
            (0.425, 0.9, 0.0),  # 17: left ankle
            (0.575, 0.9, 0.0),  # 18: right ankle
            (0.425, 0.92, 0.0),  # 19: left foot
            (0.575, 0.92, 0.0),  # 20: right foot
        ]
        landmarks.append(frame)

    return landmarks


def create_swing_motion(i: int, n: int) -> tuple[float, float]:
    """Create realistic swing motion pattern."""
    t = i / (n - 1)

    if t < 0.3:
        # Address to top of backswing
        x = 0.5 - 0.2 * (t / 0.3)
        y = 0.5 - 0.2 * (t / 0.3)  # Hands go up
    elif t < 0.5:
        # Top of backswing to impact
        phase = (t - 0.3) / 0.2
        x = 0.3 + 0.4 * phase  # Fast downswing
        y = 0.3 + 0.3 * phase
    else:
        # Follow-through
        phase = (t - 0.5) / 0.5
        x = 0.7 + 0.1 * phase
        y = 0.6 - 0.1 * phase  # Hands go up again

    return x, y


class TestComputeDTWAlignment:
    """Tests for DTW alignment computation."""

    def test_identical_sequences(self):
        landmarks = create_mock_landmarks(50)
        alignment = compute_dtw_alignment(landmarks, landmarks)

        assert alignment.distance == 0.0 or alignment.distance < 1.0
        assert len(alignment.path) >= 50

    def test_similar_sequences(self):
        def motion1(i, n):
            return 0.5 + i * 0.01, 0.5

        def motion2(i, n):
            return 0.5 + i * 0.01 + 0.001, 0.5  # Slightly offset

        landmarks1 = create_mock_landmarks(50, motion1)
        landmarks2 = create_mock_landmarks(50, motion2)

        alignment = compute_dtw_alignment(landmarks1, landmarks2)

        # Should align well
        assert alignment.distance < 100

    def test_different_lengths(self):
        landmarks1 = create_mock_landmarks(50)
        landmarks2 = create_mock_landmarks(70)

        alignment = compute_dtw_alignment(landmarks1, landmarks2)

        # Should create valid mappings
        assert len(alignment.user_to_ref) > 0
        assert len(alignment.ref_to_user) > 0

    def test_mappings_are_valid(self):
        landmarks1 = create_mock_landmarks(50)
        landmarks2 = create_mock_landmarks(60)

        alignment = compute_dtw_alignment(landmarks1, landmarks2)

        # All user frames should map to valid ref frames
        for user_frame, ref_frame in alignment.user_to_ref.items():
            assert 0 <= user_frame < 50
            assert 0 <= ref_frame < 60


class TestTransferPositions:
    """Tests for position transfer via DTW."""

    def test_transfers_positions(self):
        ref_positions = {"P1": 10, "P4": 30, "P6": 45, "P9": 80}

        # Create mock alignment
        alignment = AlignmentResult(
            path=[(i, i) for i in range(100)],
            distance=0.0,
            user_to_ref={i: i for i in range(100)},
            ref_to_user={i: i for i in range(100)},
        )

        transferred = transfer_positions(ref_positions, alignment)

        assert "P1" in transferred
        assert "P4" in transferred
        assert "P6" in transferred
        assert "P9" in transferred

    def test_handles_scaled_alignment(self):
        ref_positions = {"P1": 10, "P6": 50}

        # User is 2x speed of ref
        alignment = AlignmentResult(
            path=[(i, i * 2) for i in range(50)],
            distance=0.0,
            user_to_ref={i: i * 2 for i in range(50)},
            ref_to_user={i * 2: i for i in range(50)},
        )

        transferred = transfer_positions(ref_positions, alignment)

        # P1 at ref frame 10 should map to user frame 5
        assert transferred["P1"] == 5
        # P6 at ref frame 50 should map to user frame 25
        assert transferred["P6"] == 25

    def test_handles_missing_frames(self):
        ref_positions = {"P1": 15}

        # Sparse alignment (every other frame)
        alignment = AlignmentResult(
            path=[(i, i * 2) for i in range(50)],
            distance=0.0,
            user_to_ref={i: i * 2 for i in range(50)},
            ref_to_user={i * 2: i for i in range(50)},
        )

        transferred = transfer_positions(ref_positions, alignment)

        # Should find closest frame
        assert "P1" in transferred


class TestRefineP4:
    """Tests for P4 refinement."""

    def test_finds_hand_height_peak(self):
        # Create landmarks with clear peak at frame 30
        def motion(i, n):
            # Hands highest (lowest y) at frame 30
            if i < 30:
                y = 0.5 - 0.2 * (i / 30)  # Rising
            else:
                y = 0.3 + 0.1 * ((i - 30) / (n - 30))  # Falling
            return 0.5, y

        landmarks = create_mock_landmarks(60, motion)

        refined, confidence = refine_p4_top_of_backswing(landmarks, 25)

        # Should find frame near 30
        assert 25 <= refined <= 35
        assert confidence > 0.0

    def test_stays_near_initial(self):
        landmarks = create_mock_landmarks(60)
        initial = 30

        refined, _ = refine_p4_top_of_backswing(landmarks, initial, search_window=5)

        # Should stay within search window
        assert abs(refined - initial) <= 5


class TestRefineP6:
    """Tests for P6 refinement."""

    def test_finds_velocity_peak(self):
        # Create landmarks with velocity peak at frame 40
        def motion(i, n):
            if i < 35:
                # Slow motion
                return 0.5 + i * 0.001, 0.5
            elif i < 45:
                # Fast motion (impact zone)
                return 0.5 + 0.035 + (i - 35) * 0.02, 0.5
            else:
                # Slow again
                return 0.7 + (i - 45) * 0.002, 0.5

        landmarks = create_mock_landmarks(60, motion)
        timestamps = [i * 16.67 for i in range(60)]

        refined, confidence = refine_p6_impact(landmarks, timestamps, 35)

        # Should find frame in high velocity zone
        assert 35 <= refined <= 50

    def test_with_no_timestamps(self):
        landmarks = create_mock_landmarks(60)

        refined, confidence = refine_p6_impact(landmarks, None, 30)

        assert 0 <= refined < 60


class TestRefineP7:
    """Tests for P7 refinement."""

    def test_finds_followthrough(self):
        landmarks = create_mock_landmarks(80)
        timestamps = [i * 16.67 for i in range(80)]

        p6_frame = 40
        refined, confidence = refine_p7_followthrough(landmarks, timestamps, p6_frame)

        # P7 should be after P6
        assert refined > p6_frame

    def test_respects_min_gap(self):
        landmarks = create_mock_landmarks(80)

        p6_frame = 40
        refined, _ = refine_p7_followthrough(landmarks, None, p6_frame)

        # Should be at least 5 frames after P6
        assert refined >= p6_frame + 5


class TestRefinePositions:
    """Tests for full position refinement."""

    def test_refines_all_positions(self):
        landmarks = create_mock_landmarks(100, create_swing_motion)
        timestamps = [i * 16.67 for i in range(100)]

        transferred = {
            "P1": 10,
            "P4": 30,
            "P6": 50,
            "P9": 90,
        }

        refined = refine_positions(landmarks, timestamps, transferred)

        # Should have all P positions
        for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]:
            assert p in refined.positions

    def test_maintains_ordering(self):
        landmarks = create_mock_landmarks(100)
        timestamps = [i * 16.67 for i in range(100)]

        transferred = {
            "P1": 10,
            "P4": 30,
            "P6": 50,
            "P9": 90,
        }

        refined = refine_positions(landmarks, timestamps, transferred)

        # Positions should be in order
        prev = -1
        for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]:
            assert refined.positions[p] > prev
            prev = refined.positions[p]

    def test_assigns_methods(self):
        landmarks = create_mock_landmarks(100)
        transferred = {"P1": 10, "P4": 30, "P6": 50, "P9": 90}

        refined = refine_positions(landmarks, None, transferred)

        assert refined.method["P4"] == "hand_height_peak"
        assert refined.method["P6"] == "velocity_peak"


class TestDetectPPositionsWithDTW:
    """Tests for full DTW-based detection."""

    def test_end_to_end(self):
        # Create user and ref sequences with similar motion
        user_landmarks = create_mock_landmarks(100, create_swing_motion)
        ref_landmarks = create_mock_landmarks(80, create_swing_motion)

        ref_positions = {
            "P1": 8,
            "P4": 24,
            "P6": 40,
            "P9": 72,
        }

        user_timestamps = [i * 16.67 for i in range(100)]
        ref_timestamps = [i * 16.67 for i in range(80)]

        result = detect_p_positions_with_dtw(
            user_landmarks,
            ref_landmarks,
            ref_positions,
            user_timestamps,
            ref_timestamps,
        )

        # Should detect all positions
        assert len(result.positions) == 9
        assert all(p in result.positions for p in ["P1", "P4", "P6", "P9"])

        # Should have confidences
        assert len(result.confidences) == 9

    def test_different_speed_sequences(self):
        # User swings faster than ref
        def user_motion(i, n):
            return create_swing_motion(i * 2 % n, n)

        user_landmarks = create_mock_landmarks(50, user_motion)
        ref_landmarks = create_mock_landmarks(100, create_swing_motion)

        ref_positions = {"P1": 10, "P4": 30, "P6": 50, "P9": 90}

        result = detect_p_positions_with_dtw(
            user_landmarks,
            ref_landmarks,
            ref_positions,
        )

        # Should still detect positions
        assert len(result.positions) >= 4


class TestRefinedPositions:
    """Tests for RefinedPositions dataclass."""

    def test_create(self):
        result = RefinedPositions(
            positions={"P1": 10, "P4": 30},
            confidences={"P1": 0.8, "P4": 0.9},
            method={"P1": "dtw_transfer", "P4": "hand_height_peak"},
        )

        assert result.positions["P1"] == 10
        assert result.confidences["P4"] == 0.9
        assert result.method["P1"] == "dtw_transfer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
