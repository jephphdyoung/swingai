#!/usr/bin/env python3
"""
Unit tests for club-based P2/P6 position detection.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.club_positions import (
    ClubPositionResult,
    detect_shaft_parallel_frames,
    find_p2_backswing,
    find_p6_impact,
    find_p7_followthrough,
    SHAFT_PARALLEL_ANGLE,
    SHAFT_PARALLEL_TOLERANCE,
)


class TestDetectShaftParallelFrames:
    """Tests for shaft parallel detection."""

    def test_finds_parallel_frames(self):
        # Create angles with some parallel positions
        angles = [45.0] * 20 + [80.0] * 5 + [45.0] * 20

        parallel = detect_shaft_parallel_frames(angles)

        # Should find frames 20-24
        assert len(parallel) == 5
        assert all(20 <= f <= 24 for f in parallel)

    def test_with_tolerance(self):
        # Angle just within tolerance
        angles = [45.0] * 10 + [70.0] * 3 + [45.0] * 10  # 70° is within ±15° of 80°

        parallel = detect_shaft_parallel_frames(angles, target_angle=80.0, tolerance=15.0)

        assert len(parallel) == 3

    def test_no_parallel_frames(self):
        # All angles far from parallel
        angles = [30.0] * 50

        parallel = detect_shaft_parallel_frames(angles)

        assert len(parallel) == 0

    def test_handles_none(self):
        angles = [45.0, None, 80.0, None, 45.0]

        parallel = detect_shaft_parallel_frames(angles)

        assert 2 in parallel
        assert len(parallel) == 1


class TestFindP2Backswing:
    """Tests for P2 (backswing shaft parallel) detection."""

    def test_finds_first_parallel(self):
        # Simulate backswing: angle increases from address to top
        angles = []
        for i in range(50):
            # Start at 45°, increase toward 90°
            angle = 45 + (i / 50) * 45
            angles.append(angle)

        p2 = find_p2_backswing(angles, p1_frame=0, p4_frame=50)

        # P2 should be around frame where angle reaches ~65° (80-15 tolerance)
        # 65 = 45 + (i/50) * 45 → i = 20/45 * 50 ≈ 22
        assert p2 is not None
        assert 20 <= p2 <= 45

    def test_returns_none_if_invalid_range(self):
        angles = [45.0] * 50

        p2 = find_p2_backswing(angles, p1_frame=30, p4_frame=20)  # Invalid

        assert p2 is None

    def test_finds_closest_if_no_exact_match(self):
        # Angles never quite reach parallel
        angles = [45.0] * 10 + [65.0] * 10 + [45.0] * 10

        p2 = find_p2_backswing(angles, p1_frame=0, p4_frame=30)

        # Should find frame with angle closest to 80°
        assert p2 is not None
        assert 10 <= p2 < 20


class TestFindP6Impact:
    """Tests for P6 (impact) refinement with club angle."""

    def test_finds_most_vertical(self):
        # Simulate downswing: angle decreases toward impact, then increases
        angles = []
        for i in range(60):
            if i < 30:
                # Downswing: decreasing angle
                angle = 60 - i * 2
            else:
                # After impact: increasing angle
                angle = (i - 30) * 2

            angles.append(max(0, angle))

        p6, confidence = find_p6_impact(angles, p4_frame=0, p9_frame=60)

        # Impact should be around frame 30 (minimum angle)
        assert p6 is not None
        assert 25 <= p6 <= 35
        assert confidence > 0.5

    def test_respects_initial_estimate(self):
        angles = [30.0] * 60
        angles[25] = 5.0  # Clear minimum

        p6, _ = find_p6_impact(angles, p4_frame=0, p9_frame=60, initial_p6=30)

        # Should search around initial estimate
        assert p6 == 25

    def test_returns_initial_if_no_clear_impact(self):
        angles = [45.0] * 60  # All same angle, no clear minimum

        p6, confidence = find_p6_impact(angles, p4_frame=0, p9_frame=60, initial_p6=30)

        # Should return a frame (first in search window since all equal)
        # and have moderate confidence since 45° is not that vertical
        assert p6 is not None
        assert 15 <= p6 <= 45  # Within search window around initial_p6


class TestFindP7Followthrough:
    """Tests for follow-through shaft parallel detection."""

    def test_finds_followthrough_parallel(self):
        # After impact, angle increases back toward horizontal
        angles = [10.0] * 10  # Impact zone (vertical)
        angles += [angle for angle in range(10, 90, 2)]  # Increasing to horizontal

        p7 = find_p7_followthrough(angles, p6_frame=5, p9_frame=len(angles))

        # Should find frame where angle reaches ~80°
        assert p7 is not None
        assert p7 > 10

    def test_returns_none_if_never_horizontal(self):
        # Club stays relatively vertical
        angles = [20.0] * 50

        p7 = find_p7_followthrough(angles, p6_frame=10, p9_frame=50)

        assert p7 is None

    def test_skips_frames_right_after_impact(self):
        # Even if first frame is horizontal, skip it (too close to impact)
        angles = [80.0] * 20

        p7 = find_p7_followthrough(angles, p6_frame=0, p9_frame=20)

        # Should skip first few frames after impact
        assert p7 is None or p7 > 5


class TestClubPositionResult:
    """Tests for ClubPositionResult dataclass."""

    def test_create(self):
        result = ClubPositionResult(
            p2_frame=30,
            p6_frame=50,
            p2_follow_frame=70,
            confidence=0.75,
            detection_rate=0.6,
            angles=[45.0] * 100,
        )

        assert result.p2_frame == 30
        assert result.p6_frame == 50
        assert result.confidence == 0.75

    def test_none_values(self):
        result = ClubPositionResult(
            p2_frame=None,
            p6_frame=None,
            p2_follow_frame=None,
            confidence=0.0,
            detection_rate=0.1,
            angles=[],
        )

        assert result.p2_frame is None
        assert result.confidence == 0.0


class TestIntegration:
    """Integration tests for club position detection."""

    def test_realistic_swing_angles(self):
        """Test with realistic swing angle progression."""
        # Realistic golf swing angle pattern:
        # Address: ~45°
        # P2 (backswing parallel): ~80-85°
        # P4 (top): ~85-90° but may go past horizontal
        # Downswing: rapidly decreasing
        # Impact: ~0-15°
        # Follow-through: rapidly increasing
        # P7 (follow parallel): ~80-85°
        # Finish: ~45-60°

        angles = []

        # Address to P2 (frames 0-20)
        for i in range(20):
            angles.append(45 + i * 2)  # 45° to 83°

        # P2 to P4 (frames 20-30)
        for i in range(10):
            angles.append(83 + i * 0.5)  # 83° to 88°

        # P4 to impact (frames 30-45)
        for i in range(15):
            angles.append(88 - i * 6)  # 88° down to ~-2° (vertical)

        # Impact to P7 (frames 45-60)
        for i in range(15):
            angles.append(max(0, -2 + i * 6))  # Back up to ~88°

        # P7 to finish (frames 60-80)
        for i in range(20):
            angles.append(88 - i * 2)  # 88° down to ~48°

        # Test P2 detection
        p2 = find_p2_backswing(angles, p1_frame=0, p4_frame=30)
        assert p2 is not None
        # P2 found when angle first reaches ~65° (80-15 tolerance)
        # 65 = 45 + i*2 → i = 10
        assert 8 <= p2 <= 20

        # Test P6 detection
        p6, conf = find_p6_impact(angles, p4_frame=30, p9_frame=80)
        assert p6 is not None
        assert 40 <= p6 <= 50  # Should be around frame 45
        assert conf > 0.5

        # Test P7 detection
        p7 = find_p7_followthrough(angles, p6_frame=45, p9_frame=80)
        assert p7 is not None
        assert 55 <= p7 <= 65  # Should be around frame 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
