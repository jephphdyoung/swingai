#!/usr/bin/env python3
"""
Unit tests for the P-position evaluation harness.
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

from scripts.eval_positions import (
    EvaluationReport,
    PositionError,
    VideoMetadata,
    VideoResult,
    frame_to_timestamp,
    load_ground_truth,
    normalize_ground_truth_positions,
    timestamp_to_frame,
)


class TestTimestampConversions:
    """Tests for frame/timestamp conversion functions."""

    def test_timestamp_to_frame_30fps(self):
        assert timestamp_to_frame(0, 30) == 0
        assert timestamp_to_frame(1000, 30) == 30
        assert timestamp_to_frame(500, 30) == 15
        assert timestamp_to_frame(33.33, 30) == 1

    def test_timestamp_to_frame_60fps(self):
        assert timestamp_to_frame(0, 60) == 0
        assert timestamp_to_frame(1000, 60) == 60
        assert timestamp_to_frame(500, 60) == 30
        assert timestamp_to_frame(16.67, 60) == 1

    def test_timestamp_to_frame_120fps(self):
        assert timestamp_to_frame(0, 120) == 0
        assert timestamp_to_frame(1000, 120) == 120
        assert timestamp_to_frame(8.33, 120) == 1

    def test_frame_to_timestamp_30fps(self):
        assert frame_to_timestamp(0, 30) == 0
        assert frame_to_timestamp(30, 30) == 1000
        assert frame_to_timestamp(15, 30) == 500
        assert abs(frame_to_timestamp(1, 30) - 33.33) < 0.1

    def test_frame_to_timestamp_60fps(self):
        assert frame_to_timestamp(0, 60) == 0
        assert frame_to_timestamp(60, 60) == 1000
        assert frame_to_timestamp(30, 60) == 500

    def test_roundtrip_conversion(self):
        """Verify frame -> timestamp -> frame roundtrip."""
        for fps in [30, 60, 120, 240]:
            for frame in [0, 10, 50, 100]:
                ts = frame_to_timestamp(frame, fps)
                recovered = timestamp_to_frame(ts, fps)
                assert recovered == frame, f"Roundtrip failed for frame={frame}, fps={fps}"


class TestNormalizeGroundTruth:
    """Tests for ground truth normalization."""

    def test_frame_based_positions(self):
        positions = {
            "P1": {"frame": 10},
            "P4": {"frame": 50},
            "P6": {"frame": 70},
        }
        result = normalize_ground_truth_positions(positions, 60.0)
        assert result == {"P1": 10, "P4": 50, "P6": 70}

    def test_timestamp_based_positions(self):
        positions = {
            "P1": {"timestamp_ms": 500},
            "P4": {"timestamp_ms": 1000},
        }
        result = normalize_ground_truth_positions(positions, 60.0)
        assert result == {"P1": 30, "P4": 60}

    def test_direct_frame_numbers(self):
        positions = {
            "P1": 10,
            "P4": 50,
        }
        result = normalize_ground_truth_positions(positions, 60.0)
        assert result == {"P1": 10, "P4": 50}

    def test_mixed_formats_raises(self):
        """Each position should be consistent, but mixed formats per position work."""
        positions = {
            "P1": {"frame": 10},
            "P4": {"timestamp_ms": 1000},
        }
        result = normalize_ground_truth_positions(positions, 60.0)
        assert result["P1"] == 10
        assert result["P4"] == 60

    def test_invalid_position_raises(self):
        positions = {
            "P1": {"invalid_key": 10},
        }
        with pytest.raises(ValueError, match="must have 'frame' or 'timestamp_ms'"):
            normalize_ground_truth_positions(positions, 60.0)


class TestLoadGroundTruth:
    """Tests for loading ground truth JSON."""

    def test_load_valid_json(self):
        data = {
            "videos": [
                {
                    "path": "test.mp4",
                    "positions": {"P1": {"frame": 10}}
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            result = load_ground_truth(f.name)
            assert "videos" in result
            assert len(result["videos"]) == 1

        os.unlink(f.name)

    def test_missing_videos_key_raises(self):
        data = {"positions": {}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            with pytest.raises(ValueError, match="must have 'videos' array"):
                load_ground_truth(f.name)

        os.unlink(f.name)


class TestVideoResult:
    """Tests for VideoResult dataclass."""

    def test_success_result(self):
        metadata = VideoMetadata(fps=60.0, frame_count=200, duration_ms=3333.0, width=1920, height=1080)
        errors = [
            PositionError("P1", 12, 10, 2, 33.3),
            PositionError("P4", 48, 50, -2, -33.3),
        ]
        result = VideoResult(
            video_path="test.mp4",
            metadata=metadata,
            errors=errors,
            success=True,
        )
        assert result.success
        assert len(result.errors) == 2

    def test_failure_result(self):
        result = VideoResult(
            video_path="missing.mp4",
            metadata=VideoMetadata(0, 0, 0, 0, 0),
            success=False,
            error_message="File not found",
        )
        assert not result.success
        assert "not found" in result.error_message


class TestEvaluationReport:
    """Tests for EvaluationReport aggregation."""

    def create_sample_report(self) -> EvaluationReport:
        """Create a sample report with two videos."""
        report = EvaluationReport()

        # Video 1: some errors
        metadata1 = VideoMetadata(fps=60.0, frame_count=200, duration_ms=3333.0, width=1920, height=1080)
        errors1 = [
            PositionError("P1", 12, 10, 2, 33.3),
            PositionError("P4", 52, 50, 2, 33.3),
            PositionError("P6", 68, 70, -2, -33.3),
        ]
        report.video_results.append(VideoResult(
            video_path="video1.mp4",
            metadata=metadata1,
            errors=errors1,
            success=True,
        ))

        # Video 2: different errors
        metadata2 = VideoMetadata(fps=30.0, frame_count=100, duration_ms=3333.0, width=1280, height=720)
        errors2 = [
            PositionError("P1", 8, 10, -2, -66.7),
            PositionError("P4", 28, 25, 3, 100.0),
            PositionError("P6", 35, 35, 0, 0.0),
        ]
        report.video_results.append(VideoResult(
            video_path="video2.mp4",
            metadata=metadata2,
            errors=errors2,
            success=True,
        ))

        return report

    def test_mae_by_position(self):
        report = self.create_sample_report()
        mae_by_pos = report.compute_mae_by_position()

        # P1: errors are +2 and -2 frames -> MAE = 2.0
        assert mae_by_pos["P1"]["mae_frames"] == 2.0
        assert mae_by_pos["P1"]["n_samples"] == 2

        # P4: errors are +2 and +3 frames -> MAE = 2.5
        assert mae_by_pos["P4"]["mae_frames"] == 2.5

        # P6: errors are -2 and 0 frames -> MAE = 1.0
        assert mae_by_pos["P6"]["mae_frames"] == 1.0

    def test_overall_mae(self):
        report = self.create_sample_report()
        overall = report.compute_overall_mae()

        # All absolute errors: 2, 2, 2, 2, 3, 0 -> mean = 11/6 ≈ 1.83
        expected_mae = (2 + 2 + 2 + 2 + 3 + 0) / 6
        assert abs(overall["mae_frames"] - expected_mae) < 0.01
        assert overall["n_samples"] == 6

    def test_empty_report(self):
        report = EvaluationReport()
        mae_by_pos = report.compute_mae_by_position()
        overall = report.compute_overall_mae()

        assert mae_by_pos == {}
        assert overall["mae_frames"] == 0.0
        assert overall["n_samples"] == 0

    def test_failed_videos_excluded(self):
        report = EvaluationReport()

        # Add a failed video
        report.video_results.append(VideoResult(
            video_path="failed.mp4",
            metadata=VideoMetadata(0, 0, 0, 0, 0),
            success=False,
            error_message="Test failure",
        ))

        # Add a successful video
        metadata = VideoMetadata(fps=60.0, frame_count=200, duration_ms=3333.0, width=1920, height=1080)
        errors = [PositionError("P1", 12, 10, 2, 33.3)]
        report.video_results.append(VideoResult(
            video_path="success.mp4",
            metadata=metadata,
            errors=errors,
            success=True,
        ))

        overall = report.compute_overall_mae()
        assert overall["n_samples"] == 1  # Only the successful video


class TestPositionError:
    """Tests for PositionError calculations."""

    def test_positive_error_late(self):
        """Predicted frame > ground truth = late detection."""
        err = PositionError("P1", predicted_frame=15, ground_truth_frame=10,
                           error_frames=5, error_ms=166.7)
        assert err.error_frames == 5  # Late by 5 frames

    def test_negative_error_early(self):
        """Predicted frame < ground truth = early detection."""
        err = PositionError("P1", predicted_frame=8, ground_truth_frame=10,
                           error_frames=-2, error_ms=-66.7)
        assert err.error_frames == -2  # Early by 2 frames

    def test_zero_error(self):
        """Perfect detection."""
        err = PositionError("P1", predicted_frame=10, ground_truth_frame=10,
                           error_frames=0, error_ms=0.0)
        assert err.error_frames == 0


class TestVideoMetadata:
    """Tests for VideoMetadata."""

    def test_metadata_creation(self):
        metadata = VideoMetadata(
            fps=60.0,
            frame_count=360,
            duration_ms=6000.0,
            width=1920,
            height=1080,
        )
        assert metadata.fps == 60.0
        assert metadata.frame_count == 360
        assert metadata.duration_ms == 6000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
