#!/usr/bin/env python3
"""
Unit tests for DTW feature extraction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dtw_features import (
    DTWFeatures,
    NormalizedPose,
    compute_angle_3points,
    compute_body_scale,
    compute_line_tilt,
    compute_pelvis_center,
    compute_shoulder_rotation,
    compute_spine_tilt,
    compute_wrist_velocity,
    extract_angles,
    extract_dtw_features,
    features_to_matrix,
    normalize_feature_matrix,
    normalize_pose,
    smooth_landmarks,
)


def create_mock_landmarks(
    wrist_pos=(0.5, 0.5),
    shoulder_spread=0.2,
    hip_spread=0.15,
) -> list:
    """
    Create a realistic mock frame of 21 landmarks.

    Indices: 0=nose, 1-2=shoulders, 3-4=elbows, 5-6=wrists,
             7-12=hands, 13-14=hips, 15-16=knees, 17-18=ankles, 19-20=feet
    """
    wx, wy = wrist_pos

    # Head/torso
    nose = (0.5, 0.3, 0.0)
    left_shoulder = (0.5 - shoulder_spread/2, 0.4, 0.0)
    right_shoulder = (0.5 + shoulder_spread/2, 0.4, 0.0)

    # Arms
    left_elbow = (0.5 - shoulder_spread/2 - 0.05, 0.5, 0.0)
    right_elbow = (0.5 + shoulder_spread/2 + 0.05, 0.5, 0.0)
    left_wrist = (wx - 0.1, wy, 0.0)
    right_wrist = (wx + 0.1, wy, 0.0)

    # Hands (near wrists)
    left_pinky = (wx - 0.12, wy + 0.02, 0.0)
    right_pinky = (wx + 0.12, wy + 0.02, 0.0)
    left_index = (wx - 0.11, wy + 0.01, 0.0)
    right_index = (wx + 0.11, wy + 0.01, 0.0)
    left_thumb = (wx - 0.09, wy + 0.01, 0.0)
    right_thumb = (wx + 0.09, wy + 0.01, 0.0)

    # Lower body
    left_hip = (0.5 - hip_spread/2, 0.6, 0.0)
    right_hip = (0.5 + hip_spread/2, 0.6, 0.0)
    left_knee = (0.5 - hip_spread/2, 0.75, 0.0)
    right_knee = (0.5 + hip_spread/2, 0.75, 0.0)
    left_ankle = (0.5 - hip_spread/2, 0.9, 0.0)
    right_ankle = (0.5 + hip_spread/2, 0.9, 0.0)
    left_foot = (0.5 - hip_spread/2, 0.92, 0.0)
    right_foot = (0.5 + hip_spread/2, 0.92, 0.0)

    return [
        nose,           # 0
        left_shoulder,  # 1
        right_shoulder, # 2
        left_elbow,     # 3
        right_elbow,    # 4
        left_wrist,     # 5
        right_wrist,    # 6
        left_pinky,     # 7
        right_pinky,    # 8
        left_index,     # 9
        right_index,    # 10
        left_thumb,     # 11
        right_thumb,    # 12
        left_hip,       # 13
        right_hip,      # 14
        left_knee,      # 15
        right_knee,     # 16
        left_ankle,     # 17
        right_ankle,    # 18
        left_foot,      # 19
        right_foot,     # 20
    ]


class TestComputePelvisCenter:
    """Tests for pelvis center computation."""

    def test_centered_pose(self):
        landmarks = create_mock_landmarks()
        pelvis = compute_pelvis_center(landmarks)

        # Hip center should be at x=0.5
        assert abs(pelvis[0] - 0.5) < 0.01
        # Y should be around 0.6
        assert abs(pelvis[1] - 0.6) < 0.01

    def test_shifted_pose(self):
        # Shift whole pose to the right
        landmarks = create_mock_landmarks()
        for i in range(len(landmarks)):
            x, y, z = landmarks[i]
            landmarks[i] = (x + 0.2, y, z)

        pelvis = compute_pelvis_center(landmarks)
        assert abs(pelvis[0] - 0.7) < 0.01


class TestComputeBodyScale:
    """Tests for body scale computation."""

    def test_normal_scale(self):
        landmarks = create_mock_landmarks()
        scale = compute_body_scale(landmarks)

        # Torso length should be shoulder-to-hip distance
        # Shoulder at y=0.4, hip at y=0.6, so ~0.2
        assert 0.15 < scale < 0.25

    def test_larger_body(self):
        # Double the vertical spread
        landmarks = create_mock_landmarks()
        landmarks[1] = (landmarks[1][0], 0.2, 0.0)  # Left shoulder higher
        landmarks[2] = (landmarks[2][0], 0.2, 0.0)  # Right shoulder higher

        scale = compute_body_scale(landmarks)
        assert scale > 0.3  # Larger torso


class TestComputeShoulderRotation:
    """Tests for shoulder rotation computation."""

    def test_horizontal_shoulders(self):
        landmarks = create_mock_landmarks()
        rotation = compute_shoulder_rotation(landmarks)

        # Level shoulders = ~0 rotation
        assert abs(rotation) < 0.1

    def test_tilted_shoulders(self):
        landmarks = create_mock_landmarks()
        # Tilt right shoulder down
        landmarks[2] = (landmarks[2][0], landmarks[2][1] + 0.1, 0.0)

        rotation = compute_shoulder_rotation(landmarks)
        assert rotation > 0.1  # Positive rotation


class TestNormalizePose:
    """Tests for pose normalization."""

    def test_centered_at_pelvis(self):
        landmarks = create_mock_landmarks()
        normalized = normalize_pose(landmarks)

        # Hip center should be at origin
        left_hip = normalized.landmarks[13]
        right_hip = normalized.landmarks[14]
        hip_center = (left_hip + right_hip) / 2

        assert abs(hip_center[0]) < 0.01
        assert abs(hip_center[1]) < 0.01

    def test_scale_normalized(self):
        landmarks = create_mock_landmarks()
        normalized = normalize_pose(landmarks)

        # After normalization, torso should be ~1 unit
        left_shoulder = normalized.landmarks[1]
        right_shoulder = normalized.landmarks[2]
        left_hip = normalized.landmarks[13]
        right_hip = normalized.landmarks[14]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        assert abs(torso_length - 1.0) < 0.1


class TestComputeAngle3Points:
    """Tests for angle computation."""

    def test_straight_line(self):
        # 180 degree angle (straight line)
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 0, 0])
        p3 = np.array([2, 0, 0])

        angle = compute_angle_3points(p1, p2, p3)
        assert abs(angle - np.pi) < 0.01

    def test_right_angle(self):
        # 90 degree angle
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 0, 0])
        p3 = np.array([1, 1, 0])

        angle = compute_angle_3points(p1, p2, p3)
        assert abs(angle - np.pi/2) < 0.01

    def test_acute_angle(self):
        # 60 degree angle at p2
        # v1 = p1 - p2, v2 = p3 - p2
        # For 60°, need cos(60°) = v1·v2 / (|v1||v2|) = 0.5
        p1 = np.array([2, 0, 0])  # v1 = (1, 0, 0)
        p2 = np.array([1, 0, 0])
        p3 = np.array([1.5, np.sqrt(3)/2, 0])  # v2 = (0.5, 0.866, 0), v1·v2 = 0.5

        angle = compute_angle_3points(p1, p2, p3)
        assert abs(angle - np.pi/3) < 0.1


class TestExtractAngles:
    """Tests for angle extraction."""

    def test_extracts_all_angles(self):
        landmarks = create_mock_landmarks()
        angles = extract_angles(landmarks)

        assert "spine_tilt" in angles
        assert "shoulder_tilt" in angles
        assert "hip_tilt" in angles
        assert "left_elbow_angle" in angles
        assert "right_elbow_angle" in angles
        assert "left_knee_angle" in angles
        assert "right_knee_angle" in angles

    def test_angles_in_valid_range(self):
        landmarks = create_mock_landmarks()
        angles = extract_angles(landmarks)

        for name, angle in angles.items():
            # All angles should be between 0 and pi
            assert 0 <= angle <= np.pi, f"{name} out of range: {angle}"


class TestSmoothLandmarks:
    """Tests for landmark smoothing."""

    def test_reduces_noise(self):
        # Create noisy sequence
        n_frames = 50
        landmarks = []
        for i in range(n_frames):
            noise = np.random.normal(0, 0.01)
            lm = create_mock_landmarks(wrist_pos=(0.5 + noise, 0.5 + noise))
            landmarks.append(lm)

        smoothed = smooth_landmarks(landmarks, window_length=7)

        # Compute variance of wrist position
        orig_wrist_x = [f[5][0] for f in landmarks]
        smooth_wrist_x = [f[5][0] for f in smoothed]

        assert np.var(smooth_wrist_x) < np.var(orig_wrist_x)

    def test_short_sequence(self):
        landmarks = [create_mock_landmarks()] * 3
        smoothed = smooth_landmarks(landmarks, window_length=7)

        # Should return original if too short
        assert len(smoothed) == 3

    def test_preserves_shape(self):
        landmarks = [create_mock_landmarks()] * 20
        smoothed = smooth_landmarks(landmarks, window_length=5)

        assert len(smoothed) == len(landmarks)
        assert len(smoothed[0]) == len(landmarks[0])


class TestComputeWristVelocity:
    """Tests for wrist velocity computation."""

    def test_stationary(self):
        landmarks = [create_mock_landmarks()] * 10
        velocity = compute_wrist_velocity(landmarks, frame_idx=5)

        assert velocity == 0.0

    def test_moving(self):
        # Create sequence with wrist moving
        landmarks = []
        for i in range(10):
            x = 0.5 + i * 0.01  # Moving right
            lm = create_mock_landmarks(wrist_pos=(x, 0.5))
            landmarks.append(lm)

        timestamps = [i * 16.67 for i in range(10)]
        velocity = compute_wrist_velocity(landmarks, timestamps, frame_idx=5)

        assert velocity > 0.0

    def test_first_frame(self):
        landmarks = [create_mock_landmarks()] * 10
        velocity = compute_wrist_velocity(landmarks, frame_idx=0)

        assert velocity == 0.0


class TestExtractDTWFeatures:
    """Tests for full feature extraction."""

    def test_extracts_features(self):
        landmarks = [create_mock_landmarks()] * 20
        features = extract_dtw_features(landmarks)

        assert len(features) == 20
        assert all(isinstance(f, DTWFeatures) for f in features)

    def test_with_timestamps(self):
        landmarks = [create_mock_landmarks()] * 20
        timestamps = [i * 16.67 for i in range(20)]

        features = extract_dtw_features(landmarks, timestamps)

        assert len(features) == 20

    def test_to_vector(self):
        landmarks = [create_mock_landmarks()]
        features = extract_dtw_features(landmarks, smooth=False)

        vector = features[0].to_vector()

        # 21 landmarks * 3 coords + 8 additional features
        expected_len = 21 * 3 + 8
        assert len(vector) == expected_len


class TestFeaturesToMatrix:
    """Tests for matrix conversion."""

    def test_empty(self):
        matrix = features_to_matrix([])
        assert matrix.size == 0

    def test_shape(self):
        landmarks = [create_mock_landmarks()] * 10
        features = extract_dtw_features(landmarks, smooth=False)
        matrix = features_to_matrix(features)

        assert matrix.shape[0] == 10
        assert matrix.shape[1] == 21 * 3 + 8


class TestNormalizeFeatureMatrix:
    """Tests for feature normalization."""

    def test_zero_mean(self):
        landmarks = []
        for i in range(50):
            lm = create_mock_landmarks(wrist_pos=(0.4 + i * 0.01, 0.5))
            landmarks.append(lm)

        features = extract_dtw_features(landmarks, smooth=False)
        matrix = features_to_matrix(features)
        normalized = normalize_feature_matrix(matrix)

        # Mean should be ~0
        mean = np.mean(normalized, axis=0)
        assert np.all(np.abs(mean) < 0.01)

    def test_unit_variance(self):
        landmarks = []
        for i in range(50):
            lm = create_mock_landmarks(wrist_pos=(0.4 + i * 0.01, 0.5))
            landmarks.append(lm)

        features = extract_dtw_features(landmarks, smooth=False)
        matrix = features_to_matrix(features)
        normalized = normalize_feature_matrix(matrix)

        # Std should be ~1 for varying features
        std = np.std(normalized, axis=0)
        # At least some features should have std close to 1
        assert np.any(np.abs(std - 1.0) < 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
