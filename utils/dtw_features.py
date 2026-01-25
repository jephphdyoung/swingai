"""
DTW feature extraction and normalization.

Converts raw pose landmarks into normalized, rotation-invariant features
suitable for Dynamic Time Warping alignment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.signal import savgol_filter


# Landmark indices in our 21-landmark set
# 0: nose, 1-2: shoulders, 3-4: elbows, 5-6: wrists,
# 7-8: pinky, 9-10: index, 11-12: thumb,
# 13-14: hips, 15-16: knees, 17-18: ankles, 19-20: feet
NOSE = 0
LEFT_SHOULDER = 1
RIGHT_SHOULDER = 2
LEFT_ELBOW = 3
RIGHT_ELBOW = 4
LEFT_WRIST = 5
RIGHT_WRIST = 6
LEFT_HIP = 13
RIGHT_HIP = 14
LEFT_KNEE = 15
RIGHT_KNEE = 16
LEFT_ANKLE = 17
RIGHT_ANKLE = 18


@dataclass
class NormalizedPose:
    """Normalized pose coordinates for a single frame."""
    landmarks: np.ndarray  # (n_landmarks, 3) pelvis-centered, normalized
    scale: float           # Body scale factor used
    rotation: float        # Shoulder rotation angle (radians)


@dataclass
class DTWFeatures:
    """Feature vector for DTW alignment."""
    # Normalized coordinates (flattened)
    coords: np.ndarray  # (n_landmarks * 3,)

    # Angle features
    spine_tilt: float        # Spine angle from vertical
    shoulder_tilt: float     # Shoulder line tilt
    hip_tilt: float          # Hip line tilt
    left_elbow_angle: float  # Left elbow flexion
    right_elbow_angle: float # Right elbow flexion
    left_knee_angle: float   # Left knee flexion
    right_knee_angle: float  # Right knee flexion

    # Velocity features
    wrist_speed: float       # Average wrist speed

    def to_vector(self) -> np.ndarray:
        """Convert to flat feature vector for DTW."""
        angles = np.array([
            self.spine_tilt,
            self.shoulder_tilt,
            self.hip_tilt,
            self.left_elbow_angle,
            self.right_elbow_angle,
            self.left_knee_angle,
            self.right_knee_angle,
            self.wrist_speed,
        ])
        return np.concatenate([self.coords, angles])


def compute_pelvis_center(landmarks: list) -> np.ndarray:
    """
    Compute pelvis center from hip landmarks.

    Args:
        landmarks: Single frame of landmarks [(x, y, z), ...]

    Returns:
        (3,) array of pelvis center coordinates
    """
    left_hip = np.array(landmarks[LEFT_HIP])
    right_hip = np.array(landmarks[RIGHT_HIP])
    return (left_hip + right_hip) / 2


def compute_body_scale(landmarks: list) -> float:
    """
    Compute body scale from shoulder-to-hip distance.

    Args:
        landmarks: Single frame of landmarks

    Returns:
        Scale factor (torso length)
    """
    left_shoulder = np.array(landmarks[LEFT_SHOULDER])
    right_shoulder = np.array(landmarks[RIGHT_SHOULDER])
    left_hip = np.array(landmarks[LEFT_HIP])
    right_hip = np.array(landmarks[RIGHT_HIP])

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    # Torso length (3D distance)
    torso_length = np.linalg.norm(shoulder_center - hip_center)

    return max(torso_length, 0.01)  # Avoid division by zero


def compute_shoulder_rotation(landmarks: list) -> float:
    """
    Compute shoulder rotation angle from horizontal.

    Args:
        landmarks: Single frame of landmarks

    Returns:
        Rotation angle in radians
    """
    left_shoulder = np.array(landmarks[LEFT_SHOULDER][:2])  # Just x, y
    right_shoulder = np.array(landmarks[RIGHT_SHOULDER][:2])

    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]

    return np.arctan2(dy, dx)


def normalize_pose(landmarks: list) -> NormalizedPose:
    """
    Normalize pose to pelvis-centered, scale-normalized coordinates.

    Args:
        landmarks: Single frame of landmarks [(x, y, z), ...]

    Returns:
        NormalizedPose with centered, scaled landmarks
    """
    # Get pelvis center
    pelvis = compute_pelvis_center(landmarks)

    # Get scale
    scale = compute_body_scale(landmarks)

    # Get rotation
    rotation = compute_shoulder_rotation(landmarks)

    # Center and scale landmarks
    normalized = []
    for lm in landmarks:
        centered = np.array(lm) - pelvis
        scaled = centered / scale
        normalized.append(scaled)

    return NormalizedPose(
        landmarks=np.array(normalized),
        scale=scale,
        rotation=rotation,
    )


def compute_angle_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute angle at p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: 3D points

    Returns:
        Angle in radians (0 to pi)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def compute_spine_tilt(landmarks: list) -> float:
    """
    Compute spine tilt from vertical.

    Args:
        landmarks: Single frame of landmarks

    Returns:
        Tilt angle in radians
    """
    # Spine direction: hip center to shoulder center
    left_shoulder = np.array(landmarks[LEFT_SHOULDER])
    right_shoulder = np.array(landmarks[RIGHT_SHOULDER])
    left_hip = np.array(landmarks[LEFT_HIP])
    right_hip = np.array(landmarks[RIGHT_HIP])

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    spine_vec = shoulder_center - hip_center
    vertical = np.array([0, -1, 0])  # Up in image coords (y decreases)

    cos_angle = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def compute_line_tilt(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute tilt of line from horizontal.

    Args:
        p1, p2: Endpoints of line

    Returns:
        Tilt angle in radians
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)


def extract_angles(landmarks: list) -> dict:
    """
    Extract all angle features from landmarks.

    Args:
        landmarks: Single frame of landmarks

    Returns:
        Dict of angle features
    """
    # Convert to numpy arrays
    left_shoulder = np.array(landmarks[LEFT_SHOULDER])
    right_shoulder = np.array(landmarks[RIGHT_SHOULDER])
    left_elbow = np.array(landmarks[LEFT_ELBOW])
    right_elbow = np.array(landmarks[RIGHT_ELBOW])
    left_wrist = np.array(landmarks[LEFT_WRIST])
    right_wrist = np.array(landmarks[RIGHT_WRIST])
    left_hip = np.array(landmarks[LEFT_HIP])
    right_hip = np.array(landmarks[RIGHT_HIP])
    left_knee = np.array(landmarks[LEFT_KNEE])
    right_knee = np.array(landmarks[RIGHT_KNEE])
    left_ankle = np.array(landmarks[LEFT_ANKLE])
    right_ankle = np.array(landmarks[RIGHT_ANKLE])

    return {
        "spine_tilt": compute_spine_tilt(landmarks),
        "shoulder_tilt": compute_line_tilt(left_shoulder, right_shoulder),
        "hip_tilt": compute_line_tilt(left_hip, right_hip),
        "left_elbow_angle": compute_angle_3points(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle": compute_angle_3points(right_shoulder, right_elbow, right_wrist),
        "left_knee_angle": compute_angle_3points(left_hip, left_knee, left_ankle),
        "right_knee_angle": compute_angle_3points(right_hip, right_knee, right_ankle),
    }


def smooth_landmarks(
    landmarks: list,
    window_length: int = 7,
    polyorder: int = 2,
) -> list:
    """
    Smooth landmarks using Savitzky-Golay filter.

    Args:
        landmarks: List of frames, each frame is list of (x, y, z) tuples
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order

    Returns:
        Smoothed landmarks in same format
    """
    if len(landmarks) < window_length:
        return landmarks

    n_frames = len(landmarks)
    n_landmarks = len(landmarks[0])

    # Convert to array (n_frames, n_landmarks, 3)
    arr = np.array([
        [(lm[0], lm[1], lm[2]) for lm in frame]
        for frame in landmarks
    ])

    # Apply filter to each coordinate
    smoothed = np.zeros_like(arr)
    for lm_idx in range(n_landmarks):
        for coord_idx in range(3):
            smoothed[:, lm_idx, coord_idx] = savgol_filter(
                arr[:, lm_idx, coord_idx],
                window_length=window_length,
                polyorder=polyorder,
            )

    # Convert back to list format
    return [
        [(smoothed[i, j, 0], smoothed[i, j, 1], smoothed[i, j, 2])
         for j in range(n_landmarks)]
        for i in range(n_frames)
    ]


def compute_wrist_velocity(
    landmarks: list,
    timestamps_ms: Optional[list[float]] = None,
    frame_idx: int = 0,
) -> float:
    """
    Compute wrist velocity at a specific frame.

    Args:
        landmarks: Full landmark sequence
        timestamps_ms: Per-frame timestamps
        frame_idx: Frame to compute velocity for

    Returns:
        Wrist speed (displacement/time)
    """
    if frame_idx == 0 or frame_idx >= len(landmarks):
        return 0.0

    # Get wrist positions
    prev_left = np.array(landmarks[frame_idx - 1][LEFT_WRIST])
    prev_right = np.array(landmarks[frame_idx - 1][RIGHT_WRIST])
    curr_left = np.array(landmarks[frame_idx][LEFT_WRIST])
    curr_right = np.array(landmarks[frame_idx][RIGHT_WRIST])

    # Average displacement
    left_disp = np.linalg.norm(curr_left - prev_left)
    right_disp = np.linalg.norm(curr_right - prev_right)
    avg_disp = (left_disp + right_disp) / 2

    # Time delta
    if timestamps_ms is not None and len(timestamps_ms) > frame_idx:
        dt = timestamps_ms[frame_idx] - timestamps_ms[frame_idx - 1]
        dt = max(dt, 1.0)  # Avoid division by zero
    else:
        dt = 16.67  # Assume 60fps

    return avg_disp / dt * 1000  # Units per second


def extract_dtw_features(
    landmarks: list,
    timestamps_ms: Optional[list[float]] = None,
    smooth: bool = True,
    smooth_window: int = 7,
) -> list[DTWFeatures]:
    """
    Extract DTW-ready features from landmark sequence.

    Args:
        landmarks: List of pose landmarks per frame
        timestamps_ms: Per-frame timestamps
        smooth: Apply Savitzky-Golay smoothing
        smooth_window: Smoothing window length

    Returns:
        List of DTWFeatures per frame
    """
    if smooth and len(landmarks) >= smooth_window:
        landmarks = smooth_landmarks(landmarks, smooth_window)

    features = []

    for i, frame_lm in enumerate(landmarks):
        # Normalize pose
        normalized = normalize_pose(frame_lm)

        # Extract angles
        angles = extract_angles(frame_lm)

        # Compute wrist velocity
        wrist_speed = compute_wrist_velocity(landmarks, timestamps_ms, i)

        # Create feature object
        feat = DTWFeatures(
            coords=normalized.landmarks.flatten(),
            spine_tilt=angles["spine_tilt"],
            shoulder_tilt=angles["shoulder_tilt"],
            hip_tilt=angles["hip_tilt"],
            left_elbow_angle=angles["left_elbow_angle"],
            right_elbow_angle=angles["right_elbow_angle"],
            left_knee_angle=angles["left_knee_angle"],
            right_knee_angle=angles["right_knee_angle"],
            wrist_speed=wrist_speed,
        )
        features.append(feat)

    return features


def features_to_matrix(features: list[DTWFeatures]) -> np.ndarray:
    """
    Convert list of DTWFeatures to matrix for DTW.

    Args:
        features: List of DTWFeatures

    Returns:
        (n_frames, n_features) matrix
    """
    if not features:
        return np.array([])

    vectors = [f.to_vector() for f in features]
    return np.vstack(vectors)


def normalize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Z-score normalize feature matrix.

    Args:
        matrix: (n_frames, n_features) matrix

    Returns:
        Normalized matrix
    """
    if matrix.size == 0:
        return matrix

    mean = np.mean(matrix, axis=0, keepdims=True)
    std = np.std(matrix, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

    return (matrix - mean) / std
