"""
DTW-based P-position alignment and label transfer.

Uses Dynamic Time Warping to align user swing to reference swing,
then transfers P-position labels with local refinement.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from .dtw_features import (
    extract_dtw_features,
    features_to_matrix,
    normalize_feature_matrix,
)


@dataclass
class AlignmentResult:
    """Result of DTW alignment."""
    path: list[tuple[int, int]]  # List of (user_frame, ref_frame) pairs
    distance: float              # Total DTW distance
    user_to_ref: dict[int, int]  # Mapping from user frame to ref frame
    ref_to_user: dict[int, int]  # Mapping from ref frame to user frame


@dataclass
class RefinedPositions:
    """P-positions with refinement info."""
    positions: dict[str, int]  # Position name to frame number
    confidences: dict[str, float]  # Confidence per position
    method: dict[str, str]  # Detection method used per position


def compute_dtw_alignment(
    user_landmarks: list,
    ref_landmarks: list,
    user_timestamps: Optional[list[float]] = None,
    ref_timestamps: Optional[list[float]] = None,
    normalize: bool = True,
) -> AlignmentResult:
    """
    Compute DTW alignment between user and reference swings.

    Args:
        user_landmarks: User pose landmarks
        ref_landmarks: Reference pose landmarks
        user_timestamps: User timestamps (optional)
        ref_timestamps: Reference timestamps (optional)
        normalize: Apply Z-score normalization to features

    Returns:
        AlignmentResult with path and mappings
    """
    # Extract features
    user_features = extract_dtw_features(user_landmarks, user_timestamps)
    ref_features = extract_dtw_features(ref_landmarks, ref_timestamps)

    # Convert to matrices
    user_matrix = features_to_matrix(user_features)
    ref_matrix = features_to_matrix(ref_features)

    # Normalize if requested
    if normalize:
        user_matrix = normalize_feature_matrix(user_matrix)
        ref_matrix = normalize_feature_matrix(ref_matrix)

    # Run FastDTW
    distance, path = fastdtw(user_matrix, ref_matrix, dist=euclidean)

    # Build mappings
    user_to_ref = {}
    ref_to_user = {}

    for user_idx, ref_idx in path:
        # For user_to_ref, take the first ref frame that maps
        if user_idx not in user_to_ref:
            user_to_ref[user_idx] = ref_idx

        # For ref_to_user, take the first user frame that maps
        if ref_idx not in ref_to_user:
            ref_to_user[ref_idx] = user_idx

    return AlignmentResult(
        path=path,
        distance=distance,
        user_to_ref=user_to_ref,
        ref_to_user=ref_to_user,
    )


def transfer_positions(
    ref_positions: dict[str, int],
    alignment: AlignmentResult,
) -> dict[str, int]:
    """
    Transfer P-position labels from reference to user via DTW path.

    Args:
        ref_positions: Reference P-positions (name -> frame)
        alignment: DTW alignment result

    Returns:
        User P-positions (name -> frame)
    """
    user_positions = {}

    for pos_name, ref_frame in ref_positions.items():
        if ref_frame in alignment.ref_to_user:
            user_positions[pos_name] = alignment.ref_to_user[ref_frame]
        else:
            # Find closest mapped frame
            mapped_frames = list(alignment.ref_to_user.keys())
            if mapped_frames:
                closest = min(mapped_frames, key=lambda f: abs(f - ref_frame))
                user_positions[pos_name] = alignment.ref_to_user[closest]

    return user_positions


def refine_p4_top_of_backswing(
    landmarks: list,
    initial_p4: int,
    search_window: int = 10,
) -> tuple[int, float]:
    """
    Refine P4 (top of backswing) using hand height peak detection.

    Args:
        landmarks: User landmarks
        initial_p4: Initial P4 estimate from DTW transfer
        search_window: Frames to search around initial estimate

    Returns:
        (refined_p4, confidence)
    """
    n = len(landmarks)

    # Search window
    start = max(0, initial_p4 - search_window)
    end = min(n, initial_p4 + search_window)

    # Get wrist heights (y coordinate, lower = higher in image)
    wrist_heights = []
    for i in range(start, end):
        left_wrist_y = landmarks[i][5][1]
        right_wrist_y = landmarks[i][6][1]
        avg_y = (left_wrist_y + right_wrist_y) / 2
        wrist_heights.append(avg_y)

    if not wrist_heights:
        return initial_p4, 0.5

    # Find minimum (highest point)
    min_idx = np.argmin(wrist_heights)
    refined_p4 = start + min_idx

    # Confidence based on how clear the peak is
    if len(wrist_heights) >= 3:
        peak_height = wrist_heights[min_idx]
        mean_height = np.mean(wrist_heights)
        confidence = min(1.0, (mean_height - peak_height) / 0.1 + 0.5)
    else:
        confidence = 0.5

    return refined_p4, confidence


def refine_p6_impact(
    landmarks: list,
    timestamps_ms: Optional[list[float]],
    initial_p6: int,
    search_window: int = 15,
) -> tuple[int, float]:
    """
    Refine P6 (impact) using velocity spike detection.

    Args:
        landmarks: User landmarks
        timestamps_ms: Timestamps
        initial_p6: Initial P6 estimate from DTW transfer
        search_window: Frames to search around initial estimate

    Returns:
        (refined_p6, confidence)
    """
    n = len(landmarks)

    # Search window
    start = max(0, initial_p6 - search_window)
    end = min(n, initial_p6 + search_window)

    # Compute wrist velocities
    velocities = []
    for i in range(start, end):
        if i == 0:
            velocities.append(0.0)
            continue

        # Wrist displacement
        prev_left = np.array(landmarks[i-1][5])
        prev_right = np.array(landmarks[i-1][6])
        curr_left = np.array(landmarks[i][5])
        curr_right = np.array(landmarks[i][6])

        left_disp = np.linalg.norm(curr_left - prev_left)
        right_disp = np.linalg.norm(curr_right - prev_right)
        avg_disp = (left_disp + right_disp) / 2

        # Time delta
        if timestamps_ms and len(timestamps_ms) > i:
            dt = max(1.0, timestamps_ms[i] - timestamps_ms[i-1])
        else:
            dt = 16.67

        velocities.append(avg_disp / dt * 1000)

    if not velocities:
        return initial_p6, 0.5

    # Find maximum velocity
    max_idx = np.argmax(velocities)
    refined_p6 = start + max_idx

    # Confidence based on velocity peak prominence
    if len(velocities) >= 3:
        peak_vel = velocities[max_idx]
        mean_vel = np.mean(velocities)
        confidence = min(1.0, peak_vel / (mean_vel + 0.01))
    else:
        confidence = 0.5

    return refined_p6, confidence


def refine_p7_followthrough(
    landmarks: list,
    timestamps_ms: Optional[list[float]],
    p6_frame: int,
    search_window: int = 20,
) -> tuple[int, float]:
    """
    Refine P7 (shaft parallel in follow-through) using velocity pattern.

    P7 is typically where there's a secondary speed spike or inflection
    in the follow-through.

    Args:
        landmarks: User landmarks
        timestamps_ms: Timestamps
        p6_frame: Impact frame
        search_window: Frames to search after impact

    Returns:
        (refined_p7, confidence)
    """
    n = len(landmarks)

    # Search window after impact
    start = p6_frame + 5  # At least 5 frames after impact
    end = min(n, p6_frame + search_window)

    if start >= end:
        return min(p6_frame + 10, n - 1), 0.3

    # Compute wrist velocities
    velocities = []
    for i in range(start, end):
        if i == 0:
            velocities.append(0.0)
            continue

        prev_left = np.array(landmarks[i-1][5])
        prev_right = np.array(landmarks[i-1][6])
        curr_left = np.array(landmarks[i][5])
        curr_right = np.array(landmarks[i][6])

        left_disp = np.linalg.norm(curr_left - prev_left)
        right_disp = np.linalg.norm(curr_right - prev_right)
        avg_disp = (left_disp + right_disp) / 2

        if timestamps_ms and len(timestamps_ms) > i:
            dt = max(1.0, timestamps_ms[i] - timestamps_ms[i-1])
        else:
            dt = 16.67

        velocities.append(avg_disp / dt * 1000)

    if not velocities:
        return start, 0.3

    # Look for local maximum (secondary peak)
    # If no clear peak, use 1/3 of the way through the window
    max_idx = np.argmax(velocities)
    default_idx = len(velocities) // 3

    # Use max if it's prominent, otherwise use default
    if len(velocities) >= 3:
        peak_vel = velocities[max_idx]
        mean_vel = np.mean(velocities)
        if peak_vel > mean_vel * 1.2:
            refined_p7 = start + max_idx
            confidence = 0.7
        else:
            refined_p7 = start + default_idx
            confidence = 0.4
    else:
        refined_p7 = start + max_idx
        confidence = 0.4

    return refined_p7, confidence


def refine_positions(
    landmarks: list,
    timestamps_ms: Optional[list[float]],
    transferred_positions: dict[str, int],
) -> RefinedPositions:
    """
    Apply local refinement to DTW-transferred positions.

    Args:
        landmarks: User pose landmarks
        timestamps_ms: User timestamps
        transferred_positions: Initial positions from DTW transfer

    Returns:
        RefinedPositions with refined frames and confidence
    """
    n = len(landmarks)
    refined = {}
    confidences = {}
    methods = {}

    # P1 (address) - use transferred, no refinement needed
    if "P1" in transferred_positions:
        refined["P1"] = transferred_positions["P1"]
        confidences["P1"] = 0.8
        methods["P1"] = "dtw_transfer"

    # P4 (top of backswing) - refine with hand height
    if "P4" in transferred_positions:
        p4, conf = refine_p4_top_of_backswing(
            landmarks,
            transferred_positions["P4"],
        )
        refined["P4"] = p4
        confidences["P4"] = conf
        methods["P4"] = "hand_height_peak"

    # P6 (impact) - refine with velocity spike
    if "P6" in transferred_positions:
        p6, conf = refine_p6_impact(
            landmarks,
            timestamps_ms,
            transferred_positions["P6"],
        )
        refined["P6"] = p6
        confidences["P6"] = conf
        methods["P6"] = "velocity_peak"

    # P7 (follow-through) - refine with velocity pattern
    p6_frame = refined.get("P6", transferred_positions.get("P6", n // 2))
    if "P7" in transferred_positions:
        p7, conf = refine_p7_followthrough(
            landmarks,
            timestamps_ms,
            p6_frame,
        )
        refined["P7"] = p7
        confidences["P7"] = conf
        methods["P7"] = "velocity_pattern"

    # Interpolate remaining positions
    p1 = refined.get("P1", 0)
    p4 = refined.get("P4", n // 3)
    p6 = refined.get("P6", n // 2)
    p9 = transferred_positions.get("P9", n - 1)

    # P2, P3 - distribute between P1 and P4
    backswing_len = p4 - p1
    if "P2" not in refined:
        refined["P2"] = p1 + backswing_len // 3
        confidences["P2"] = 0.6
        methods["P2"] = "interpolated"
    if "P3" not in refined:
        refined["P3"] = p1 + 2 * backswing_len // 3
        confidences["P3"] = 0.6
        methods["P3"] = "interpolated"

    # P5 - between P4 and P6
    if "P5" not in refined:
        refined["P5"] = p4 + (p6 - p4) // 2
        confidences["P5"] = 0.6
        methods["P5"] = "interpolated"

    # P7, P8, P9 - distribute between P6 and P9
    followthrough_len = p9 - p6
    if "P7" not in refined:
        refined["P7"] = p6 + followthrough_len // 3
        confidences["P7"] = 0.5
        methods["P7"] = "interpolated"
    if "P8" not in refined:
        refined["P8"] = p6 + 2 * followthrough_len // 3
        confidences["P8"] = 0.5
        methods["P8"] = "interpolated"
    if "P9" not in refined:
        refined["P9"] = p9
        confidences["P9"] = 0.7
        methods["P9"] = "dtw_transfer"

    # Ensure ordering
    positions_ordered = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    prev_frame = -1
    for pos in positions_ordered:
        if pos in refined:
            if refined[pos] <= prev_frame:
                refined[pos] = prev_frame + 1
            prev_frame = refined[pos]
            # Clamp to valid range
            refined[pos] = max(0, min(refined[pos], n - 1))

    return RefinedPositions(
        positions=refined,
        confidences=confidences,
        method=methods,
    )


def detect_p_positions_with_dtw(
    user_landmarks: list,
    ref_landmarks: list,
    ref_positions: dict[str, int],
    user_timestamps: Optional[list[float]] = None,
    ref_timestamps: Optional[list[float]] = None,
) -> RefinedPositions:
    """
    Detect P-positions using DTW alignment to reference + local refinement.

    This is the main entry point for DTW-based P-position detection.

    Args:
        user_landmarks: User pose landmarks
        ref_landmarks: Reference pose landmarks
        ref_positions: Known P-positions in reference (name -> frame)
        user_timestamps: User timestamps (optional)
        ref_timestamps: Reference timestamps (optional)

    Returns:
        RefinedPositions with detected positions, confidences, and methods
    """
    # Step 1: DTW alignment
    alignment = compute_dtw_alignment(
        user_landmarks,
        ref_landmarks,
        user_timestamps,
        ref_timestamps,
    )

    # Step 2: Transfer positions
    transferred = transfer_positions(ref_positions, alignment)

    # Step 3: Local refinement
    refined = refine_positions(user_landmarks, user_timestamps, transferred)

    return refined
