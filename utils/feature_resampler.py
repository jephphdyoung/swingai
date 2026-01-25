"""
Feature resampling utilities.

Resamples pose features to a fixed rate (default 60Hz) for consistent
comparison across videos with different frame rates.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import interpolate


@dataclass
class ResampledFeatures:
    """Result of feature resampling."""
    landmarks: list  # Resampled landmarks
    timestamps_ms: list[float]  # New uniform timestamps
    original_duration_ms: float  # Original sequence duration
    n_frames: int  # Number of resampled frames
    target_fps: float  # Target frame rate


def resample_landmarks(
    landmarks: list,
    timestamps_ms: list[float],
    target_fps: float = 60.0,
) -> ResampledFeatures:
    """
    Resample pose landmarks to a fixed frame rate.

    Uses linear interpolation to resample landmarks at uniform intervals.

    Args:
        landmarks: List of pose landmarks per frame (each frame is list of (x, y, z) tuples)
        timestamps_ms: Per-frame timestamps in milliseconds
        target_fps: Target frame rate (default 60Hz)

    Returns:
        ResampledFeatures with uniformly sampled landmarks
    """
    if not landmarks or not timestamps_ms:
        return ResampledFeatures(
            landmarks=[],
            timestamps_ms=[],
            original_duration_ms=0.0,
            n_frames=0,
            target_fps=target_fps,
        )

    n_frames = len(landmarks)
    n_landmarks = len(landmarks[0])

    # Convert to numpy for interpolation
    # Shape: (n_frames, n_landmarks, 3)
    lm_array = np.array([
        [(lm[0], lm[1], lm[2]) for lm in frame]
        for frame in landmarks
    ])

    # Original timestamps
    t_orig = np.array(timestamps_ms)
    duration_ms = t_orig[-1] - t_orig[0]

    # Target timestamps at uniform intervals
    interval_ms = 1000.0 / target_fps
    n_target = int(duration_ms / interval_ms) + 1
    t_target = np.linspace(t_orig[0], t_orig[-1], n_target)

    # Interpolate each landmark coordinate
    resampled = np.zeros((n_target, n_landmarks, 3))

    for lm_idx in range(n_landmarks):
        for coord_idx in range(3):  # x, y, z
            values = lm_array[:, lm_idx, coord_idx]

            # Create interpolation function
            interp_func = interpolate.interp1d(
                t_orig,
                values,
                kind='linear',
                fill_value='extrapolate',
            )

            # Interpolate at target timestamps
            resampled[:, lm_idx, coord_idx] = interp_func(t_target)

    # Convert back to list format
    resampled_landmarks = [
        [(resampled[i, j, 0], resampled[i, j, 1], resampled[i, j, 2])
         for j in range(n_landmarks)]
        for i in range(n_target)
    ]

    # Re-zero timestamps
    resampled_timestamps = [t - t_target[0] for t in t_target]

    return ResampledFeatures(
        landmarks=resampled_landmarks,
        timestamps_ms=resampled_timestamps,
        original_duration_ms=duration_ms,
        n_frames=n_target,
        target_fps=target_fps,
    )


def resample_to_length(
    landmarks: list,
    target_length: int,
) -> list:
    """
    Resample landmarks to a specific number of frames.

    Useful for making two sequences the same length for comparison.

    Args:
        landmarks: List of pose landmarks per frame
        target_length: Desired number of frames

    Returns:
        Resampled landmarks with exactly target_length frames
    """
    if not landmarks:
        return []

    n_frames = len(landmarks)
    if n_frames == target_length:
        return landmarks

    n_landmarks = len(landmarks[0])

    # Convert to numpy
    lm_array = np.array([
        [(lm[0], lm[1], lm[2]) for lm in frame]
        for frame in landmarks
    ])

    # Original and target indices
    t_orig = np.linspace(0, 1, n_frames)
    t_target = np.linspace(0, 1, target_length)

    # Interpolate
    resampled = np.zeros((target_length, n_landmarks, 3))

    for lm_idx in range(n_landmarks):
        for coord_idx in range(3):
            values = lm_array[:, lm_idx, coord_idx]
            interp_func = interpolate.interp1d(
                t_orig, values, kind='linear', fill_value='extrapolate'
            )
            resampled[:, lm_idx, coord_idx] = interp_func(t_target)

    # Convert back to list format
    return [
        [(resampled[i, j, 0], resampled[i, j, 1], resampled[i, j, 2])
         for j in range(n_landmarks)]
        for i in range(target_length)
    ]


def compute_uniform_timestamps(
    duration_ms: float,
    fps: float = 60.0,
) -> list[float]:
    """
    Generate uniform timestamps for a given duration and fps.

    Args:
        duration_ms: Total duration in milliseconds
        fps: Frame rate

    Returns:
        List of timestamps at uniform intervals
    """
    interval_ms = 1000.0 / fps
    n_frames = int(duration_ms / interval_ms) + 1
    return [i * interval_ms for i in range(n_frames)]


def estimate_fps_from_timestamps(timestamps_ms: list[float]) -> float:
    """
    Estimate frame rate from timestamps.

    Args:
        timestamps_ms: List of timestamps

    Returns:
        Estimated FPS
    """
    if len(timestamps_ms) < 2:
        return 60.0

    deltas = [timestamps_ms[i] - timestamps_ms[i-1] for i in range(1, len(timestamps_ms))]
    median_delta = np.median(deltas)

    if median_delta <= 0:
        return 60.0

    return 1000.0 / median_delta


def align_to_common_rate(
    landmarks1: list,
    timestamps1: list[float],
    landmarks2: list,
    timestamps2: list[float],
    target_fps: float = 60.0,
) -> tuple[ResampledFeatures, ResampledFeatures]:
    """
    Resample two landmark sequences to a common frame rate.

    Args:
        landmarks1: First landmark sequence
        timestamps1: First timestamp sequence
        landmarks2: Second landmark sequence
        timestamps2: Second timestamp sequence
        target_fps: Target frame rate

    Returns:
        Tuple of (resampled1, resampled2)
    """
    resampled1 = resample_landmarks(landmarks1, timestamps1, target_fps)
    resampled2 = resample_landmarks(landmarks2, timestamps2, target_fps)

    return resampled1, resampled2
