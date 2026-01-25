"""
Swing detection and waggle trimming utilities.

Detects the main golf swing in a video and trims waggle/practice swings.
Uses wrist speed with hysteresis for robust start-of-takeaway detection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SwingWindow:
    """Detected swing window within a video."""
    start_frame: int      # First frame of detected swing (P1 candidate)
    end_frame: int        # Last frame of swing
    takeaway_frame: int   # Frame where takeaway motion begins
    impact_frame: int     # Estimated impact frame
    confidence: float     # Detection confidence (0-1)


@dataclass
class SwingDetectionParams:
    """Parameters for swing detection."""
    # Baseline estimation
    baseline_duration_ms: float = 1000.0  # First N ms to estimate baseline motion

    # Thresholds (in MAD units)
    start_threshold_mad: float = 3.0  # MAD multiplier for motion start
    keep_threshold_mad: float = 1.5   # MAD multiplier to stay in motion

    # Sustained motion requirements
    min_motion_duration_ms: float = 150.0  # Minimum sustained motion duration
    max_motion_duration_ms: float = 3000.0  # Maximum swing duration

    # Pre-window for finding stable P1
    pre_window_ms: float = 500.0  # Look back this far for stable address

    # Motion burst detection
    min_burst_frames: int = 5  # Minimum frames in a motion burst
    burst_gap_frames: int = 10  # Gap between bursts to consider separate


def compute_wrist_speed(
    landmarks: list,
    timestamps_ms: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Compute wrist speed from pose landmarks.

    Args:
        landmarks: List of pose landmarks per frame
        timestamps_ms: Optional per-frame timestamps

    Returns:
        Array of wrist speeds (units per second if timestamps provided)
    """
    n = len(landmarks)
    if n < 2:
        return np.zeros(1)

    # Extract wrist positions (indices 5, 6 in our 21-landmark set)
    left_wrist = np.array([[frame[5][0], frame[5][1]] for frame in landmarks])
    right_wrist = np.array([[frame[6][0], frame[6][1]] for frame in landmarks])
    avg_wrist = (left_wrist + right_wrist) / 2

    # Compute time deltas
    if timestamps_ms is not None and len(timestamps_ms) == n:
        dt = np.zeros(n)
        for i in range(1, n):
            dt[i] = max(1.0, timestamps_ms[i] - timestamps_ms[i-1])  # ms
    else:
        dt = np.ones(n) * 16.67  # Assume 60fps

    # Compute speeds
    speeds = np.zeros(n)
    for i in range(1, n):
        displacement = np.linalg.norm(avg_wrist[i] - avg_wrist[i-1])
        speeds[i] = displacement / dt[i] * 1000  # Units per second

    return speeds


def estimate_baseline(
    speeds: np.ndarray,
    timestamps_ms: Optional[list[float]] = None,
    duration_ms: float = 1000.0,
) -> tuple[float, float]:
    """
    Estimate baseline motion statistics from initial frames.

    Args:
        speeds: Array of speeds
        timestamps_ms: Per-frame timestamps
        duration_ms: Duration to use for baseline estimation

    Returns:
        (median, mad) - median and Median Absolute Deviation
    """
    if timestamps_ms is not None:
        # Find frames within duration
        end_idx = 1
        for i, ts in enumerate(timestamps_ms):
            if ts <= duration_ms:
                end_idx = i + 1
            else:
                break
    else:
        # Assume 60fps
        end_idx = int(duration_ms / 16.67)

    end_idx = max(10, min(end_idx, len(speeds) // 3))  # At least 10 frames, at most 1/3

    baseline_speeds = speeds[:end_idx]
    median = np.median(baseline_speeds)
    mad = np.median(np.abs(baseline_speeds - median))

    # Ensure MAD is not zero
    mad = max(mad, 0.001)

    return median, mad


def detect_motion_bursts(
    speeds: np.ndarray,
    median: float,
    mad: float,
    params: SwingDetectionParams,
    timestamps_ms: Optional[list[float]] = None,
) -> list[tuple[int, int, float]]:
    """
    Detect bursts of motion above threshold.

    Args:
        speeds: Array of speeds
        median: Baseline median
        mad: Baseline MAD
        params: Detection parameters
        timestamps_ms: Per-frame timestamps

    Returns:
        List of (start_frame, end_frame, peak_speed) tuples
    """
    n = len(speeds)
    start_threshold = median + params.start_threshold_mad * mad
    keep_threshold = median + params.keep_threshold_mad * mad

    bursts = []
    in_burst = False
    burst_start = 0
    peak_speed = 0.0

    for i in range(n):
        if not in_burst:
            # Look for motion start
            if speeds[i] > start_threshold:
                in_burst = True
                burst_start = i
                peak_speed = speeds[i]
        else:
            # Track peak speed
            peak_speed = max(peak_speed, speeds[i])

            # Check if motion ended (with hysteresis)
            if speeds[i] < keep_threshold:
                # Check if burst is long enough
                burst_len = i - burst_start
                if burst_len >= params.min_burst_frames:
                    bursts.append((burst_start, i, peak_speed))

                in_burst = False
                peak_speed = 0.0

    # Handle burst that goes to end
    if in_burst:
        burst_len = n - burst_start
        if burst_len >= params.min_burst_frames:
            bursts.append((burst_start, n - 1, peak_speed))

    return bursts


def find_main_swing_burst(
    bursts: list[tuple[int, int, float]],
    timestamps_ms: Optional[list[float]] = None,
    params: Optional[SwingDetectionParams] = None,
) -> Optional[tuple[int, int, float]]:
    """
    Find the main swing burst from detected motion bursts.

    Heuristics:
    - Prefer bursts with duration in expected swing range (0.5-3s)
    - Prefer bursts with higher peak speed
    - The main swing is usually the largest sustained motion

    Args:
        bursts: List of (start, end, peak_speed) tuples
        timestamps_ms: Per-frame timestamps
        params: Detection parameters

    Returns:
        (start, end, peak_speed) of main swing, or None
    """
    if not bursts:
        return None

    if params is None:
        params = SwingDetectionParams()

    # Score each burst
    scored_bursts = []
    for start, end, peak_speed in bursts:
        # Duration score
        if timestamps_ms is not None:
            duration_ms = timestamps_ms[end] - timestamps_ms[start]
        else:
            duration_ms = (end - start) * 16.67

        # Ideal duration is 500-2000ms for a swing
        if 500 <= duration_ms <= 2000:
            duration_score = 1.0
        elif 300 <= duration_ms <= 3000:
            duration_score = 0.7
        else:
            duration_score = 0.3

        # Speed score (normalized)
        speed_score = min(peak_speed / 0.1, 1.0)  # 0.1 units/s is high speed

        # Length score (longer is better, up to a point)
        length_score = min((end - start) / 100, 1.0)

        # Combined score
        score = duration_score * 0.4 + speed_score * 0.4 + length_score * 0.2
        scored_bursts.append((score, start, end, peak_speed))

    # Return highest scored burst
    scored_bursts.sort(reverse=True)
    _, start, end, peak_speed = scored_bursts[0]
    return (start, end, peak_speed)


def find_address_frame(
    speeds: np.ndarray,
    takeaway_frame: int,
    timestamps_ms: Optional[list[float]] = None,
    params: Optional[SwingDetectionParams] = None,
) -> int:
    """
    Find the address (P1) frame - last stable frame before takeaway.

    Args:
        speeds: Array of speeds
        takeaway_frame: Frame where takeaway motion begins
        timestamps_ms: Per-frame timestamps
        params: Detection parameters

    Returns:
        Frame number for P1 (address)
    """
    if params is None:
        params = SwingDetectionParams()

    # Look back from takeaway to find stable region
    if timestamps_ms is not None:
        # Find frames within pre_window
        pre_frames = 1
        takeaway_time = timestamps_ms[takeaway_frame]
        for i in range(takeaway_frame - 1, -1, -1):
            if takeaway_time - timestamps_ms[i] <= params.pre_window_ms:
                pre_frames = takeaway_frame - i
            else:
                break
    else:
        pre_frames = int(params.pre_window_ms / 16.67)

    # Search window
    search_start = max(0, takeaway_frame - pre_frames)
    search_end = takeaway_frame

    if search_end <= search_start:
        return max(0, takeaway_frame - 5)

    # Find frame with minimum speed in search window (most stable)
    search_region = speeds[search_start:search_end]
    min_idx = np.argmin(search_region)

    return search_start + min_idx


def detect_swing_window(
    landmarks: list,
    timestamps_ms: Optional[list[float]] = None,
    params: Optional[SwingDetectionParams] = None,
) -> Optional[SwingWindow]:
    """
    Detect the main swing window in pose landmark sequence.

    Args:
        landmarks: List of pose landmarks per frame
        timestamps_ms: Optional per-frame timestamps in milliseconds
        params: Detection parameters

    Returns:
        SwingWindow with detected boundaries, or None if no swing found
    """
    if params is None:
        params = SwingDetectionParams()

    n = len(landmarks)
    if n < 30:  # Need at least 30 frames
        return None

    # Compute wrist speeds
    speeds = compute_wrist_speed(landmarks, timestamps_ms)

    # Smooth speeds
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smooth_speeds = np.convolve(speeds, kernel, mode='same')

    # Estimate baseline from initial frames
    median, mad = estimate_baseline(smooth_speeds, timestamps_ms, params.baseline_duration_ms)

    # Detect motion bursts
    bursts = detect_motion_bursts(smooth_speeds, median, mad, params, timestamps_ms)

    if not bursts:
        return None

    # Find the main swing burst
    main_burst = find_main_swing_burst(bursts, timestamps_ms, params)

    if main_burst is None:
        return None

    takeaway_frame, end_frame, peak_speed = main_burst

    # Find address frame (P1)
    address_frame = find_address_frame(smooth_speeds, takeaway_frame, timestamps_ms, params)

    # Estimate impact frame (max speed in the burst)
    burst_speeds = smooth_speeds[takeaway_frame:end_frame+1]
    impact_offset = np.argmax(burst_speeds)
    impact_frame = takeaway_frame + impact_offset

    # Calculate confidence based on:
    # - Speed contrast (peak vs baseline)
    # - Duration within expected range
    speed_contrast = (peak_speed - median) / mad if mad > 0 else 0
    confidence = min(speed_contrast / 10.0, 1.0)  # 10 MAD = 100% confidence

    return SwingWindow(
        start_frame=address_frame,
        end_frame=end_frame,
        takeaway_frame=takeaway_frame,
        impact_frame=impact_frame,
        confidence=confidence,
    )


def trim_to_swing(
    landmarks: list,
    timestamps_ms: Optional[list[float]] = None,
    params: Optional[SwingDetectionParams] = None,
) -> tuple[list, Optional[list[float]], Optional[SwingWindow]]:
    """
    Trim landmarks to the detected swing window.

    Args:
        landmarks: Full pose landmark sequence
        timestamps_ms: Full timestamp sequence
        params: Detection parameters

    Returns:
        (trimmed_landmarks, trimmed_timestamps, swing_window)
        If no swing detected, returns original data with None window
    """
    window = detect_swing_window(landmarks, timestamps_ms, params)

    if window is None:
        return landmarks, timestamps_ms, None

    # Trim landmarks
    trimmed_landmarks = landmarks[window.start_frame:window.end_frame + 1]

    # Trim timestamps
    if timestamps_ms is not None:
        trimmed_timestamps = timestamps_ms[window.start_frame:window.end_frame + 1]
        # Re-zero timestamps
        t0 = trimmed_timestamps[0]
        trimmed_timestamps = [t - t0 for t in trimmed_timestamps]
    else:
        trimmed_timestamps = None

    return trimmed_landmarks, trimmed_timestamps, window
