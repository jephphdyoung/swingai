"""
Club-based P2/P6 position detection.

Uses golf club shaft angle detection to find shaft-parallel positions:
- P2: Club parallel to ground in backswing
- P6: Impact (can refine with club angle)
- P2-equivalent in follow-through

Integration with existing club_detector.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .club_detector import track_club_in_video, get_club_angle, interpolate_angles


# Club shaft angles for key positions (degrees from vertical)
# 0° = vertical (pointing down), 90° = horizontal
SHAFT_PARALLEL_ANGLE = 80.0  # ~80° from vertical = nearly horizontal
SHAFT_PARALLEL_TOLERANCE = 15.0  # ±15° tolerance

# Impact typically has club near vertical
IMPACT_ANGLE_RANGE = (0.0, 30.0)  # Club within 30° of vertical at impact


@dataclass
class ClubPositionResult:
    """Result of club-based position detection."""
    p2_frame: Optional[int]  # Shaft parallel in backswing
    p6_frame: Optional[int]  # Impact (refined)
    p2_follow_frame: Optional[int]  # Shaft parallel in follow-through (near P7)
    confidence: float  # Overall confidence in club detection
    detection_rate: float  # Fraction of frames with club detected
    angles: list[float]  # Interpolated club angles per frame


def detect_shaft_parallel_frames(
    angles: list[float],
    target_angle: float = SHAFT_PARALLEL_ANGLE,
    tolerance: float = SHAFT_PARALLEL_TOLERANCE,
) -> list[int]:
    """
    Find frames where club shaft is approximately parallel to ground.

    Args:
        angles: Club angles per frame (degrees from vertical)
        target_angle: Target angle for parallel (default 80°)
        tolerance: Angle tolerance in degrees

    Returns:
        List of frame indices where shaft is parallel
    """
    parallel_frames = []

    for i, angle in enumerate(angles):
        if angle is None:
            continue
        if abs(angle - target_angle) <= tolerance:
            parallel_frames.append(i)

    return parallel_frames


def find_p2_backswing(
    angles: list[float],
    p1_frame: int,
    p4_frame: int,
) -> Optional[int]:
    """
    Find P2 (shaft parallel in backswing) between P1 and P4.

    P2 is when the club first reaches horizontal during the backswing.

    Args:
        angles: Club angles per frame
        p1_frame: Address frame
        p4_frame: Top of backswing frame

    Returns:
        Frame number for P2, or None if not found
    """
    if p1_frame >= p4_frame:
        return None

    # Search between P1 and P4
    search_angles = angles[p1_frame:p4_frame]

    # Find first frame where shaft approaches horizontal
    # During backswing, angle increases from ~45° (address) toward 90° (horizontal)
    for i, angle in enumerate(search_angles):
        if angle is None:
            continue
        if angle >= SHAFT_PARALLEL_ANGLE - SHAFT_PARALLEL_TOLERANCE:
            return p1_frame + i

    # Fallback: find frame closest to target angle
    best_frame = None
    best_diff = float('inf')

    for i, angle in enumerate(search_angles):
        if angle is None:
            continue
        diff = abs(angle - SHAFT_PARALLEL_ANGLE)
        if diff < best_diff:
            best_diff = diff
            best_frame = p1_frame + i

    return best_frame


def find_p6_impact(
    angles: list[float],
    p4_frame: int,
    p9_frame: int,
    initial_p6: Optional[int] = None,
) -> tuple[Optional[int], float]:
    """
    Find/refine P6 (impact) using club angle.

    At impact, the club should be near vertical (low angle from vertical).

    Args:
        angles: Club angles per frame
        p4_frame: Top of backswing frame
        p9_frame: Finish frame
        initial_p6: Initial P6 estimate to refine

    Returns:
        (frame, confidence) tuple
    """
    # Search window
    if initial_p6 is not None:
        search_start = max(p4_frame, initial_p6 - 15)
        search_end = min(p9_frame, initial_p6 + 15)
    else:
        # Search middle third of downswing/impact zone
        search_start = p4_frame
        search_end = p4_frame + (p9_frame - p4_frame) // 2

    if search_start >= search_end:
        return initial_p6, 0.3

    # Find frame with most vertical club (smallest angle)
    best_frame = None
    best_angle = float('inf')

    for i in range(search_start, search_end):
        if i >= len(angles):
            break
        angle = angles[i]
        if angle is None:
            continue
        if angle < best_angle:
            best_angle = angle
            best_frame = i

    # Confidence based on how vertical the club is at impact
    if best_frame is not None and best_angle < 45:
        confidence = 1.0 - (best_angle / 90.0)  # More vertical = more confident
    else:
        confidence = 0.3

    return best_frame if best_frame is not None else initial_p6, confidence


def find_p7_followthrough(
    angles: list[float],
    p6_frame: int,
    p9_frame: int,
) -> Optional[int]:
    """
    Find shaft-parallel frame in follow-through (approximately P7).

    Args:
        angles: Club angles per frame
        p6_frame: Impact frame
        p9_frame: Finish frame

    Returns:
        Frame number, or None if not found
    """
    if p6_frame >= p9_frame:
        return None

    # Search from impact to finish
    # In follow-through, club goes from vertical back to horizontal
    search_angles = angles[p6_frame:p9_frame]

    # Find first frame after impact where shaft returns to horizontal
    for i, angle in enumerate(search_angles):
        if angle is None:
            continue
        if i > 5 and angle >= SHAFT_PARALLEL_ANGLE - SHAFT_PARALLEL_TOLERANCE:
            return p6_frame + i

    return None


def detect_club_positions(
    video_path: str,
    landmarks: list,
    p_positions: dict[str, int],
) -> ClubPositionResult:
    """
    Detect P2, P6 (refined), and follow-through shaft-parallel using club tracking.

    Args:
        video_path: Path to video file
        landmarks: Pose landmarks per frame
        p_positions: Initial P-positions (P1, P4, P6, P9 at minimum)

    Returns:
        ClubPositionResult with detected/refined positions
    """
    # Track club through video
    club_lines = track_club_in_video(video_path, landmarks)

    if not club_lines:
        return ClubPositionResult(
            p2_frame=None,
            p6_frame=None,
            p2_follow_frame=None,
            confidence=0.0,
            detection_rate=0.0,
            angles=[],
        )

    # Compute detection rate
    detected_count = sum(1 for c in club_lines if c is not None)
    detection_rate = detected_count / len(club_lines)

    # Get angles
    raw_angles = [get_club_angle(line) for line in club_lines]
    angles = interpolate_angles(raw_angles)

    # Get key position frames
    p1 = p_positions.get("P1", 0)
    p4 = p_positions.get("P4", len(landmarks) // 3)
    p6 = p_positions.get("P6", len(landmarks) // 2)
    p9 = p_positions.get("P9", len(landmarks) - 1)

    # Confidence based on detection rate
    # Too high (>90%) = likely false positives (body edges)
    # Too low (<30%) = unreliable
    if detection_rate > 0.9:
        base_confidence = 0.3  # Suspicious
    elif detection_rate < 0.3:
        base_confidence = 0.2  # Unreliable
    else:
        base_confidence = 0.5 + (detection_rate - 0.3) * 0.5  # 0.5 to 0.8

    # Find P2 (backswing shaft parallel)
    p2_frame = find_p2_backswing(angles, p1, p4)

    # Refine P6 (impact)
    p6_refined, p6_confidence = find_p6_impact(angles, p4, p9, p6)

    # Find follow-through shaft parallel (near P7)
    p2_follow = find_p7_followthrough(angles, p6_refined or p6, p9)

    # Overall confidence
    confidence = base_confidence * (0.5 + p6_confidence * 0.5)

    return ClubPositionResult(
        p2_frame=p2_frame,
        p6_frame=p6_refined,
        p2_follow_frame=p2_follow,
        confidence=confidence,
        detection_rate=detection_rate,
        angles=angles,
    )


def refine_positions_with_club(
    video_path: str,
    landmarks: list,
    positions: dict[str, int],
    timestamps_ms: Optional[list[float]] = None,
) -> dict[str, int]:
    """
    Refine P-positions using club tracking data.

    This is a high-level function that integrates club detection
    with existing position detection.

    Args:
        video_path: Path to video file
        landmarks: Pose landmarks
        positions: Initial P-positions to refine
        timestamps_ms: Optional timestamps

    Returns:
        Refined positions dict
    """
    result = detect_club_positions(video_path, landmarks, positions)

    # Only use club-based positions if confidence is reasonable
    if result.confidence < 0.4:
        print(f"  Club detection confidence too low ({result.confidence:.0%}), using pose-only positions")
        return positions

    refined = dict(positions)

    # Update P2 if detected
    if result.p2_frame is not None:
        p1 = positions.get("P1", 0)
        p4 = positions.get("P4", len(landmarks) // 3)
        # Validate P2 is between P1 and P4
        if p1 < result.p2_frame < p4:
            refined["P2"] = result.p2_frame
            print(f"  P2 refined with club: frame {result.p2_frame}")

    # Update P6 if refined
    if result.p6_frame is not None:
        p4 = positions.get("P4", len(landmarks) // 3)
        p9 = positions.get("P9", len(landmarks) - 1)
        # Validate P6 is between P4 and P9
        if p4 < result.p6_frame < p9:
            refined["P6"] = result.p6_frame
            print(f"  P6 refined with club: frame {result.p6_frame}")

    # P7 approximation using follow-through shaft parallel
    if result.p2_follow_frame is not None:
        p6 = refined.get("P6", positions.get("P6", len(landmarks) // 2))
        p9 = positions.get("P9", len(landmarks) - 1)
        if p6 < result.p2_follow_frame < p9:
            refined["P7"] = result.p2_follow_frame
            print(f"  P7 refined with club: frame {result.p2_follow_frame}")

    return refined
