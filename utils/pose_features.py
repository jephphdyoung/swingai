import numpy as np
from typing import Optional


def detect_p_positions(
    landmarks,
    impact_override: Optional[int] = None,
    timestamps_ms: Optional[list[float]] = None,
):
    """
    Detect golf P positions (P1-P9) based on pose landmarks.

    Uses multiple body parts (wrists, elbows, shoulders) for robust detection:
    1. Find P6 (impact) - max combined upper body velocity
    2. Find P4 (top of backswing) - combination of hand height + shoulder rotation
    3. Find P1 (address) - stable position before backswing motion begins
    4. Distribute other positions proportionally

    Args:
        landmarks: List of pose landmarks per frame
        impact_override: Optional frame number to use as P6 (impact)
        timestamps_ms: Optional per-frame timestamps in milliseconds.
                      If provided, velocities are computed using actual dt.
                      If None, assumes constant frame rate (legacy behavior).

    Returns:
        Dict mapping position names (P1-P9) to frame numbers
    """
    n = len(landmarks)
    if n < 20:
        return {"P1": 0, "P4": n//3, "P6": n//2, "P9": n-1}

    # Compute time deltas between frames
    if timestamps_ms is not None and len(timestamps_ms) == n:
        # Use actual timestamps for accurate velocity calculation
        dt = np.zeros(n)
        for i in range(1, n):
            dt[i] = timestamps_ms[i] - timestamps_ms[i-1]
            # Clamp to reasonable range (avoid division issues)
            dt[i] = max(1.0, min(100.0, dt[i]))  # 1ms to 100ms
    else:
        # Legacy: assume constant frame rate (~33ms for 30fps)
        dt = np.ones(n) * 33.33

    # Extract body positions
    # Landmark indices: 1,2=shoulders, 3,4=elbows, 5,6=wrists, 13,14=hips
    left_shoulder = np.array([[frame[1][0], frame[1][1]] for frame in landmarks])
    right_shoulder = np.array([[frame[2][0], frame[2][1]] for frame in landmarks])
    left_elbow = np.array([[frame[3][0], frame[3][1]] for frame in landmarks])
    right_elbow = np.array([[frame[4][0], frame[4][1]] for frame in landmarks])
    left_wrist = np.array([[frame[5][0], frame[5][1]] for frame in landmarks])
    right_wrist = np.array([[frame[6][0], frame[6][1]] for frame in landmarks])
    left_hip = np.array([[frame[13][0], frame[13][1]] for frame in landmarks])
    right_hip = np.array([[frame[14][0], frame[14][1]] for frame in landmarks])

    # Compute averages
    avg_wrist = (left_wrist + right_wrist) / 2
    avg_elbow = (left_elbow + right_elbow) / 2
    avg_shoulder = (left_shoulder + right_shoulder) / 2
    avg_hip = (left_hip + right_hip) / 2

    # Compute combined upper body velocity (wrists + elbows + shoulders)
    # Weight wrists more heavily as they move most during swing
    # Velocity = displacement / time (normalized by dt for VFR handling)
    combined_vel = np.zeros(n)
    for i in range(1, n):
        # Displacement in normalized coordinates
        wrist_disp = np.linalg.norm(avg_wrist[i] - avg_wrist[i-1])
        elbow_disp = np.linalg.norm(avg_elbow[i] - avg_elbow[i-1])
        shoulder_disp = np.linalg.norm(avg_shoulder[i] - avg_shoulder[i-1])

        # Convert to velocity (units per ms) then scale to comparable range
        # Multiply by 33.33 to normalize to ~30fps equivalent
        time_scale = 33.33 / dt[i]
        combined_vel[i] = (wrist_disp * 0.5 + elbow_disp * 0.3 + shoulder_disp * 0.2) * time_scale

    smooth_vel = np.convolve(combined_vel, np.ones(5)/5, mode='same')

    # Compute shoulder rotation (X distance between shoulders relative to hips)
    # In face-on view, shoulder turn shows as change in shoulder X positions
    shoulder_spread = np.abs(right_shoulder[:, 0] - left_shoulder[:, 0])
    hip_spread = np.abs(right_hip[:, 0] - left_hip[:, 0])
    # Shoulder rotation ratio: lower = more rotated (shoulders closer together in face-on)
    shoulder_rotation = shoulder_spread / (hip_spread + 0.001)
    smooth_rotation = np.convolve(shoulder_rotation, np.ones(5)/5, mode='same')

    # Combined arm height (average Y of wrists and elbows)
    # Lower Y = higher position in frame
    arm_height = (avg_wrist[:, 1] + avg_elbow[:, 1]) / 2

    # STEP 1: P6 = impact (max combined velocity)
    if impact_override is not None:
        p6 = impact_override
    else:
        p6 = int(np.argmax(smooth_vel))

    # STEP 2: P4 = top of backswing
    # Find where arms are at their highest position (min Y) before impact
    # But constrain the search to a reasonable window (not at very start of video)
    if p6 > 20:
        # Search window: from 20% to 95% of the way to impact
        # This avoids finding "top" at the very start during setup/waggle
        search_start = max(10, int(p6 * 0.2))
        search_end = int(p6 * 0.95)

        if search_end > search_start:
            # Find minimum arm height (highest position) in search window
            search_region = arm_height[search_start:search_end]
            p4 = search_start + int(np.argmin(search_region))
        else:
            p4 = max(1, p6 // 2)
    else:
        p4 = max(1, p6 // 2)

    # Verify P4 is reasonable: arm height at P4 should be lower than surrounding frames
    # If not, fall back to simpler detection
    if p4 > 5 and p4 < p6 - 5:
        p4_height = arm_height[p4]
        before_height = np.mean(arm_height[max(0, p4-10):p4])
        after_height = np.mean(arm_height[p4:min(p6, p4+10)])
        if not (p4_height < before_height and p4_height < after_height):
            # P4 doesn't look like a peak - use simple wrist-only detection
            p4 = int(np.argmin(avg_wrist[:p6, 1]))

    # STEP 3: P1 = find ADDRESS position
    # Strategy: Work backwards from P4 to find where the backswing climb started
    # This works for both videos with and without waggle

    # Compute arm height velocity (negative = arms going up, positive = arms going down)
    # Normalized by time delta for VFR handling
    arm_height_vel = np.zeros(n)
    for i in range(1, n):
        # Displacement / time, scaled to ~30fps equivalent
        time_scale = 33.33 / dt[i]
        arm_height_vel[i] = (arm_height[i] - arm_height[i-1]) * time_scale
    smooth_arm_vel = np.convolve(arm_height_vel, np.ones(7)/7, mode='same')

    # From P4, go backwards to find where the sustained upward motion (negative arm_height_vel) started
    # The address is where arms were going DOWN (positive vel) or stable before climbing started
    p1 = 0

    # Look for where arm_height_vel is POSITIVE (arms going down) - this is pre-backswing
    # Require multiple consecutive frames to avoid noise
    descend_threshold = 0.0005  # Positive = arms going down
    consecutive_descend = 0
    required_consecutive = 5

    for i in range(p4 - 15, -1, -1):
        # Check if arms are going DOWN (positive velocity) or very stable
        if smooth_arm_vel[i] > descend_threshold:
            consecutive_descend += 1
            if consecutive_descend >= required_consecutive:
                # Found where backswing started - P1 is stable frame in this area
                address_area = i + required_consecutive
                stable_start = max(0, address_area - 5)
                stable_end = min(address_area + 15, p4 - 15)
                if stable_end > stable_start:
                    stable_region = smooth_vel[stable_start:stable_end]
                    p1 = stable_start + int(np.argmin(stable_region))
                else:
                    p1 = address_area
                break
        else:
            consecutive_descend = 0  # Reset

    # Ensure minimum backswing length
    if p4 - p1 < 20:
        p1 = max(0, p4 - 30)

    # STEP 4: P9 = end of follow through
    # Find where velocity drops back to low levels after impact
    # This is where the swing motion actually ends
    vel_threshold = np.percentile(smooth_vel, 30)  # Low velocity threshold
    p9 = n - 1  # Default to last frame

    # Search from impact forward, find where velocity stabilizes
    for i in range(p6 + 20, n):  # Start 20 frames after impact
        # Check if velocity has been low for several consecutive frames
        if i + 5 < n:
            window_vel = smooth_vel[i:i+5]
            if np.all(window_vel < vel_threshold * 1.5):
                p9 = i
                break

    # Ensure P9 is at least 30 frames after P6
    p9 = max(p6 + 30, min(p9, n - 1))

    # Distribute intermediate positions proportionally
    backswing_len = p4 - p1
    p2 = p1 + backswing_len // 3
    p3 = p1 + 2 * backswing_len // 3

    p5 = p4 + (p6 - p4) // 2

    followthrough_len = p9 - p6
    p7 = p6 + followthrough_len // 3
    p8 = p6 + 2 * followthrough_len // 3

    positions = {
        "P1": max(0, p1),
        "P2": max(p1+1, p2),
        "P3": max(p2+1, p3),
        "P4": p4,
        "P5": max(p4+1, p5),
        "P6": p6,
        "P7": max(p6+1, p7),
        "P8": max(p7+1, p8),
        "P9": min(p9, n-1),
    }

    print(f"  P1={p1}, P4={p4}, P6={p6} (backswing={p4-p1} frames, downswing={p6-p4} frames)")

    return positions
