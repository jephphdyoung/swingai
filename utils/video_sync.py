import numpy as np


def find_impact_frame(landmarks):
    """
    Find the impact frame by detecting maximum wrist velocity.
    """
    if len(landmarks) < 10:
        return len(landmarks) // 2

    left_wrist = np.array([[frame[5][0], frame[5][1]] for frame in landmarks])
    right_wrist = np.array([[frame[6][0], frame[6][1]] for frame in landmarks])

    velocity = np.zeros(len(landmarks))
    for i in range(1, len(landmarks)):
        left_v = np.linalg.norm(left_wrist[i] - left_wrist[i-1])
        right_v = np.linalg.norm(right_wrist[i] - right_wrist[i-1])
        velocity[i] = left_v + right_v

    kernel = np.ones(5) / 5
    smoothed = np.convolve(velocity, kernel, mode='same')

    return int(np.argmax(smoothed))


def sync_on_p_positions(user_p, ref_p):
    """
    Sync videos based on P positions.
    Aligns P1->P1, P4->P4, P6->P6, P9->P9, P10->P10.
    Only includes the swing (P1 to P10), cropping waggle.
    """
    # Get key frames
    u_p1, u_p4, u_p6, u_p9 = user_p['P1'], user_p['P4'], user_p['P6'], user_p['P9']
    r_p1, r_p4, r_p6, r_p9 = ref_p['P1'], ref_p['P4'], ref_p['P6'], ref_p['P9']
    u_p10 = user_p.get('P10', u_p9)
    r_p10 = ref_p.get('P10', r_p9)

    print(f"User P positions: P1={u_p1}, P4={u_p4}, P6={u_p6}, P9={u_p9}, P10={u_p10}")
    print(f"Ref P positions:  P1={r_p1}, P4={r_p4}, P6={r_p6}, P9={r_p9}, P10={r_p10}")

    alignment = []

    # Phase 1: P1 to P4 (backswing)
    u_backswing = u_p4 - u_p1
    r_backswing = r_p4 - r_p1
    if u_backswing > 0 and r_backswing > 0:
        for i in range(u_backswing + 1):
            t = i / u_backswing
            u_idx = u_p1 + i
            r_idx = r_p1 + int(t * r_backswing)
            alignment.append((u_idx, r_idx))

    # Phase 2: P4 to P6 (downswing)
    u_downswing = u_p6 - u_p4
    r_downswing = r_p6 - r_p4
    if u_downswing > 0 and r_downswing > 0:
        for i in range(1, u_downswing + 1):
            t = i / u_downswing
            u_idx = u_p4 + i
            r_idx = r_p4 + int(t * r_downswing)
            alignment.append((u_idx, r_idx))

    # Phase 3: P6 to P9 (follow-through)
    u_followthrough = u_p9 - u_p6
    r_followthrough = r_p9 - r_p6
    if u_followthrough > 0 and r_followthrough > 0:
        for i in range(1, u_followthrough + 1):
            t = i / u_followthrough
            u_idx = u_p6 + i
            r_idx = r_p6 + int(t * r_followthrough)
            alignment.append((u_idx, r_idx))

    # Phase 4: P9 to P10 (finish hold)
    u_finish = u_p10 - u_p9
    r_finish = r_p10 - r_p9
    if u_finish > 0 and r_finish > 0:
        for i in range(1, u_finish + 1):
            t = i / u_finish
            u_idx = u_p9 + i
            r_idx = r_p9 + int(t * r_finish)
            alignment.append((u_idx, r_idx))

    print(f"Alignment: {len(alignment)} pairs (P1 to P10, waggle cropped)")

    return alignment


# Keep old function for compatibility
def sync_poses(user_landmarks, ref_landmarks, user_p=None, ref_p=None,
               user_impact=None, ref_impact=None):
    """Legacy sync - use sync_on_p_positions instead."""
    print(f"Frames: user={len(user_landmarks)}, ref={len(ref_landmarks)}")

    if user_impact is None:
        user_impact = find_impact_frame(user_landmarks)
    if ref_impact is None:
        ref_impact = find_impact_frame(ref_landmarks)

    print(f"Impact frames: user={user_impact}, ref={ref_impact}")

    user_len = len(user_landmarks)
    ref_len = len(ref_landmarks)

    alignment = []

    if user_impact > 0 and ref_impact > 0:
        for u_idx in range(user_impact + 1):
            t = u_idx / user_impact
            r_idx = int(t * ref_impact)
            alignment.append((u_idx, r_idx))

    user_after = user_len - 1 - user_impact
    ref_after = ref_len - 1 - ref_impact

    if user_after > 0 and ref_after > 0:
        for i in range(1, user_after + 1):
            t = i / user_after
            u_idx = user_impact + i
            r_idx = ref_impact + int(t * ref_after)
            alignment.append((u_idx, r_idx))

    print(f"Final alignment length: {len(alignment)}")

    return alignment
