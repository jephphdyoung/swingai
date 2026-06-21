import cv2
import numpy as np
import subprocess
import os

from utils.overlay import composite_overlay, transform_points

# Skeleton colors for overlay mode (BGR): user vs reference, distinct so the two
# swings are easy to tell apart when ghosted on top of each other.
OVERLAY_USER_COLOR = (255, 170, 0)   # azure/blue
OVERLAY_REF_COLOR = (0, 80, 255)     # red-orange

# Landmark indices in our extracted array:
# 0: nose, 1: L shoulder, 2: R shoulder, 3: L elbow, 4: R elbow,
# 5: L wrist, 6: R wrist, 7: L pinky, 8: R pinky, 9: L index, 10: R index,
# 11: L thumb, 12: R thumb, 13: L hip, 14: R hip, 15: L knee, 16: R knee,
# 17: L ankle, 18: R ankle, 19: L foot, 20: R foot

# Colors (BGR format) for each landmark
LANDMARK_COLORS = [
    (0, 255, 255),  # nose/head - yellow
    (255, 0, 0),    # shoulders - blue
    (255, 0, 0),
    (0, 165, 255),  # elbows - orange
    (0, 165, 255),
    (0, 255, 255),  # wrists - yellow
    (0, 255, 255),
    (255, 0, 255),  # pinky - magenta
    (255, 0, 255),
    (0, 255, 0),    # index finger - green
    (0, 255, 0),
    (0, 0, 255),    # thumb - red
    (0, 0, 255),
    (255, 255, 0),  # hips - cyan
    (255, 255, 0),
    (0, 128, 255),  # knees - dark orange
    (0, 128, 255),
    (128, 0, 128),  # ankles - purple
    (128, 0, 128),
    (255, 128, 0),  # feet - light blue
    (255, 128, 0),
]

# Skeleton connections: pairs of landmark indices to connect with lines
SKELETON_CONNECTIONS = [
    # Torso
    (1, 2),    # shoulder to shoulder
    (13, 14),  # hip to hip
    (1, 13),   # left shoulder to left hip
    (2, 14),   # right shoulder to right hip
    # Left arm
    (1, 3),    # left shoulder to left elbow
    (3, 5),    # left elbow to left wrist
    (5, 7),    # left wrist to left pinky
    (5, 9),    # left wrist to left index
    (5, 11),   # left wrist to left thumb
    # Right arm
    (2, 4),    # right shoulder to right elbow
    (4, 6),    # right elbow to right wrist
    (6, 8),    # right wrist to right pinky
    (6, 10),   # right wrist to right index
    (6, 12),   # right wrist to right thumb
    # Left leg
    (13, 15),  # left hip to left knee
    (15, 17),  # left knee to left ankle
    (17, 19),  # left ankle to left foot
    # Right leg
    (14, 16),  # right hip to right knee
    (16, 18),  # right knee to right ankle
    (18, 20),  # right ankle to right foot
]

def _draw_skeleton_px(frame, points, line_color=(200, 200, 200), dot_color=None):
    """Draw a skeleton from pre-computed pixel points.

    line_color: color for all connection lines.
    dot_color:  if None, use the per-landmark LANDMARK_COLORS; otherwise a single
                uniform color for every joint dot.
    """
    # Draw skeleton lines first (so dots appear on top)
    for i1, i2 in SKELETON_CONNECTIONS:
        if i1 < len(points) and i2 < len(points):
            cv2.line(frame, points[i1], points[i2], line_color, 2)

    # Draw neck line (nose to midpoint of shoulders)
    if len(points) > 2:
        neck_pt = ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2)
        cv2.line(frame, points[0], neck_pt, line_color, 2)

    # Draw spine (midpoint shoulders to midpoint hips)
    if len(points) > 14:
        shoulder_mid = ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2)
        hip_mid = ((points[13][0] + points[14][0]) // 2, (points[13][1] + points[14][1]) // 2)
        cv2.line(frame, shoulder_mid, hip_mid, line_color, 2)

    # Draw landmark dots
    for i, pt in enumerate(points):
        if dot_color is not None:
            color = dot_color
        else:
            color = LANDMARK_COLORS[i] if i < len(LANDMARK_COLORS) else (0, 255, 0)
        cv2.circle(frame, pt, 5, color, -1)

    return frame


def draw_landmarks(frame, landmarks, color=None):
    """Draw a skeleton from normalized landmarks.

    color: if given, draw lines and dots in that single color (used for the
           two-color overlay); otherwise use the default gray lines + multi-color
           dots (used for the side-by-side view).
    """
    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    if color is not None:
        return _draw_skeleton_px(frame, points, line_color=color, dot_color=color)
    return _draw_skeleton_px(frame, points)

def interpolate_landmarks(landmarks1, landmarks2, t):
    """Linearly interpolate between two landmark sets."""
    return [
        (l1[0] + t * (l2[0] - l1[0]),
         l1[1] + t * (l2[1] - l1[1]),
         l1[2] + t * (l2[2] - l1[2]))
        for l1, l2 in zip(landmarks1, landmarks2)
    ]

def blend_frames(frame1, frame2, t):
    """Blend two frames with weight t (0=frame1, 1=frame2)."""
    return cv2.addWeighted(frame1, 1 - t, frame2, t, 0)

def draw_p_label(frame, label, position="top"):
    """Draw P position label on frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Position: top center
    x = (w - text_w) // 2
    y = text_h + 30

    # Draw background rectangle
    cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    return frame


def _compose(frame_u, frame_r, lm_u, lm_r, render_mode, draw_skeleton, overlay_transform):
    """Combine an aligned user/reference frame pair into one output frame.

    render_mode "sidebyside": reference on the left, user on the right (hstack).
    render_mode "overlay":    reference ghosted on top of the user via overlay_transform.
    """
    if render_mode == "overlay":
        out, m = composite_overlay(frame_u, frame_r, overlay_transform)
        if draw_skeleton:
            h_u, w_u = frame_u.shape[:2]
            h_r, w_r = frame_r.shape[:2]
            user_pts = [(int(x * w_u), int(y * h_u)) for x, y, _ in lm_u]
            _draw_skeleton_px(out, user_pts,
                              line_color=OVERLAY_USER_COLOR, dot_color=OVERLAY_USER_COLOR)
            ref_pts = transform_points(lm_r, w_r, h_r, m)
            _draw_skeleton_px(out, ref_pts,
                              line_color=OVERLAY_REF_COLOR, dot_color=OVERLAY_REF_COLOR)
        return out

    # Side-by-side (default)
    fu = frame_u.copy()
    fr = frame_r.copy()
    if draw_skeleton:
        draw_landmarks(fu, lm_u)
        draw_landmarks(fr, lm_r)

    target_height = min(fu.shape[0], fr.shape[0])
    if fu.shape[0] != target_height:
        scale = target_height / fu.shape[0]
        fu = cv2.resize(fu, (int(fu.shape[1] * scale), target_height))
    if fr.shape[0] != target_height:
        scale = target_height / fr.shape[0]
        fr = cv2.resize(fr, (int(fr.shape[1] * scale), target_height))

    return np.hstack((fr, fu))


def generate_comparison_video(user_video_path, ref_video_path, alignment, user_poses, ref_poses,
                               output_fps=24, slowdown=3, user_p_positions=None, ref_p_positions=None,
                               pause_duration=5, draw_skeleton=True, output_filename="comparison_output.mp4",
                               render_mode="sidebyside", overlay_transform=None):
    """
    Generate comparison video with synchronized swings.

    render_mode: "sidebyside" (default) or "overlay".
    overlay_transform: required when render_mode == "overlay"; dict with keys
        left/top/width/mirror/alpha (see utils.overlay).

    slowdown: multiplier for how many frames to generate per alignment pair (higher = slower playback)
    user_p_positions: dict of P position names to frame indices for user video
    ref_p_positions: dict of P position names to frame indices for reference video
    pause_duration: seconds to pause at each P position
    draw_skeleton: whether to draw pose landmarks on the video
    output_filename: name of the output file

    Returns:
        tuple: (output_path, p_timestamps) where p_timestamps maps P-names to seconds in output video
    """
    cap_user = cv2.VideoCapture(user_video_path)
    cap_ref = cv2.VideoCapture(ref_video_path)

    # Pre-read all frames
    user_frames = []
    while True:
        ret, frame = cap_user.read()
        if not ret:
            break
        user_frames.append(frame)

    ref_frames = []
    while True:
        ret, frame = cap_ref.read()
        if not ret:
            break
        ref_frames.append(frame)

    cap_user.release()
    cap_ref.release()

    print(f"Loaded {len(user_frames)} user frames, {len(ref_frames)} ref frames")
    print(f"Alignment pairs: {len(alignment)}, slowdown: {slowdown}x")

    # Build lookup of P positions by frame index
    user_p_lookup = {}
    ref_p_lookup = {}
    if user_p_positions:
        for name, frame_idx in user_p_positions.items():
            user_p_lookup[frame_idx] = name
    if ref_p_positions:
        for name, frame_idx in ref_p_positions.items():
            ref_p_lookup[frame_idx] = name

    # Build sorted list of P-position boundaries for persistent labels
    p_boundaries = []
    if user_p_positions:
        for name, frame_idx in sorted(user_p_positions.items(), key=lambda x: x[1]):
            p_boundaries.append((frame_idx, name))

    def _current_p_label(u_idx):
        """Return the most recent P-position label at or before u_idx."""
        label = None
        for frame_idx, name in p_boundaries:
            if frame_idx <= u_idx:
                label = name
            else:
                break
        return label

    paused_positions = set()
    pause_frames = int(pause_duration * output_fps)

    print(f"Will pause for {pause_duration}s ({pause_frames} frames) at each P position")

    output_frames = []
    p_timestamps = {}  # Maps P-name to timestamp (seconds) in output video
    output_frame_count = 0

    for i in range(len(alignment) - 1):
        u_idx, r_idx = alignment[i]
        u_idx_next, r_idx_next = alignment[i + 1]

        if u_idx >= len(user_frames) or r_idx >= len(ref_frames):
            continue
        if u_idx >= len(user_poses) or r_idx >= len(ref_poses):
            continue
        if u_idx_next >= len(user_poses) or r_idx_next >= len(ref_poses):
            continue

        # Check if this exact frame is a P position (for pausing)
        pause_label = None
        if u_idx in user_p_lookup and user_p_lookup[u_idx] not in paused_positions:
            pause_label = user_p_lookup[u_idx]
            paused_positions.add(pause_label)

        # Persistent label: show which P-position range we're in
        current_label = _current_p_label(u_idx)

        for j in range(slowdown):
            t = j / slowdown

            interp_user_lm = interpolate_landmarks(user_poses[u_idx], user_poses[u_idx_next], t)
            interp_ref_lm = interpolate_landmarks(ref_poses[r_idx], ref_poses[r_idx_next], t)

            combined = _compose(user_frames[u_idx], ref_frames[r_idx],
                                interp_user_lm, interp_ref_lm,
                                render_mode, draw_skeleton, overlay_transform)
            if current_label:
                combined = draw_p_label(combined, current_label)
            output_frames.append(combined)
            output_frame_count += 1

        # Insert pause at P position
        if pause_label:
            # Record timestamp where this P-position appears in output video
            p_timestamps[pause_label] = output_frame_count / output_fps

            combined = _compose(user_frames[u_idx], ref_frames[r_idx],
                                user_poses[u_idx], ref_poses[r_idx],
                                render_mode, draw_skeleton, overlay_transform)
            combined = draw_p_label(combined, pause_label)

            print(f"Adding {pause_duration}s pause at {pause_label} (user frame {u_idx})")
            for _ in range(pause_frames):
                output_frames.append(combined.copy())
                output_frame_count += 1

    if not output_frames:
        raise RuntimeError("No frames were processed.")

    print(f"Output frames: {len(output_frames)}")

    height, width, _ = output_frames[0].shape
    temp_path = "temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))

    for frame in output_frames:
        out.write(frame)
    out.release()

    final_path = output_filename

    # Convert using libx264 with settings optimized for smooth playback
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", temp_path,
        "-vcodec", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    os.remove(temp_path)
    return final_path, p_timestamps
