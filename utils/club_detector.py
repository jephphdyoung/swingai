import cv2
import numpy as np


def detect_club_shaft(frame, hand_positions, prev_line=None):
    """
    Detect the golf club shaft in a frame using line detection.

    Args:
        frame: BGR image
        hand_positions: list of (x, y) normalized coordinates for hands/wrists
        prev_line: previous frame's detected line for temporal consistency

    Returns:
        (x1, y1, x2, y2) line endpoints in pixel coords, or None if not detected
    """
    h, w = frame.shape[:2]
    min_club_length = min(w, h) * 0.15  # Club should be at least 15% of frame

    # Convert hand positions to pixels
    hands_px = [(int(x * w), int(y * h)) for x, y in hand_positions]

    # Create region of interest around hands
    if len(hands_px) >= 2:
        # Use both wrists to define grip area
        grip_x = (hands_px[0][0] + hands_px[1][0]) // 2
        grip_y = (hands_px[0][1] + hands_px[1][1]) // 2
    else:
        grip_x, grip_y = hands_px[0] if hands_px else (w // 2, h // 2)

    # Search area: extend from hands, biased downward where clubhead usually is
    margin = int(min(w, h) * 0.45)
    x1 = max(0, grip_x - margin)
    y1 = max(0, grip_y - int(margin * 0.3))  # Less margin above hands
    x2 = min(w, grip_x + margin)
    y2 = min(h, grip_y + margin)  # More margin below hands

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Multi-scale edge detection for better shaft detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast on metallic shafts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive edge detection thresholds based on image statistics
    median_val = np.median(gray)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3)

    # Detect lines with adjusted parameters for thin shafts
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=int(min_club_length * 0.5), maxLineGap=15)

    if lines is None:
        return None

    # Find the best line with improved scoring
    best_line = None
    best_score = 0

    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]

        # Convert back to full frame coordinates
        lx1, lx2 = lx1 + x1, lx2 + x1
        ly1, ly2 = ly1 + y1, ly2 + y1

        # Line length - reject if too short
        length = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
        if length < min_club_length:
            continue

        # Ensure one endpoint is near the grip (hands)
        dist_p1_to_grip = np.sqrt((lx1 - grip_x)**2 + (ly1 - grip_y)**2)
        dist_p2_to_grip = np.sqrt((lx2 - grip_x)**2 + (ly2 - grip_y)**2)
        min_dist_to_grip = min(dist_p1_to_grip, dist_p2_to_grip)

        # Club must connect to hands - reject if neither endpoint is close
        max_grip_distance = min(w, h) * 0.15
        if min_dist_to_grip > max_grip_distance:
            continue

        # Reorient line so p1 is the grip end, p2 is the clubhead end
        if dist_p2_to_grip < dist_p1_to_grip:
            lx1, ly1, lx2, ly2 = lx2, ly2, lx1, ly1

        # Angle scoring - REJECT horizontal lines, prefer angled/vertical
        angle = np.arctan2(ly2 - ly1, lx2 - lx1)
        angle_from_horizontal = abs(angle)  # 0 = horizontal, pi/2 = vertical

        # Reject lines that are too horizontal (club shaft is never horizontal)
        # Allow angles from 20° to 160° from horizontal
        min_angle_from_horizontal = np.radians(20)
        if angle_from_horizontal < min_angle_from_horizontal:
            continue
        if angle_from_horizontal > np.pi - min_angle_from_horizontal:
            continue

        # Score: prefer more vertical lines (closer to 90°)
        angle_score = abs(np.sin(angle))  # 1 at vertical, 0 at horizontal

        # Bonus for lines extending downward from grip (typical club orientation)
        downward_bonus = 1.0 if ly2 > ly1 else 0.7

        # Temporal consistency bonus - prefer lines similar to previous detection
        temporal_score = 1.0
        if prev_line is not None:
            prev_angle = np.arctan2(prev_line[3] - prev_line[1], prev_line[2] - prev_line[0])
            angle_diff = abs(angle - prev_angle)
            # Allow more angle change during swing (up to 30 degrees per frame)
            temporal_score = max(0.3, 1.0 - angle_diff / np.radians(45))

        # Final score
        score = (length * angle_score * downward_bonus * temporal_score) / (min_dist_to_grip + 1)

        if score > best_score:
            best_score = score
            best_line = (lx1, ly1, lx2, ly2)

    return best_line


def get_club_angle(club_line):
    """Get the angle of the club shaft in degrees from vertical."""
    if club_line is None:
        return None
    x1, y1, x2, y2 = club_line
    # Angle from vertical (0 = vertical, 90 = horizontal)
    angle = np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1)))
    return angle


def track_club_in_video(video_path, landmarks):
    """
    Track the club shaft throughout a video with temporal consistency.

    Args:
        video_path: path to video file
        landmarks: list of pose landmarks per frame

    Returns:
        list of club lines (or None) for each frame
    """
    cap = cv2.VideoCapture(video_path)
    club_lines = []
    prev_line = None

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(landmarks):
            lm = landmarks[frame_idx]
            # Get wrist positions (indices 5, 6 = left/right wrist)
            hand_positions = [(lm[5][0], lm[5][1]), (lm[6][0], lm[6][1])]
            club_line = detect_club_shaft(frame, hand_positions, prev_line)
            if club_line is not None:
                prev_line = club_line
        else:
            club_line = None

        club_lines.append(club_line)
        frame_idx += 1

    cap.release()

    # Post-process: smooth detected lines and fill small gaps
    club_lines = smooth_club_trajectory(club_lines)

    return club_lines


def smooth_club_trajectory(club_lines, max_gap=3):
    """
    Smooth the club trajectory and fill small gaps in detection.

    Args:
        club_lines: list of detected club lines (or None)
        max_gap: maximum gap size to interpolate

    Returns:
        smoothed club lines
    """
    if not club_lines:
        return club_lines

    result = list(club_lines)
    n = len(result)

    # Fill small gaps by interpolation
    i = 0
    while i < n:
        if result[i] is None:
            # Find the gap extent
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i

            gap_size = gap_end - gap_start
            # Only fill small gaps with valid endpoints
            if gap_size <= max_gap and gap_start > 0 and gap_end < n:
                prev_line = result[gap_start - 1]
                next_line = result[gap_end]
                if prev_line is not None and next_line is not None:
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_size + 1)
                        result[j] = tuple(
                            int(prev_line[k] + t * (next_line[k] - prev_line[k]))
                            for k in range(4)
                        )
        else:
            i += 1

    # Apply moving average smoothing to detected lines
    smoothed = []
    window = 3
    for i in range(n):
        if result[i] is None:
            smoothed.append(None)
            continue

        # Collect nearby valid lines
        nearby = []
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if result[j] is not None:
                nearby.append(result[j])

        if nearby:
            avg_line = tuple(
                int(np.mean([line[k] for line in nearby]))
                for k in range(4)
            )
            smoothed.append(avg_line)
        else:
            smoothed.append(result[i])

    return smoothed


def find_impact_from_club(club_lines, fallback_landmarks=None):
    """
    Find impact frame based on club shaft velocity/angle change.
    Impact is when the club is moving fastest and near vertical.

    Returns None if club detection is unreliable (caller should use wrist-based).

    NOTE: Hough line detection often picks up body edges rather than the actual
    club shaft. This function is conservative and will return None (triggering
    wrist-based fallback) if the club angles don't follow expected patterns.
    """
    if not club_lines or len(club_lines) < 10:
        return None

    # Count how many frames have detected clubs
    detected_count = sum(1 for c in club_lines if c is not None)
    detection_rate = detected_count / len(club_lines)

    # High detection rate (>90%) usually means we're detecting body edges, not clubs
    # Real club detection typically has some dropout frames
    if detection_rate > 0.9:
        print(f"  Club detection rate suspiciously high ({detection_rate:.1%}), likely false positives")
        return None

    # If less than 30% detection, club tracking is unreliable
    if detection_rate < 0.3:
        print(f"  Club detection rate too low ({detection_rate:.1%}), skipping")
        return None

    # Compute club angle for each frame
    angles = []
    for line in club_lines:
        if line is not None:
            angles.append(get_club_angle(line))
        else:
            angles.append(None)

    # Interpolate missing angles
    angles = interpolate_angles(angles)

    # Compute angular velocity
    angular_velocity = np.zeros(len(angles))
    for i in range(1, len(angles)):
        if angles[i] is not None and angles[i-1] is not None:
            angular_velocity[i] = abs(angles[i] - angles[i-1])

    # Smooth
    kernel = np.ones(5) / 5
    smoothed = np.convolve(angular_velocity, kernel, mode='same')

    # Impact = max angular velocity
    impact = int(np.argmax(smoothed))

    # Sanity check: impact should be in middle portion of video
    if impact < len(club_lines) * 0.1 or impact > len(club_lines) * 0.9:
        print(f"  Club-based impact at {impact} seems wrong, skipping")
        return None

    return impact


def interpolate_angles(angles):
    """Fill in None values by linear interpolation."""
    result = angles.copy()

    # Find first and last valid values
    first_valid = None
    last_valid = None
    for i, a in enumerate(angles):
        if a is not None:
            if first_valid is None:
                first_valid = i
            last_valid = i

    if first_valid is None:
        return [45.0] * len(angles)  # Default

    # Fill before first valid
    for i in range(first_valid):
        result[i] = angles[first_valid]

    # Fill after last valid
    for i in range(last_valid + 1, len(angles)):
        result[i] = angles[last_valid]

    # Interpolate middle
    prev_valid = first_valid
    for i in range(first_valid + 1, last_valid + 1):
        if result[i] is None:
            # Find next valid
            next_valid = i + 1
            while next_valid < len(angles) and result[next_valid] is None:
                next_valid += 1
            if next_valid < len(angles):
                t = (i - prev_valid) / (next_valid - prev_valid)
                result[i] = result[prev_valid] + t * (result[next_valid] - result[prev_valid])
        else:
            prev_valid = i

    return result


def draw_club_shaft(frame, club_line, color=(0, 255, 0), thickness=3):
    """Draw the detected club shaft on the frame."""
    if club_line is None:
        return frame

    x1, y1, x2, y2 = [int(v) for v in club_line]

    # Draw the line with a glow effect
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 4)  # Black outline
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)  # Main line

    # Draw endpoints
    cv2.circle(frame, (x1, y1), 5, color, -1)
    cv2.circle(frame, (x2, y2), 5, color, -1)

    return frame
