import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import os
from dataclasses import dataclass
from typing import Optional

# MediaPipe pose landmark indices
# Order matters - used for skeleton connections in video_renderer.py
SELECTED_LANDMARK_INDICES = [
    0,              # nose (head)
    11, 12,         # shoulders (left/right)
    13, 14,         # elbows (left/right)
    15, 16,         # wrists (left/right)
    17, 18,         # pinky (left/right)
    19, 20,         # index finger (left/right)
    21, 22,         # thumb (left/right)
    23, 24,         # hips (left/right)
    25, 26,         # knees (left/right)
    27, 28,         # ankles (left/right)
    31, 32,         # foot index (left/right)
]

# Path to the pose landmarker model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pose_landmarker.task")


@dataclass
class PoseExtractionResult:
    """Result of pose extraction with timing information."""
    landmarks: list  # List of frames, each frame is list of (x, y, z) tuples
    timestamps_ms: list[float]  # Per-frame timestamps in milliseconds
    fps: float  # Detected/used frame rate
    frame_count: int


def extract_pose(video_path: str, timestamps_ms: Optional[list[float]] = None) -> list:
    """
    Extract pose landmarks from video.

    This is the legacy interface that returns just landmarks for backwards compatibility.

    Args:
        video_path: Path to video file
        timestamps_ms: Optional pre-computed timestamps (from preprocessor)

    Returns:
        List of landmarks per frame
    """
    result = extract_pose_with_timestamps(video_path, timestamps_ms)
    return result.landmarks


def extract_pose_with_timestamps(
    video_path: str,
    timestamps_ms: Optional[list[float]] = None
) -> PoseExtractionResult:
    """
    Extract pose landmarks from video with timing information.

    Args:
        video_path: Path to video file
        timestamps_ms: Optional pre-computed timestamps (from video_preprocessor)
                      If None, timestamps are estimated from fps

    Returns:
        PoseExtractionResult with landmarks and timestamps
    """
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    landmarks = []
    actual_timestamps = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use provided timestamps or estimate from fps
        if timestamps_ms is not None and frame_idx < len(timestamps_ms):
            timestamp = timestamps_ms[frame_idx]
        else:
            timestamp = frame_idx * 1000 / fps

        actual_timestamps.append(timestamp)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe requires integer timestamp
        results = pose_landmarker.detect_for_video(mp_image, int(timestamp))

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            all_landmarks = results.pose_landmarks[0]
            if len(all_landmarks) == 33:
                frame_landmarks = [
                    (lm.x, lm.y, lm.z)
                    for idx, lm in enumerate(all_landmarks)
                    if idx in SELECTED_LANDMARK_INDICES
                ]
                landmarks.append(frame_landmarks)
            else:
                # Use previous frame's landmarks if available, else zeros
                if landmarks:
                    landmarks.append(landmarks[-1])
                else:
                    landmarks.append([(0.5, 0.5, 0.0)] * len(SELECTED_LANDMARK_INDICES))
        else:
            # No pose detected - use previous frame's landmarks to maintain alignment
            if landmarks:
                landmarks.append(landmarks[-1])
            else:
                landmarks.append([(0.5, 0.5, 0.0)] * len(SELECTED_LANDMARK_INDICES))

        frame_idx += 1

    cap.release()
    pose_landmarker.close()

    return PoseExtractionResult(
        landmarks=landmarks,
        timestamps_ms=actual_timestamps,
        fps=fps,
        frame_count=len(landmarks),
    )
