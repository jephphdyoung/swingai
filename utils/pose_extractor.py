import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import os

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

def extract_pose(video_path):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    landmarks = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx * 1000 / fps)

        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

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
    return landmarks
