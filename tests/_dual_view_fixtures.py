"""Deterministic fixtures for the dual-view tests: tiny videos + a fake pose backend."""

import math

import cv2
import numpy as np

from utils.landmarks import LANDMARK_NAMES
from utils.pose_data import FrameObservation


def write_video(path, n_frames, fps, size=(320, 240)):
    """MJPG AVI whose frames are distinguishable (frame number drawn on each)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened(), "cv2 cannot write MJPG"
    for i in range(n_frames):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.putText(img, str(i), (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    return str(path)


def skeleton(progress, view="face_on"):
    """Normalized landmarks for a right-handed golfer; arms rise with progress 0..1."""
    lift = math.radians(170 * progress)            # lead arm swings up
    pts = {
        "nose": (0.50, 0.20), "left_ear": (0.53, 0.18), "right_ear": (0.47, 0.18),
        "left_shoulder": (0.58, 0.32), "right_shoulder": (0.42, 0.32),
        "left_hip": (0.56, 0.58), "right_hip": (0.44, 0.58),
        "left_knee": (0.56, 0.75), "right_knee": (0.44, 0.75),
        "left_ankle": (0.56, 0.92), "right_ankle": (0.44, 0.92),
        "left_foot_index": (0.58, 0.95), "right_foot_index": (0.42, 0.95),
    }
    if view == "dtl":
        pts["left_hip"], pts["right_hip"] = (0.40, 0.58), (0.39, 0.585)
        pts["left_shoulder"], pts["right_shoulder"] = (0.52, 0.32), (0.50, 0.325)
    sx, sy = pts["left_shoulder"]
    arm = 0.22
    wrist = (sx + arm * math.sin(lift) * 0.3, sy + arm * math.cos(lift))
    elbow = ((sx + wrist[0]) / 2, (sy + wrist[1]) / 2)
    pts["left_elbow"], pts["left_wrist"] = elbow, wrist
    pts["right_elbow"] = (pts["right_shoulder"][0], 0.45)
    pts["right_wrist"] = (wrist[0] - 0.02, wrist[1] + 0.01)
    for name in ("pinky", "index", "thumb"):
        pts[f"left_{name}"] = pts["left_wrist"]
        pts[f"right_{name}"] = pts["right_wrist"]
    return pts


class FakeBackend:
    """Stands in for MediaPipe. Counts frames it was asked to process."""

    name = "fake"
    min_confidence = 0.5
    score_is_visibility = True

    def __init__(self, view="face_on", n_frames=100, missing=(), identity="fake-backend-1"):
        self.view, self.n_frames, self.missing = view, n_frames, set(missing)
        self._identity = identity
        self.calls = 0

    def identity(self):
        return self._identity

    def open(self):
        self._i = 0

    def detect(self, rgb_frame, timestamp_ms):
        i = self._i
        self._i += 1
        self.calls += 1
        if i in self.missing:
            return None
        pts = skeleton(i / max(1, self.n_frames - 1), self.view)
        n = len(LANDMARK_NAMES)
        xy = np.array([pts.get(name, (np.nan, np.nan)) for name in LANDMARK_NAMES], float)
        conf = np.full(n, 0.9)
        return FrameObservation(xy_norm=xy, z=np.zeros(n), confidence=conf)

    def close(self):
        pass
