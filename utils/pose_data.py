"""Per-frame pose observations with validity, behind a swappable backend.

This is the pose layer for the dual-view comparison. It differs from the
legacy `pose_extractor.extract_pose` (still used by the Analyze tab) in three
ways that matter for measurement:

- **No carry-forward.** A frame where the backend found no pose is recorded as
  `detected=False` with NaN coordinates. It is never filled with the previous
  frame's landmarks, so it can't be mistaken for a fresh observation.
- **Named landmarks** (`utils.landmarks`), in pixel coordinates, so angles are
  not distorted by a non-square frame.
- **Per-landmark confidence** where the backend provides one.

Extraction is slow (a 1,245-frame 240fps clip is minutes on CPU), so results
are cached on disk keyed by the video's identity and the backend's identity.
Changing annotations, the selected P-position or the selected measurement
never touches inference.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from typing import Callable, Optional, Protocol

import numpy as np

from utils.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, MEDIAPIPE_INDEX

# Landmarks below this confidence are treated as not observed.
DEFAULT_MIN_CONFIDENCE = 0.5

# Bump when the cached array layout changes.
CACHE_SCHEMA = 1


@dataclass
class FrameObservation:
    """One backend detection, already in LANDMARK_NAMES order.

    xy_norm:    (J, 2) x/y as fractions of frame width/height.
    z:          (J,)   backend-relative depth (MediaPipe: hip-centred, not metric).
    confidence: (J,)   0..1, NaN where the backend gives none.
    Any landmark the backend can't produce is NaN in all three.
    """
    xy_norm: np.ndarray
    z: np.ndarray
    confidence: np.ndarray


class PoseBackend(Protocol):
    """What a pose model has to provide. MediaPipe today; RTMPose next."""

    name: str

    def identity(self) -> str:
        """Changes whenever outputs could change (model file, library version)."""

    def open(self) -> None: ...

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: float) -> Optional[FrameObservation]:
        """Return the observation for this frame, or None when no pose was found."""

    def close(self) -> None: ...


@dataclass
class PoseSequence:
    """All frames of one video. Arrays are indexed [frame, landmark, ...]."""
    video_path: str
    backend: str
    fps: float
    width: int
    height: int
    timestamps_ms: np.ndarray   # (N,) container presentation time
    xy: np.ndarray              # (N, J, 2) pixels; NaN where not observed
    z: np.ndarray               # (N, J)
    confidence: np.ndarray      # (N, J)
    detected: np.ndarray        # (N,) bool -- backend returned a pose
    view: Optional[str] = None          # "dtl" | "face_on"; set by the caller
    handedness: str = "right"           # set by the caller; not part of the cache

    @property
    def frame_count(self):
        return len(self.timestamps_ms)

    def is_valid(self, frame, name, min_confidence=DEFAULT_MIN_CONFIDENCE):
        if not 0 <= frame < self.frame_count or not self.detected[frame]:
            return False
        j = LANDMARK_INDEX[name]
        if np.isnan(self.xy[frame, j]).any():
            return False
        conf = self.confidence[frame, j]
        return bool(np.isnan(conf) or conf >= min_confidence)

    def point(self, frame, name, min_confidence=DEFAULT_MIN_CONFIDENCE):
        """Pixel (x, y) of a landmark, or None when not validly observed."""
        if not self.is_valid(frame, name, min_confidence):
            return None
        return self.xy[frame, LANDMARK_INDEX[name]].copy()

    def points(self, frame, min_confidence=DEFAULT_MIN_CONFIDENCE):
        """{name: (x, y)} for every validly observed landmark in a frame."""
        out = {}
        for name in LANDMARK_NAMES:
            p = self.point(frame, name, min_confidence)
            if p is not None:
                out[name] = p
        return out

    def nearest_frame(self, timestamp_ms):
        return nearest_frame(self.timestamps_ms, timestamp_ms)


def nearest_frame(timestamps_ms, timestamp_ms):
    """Index of the frame whose timestamp is closest to `timestamp_ms`."""
    ts = np.asarray(timestamps_ms, dtype=float)
    if ts.size == 0:
        raise ValueError("no frames")
    return int(np.argmin(np.abs(ts - float(timestamp_ms))))


def frame_at_timestamp(timestamps_ms, timestamp_ms):
    """Nearest frame, or None when the instant falls outside this video.

    "Outside" is more than half a frame interval before the first frame or
    after the last, so a time from a longer video is never clamped onto this
    video's first or last frame.
    """
    ts = np.asarray(timestamps_ms, dtype=float)
    if ts.size == 0:
        return None
    t = float(timestamp_ms)
    half = float(np.median(np.diff(ts))) / 2 if ts.size > 1 else 0.0
    if t < ts[0] - half or t > ts[-1] + half:
        return None
    return nearest_frame(ts, t)


# --- MediaPipe ------------------------------------------------------------

@lru_cache(maxsize=None)
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MediaPipeBackend:
    """MediaPipe Pose Landmarker in VIDEO mode (temporal tracking on)."""

    name = "mediapipe"

    def __init__(self, model_path=None):
        from utils.pose_extractor import MODEL_PATH
        self.model_path = model_path or MODEL_PATH
        self._landmarker = None

    def identity(self):
        import mediapipe as mp
        return f"mediapipe-{mp.__version__}-{_file_sha256(self.model_path)[:16]}"

    def open(self):
        from mediapipe.tasks.python import BaseOptions, vision
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb_frame, timestamp_ms):
        import mediapipe as mp
        # VIDEO mode needs strictly increasing integer milliseconds.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, ts)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        if len(landmarks) != 33:
            return None
        return mediapipe_observation(landmarks)

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def mediapipe_observation(landmarks):
    """Map MediaPipe's 33 landmarks onto LANDMARK_NAMES."""
    n = len(LANDMARK_NAMES)
    xy = np.full((n, 2), np.nan)
    z = np.full(n, np.nan)
    conf = np.full(n, np.nan)
    for j, name in enumerate(LANDMARK_NAMES):
        lm = landmarks[MEDIAPIPE_INDEX[name]]
        xy[j] = (lm.x, lm.y)
        z[j] = lm.z
        vis = getattr(lm, "visibility", None)
        if vis is not None:
            conf[j] = vis
    return FrameObservation(xy_norm=xy, z=z, confidence=conf)


# --- Extraction -----------------------------------------------------------

def extract_pose_sequence(video_path, backend: PoseBackend,
                          progress: Optional[Callable[[int, int], None]] = None):
    """Run `backend` over every frame of a video. No carry-forward."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(LANDMARK_NAMES)

    timestamps, xys, zs, confs, detected = [], [], [], [], []
    backend.open()
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = _frame_timestamp(cap, idx, fps, timestamps)
            timestamps.append(ts)
            obs = backend.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)
            if obs is None:
                xys.append(np.full((n, 2), np.nan))
                zs.append(np.full(n, np.nan))
                confs.append(np.full(n, np.nan))
                detected.append(False)
            else:
                xys.append(obs.xy_norm * (width, height))
                zs.append(obs.z)
                confs.append(obs.confidence)
                detected.append(True)
            idx += 1
            if progress is not None:
                progress(idx, total)
    finally:
        backend.close()
        cap.release()

    return PoseSequence(
        video_path=video_path,
        backend=backend.identity(),
        fps=float(fps),
        width=width,
        height=height,
        timestamps_ms=np.asarray(timestamps, dtype=float),
        xy=np.asarray(xys, dtype=float).reshape(-1, n, 2),
        z=np.asarray(zs, dtype=float).reshape(-1, n),
        confidence=np.asarray(confs, dtype=float).reshape(-1, n),
        detected=np.asarray(detected, dtype=bool),
    )


def _frame_timestamp(cap, idx, fps, previous):
    """Container presentation time of the frame just read, in ms.

    Falls back to idx/fps when the container gives nothing usable (0 after the
    first frame, or non-increasing), so timestamps are always monotonic.
    """
    import cv2
    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
    fallback = idx * 1000.0 / fps
    if idx == 0:
        return ts if ts and ts > 0 else 0.0
    if not ts or ts <= previous[-1]:
        return max(fallback, previous[-1] + 1e-3)
    return ts


# --- Cache ----------------------------------------------------------------

def video_identity(video_path):
    """Name, size and mtime: stable across host vs. container mount points."""
    st = os.stat(video_path)
    return {"name": os.path.basename(video_path), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def cache_key(video_path, backend_identity):
    payload = json.dumps({"schema": CACHE_SCHEMA, "video": video_identity(video_path),
                          "backend": backend_identity}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def default_cache_dir():
    from utils.paths import get_output_dir
    return os.path.join(get_output_dir(), "pose_cache")


def _cache_file(cache_dir, key):
    return os.path.join(cache_dir, f"{key}.npz")


def save_pose_sequence(seq: PoseSequence, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"video_path": seq.video_path, "backend": seq.backend, "fps": seq.fps,
            "width": seq.width, "height": seq.height, "schema": CACHE_SCHEMA,
            "landmarks": list(LANDMARK_NAMES)}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, timestamps_ms=seq.timestamps_ms, xy=seq.xy, z=seq.z,
                        confidence=seq.confidence, detected=seq.detected,
                        meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def load_pose_sequence(path, video_path=None):
    """Load a cached sequence, or None if missing or from another layout."""
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != CACHE_SCHEMA or meta.get("landmarks") != list(LANDMARK_NAMES):
            return None
        return PoseSequence(
            video_path=video_path or meta["video_path"],
            backend=meta["backend"],
            fps=meta["fps"],
            width=meta["width"],
            height=meta["height"],
            timestamps_ms=data["timestamps_ms"],
            xy=data["xy"],
            z=data["z"],
            confidence=data["confidence"],
            detected=data["detected"].astype(bool),
        )


def cached_pose_path(video_path, backend: Optional[PoseBackend] = None, cache_dir=None):
    backend = backend or MediaPipeBackend()
    return _cache_file(cache_dir or default_cache_dir(), cache_key(video_path, backend.identity()))


def get_pose_sequence(video_path, backend: Optional[PoseBackend] = None, cache_dir=None,
                      progress=None, extract_if_missing=True):
    """Cached pose for a video; runs inference only on a cache miss.

    Returns None on a miss when `extract_if_missing` is False.
    """
    backend = backend or MediaPipeBackend()
    path = cached_pose_path(video_path, backend, cache_dir)
    seq = load_pose_sequence(path, video_path)
    if seq is not None:
        return seq
    if not extract_if_missing:
        return None
    seq = extract_pose_sequence(video_path, backend, progress)
    save_pose_sequence(seq, path)
    return seq
