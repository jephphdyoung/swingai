"""RTMPose-m (Body7, Halpe26 keypoints) behind the PoseBackend protocol.

Two ONNX networks run through ONNX Runtime's CUDA execution provider:

    YOLOX-m (HumanArt)  -> person boxes on the full frame (letterboxed to 640)
    RTMPose-m Halpe26   -> 26 keypoints inside the golfer's box (crop 192x256)

Pre- and post-processing are implemented here in numpy/OpenCV, exactly as the
models' own mmdeploy `pipeline.json` specifies, so the coordinate maths is
unit-testable without a GPU:

    pose:     box padded 1.25x, aspect fixed to 192:256, affine crop,
              BGR->RGB, ImageNet mean/std; SimCC decode at split ratio 2
    detector: keep-ratio resize to 640, pad to square with 114, BGR, no norm

Keypoints are mapped back through the inverse of the crop transform, so they
come out in original-frame pixels.

Confidence is the SimCC peak height (the smaller of the x and y maxima). It is
an uncalibrated score, not a probability and not comparable to MediaPipe's
visibility; the default threshold is therefore backend-specific
(RTMPOSE_MIN_CONFIDENCE).

GPU only by default: if the CUDA execution provider is unavailable, or either
network's session lands on the CPU or fails its warm-up run, `open()` raises
RTMPoseUnavailable naming that network instead of silently using the CPU.
"""

from functools import lru_cache
import hashlib
import json
import os

import numpy as np

from utils.landmarks import LANDMARK_NAMES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "pose": {
        "file": "rtmpose-m_body7-halpe26_256x192.onnx",
        "archive": "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "archive_sha256": "55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18",
        "onnx_sha256": "26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf",
    },
    "detector": {
        "file": "yolox-m_humanart_640.onnx",
        "archive": "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "archive_sha256": "a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde",
        "onnx_sha256": "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
    },
}

NETWORKS = ("detector", "pose")

# Everything that changes outputs for given weights. Part of the cache key.
CONFIG = {
    "pose_input_wh": (192, 256),
    "bbox_padding": 1.25,
    "mean_rgb": (123.675, 116.28, 103.53),
    "std_rgb": (58.395, 57.12, 57.375),
    "simcc_split_ratio": 2.0,
    "det_input_wh": (640, 640),
    "det_pad_value": 114.0,
    "det_score_threshold": 0.5,
    "golfer_iou_keep": 0.3,
}

# SimCC peak below this = not observed. rtmlib's default keypoint threshold.
RTMPOSE_MIN_CONFIDENCE = 0.3

HALPE26 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)

# SwingAI landmark -> Halpe26 keypoint. None = the model has no such point;
# it is reported as unobserved, never estimated.
# foot_index: MediaPipe's point is the front of the foot. Halpe's big toe goes in
# that column only so the skeleton can draw a foot; the two are not the same
# point and are never scored against each other (utils/label_points.py).
# Hips: COCO-family hip keypoints sit on the hip joint, MediaPipe's somewhat
# differently -- compare values within one backend.
HALPE26_FOR = {name: (name if name in HALPE26 else None) for name in LANDMARK_NAMES}
HALPE26_FOR.update({
    "left_foot_index": "left_big_toe",
    "right_foot_index": "right_big_toe",
})


class RTMPoseUnavailable(RuntimeError):
    """RTMPose can't run here (no onnxruntime-gpu, no GPU, missing models)."""


class RTMPoseInferenceError(RTMPoseUnavailable):
    """A network failed mid-video (e.g. CUDA out of memory). Nothing is cached."""


def default_model_dir():
    return os.environ.get("SWINGAI_MODEL_DIR") or os.path.join(REPO_ROOT, "models", "rtmpose")


@lru_cache(maxsize=None)
def _sha256(path, mtime_ns, size):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    st = os.stat(path)
    return _sha256(path, st.st_mtime_ns, st.st_size)


# --- pure geometry (no onnxruntime needed) --------------------------------

def letterbox(bgr, size=CONFIG["det_input_wh"], pad_value=CONFIG["det_pad_value"]):
    """Keep-ratio resize into the top-left of a padded square. Returns (img, ratio)."""
    import cv2
    w, h = size
    ratio = min(w / bgr.shape[1], h / bgr.shape[0])
    nw, nh = int(round(bgr.shape[1] * ratio)), int(round(bgr.shape[0] * ratio))
    out = np.full((h, w, 3), pad_value, dtype=np.float32)
    out[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, ratio


def bbox_center_scale(bbox, padding=CONFIG["bbox_padding"]):
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    scale = np.array([x1 - x0, y1 - y0]) * padding
    return center, scale


def fix_aspect(scale, aspect):
    """Grow the shorter side so w/h == aspect (mmpose _fix_aspect_ratio)."""
    w, h = float(scale[0]), float(scale[1])
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return np.array([w, h])


def crop_matrix(center, scale, out_wh):
    """2x3 affine taking frame pixels to crop pixels (no rotation).

    Matches mmpose get_warp_matrix(rot=0): the box centre lands on the crop
    centre and the box width spans the crop width.
    """
    s = out_wh[0] / scale[0]
    return np.array([[s, 0.0, out_wh[0] / 2 - s * center[0]],
                     [0.0, s, out_wh[1] / 2 - s * center[1]]])


def crop_to_frame(points, center, scale, out_wh):
    """Inverse of crop_matrix for (K, 2) crop-pixel points."""
    points = np.asarray(points, dtype=float)
    return points / np.asarray(out_wh, float) * scale + center - scale / 2


def decode_simcc(simcc_x, simcc_y, split_ratio=CONFIG["simcc_split_ratio"]):
    """(K, Wx), (K, Wy) -> (K, 2) crop-pixel locations and (K,) scores."""
    x_idx = np.argmax(simcc_x, axis=1)
    y_idx = np.argmax(simcc_y, axis=1)
    x_val = np.max(simcc_x, axis=1)
    y_val = np.max(simcc_y, axis=1)
    scores = np.minimum(x_val, y_val).astype(float)
    locs = np.stack([x_idx, y_idx], axis=1).astype(float) / split_ratio
    locs[scores <= 0] = np.nan
    return locs, scores


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_golfer(boxes, previous=None, score_threshold=CONFIG["det_score_threshold"],
                  iou_keep=CONFIG["golfer_iou_keep"]):
    """One person box (x0, y0, x1, y1, score) or None.

    Follows the previous frame's golfer when a box overlaps it enough;
    otherwise takes the largest confident person (the golfer is the subject
    nearest the camera in these captures).
    """
    boxes = [b for b in np.asarray(boxes, float).reshape(-1, 5) if b[4] >= score_threshold]
    if not boxes:
        return None
    if previous is not None:
        best = max(boxes, key=lambda b: _iou(b, previous))
        if _iou(best, previous) >= iou_keep:
            return best
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def halpe26_observation(keypoints_px, scores, width, height):
    """Halpe26 (26, 2) frame pixels + (26,) scores -> FrameObservation."""
    from utils.pose_data import FrameObservation
    n = len(LANDMARK_NAMES)
    xy = np.full((n, 2), np.nan)
    conf = np.full(n, np.nan)
    for j, name in enumerate(LANDMARK_NAMES):
        src = HALPE26_FOR[name]
        if src is None:
            continue
        k = HALPE26.index(src)
        if np.isnan(keypoints_px[k]).any():
            continue
        xy[j] = keypoints_px[k] / (width, height)
        conf[j] = scores[k]
    return FrameObservation(xy_norm=xy, z=np.full(n, np.nan), confidence=conf)


# --- backend --------------------------------------------------------------

class RTMPoseBackend:
    """RTMPose-m Halpe26 + YOLOX-m on ONNX Runtime (CUDA by default)."""

    name = "rtmpose"
    min_confidence = RTMPOSE_MIN_CONFIDENCE
    score_is_visibility = False

    def __init__(self, model_dir=None, device="cuda", profile_dir=None):
        if device not in ("cuda", "cpu"):
            raise ValueError(device)
        self.model_dir = model_dir or default_model_dir()
        self.device = device
        self.profile_dir = profile_dir      # ONNX Runtime per-node profiles (verify_gpu.py)
        self._pose = self._det = None
        self.last_box = None
        self.providers = {}                 # network -> session providers, after open()

    def _paths(self):
        paths = {k: os.path.join(self.model_dir, spec["file"]) for k, spec in MODELS.items()}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise RTMPoseUnavailable(
                "RTMPose model files missing: " + ", ".join(missing) +
                ". Run: python scripts/fetch_rtmpose_models.py")
        return paths

    @staticmethod
    def _ort():
        try:
            import onnxruntime
        except ImportError as e:
            raise RTMPoseUnavailable(
                "onnxruntime-gpu is not installed in this environment. RTMPose runs in the "
                "GPU image: SWINGAI_GPU=1 ./run.sh (see docs/gpu-pose.md).") from e
        return onnxruntime

    def identity(self):
        ort = self._ort()
        paths = self._paths()
        shas = {k: file_sha256(p) for k, p in paths.items()}
        for k, spec in MODELS.items():
            if shas[k] != spec["onnx_sha256"]:
                raise RTMPoseUnavailable(f"{paths[k]} does not match its pinned checksum")
        cfg = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:10]
        from utils.pose_data import runtime_fingerprint
        return (f"rtmpose-m-halpe26-{shas['pose'][:12]}+yolox-m-{shas['detector'][:12]}"
                f"-cfg{cfg}-ort{ort.__version__}-{self.device}-env{runtime_fingerprint()}")

    def open(self):
        ort = self._ort()
        paths = self._paths()
        if self.device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RTMPoseUnavailable(
                    f"ONNX Runtime {ort.__version__} has no CUDA execution provider "
                    f"(available: {ort.get_available_providers()}).")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        sessions, self.providers = {}, {}
        # Each network is created, checked and warmed up on its own, so a
        # failure names the network: ONNX Runtime falls back to the CPU per
        # session, and one network on the GPU says nothing about the other.
        for net in NETWORKS:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            if self.device == "cpu":
                # The two sessions alternate every frame; the idle one's
                # spin-waiting thread pool starves the other (measured 311 vs
                # 56 ms per frame). Same outputs either way.
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            if self.profile_dir:
                opts.enable_profiling = True
                opts.profile_file_prefix = os.path.join(self.profile_dir, f"rtmpose_{net}")
            try:
                sessions[net] = ort.InferenceSession(paths[net], opts, providers=providers)
            except Exception as e:
                raise RTMPoseUnavailable(
                    f"could not create the {net} ONNX Runtime session: {e}") from e
            self.providers[net] = sessions[net].get_providers()
            if self.device == "cuda" and self.providers[net][0] != "CUDAExecutionProvider":
                raise RTMPoseUnavailable(
                    f"CUDA provider not active for the {net} (got {self.providers[net]})")
        self._det, self._pose = sessions["detector"], sessions["pose"]
        # Fail now, not mid-video, if the GPU can't actually run the graphs
        # (e.g. no kernels for this architecture).
        warmup = {"detector": lambda: self._run_det(np.zeros((480, 640, 3), np.uint8)),
                  "pose": lambda: self._run_pose(np.zeros((256, 192, 3), np.float32))}
        for net in NETWORKS:
            try:
                warmup[net]()
            except Exception as e:
                self.close()
                raise RTMPoseUnavailable(f"{net} inference failed on {self.device}: {e}") from e
        self.last_box = None

    def end_profiling(self):
        """{network: profile JSON path}; only when created with `profile_dir`."""
        return {"detector": self._det.end_profiling(), "pose": self._pose.end_profiling()}

    def _run_det(self, bgr):
        img, ratio = letterbox(bgr)
        inp = img.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self._det.run(None, {self._det.get_inputs()[0].name: inp})
        dets, labels = outputs[0][0], outputs[1][0]
        dets = dets[labels == 0].astype(float)
        if dets.size:
            dets[:, :4] /= ratio
        return dets.reshape(-1, 5)

    def _run_pose(self, crop_rgb):
        mean = np.asarray(CONFIG["mean_rgb"], np.float32)
        std = np.asarray(CONFIG["std_rgb"], np.float32)
        inp = ((crop_rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._pose.run(None, {self._pose.get_inputs()[0].name: inp})
        return simcc_x[0], simcc_y[0]

    def crop(self, rgb_frame, box):
        """Pose-network input for one person box: (crop, center, scale)."""
        import cv2
        out_wh = CONFIG["pose_input_wh"]
        center, scale = bbox_center_scale(box)
        scale = fix_aspect(scale, out_wh[0] / out_wh[1])
        crop = cv2.warpAffine(rgb_frame, crop_matrix(center, scale, out_wh), out_wh,
                              flags=cv2.INTER_LINEAR)
        return crop, center, scale

    def _infer(self, net, fn, arg, timestamp_ms):
        try:
            return fn(arg)
        except Exception as e:
            raise RTMPoseInferenceError(
                f"{net} inference failed on {self.device} at {timestamp_ms:.1f} ms: {e}") from e

    def detect(self, rgb_frame, timestamp_ms):
        h, w = rgb_frame.shape[:2]
        boxes = self._infer("detector", self._run_det,
                            np.ascontiguousarray(rgb_frame[..., ::-1]), timestamp_ms)
        box = select_golfer(boxes, self.last_box)
        self.last_box = box
        if box is None:
            return None
        crop, center, scale = self.crop(rgb_frame, box)
        simcc_x, simcc_y = self._infer("pose", self._run_pose, crop, timestamp_ms)
        locs, scores = decode_simcc(simcc_x, simcc_y)
        keypoints = crop_to_frame(locs, center, scale, CONFIG["pose_input_wh"])
        return halpe26_observation(keypoints, scores, w, h)

    def close(self):
        self._pose = self._det = None
