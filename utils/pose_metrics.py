"""Backend-agnostic quality metrics for PoseSequences.

Two kinds, kept apart on purpose:

- **Ground-truth accuracy** (`label_errors`): distance to human-verified joint
  locations, in units of the P1 torso length from the same labels. This is
  the only measure of correctness.
- **Behaviour without ground truth** (detection/valid rates, temporal spikes,
  left/right swaps, cross-backend agreement): useful for spotting failures,
  but a smooth, confident, mutually-agreeing track can still be wrong.
"""

import numpy as np

from utils.landmarks import LANDMARK_INDEX

# Joints both backends produce (RTMPose has no hand points; feet differ).
COMMON_JOINTS = (
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
SIDED_PAIRS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")

# A point that sits this far (torso lengths) off the line between its
# neighbouring frames is a spike, not motion.
SPIKE_THRESHOLD = 0.2
# Left/right swap: relabelling the pair explains the motion at least this
# much better, and the pair moved at least SWAP_MIN_MOTION torso lengths.
SWAP_RATIO = 0.5
SWAP_MIN_MOTION = 0.1


def valid_mask(seq, names=COMMON_JOINTS, min_confidence=None):
    """(N, J) bool: landmark validly observed under the backend's threshold."""
    thr = seq.default_min_confidence if min_confidence is None else min_confidence
    idx = [LANDMARK_INDEX[n] for n in names]
    xy_ok = ~np.isnan(seq.xy[:, idx]).any(axis=2)
    conf = seq.confidence[:, idx]
    conf_ok = np.isnan(conf) | (conf >= thr)
    return xy_ok & conf_ok & seq.detected[:, None]


def torso_lengths(seq, min_confidence=None):
    """(N,) shoulder-centre to hip-centre distance; NaN where not observed."""
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    ok = valid_mask(seq, names, min_confidence).all(axis=1)
    i = [LANDMARK_INDEX[n] for n in names]
    s = (seq.xy[:, i[0]] + seq.xy[:, i[1]]) / 2
    h = (seq.xy[:, i[2]] + seq.xy[:, i[3]]) / 2
    out = np.linalg.norm(s - h, axis=1)
    out[~ok] = np.nan
    return out


def body_scale(seq, min_confidence=None):
    """Median torso length over the video (for normalising behaviour metrics)."""
    t = torso_lengths(seq, min_confidence)
    return float(np.nanmedian(t)) if np.isfinite(t).any() else float("nan")


def detection_rate(seq):
    return float(np.mean(seq.detected)) if seq.frame_count else 0.0


def valid_rate(seq, names=COMMON_JOINTS, min_confidence=None):
    """{joint: fraction of frames validly observed}."""
    m = valid_mask(seq, names, min_confidence)
    return {n: float(m[:, j].mean()) for j, n in enumerate(names)}


def spikes(seq, names=COMMON_JOINTS, min_confidence=None, scale=None):
    """Per joint: distance of each point from the midpoint of its neighbours.

    Returns {joint: (p99, rate above SPIKE_THRESHOLD)} in torso lengths, over
    frames where the joint is valid in t-1, t and t+1.
    """
    scale = body_scale(seq, min_confidence) if scale is None else scale
    m = valid_mask(seq, names, min_confidence)
    out = {}
    for j, n in enumerate(names):
        xy = seq.xy[:, LANDMARK_INDEX[n]]
        ok = m[:-2, j] & m[1:-1, j] & m[2:, j]
        d = np.linalg.norm(xy[1:-1] - (xy[:-2] + xy[2:]) / 2, axis=1)[ok] / scale
        out[n] = (float(np.percentile(d, 99)) if d.size else float("nan"),
                  float(np.mean(d > SPIKE_THRESHOLD)) if d.size else float("nan"))
    return out


def lr_swaps(seq, min_confidence=None, scale=None):
    """Count frame-to-frame left/right label swaps per joint pair.

    Returns {pair: (swap events, frame pairs checked)}.
    """
    scale = body_scale(seq, min_confidence) if scale is None else scale
    out = {}
    for part in SIDED_PAIRS:
        names = (f"left_{part}", f"right_{part}")
        m = valid_mask(seq, names, min_confidence).all(axis=1)
        L = seq.xy[:, LANDMARK_INDEX[names[0]]]
        R = seq.xy[:, LANDMARK_INDEX[names[1]]]
        ok = m[:-1] & m[1:]
        keep = (np.linalg.norm(L[1:] - L[:-1], axis=1) + np.linalg.norm(R[1:] - R[:-1], axis=1))
        swap = (np.linalg.norm(L[1:] - R[:-1], axis=1) + np.linalg.norm(R[1:] - L[:-1], axis=1))
        event = ok & (keep > SWAP_MIN_MOTION * scale) & (swap < SWAP_RATIO * keep)
        out[part] = (int(event.sum()), int(ok.sum()))
    return out


def agreement(a, b, names=COMMON_JOINTS, scale=None):
    """{joint: median distance between two backends' points} in torso lengths.

    Descriptive only: two models agreeing says nothing about which is right.
    """
    if a.frame_count != b.frame_count:
        raise ValueError("sequences cover different frame counts")
    scale = body_scale(a) if scale is None else scale
    ma, mb = valid_mask(a, names), valid_mask(b, names)
    out = {}
    for j, n in enumerate(names):
        k = LANDMARK_INDEX[n]
        ok = ma[:, j] & mb[:, j]
        d = np.linalg.norm(a.xy[ok, k] - b.xy[ok, k], axis=1) / scale
        out[n] = float(np.median(d)) if d.size else float("nan")
    return out


def label_errors(seq, labels, scale):
    """Errors against human labels for one video.

    labels: {frame: {joint: {"x", "y", "visible": bool}}}; joints a labeller
    could not place are simply absent. scale: P1 torso length from the labels.
    Returns a list of dicts (frame, joint, visible, error | None) where error
    is None when the backend gave no valid point (counted as a miss).
    """
    rows = []
    for frame, joints in labels.items():
        frame = int(frame)
        for name, lab in joints.items():
            if name not in COMMON_JOINTS:
                continue
            pred = seq.point(frame, name) if frame < seq.frame_count else None
            err = None
            if pred is not None:
                err = float(np.hypot(pred[0] - lab["x"], pred[1] - lab["y"]) / scale)
            rows.append({"frame": frame, "joint": name, "visible": bool(lab.get("visible", True)),
                         "error": err})
    return rows


def label_torso(frame_labels):
    """Torso length from one labelled frame, or None if a torso point is missing."""
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(n in frame_labels for n in need):
        return None
    p = {n: np.array([frame_labels[n]["x"], frame_labels[n]["y"]], float) for n in need}
    t = np.linalg.norm((p["left_shoulder"] + p["right_shoulder"]) / 2
                       - (p["left_hip"] + p["right_hip"]) / 2)
    return float(t) if t > 0 else None


def summarize_errors(rows, thresholds=(0.05, 0.1)):
    """Mean/median error, PCK at the thresholds, and miss rate."""
    if not rows:
        return {"n": 0}
    errs = np.array([r["error"] for r in rows if r["error"] is not None])
    out = {"n": len(rows), "missing": 1 - len(errs) / len(rows)}
    if errs.size:
        out["mean"] = float(errs.mean())
        out["median"] = float(np.median(errs))
    for t in thresholds:
        # A miss counts as a failure for PCK.
        out[f"pck@{t}"] = float(np.sum(errs <= t) / len(rows))
    return out
