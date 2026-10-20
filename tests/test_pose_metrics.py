"""Tests for the backend-agnostic pose quality metrics."""

import numpy as np
import pytest

from utils import pose_metrics as pm
from utils.landmarks import LANDMARK_INDEX, LANDMARK_NAMES
from utils.pose_data import PoseSequence


def seq_from(points_per_frame, conf=0.9, detected=None, min_conf=0.5):
    """PoseSequence from [{name: (x, y)}] frames."""
    n, j = len(points_per_frame), len(LANDMARK_NAMES)
    xy = np.full((n, j, 2), np.nan)
    c = np.full((n, j), np.nan)
    for f, pts in enumerate(points_per_frame):
        for name, p in pts.items():
            xy[f, LANDMARK_INDEX[name]] = p
            c[f, LANDMARK_INDEX[name]] = conf
    return PoseSequence(video_path="v", backend="t", fps=30.0, width=640, height=480,
                        timestamps_ms=np.arange(n) * 33.3, xy=xy, z=np.full((n, j), np.nan),
                        confidence=c,
                        detected=np.ones(n, bool) if detected is None else np.asarray(detected),
                        default_min_confidence=min_conf)


def body(dx=0.0, dy=0.0, swap_wrists=False):
    pts = {
        "nose": (100, 40), "left_ear": (105, 35), "right_ear": (95, 35),
        "left_shoulder": (120, 100), "right_shoulder": (80, 100),
        "left_elbow": (125, 150), "right_elbow": (75, 150),
        "left_wrist": (130, 200), "right_wrist": (70, 200),
        "left_hip": (115, 200), "right_hip": (85, 200),
        "left_knee": (115, 300), "right_knee": (85, 300),
        "left_ankle": (115, 400), "right_ankle": (85, 400),
    }
    if swap_wrists:
        pts["left_wrist"], pts["right_wrist"] = pts["right_wrist"], pts["left_wrist"]
    return {k: (x + dx, y + dy) for k, (x, y) in pts.items()}


def test_torso_and_scale():
    s = seq_from([body(), body(dx=10)])
    assert pm.torso_lengths(s) == pytest.approx([100, 100])
    assert pm.body_scale(s) == pytest.approx(100)


def test_detection_and_valid_rates_use_the_backend_threshold():
    frames = [body(), body(), {}]
    s = seq_from(frames, conf=0.4, detected=[True, True, False], min_conf=0.3)
    assert pm.detection_rate(s) == pytest.approx(2 / 3)
    assert pm.valid_rate(s)["left_wrist"] == pytest.approx(2 / 3)
    s.default_min_confidence = 0.5
    assert pm.valid_rate(s)["left_wrist"] == 0.0


def test_smooth_motion_has_no_spikes_but_a_jump_does():
    smooth = seq_from([body(dx=i * 5) for i in range(10)])
    assert all(rate == 0 for _, rate in pm.spikes(smooth).values())
    frames = [body(dx=i * 5) for i in range(10)]
    frames[5]["left_wrist"] = (frames[5]["left_wrist"][0] + 60, frames[5]["left_wrist"][1])
    jumpy = seq_from(frames)
    p99, rate = pm.spikes(jumpy)["left_wrist"]
    assert rate > 0 and p99 > pm.SPIKE_THRESHOLD
    assert pm.spikes(jumpy)["right_wrist"][1] == 0


def test_left_right_swap_is_counted():
    frames = [body(dx=i) for i in range(6)]
    frames[3] = body(dx=3, swap_wrists=True)
    swaps = pm.lr_swaps(seq_from(frames))
    assert swaps["wrist"][0] == 2          # into the swap and back out
    assert swaps["knee"][0] == 0


def test_agreement_is_zero_for_identical_and_scaled_otherwise():
    a = seq_from([body() for _ in range(3)])
    b = seq_from([body(dx=20) for _ in range(3)])
    assert pm.agreement(a, a)["nose"] == 0
    assert pm.agreement(a, b)["nose"] == pytest.approx(0.2)


def test_label_errors_and_misses():
    s = seq_from([body(), body()])
    s.confidence[1, LANDMARK_INDEX["left_knee"]] = 0.1     # below threshold -> miss
    labels = {"0": {"nose": {"x": 100, "y": 50, "visible": True}},
              "1": {"left_knee": {"x": 115, "y": 300, "visible": False},
                    "left_thumb": {"x": 0, "y": 0}}}       # not a common joint: ignored
    rows = pm.label_errors(s, labels, scale=100.0)
    by = {(r["frame"], r["joint"]): r for r in rows}
    assert by[(0, "nose")]["error"] == pytest.approx(0.1)
    assert by[(1, "left_knee")]["error"] is None
    assert not by[(1, "left_knee")]["visible"]
    assert len(rows) == 2
    summary = pm.summarize_errors(rows)
    assert summary["missing"] == pytest.approx(0.5)
    assert summary["pck@0.1"] == pytest.approx(0.5)
    assert summary["pck@0.05"] == 0.0


def test_label_torso():
    lab = {n: {"x": x, "y": y} for n, (x, y) in body().items()}
    assert pm.label_torso(lab) == pytest.approx(100)
    del lab["left_hip"]
    assert pm.label_torso(lab) is None
