"""GPU integration tests: real RTMPose inference on the NVIDIA GPU.

Skipped unless onnxruntime has a working CUDA provider and the models are
present -- i.e. they run only in the GPU image with the GPU attached:

    SWINGAI_GPU=1 ./run.sh   (or)
    podman --cdi-spec-dir ~/.local/share/swingai/cdi run --rm --device nvidia.com/gpu=all \\
      --security-opt label=disable -v "$PWD":/app swingai-gpu python -m pytest tests/test_gpu_rtmpose.py
"""

import importlib.util
import json
import os

import cv2
import numpy as np
import pytest

from utils import pose_rtmpose as rt
from utils.landmarks import LANDMARK_INDEX


def _gpu_backend():
    try:
        b = rt.RTMPoseBackend(device="cuda")
        b.open()
        return b
    except rt.RTMPoseUnavailable as e:
        pytest.skip(f"no GPU RTMPose here: {e}")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = "videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4"


@pytest.fixture(scope="module")
def backend():
    b = _gpu_backend()
    yield b
    b.close()


def _frame(path, idx):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_both_networks_run_on_cuda(backend):
    assert set(backend.providers) == set(rt.NETWORKS)
    for net, providers in backend.providers.items():
        assert providers[0] == "CUDAExecutionProvider", net
    assert "-cuda-env" in backend.identity()


def _verify_gpu():
    spec = importlib.util.spec_from_file_location(
        "verify_gpu", os.path.join(REPO, "scripts", "verify_gpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """verify_gpu.py's per-network evidence, gathered on real frames."""
    vg = _verify_gpu()
    gpu = rt.RTMPoseBackend(device="cuda", profile_dir=str(tmp_path_factory.mktemp("prof")))
    try:
        gpu.open()
    except rt.RTMPoseUnavailable as e:
        pytest.skip(f"no GPU RTMPose here: {e}")
    imgs = [_frame(VIDEO, i) for i in (395, 652, 730)]
    for img in imgs:
        assert gpu.detect(img, 0) is not None
    profiles = {}
    for net, path in gpu.end_profiling().items():
        with open(path) as f:
            profiles[net] = vg.tally_profile(json.load(f))
    cpu = rt.RTMPoseBackend(device="cpu")
    cpu.open()
    reports = {net: vg.verify_network(net, gpu, cpu, profiles[net], imgs[1], imgs)
               for net in rt.NETWORKS}
    yield vg, reports
    gpu.close()
    cpu.close()


@pytest.mark.parametrize("net", rt.NETWORKS)
def test_each_network_executes_cuda_kernels(evidence, net):
    vg, reports = evidence
    r = reports[net]
    assert "Conv" in r["profile"][vg.CUDA]["ops"]          # the real work, on CUDA
    assert vg.judge_network(net, r) == [], vg.describe(net, r)


def test_verification_covers_both_networks(evidence):
    vg, reports = evidence
    assert vg.judge(reports) == []


def test_real_frame_keypoints_lie_on_the_golfer(backend):
    rgb = _frame(VIDEO, 395)                       # address
    obs = backend.detect(rgb, 0)
    assert obs is not None
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = backend.last_box[:4]
    pts = obs.xy_norm * (w, h)
    for name in ("nose", "left_shoulder", "right_hip", "left_ankle"):
        x, y = pts[LANDMARK_INDEX[name]]
        assert x0 - 5 <= x <= x1 + 5 and y0 - 5 <= y <= y1 + 5, name
    # head above shoulders above hips above ankles, in image coordinates
    y = lambda n: pts[LANDMARK_INDEX[n]][1]  # noqa: E731
    assert y("nose") < y("left_shoulder") < y("left_hip") < y("left_ankle")
    assert np.isnan(pts[LANDMARK_INDEX["left_thumb"]]).all()


def test_cuda_matches_cpu(backend):
    rgb = _frame(VIDEO, 652)
    cpu = rt.RTMPoseBackend(device="cpu")
    cpu.open()
    a, b = backend.detect(rgb, 0), cpu.detect(rgb, 0)
    h, w = rgb.shape[:2]
    diff = np.nanmax(np.abs((a.xy_norm - b.xy_norm) * (w, h)))
    assert diff <= 3.0      # one SimCC bin at this crop scale is ~2 px
