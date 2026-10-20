"""Prove that RTMPose actually executes on the NVIDIA GPU -- not just that CUDA exists.

Run inside the GPU image (SWINGAI_GPU=1 ./run.sh verify-gpu does this):

    python scripts/verify_gpu.py [video]

Evidence collected:
  1. ONNX Runtime per-node profile: which execution provider ran each operator.
  2. nvidia-smi: GPU memory held by *this* process while the sessions are live.
  3. Timing on real frames, CUDA vs CPU.
  4. Output parity: CUDA keypoints match CPU keypoints on the same frames.
Exits non-zero if any check fails.
"""

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from utils.pose_rtmpose import RTMPoseBackend, RTMPoseUnavailable  # noqa: E402

DEFAULT_VIDEO = "videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4"
N_FRAMES = 60


def frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def run(backend, imgs):
    t = time.perf_counter()
    obs = [backend.detect(f, 0) for f in imgs]
    return obs, (time.perf_counter() - t) / len(imgs)


def gpu_memory_of_this_process():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, "nvidia-smi unavailable"
    rows = [r.split(",") for r in out.strip().splitlines() if r.strip()]
    # Inside a container the PID namespace differs; with one inference process
    # running, any compute app is ours. Report all.
    return sum(int(r[1]) for r in rows) if rows else 0, out.strip() or "(no compute apps)"


def profile_providers(backend, img):
    """Re-create the pose session with profiling on and tally node providers."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = "/tmp/rtmpose_profile"
    opts.log_severity_level = 3
    path = backend._pose._model_path if hasattr(backend._pose, "_model_path") else \
        os.path.join(backend.model_dir, "rtmpose-m_body7-halpe26_256x192.onnx")
    sess = ort.InferenceSession(path, opts, providers=["CUDAExecutionProvider",
                                                       "CPUExecutionProvider"])
    inp = np.random.rand(1, 3, 256, 192).astype(np.float32)
    for _ in range(3):
        sess.run(None, {sess.get_inputs()[0].name: inp})
    prof = sess.end_profiling()
    events = json.load(open(prof))
    tally, dur = {}, {}
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            p = e.get("args", {}).get("provider", "?")
            tally[p] = tally.get(p, 0) + 1
            dur[p] = dur.get(p, 0) + e.get("dur", 0)
    return tally, dur


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not os.path.exists(video):
        video = sorted(glob.glob("videos/**/*.mp4", recursive=True))[0]
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}; available providers {ort.get_available_providers()}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {smi.stdout.strip() or smi.stderr.strip()}")

    failures = []
    gpu = RTMPoseBackend(device="cuda")
    t = time.perf_counter()
    try:
        gpu.open()
    except RTMPoseUnavailable as e:
        print(f"FAIL: {e}")
        return 1
    print(f"CUDA sessions: providers {gpu.providers}; init {time.perf_counter() - t:.2f}s")

    imgs = frames(video, N_FRAMES)
    print(f"video: {video} ({len(imgs)} frames sampled, {imgs[0].shape[1]}x{imgs[0].shape[0]})")
    run(gpu, imgs[:5])                                  # warm-up
    gpu_obs, gpu_dt = run(gpu, imgs)
    mem, raw = gpu_memory_of_this_process()
    print(f"nvidia-smi compute apps while sessions live: {raw!r}")
    if not mem:
        failures.append("no GPU memory held by the inference process")

    tally, dur = profile_providers(gpu, imgs[0])
    print(f"pose graph node executions by provider: {tally}")
    total = sum(dur.values()) or 1
    cuda_share = dur.get("CUDAExecutionProvider", 0) / total
    print(f"share of node time on CUDA: {cuda_share:.1%}")
    if cuda_share < 0.9:
        failures.append(f"only {cuda_share:.0%} of node time ran on CUDA")

    cpu = RTMPoseBackend(device="cpu")
    cpu.open()
    run(cpu, imgs[:2])
    cpu_obs, cpu_dt = run(cpu, imgs[:20])
    print(f"per-frame (detector + pose): CUDA {gpu_dt * 1000:.1f} ms, CPU {cpu_dt * 1000:.1f} ms "
          f"({cpu_dt / gpu_dt:.1f}x)")

    diffs = []
    for g, c in zip(gpu_obs[:20], cpu_obs):
        if g is None or c is None:
            if (g is None) != (c is None):
                failures.append("CUDA and CPU disagree on whether a person was found")
            continue
        h, w = imgs[0].shape[:2]
        d = np.abs((g.xy_norm - c.xy_norm) * (w, h))
        diffs.append(np.nanmax(d))
    if diffs:
        print(f"CUDA vs CPU keypoint difference: max {max(diffs):.2f} px, "
              f"median {np.median(diffs):.2f} px over {len(diffs)} frames")
        if max(diffs) > 3.0:
            failures.append(f"CUDA/CPU keypoints differ by {max(diffs):.1f} px")
    found = sum(o is not None for o in gpu_obs)
    print(f"golfer detected in {found}/{len(gpu_obs)} sampled frames")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: RTMPose inference executed on the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
