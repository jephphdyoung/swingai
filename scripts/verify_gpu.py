"""Prove that BOTH RTMPose networks execute CUDA kernels on the NVIDIA GPU.

Run inside the GPU image (SWINGAI_GPU=1 ./run.sh verify-gpu does this):

    python scripts/verify_gpu.py [video] [--json report.json]

The video defaults to the committed videos/reference/GW_faceon.mp4. A video
that is missing, can't be opened or yields no frames is a failure (exit 2);
no other video is ever substituted.

The YOLOX-m person detector and the RTMPose-m keypoint network are separate
ONNX Runtime sessions. ONNX Runtime places each session -- and each node in
it -- on an execution provider independently, so one network on the GPU says
nothing about the other. Each is checked on its own:

  1. Session providers: CUDA first.
  2. Per-node profile of the backend's *own* sessions on real frames: which
     provider executed every node. At least 90% of node time on CUDA, and no
     compute-heavy op (Conv, MatMul, Gemm) executed on the CPU.
  3. Kernel execution: NVML `utilization.gpu`, defined as the percent of time
     one or more kernels was executing on the GPU, sampled while only this
     network runs back to back. It must rise clearly above both an idle
     baseline and a negative control -- the same network, same loop, on the
     CPU provider. (The onnxruntime-gpu wheel is built without CUPTI, so its
     profile names the provider per node but cannot list kernels; NVML is the
     kernel-level evidence.)
  4. Output parity: CUDA and CPU outputs agree on identical inputs.
  5. Timing, CUDA vs CPU (reported, not judged).

End to end: GPU memory held while the sessions are live, the golfer found in
the sampled frames, and full-pipeline CUDA/CPU keypoint parity.
Exits non-zero, naming the network, if any check fails.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cv2  # noqa: E402

from utils.pose_rtmpose import (  # noqa: E402
    CONFIG, NETWORKS, RTMPoseBackend, RTMPoseUnavailable, crop_to_frame, decode_simcc,
    select_golfer,
)

# Committed to the repo, so the proof never depends on local footage.
DEFAULT_VIDEO = "videos/reference/GW_faceon.mp4"
N_FRAMES = 60
N_PARITY = 20
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"

MIN_CUDA_SHARE = 0.9            # of profiled node time
HEAVY_OPS = {"Conv", "ConvTranspose", "FusedConv", "MatMul", "FusedMatMul", "Gemm"}
MIN_UTIL_RISE = 20.0            # percentage points over max(idle, CPU control)
UTIL_WINDOW_S = 3.0
UTIL_SETTLE_S = 1.0             # NVML averages over a trailing window; drop its lag
# Detector: golfer-box corners in frame pixels. YOLOX sees the frame at <=640
# px, so one letterbox pixel is ~2.3 frame pixels on 1440-wide video.
MAX_BOX_PARITY_PX = 5.0
# Pose: one SimCC bin at this crop scale is ~2 frame pixels.
MAX_KEYPOINT_PARITY_PX = 3.0
PARITY_LIMIT = {"detector": MAX_BOX_PARITY_PX, "pose": MAX_KEYPOINT_PARITY_PX}


# --- judging (pure; unit-tested without a GPU) ------------------------------

def tally_profile(events):
    """ONNX Runtime profile events -> {provider: {"nodes", "us", "ops": {op: n}}}."""
    out = {}
    for e in events:
        if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
            continue
        args = e.get("args", {})
        t = out.setdefault(args.get("provider", "?"), {"nodes": 0, "us": 0, "ops": {}})
        t["nodes"] += 1
        t["us"] += e.get("dur", 0)
        op = args.get("op_name", "?")
        t["ops"][op] = t["ops"].get(op, 0) + 1
    return out


def parse_utilization(text):
    """`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits` lines -> ints."""
    return [int(x) for x in text.split() if x.strip().isdigit()]


def judge_network(net, r):
    """Failure messages for one network's evidence (empty list = passed)."""
    fails = []
    prov = r.get("providers") or []
    if not prov or prov[0] != CUDA:
        fails.append(f"{net}: session providers {prov}, CUDA is not first")

    prof = r.get("profile") or {}
    total = sum(t["us"] for t in prof.values())
    cuda = prof.get(CUDA, {"nodes": 0, "us": 0})
    if not cuda["nodes"]:
        fails.append(f"{net}: no profiled node executed on CUDA")
    elif total and cuda["us"] / total < MIN_CUDA_SHARE:
        fails.append(f"{net}: only {cuda['us'] / total:.0%} of node time ran on CUDA")
    heavy = sorted(op for op in prof.get(CPU, {}).get("ops", {}) if op in HEAVY_OPS)
    if heavy:
        fails.append(f"{net}: compute ops executed on the CPU: {', '.join(heavy)}")

    u = r.get("utilization") or {}
    idle, on_gpu, on_cpu = u.get("idle"), u.get("cuda"), u.get("cpu")
    if None in (idle, on_gpu, on_cpu):
        fails.append(f"{net}: no NVML utilization samples (is nvidia-smi available?)")
    elif on_gpu - max(idle, on_cpu) < MIN_UTIL_RISE:
        fails.append(f"{net}: GPU kernel activity {on_gpu:.0f}% while running is not clearly "
                     f"above idle {idle:.0f}% / CPU-provider control {on_cpu:.0f}%")

    parity = r.get("parity_px")
    if parity is None:
        fails.append(f"{net}: no CUDA/CPU parity measured")
    elif parity > PARITY_LIMIT[net]:
        fails.append(f"{net}: CUDA and CPU outputs differ by {parity:.1f} px "
                     f"(limit {PARITY_LIMIT[net]:.0f})")
    return fails


def judge(reports):
    """Failures across networks. A network with no evidence fails; one network
    passing never stands in for the other."""
    fails = []
    for net in NETWORKS:
        if net not in reports:
            fails.append(f"{net}: not verified (no evidence collected)")
        else:
            fails += judge_network(net, reports[net])
    return fails


# --- measurement -------------------------------------------------------------

class VideoUnusable(Exception):
    """The verification video is missing, can't be opened, or yields no frames."""


def frames(path, n):
    """Up to `n` RGB frames sampled evenly across `path`; raises VideoUnusable."""
    if not os.path.isfile(path):
        raise VideoUnusable(f"{path}: no such file")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoUnusable(f"{path}: OpenCV can't open it as a video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = []
        for idx in np.linspace(0, total - 1, n).astype(int) if total > 0 else []:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not out:
        raise VideoUnusable(f"{path}: no frame could be decoded ({total} reported)")
    return out


def video_line(path, imgs):
    h, w = imgs[0].shape[:2]
    return f"video: {os.path.abspath(path)} ({len(imgs)} frames sampled, {w}x{h})"


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


def utilization_while(fn, seconds=UTIL_WINDOW_S, settle=UTIL_SETTLE_S):
    """Median NVML utilization.gpu while fn() runs back to back, plus ms per call."""
    interval_ms = 100
    try:
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return {"median": None, "samples": 0, "ms_per_run": None}
    t0 = time.monotonic()
    runs = 0
    while time.monotonic() - t0 < seconds:
        fn()
        runs += 1
    elapsed = time.monotonic() - t0
    smi.terminate()
    out, _ = smi.communicate(timeout=10)
    # Samples arrive every interval_ms (output may be buffered, so position,
    # not arrival time, says when each was taken).
    vals = parse_utilization(out)[int(settle * 1000 / interval_ms):]
    return {"median": float(statistics.median(vals)) if vals else None,
            "samples": len(vals), "ms_per_run": elapsed / runs * 1000}


def golfer_box(backend, rgb):
    return select_golfer(backend._run_det(np.ascontiguousarray(rgb[..., ::-1])))


def box_parity(gpu, cpu, imgs):
    """Max golfer-box corner difference, CUDA vs CPU detector, frame pixels."""
    worst = 0.0
    for rgb in imgs:
        a, b = golfer_box(gpu, rgb), golfer_box(cpu, rgb)
        if (a is None) != (b is None):
            return float("inf")
        if a is not None:
            worst = max(worst, float(np.max(np.abs(a[:4] - b[:4]))))
    return worst


def keypoint_parity(gpu, cpu, imgs):
    """Max keypoint difference, CUDA vs CPU pose network on identical crops, frame px."""
    worst, out_wh = 0.0, CONFIG["pose_input_wh"]
    for rgb in imgs:
        box = golfer_box(gpu, rgb)
        if box is None:
            continue
        crop, center, scale = gpu.crop(rgb, box)
        pts = []
        for b in (gpu, cpu):
            locs, _ = decode_simcc(*b._run_pose(crop))
            pts.append(crop_to_frame(locs, center, scale, out_wh))
        worst = max(worst, float(np.nanmax(np.abs(pts[0] - pts[1]))))
    return worst


def verify_network(net, gpu, cpu, profile, rgb, parity_imgs):
    """All evidence for one network. `rgb` must contain the golfer."""
    if net == "detector":
        inp = np.ascontiguousarray(rgb[..., ::-1])
        call = {"cuda": lambda: gpu._run_det(inp), "cpu": lambda: cpu._run_det(inp)}
        parity = box_parity(gpu, cpu, parity_imgs)
    else:
        crop, _, _ = gpu.crop(rgb, golfer_box(gpu, rgb))
        call = {"cuda": lambda: gpu._run_pose(crop), "cpu": lambda: cpu._run_pose(crop)}
        parity = keypoint_parity(gpu, cpu, parity_imgs)
    idle = utilization_while(lambda: time.sleep(0.02))
    on_gpu = utilization_while(call["cuda"])
    on_cpu = utilization_while(call["cpu"])
    return {
        "providers": gpu.providers[net],
        "profile": profile,
        "utilization": {"idle": idle["median"], "cuda": on_gpu["median"],
                        "cpu": on_cpu["median"],
                        "samples": [idle["samples"], on_gpu["samples"], on_cpu["samples"]]},
        "ms": {"cuda": on_gpu["ms_per_run"], "cpu": on_cpu["ms_per_run"]},
        "parity_px": parity,
    }


def describe(net, r):
    prof = r["profile"]
    total = sum(t["us"] for t in prof.values()) or 1
    nodes = {p.replace("ExecutionProvider", ""): t["nodes"] for p, t in prof.items()}
    cpu_ops = prof.get(CPU, {}).get("ops", {})
    u, ms = r["utilization"], r["ms"]
    return "\n".join([
        f"[{net}]",
        f"  session providers:      {r['providers']}",
        f"  profiled node runs:     {nodes}; "
        f"{prof.get(CUDA, {'us': 0})['us'] / total:.1%} of node time on CUDA",
        f"  ops executed on CPU:    {cpu_ops or 'none'}",
        f"  GPU kernel activity:    idle {u['idle']}%, CPU-provider run {u['cpu']}%, "
        f"CUDA run {u['cuda']}%  (NVML samples {u['samples']})",
        f"  per call:               CUDA {ms['cuda']:.1f} ms, CPU {ms['cpu']:.1f} ms "
        f"({ms['cpu'] / ms['cuda']:.1f}x)",
        f"  CUDA vs CPU output:     max {r['parity_px']:.2f} px "
        f"(limit {PARITY_LIMIT[net]:.0f})",
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="?", default=os.path.join(REPO, DEFAULT_VIDEO))
    ap.add_argument("--json", help="also write the evidence as JSON here")
    args = ap.parse_args(argv)
    video = os.path.abspath(args.video)
    # Checked before any GPU work, and never replaced by another video.
    try:
        imgs = frames(video, N_FRAMES)
    except VideoUnusable as e:
        print(f"FAIL: verification video unusable: {e}")
        return 2
    print(video_line(video, imgs))
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}; available providers {ort.get_available_providers()}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {smi.stdout.strip() or smi.stderr.strip()}")

    failures = []
    profile_dir = tempfile.mkdtemp(prefix="rtmpose_profile_")
    gpu = RTMPoseBackend(device="cuda", profile_dir=profile_dir)
    t = time.perf_counter()
    try:
        gpu.open()
    except RTMPoseUnavailable as e:
        print(f"FAIL: {e}")
        return 1
    print(f"CUDA sessions: {gpu.providers}; init {time.perf_counter() - t:.2f}s")

    run(gpu, imgs[:5])                                  # warm-up
    gpu_obs, gpu_dt = run(gpu, imgs)
    mem, raw = gpu_memory_of_this_process()
    print(f"nvidia-smi compute apps while sessions live: {raw!r}")
    if not mem:
        failures.append("no GPU memory held by the inference process")
    # Profiles of the sessions that just processed the real frames.
    profiles = {}
    for net, path in gpu.end_profiling().items():
        with open(path) as f:
            profiles[net] = tally_profile(json.load(f))

    cpu = RTMPoseBackend(device="cpu")
    cpu.open()
    with_golfer = [img for img, o in zip(imgs, gpu_obs) if o is not None]
    if not with_golfer:
        print("FAIL: no golfer detected in any sampled frame; nothing to verify the pose "
              "network on")
        return 1
    rgb = with_golfer[len(with_golfer) // 2]
    parity_imgs = imgs[:N_PARITY]
    reports = {}
    for net in NETWORKS:
        reports[net] = verify_network(net, gpu, cpu, profiles[net], rgb, parity_imgs)
        print(describe(net, reports[net]))
    failures += judge(reports)

    run(cpu, imgs[:2])
    cpu_obs, cpu_dt = run(cpu, parity_imgs)
    print(f"end to end per frame (detector + pose): CUDA {gpu_dt * 1000:.1f} ms, "
          f"CPU {cpu_dt * 1000:.1f} ms ({cpu_dt / gpu_dt:.1f}x)")
    diffs = []
    h, w = imgs[0].shape[:2]
    for g, c in zip(gpu_obs[:N_PARITY], cpu_obs):
        if g is None or c is None:
            if (g is None) != (c is None):
                failures.append("CUDA and CPU disagree on whether a person was found")
            continue
        diffs.append(np.nanmax(np.abs((g.xy_norm - c.xy_norm) * (w, h))))
    if diffs:
        print(f"end to end CUDA vs CPU keypoints: max {max(diffs):.2f} px, "
              f"median {np.median(diffs):.2f} px over {len(diffs)} frames")
        if max(diffs) > MAX_KEYPOINT_PARITY_PX:
            failures.append(f"end to end CUDA/CPU keypoints differ by {max(diffs):.1f} px")
    found = sum(o is not None for o in gpu_obs)
    print(f"golfer detected in {found}/{len(gpu_obs)} sampled frames")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"video": video, "onnxruntime": ort.__version__, "gpu": smi.stdout.strip(),
                       "networks": reports, "failures": failures}, f, indent=1)

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: both networks (YOLOX detector and RTMPose) executed CUDA kernels on the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
