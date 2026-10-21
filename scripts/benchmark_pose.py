"""Benchmark MediaPipe vs RTMPose on the same videos: speed, behaviour, accuracy.

Run in the GPU image (MediaPipe runs on CPU there, with the same pinned
version as the CPU image):

    python scripts/benchmark_pose.py --out docs/benchmarks/pose-backends.json \
        [--labels data/joint_labels/labels.json] [videos...]

Each backend processes every frame of every video from scratch (timed), the
result is written to the normal pose cache (so the app reuses it), and the
cache load is timed separately. Accuracy is computed only against
human-verified labels; without them the accuracy section says so.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import joint_labels as jl  # noqa: E402
from utils import pose_metrics as pm  # noqa: E402
from utils.paths import repo_relative  # noqa: E402
from utils.pose_data import (  # noqa: E402
    cached_pose_path, extract_pose_sequence, get_pose_sequence, make_backend,
    save_pose_sequence,
)

DEFAULT_VIDEOS = [
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4", "dtl"),
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4", "face_on"),
    ("videos/reference/GW_DTL.mp4", "dtl"),
    ("videos/reference/GW_faceon.mp4", "face_on"),
    ("videos/reference/GW_2009_faceon.mp4", "face_on"),
]


class GpuMemorySampler:
    """Peak GPU memory held by compute processes while running."""

    def __init__(self, interval=0.2):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def _sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return sum(int(x) for x in out.split() if x.strip().isdigit())
        except (OSError, subprocess.TimeoutExpired):
            return 0

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, self._sample())
                self._stop.wait(self.interval)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def behaviour(seq):
    scale = pm.body_scale(seq)
    valid = pm.valid_rate(seq)
    spikes = pm.spikes(seq, scale=scale)
    swaps = pm.lr_swaps(seq, scale=scale)
    return {
        "frames": seq.frame_count,
        "detection_rate": pm.detection_rate(seq),
        "valid_rate_mean": float(np.mean(list(valid.values()))),
        "valid_rate": valid,
        "spike_rate_mean": float(np.nanmean([r for _, r in spikes.values()])),
        "spike_p99_median": float(np.nanmedian([p for p, _ in spikes.values()])),
        "spikes": spikes,
        "lr_swaps": swaps,
        "lr_swap_events": int(sum(e for e, _ in swaps.values())),
        "body_scale_px": scale,
        "threshold": seq.default_min_confidence,
    }


def accuracy(seq, key, video_labels):
    if jl.video_scale(video_labels) is None:
        return {"error": "no labelled P1 torso for this video"}
    # Independent (blind) frames only: assisted labels lean toward the model accepted.
    rows = [r for r in jl.score_video(seq, key, video_labels)
            if jl.frame_mode(video_labels, r["frame"]) == "blind"]
    return jl.group_summary(rows, pm.COMMON_JOINTS)


def run(videos, labels, backends=("mediapipe", "rtmpose")):
    results = {"videos": [], "backends": {}}
    for name in backends:
        b = make_backend(name)
        t = time.perf_counter()
        b.open()
        init = time.perf_counter() - t
        b.close()
        results["backends"][name] = {"identity": b.identity(), "init_s": init}
        print(f"{name}: init {init:.2f}s  {b.identity()}", flush=True)

    for path, view in videos:
        key = repo_relative(path)
        entry = {"video": key, "view": view, "backends": {}}
        seqs = {}
        for name in backends:
            b = make_backend(name)
            with GpuMemorySampler() as mem:
                t = time.perf_counter()
                seq = extract_pose_sequence(path, b)
                elapsed = time.perf_counter() - t
            save_pose_sequence(seq, cached_pose_path(path, b))
            t = time.perf_counter()
            cached = get_pose_sequence(path, b, extract_if_missing=False)
            load = time.perf_counter() - t
            assert cached is not None and cached.backend == seq.backend
            seqs[name] = seq
            # MediaPipe runs on the CPU; the sampler would only see the CUDA
            # context the process already holds from RTMPose, so don't report it.
            gpu_mem = mem.peak if name == "rtmpose" else None
            r = {"process_s": elapsed, "fps": seq.frame_count / elapsed,
                 "gpu_mem_peak_mib": gpu_mem, "cache_load_s": load,
                 **behaviour(seq)}
            vl = labels.get(key)
            r["accuracy"] = accuracy(seq, key, vl) if vl else None
            entry["backends"][name] = r
            print(f"  {os.path.basename(path)[:44]:44s} {name:9s} {seq.frame_count:5d} f "
                  f"{elapsed:6.1f}s {r['fps']:6.1f} fps  gpu {gpu_mem if gpu_mem is not None else '-':>5} MiB  "
                  f"cache {load * 1000:5.0f} ms  det {r['detection_rate']:.3f} "
                  f"valid {r['valid_rate_mean']:.3f} spikes {r['spike_rate_mean']:.4f} "
                  f"swaps {r['lr_swap_events']}", flush=True)
        if len(seqs) == 2:
            a, b2 = (seqs[n] for n in backends)
            entry["agreement"] = pm.agreement(a, b2, scale=pm.body_scale(a))
        results["videos"].append(entry)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--view", action="append", default=[],
                    help="view for each extra video, in order (dtl|face_on)")
    ap.add_argument("--labels", default="data/joint_labels/labels.json")
    ap.add_argument("--manifest", default="data/joint_labels/manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    videos = list(DEFAULT_VIDEOS)
    for i, v in enumerate(args.videos):
        videos.append((v, args.view[i] if i < len(args.view) else "unknown"))
    videos = [(p, v) for p, v in videos if os.path.exists(p)]

    labels = {}
    if os.path.exists(args.labels):
        try:
            # The benchmark frames only: sequence frames are labelled for other work.
            labels = jl.benchmark_labels(jl.load_labels(args.labels, args.manifest))
        except jl.LabelError as e:
            sys.exit("invalid labels, refusing to score against them:\n  - "
                     + "\n  - ".join(e.problems))
        print(f"labels: {len(labels)} video(s) with human-verified joints")
    else:
        print(f"labels: none at {args.labels} -- accuracy will not be computed")

    results = run(videos, labels)
    results["labels_file"] = args.labels if labels else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
