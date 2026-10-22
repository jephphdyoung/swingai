"""Render benchmark_pose.py JSON as Markdown tables.

    python scripts/benchmark_report.py docs/benchmarks/pose-backends-2026-10-20.json
"""

import json
import os
import sys

BK = {"mediapipe": "MediaPipe", "rtmpose": "RTMPose"}
DTL_LEAD = ("left_elbow", "left_wrist", "left_knee")


def short(video):
    return os.path.basename(video).replace("Jeff Young - ", "").replace(".mp4", "")


def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def render(d):
    out = []
    out.append("### Accuracy against human-verified joints\n")
    labelled = [v for v in d["videos"]
                if any(r.get("accuracy") for r in v["backends"].values())]
    if not labelled:
        out.append("**Not computed: no human-verified joint labels exist yet.** "
                   "See *Ground truth* above.\n")
    else:
        out.append("| Video | View | Model | Joints | Missing | Mean err | Median | PCK@0.05 | "
                   "PCK@0.1 | Visible PCK@0.1 | Occluded PCK@0.1 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for v in labelled:
            for name, r in v["backends"].items():
                a = r["accuracy"]
                if "all" not in a:
                    out.append(f"| {short(v['video'])} | {v['view']} | {BK[name]} | "
                               f"{a.get('error')} |||||||||")
                    continue
                al, vi, oc = a["all"], a["visible"], a["occluded"]
                out.append(
                    f"| {short(v['video'])} | {v['view']} | {BK[name]} | {al['n']} | "
                    f"{al['missing']:.1%} | {fmt(al.get('mean'), '.3f')} | "
                    f"{fmt(al.get('median'), '.3f')} | {al['pck@0.05']:.1%} | "
                    f"{al['pck@0.1']:.1%} | {fmt(vi.get('pck@0.1'), '.1%')} | "
                    f"{fmt(oc.get('pck@0.1'), '.1%')} |")
        out.append("\nErrors in P1 torso lengths (from the labels). PCK counts a missing "
                   "point as a failure.\n")

    out.append("### Performance\n")
    init = d["backends"]
    out.append(f"Model initialisation: MediaPipe {init['mediapipe']['init_s']:.2f} s, "
               f"RTMPose {init['rtmpose']['init_s']:.2f} s (session creation + warm-up).\n")
    out.append("| Video | Frames | Model | Process (s) | fps | GPU peak (MiB) | Cache load (ms) |")
    out.append("|---|---|---|---|---|---|---|")
    for v in d["videos"]:
        for name, r in v["backends"].items():
            out.append(f"| {short(v['video'])} | {r['frames']} | {BK[name]} | "
                       f"{r['process_s']:.1f} | {r['fps']:.1f} | "
                       f"{fmt(r['gpu_mem_peak_mib'], 'd')} | {r['cache_load_s'] * 1000:.0f} |")
    out.append("\nMediaPipe runs on the CPU, so it has no GPU figure. RTMPose's figure is the whole "
               "process (CUDA context + both sessions).\n")

    out.append("### Behaviour without ground truth\n")
    out.append("| Video | View | Model | Detected | Valid (mean) | DTL lead elbow/wrist/knee valid | "
               "Spike rate | Spike p99 | L/R swaps |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for v in d["videos"]:
        for name, r in v["backends"].items():
            lead = "—"
            if v["view"] == "dtl":
                lead = " / ".join(f"{r['valid_rate'][j]:.0%}" for j in DTL_LEAD)
            out.append(f"| {short(v['video'])} | {v['view']} | {BK[name]} | "
                       f"{r['detection_rate']:.1%} | {r['valid_rate_mean']:.1%} | {lead} | "
                       f"{r['spike_rate_mean']:.2%} | {r['spike_p99_median']:.3f} | "
                       f"{r['lr_swap_events']} |")
    out.append("\nValid = landmark above the backend's own threshold (MediaPipe 0.5, RTMPose 0.3); "
               "the two are not comparable as confidence. Spike = a point more than 0.2 torso "
               "lengths off the midpoint of its neighbouring frames; p99 is the median over "
               "joints of each joint's 99th-percentile deviation. L/R swap = relabelling the "
               "left/right pair explains the frame-to-frame motion at least twice as well.\n")

    out.append("### Cross-model agreement (not accuracy)\n")
    joints = ("nose", "left_shoulder", "left_elbow", "left_wrist", "right_wrist",
              "left_hip", "left_knee", "left_ankle")
    out.append("| Video | View | " + " | ".join(j.replace("_", " ") for j in joints) + " |")
    out.append("|---|---|" + "---|" * len(joints))
    for v in d["videos"]:
        a = v.get("agreement", {})
        out.append(f"| {short(v['video'])} | {v['view']} | "
                   + " | ".join(fmt(a.get(j), ".3f") for j in joints) + " |")
    out.append("\nMedian distance between the two models' points, in torso lengths, on frames "
               "where both are valid.\n")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(json.load(open(sys.argv[1]))))
