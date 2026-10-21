"""Score MediaPipe and RTMPose against human-verified joint labels, per view.

Run in the GPU image (RTMPose needs it; MediaPipe's cache is per environment):

    python scripts/evaluate_joint_labels.py \
        --out docs/benchmarks/joint-accuracy-<date>.json [--md ...md] [--extract]

Input is `data/joint_labels/labels.json`, exported from labeler.html by the
person who placed the joints. Only the benchmark frames' 15 body joints are
scored; feet, hands/club points and sequence frames are validated but not
part of this benchmark. Without it, or if it fails validation, this
stops and says why -- it never scores against anything else.

Steps:
  1. Validate labels.json against data/joint_labels/manifest.json.
  2. Check every labelled frame decodes, sequentially as the models read it,
     to exactly the image that was labelled (SHA-1 in the manifest).
  3. Load both models' pose sequences from the pose cache (or extract them
     with --extract).
  4. Report, separately for DTL and face-on: normalized errors, frame and
     joint misses, PCK@0.05/0.1/0.2, visible vs occluded, per joint, and the
     RTMPose - MediaPipe PCK difference with a 95% bootstrap interval.
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import joint_labels as jl  # noqa: E402
from utils.pose_data import BACKEND_LABELS, get_pose_sequence, make_backend  # noqa: E402

BACKENDS = ("mediapipe", "rtmpose")
VIEW_NAMES = {"dtl": "Down the line", "face_on": "Face-on"}


def frame_alignment_problems(labels, manifest):
    """Labelled frames whose decoded pixels here differ from the labelled image."""
    problems = []
    for video, v in labels.items():
        if not v["frames"]:
            continue
        if not os.path.exists(video):
            problems.append(f"{video}: video not found")
            continue
        decoded = jl.decode_frames(video, v["frames"])
        for frame in sorted(v["frames"]):
            expected = manifest["frames"][(video, frame)].get("image_sha1")
            if frame not in decoded:
                problems.append(f"{video}: ends before frame {frame}")
            elif expected is None:
                problems.append(f"{video} frame {frame}: manifest has no image digest; "
                                "regenerate it with scripts/make_joint_label_set.py")
            elif jl.frame_digest(decoded[frame]) != expected:
                problems.append(f"{video} frame {frame}: decodes to a different image here "
                                "than the one that was labelled")
    return problems


def load_sequences(labels, extract):
    sequences, identities, missing = {}, {}, []
    for name in BACKENDS:
        backend = make_backend(name)
        identities[name] = backend.identity()       # RTMPoseUnavailable propagates
        sequences[name] = {}
        for video, v in labels.items():
            if jl.video_scale(v) is None:
                continue
            seq = get_pose_sequence(video, backend, extract_if_missing=extract)
            if seq is None:
                missing.append(f"{name}: {video}")
            else:
                sequences[name][video] = seq
    return sequences, identities, missing


def pct(v):
    return "—" if v is None else f"{v:.1%}"


def num(v):
    return "—" if v is None else f"{v:.3f}"


def render(result):
    rep, cov = result["report"], result["coverage"]
    out = [f"Labels: `{result['labels_file']}`, verified by {result['verified_by']}. "
           f"{cov['frames_labelled']}/{cov['frames_expected']} frames, "
           f"{cov['joints_labelled']}/{cov['joints_possible']} joints placed.\n"]
    if cov.get("frames_confirmed_from_earlier_export"):
        out[0] = out[0].rstrip("\n") + (
            f" {cov['frames_confirmed_from_earlier_export']} of these frames include points "
            "carried over from an earlier export and kept after checking on the same image.\n")
    if cov.get("frames_outside_benchmark"):
        out.append(f"{cov['frames_outside_benchmark']} labelled frames outside the benchmark "
                   "set (sequences) are not scored here.\n")
    for video, why in cov["unscorable"].items():
        out.append(f"- Not scored: `{video}`: {why}")
    if cov["unscorable"]:
        out.append("")
    for view, by_backend in rep["by_view"].items():
        out.append(f"### {VIEW_NAMES[view]}\n")
        out.append("| Model | Frames (missed) | Joints | Joint misses | Mean err | Median err | "
                   "PCK@0.05 | PCK@0.1 | PCK@0.2 | Visible PCK@0.1 | Occluded: n, answered, "
                   "PCK@0.1 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for b, g in by_backend.items():
            a, vi, oc = g["all"], g["visible"], g["occluded"]
            occ = (f"{oc['n']}, {pct(oc.get('answered'))}, {pct(oc.get('pck@0.1'))}"
                   if oc.get("n") else "0")
            out.append(f"| {BACKEND_LABELS[b]} | {a['frames']} ({a['frames_missed']}) | {a['n']} | "
                       f"{a['joints_missed']} ({pct(a['missing'])}) | {num(a.get('mean'))} | "
                       f"{num(a.get('median'))} | {pct(a['pck@0.05'])} | {pct(a['pck@0.1'])} | "
                       f"{pct(a['pck@0.2'])} | {pct(vi.get('pck@0.1'))} | {occ} |")
        for pair, by_t in rep["comparison"].get(view, {}).items():
            cells = []
            for t, c in by_t.items():
                if c:
                    cells.append(f"{t} {c['difference']:+.1%} "
                                 f"[{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}]")
            out.append(f"\n{pair.replace('rtmpose', 'RTMPose').replace('mediapipe', 'MediaPipe')}"
                       f" PCK, 95% interval over {by_t['pck@0.1']['frames']} labelled frames: "
                       + "; ".join(cells) + ". An interval containing 0 is no evidence of a "
                       "difference.")
        out.append("\n| Joint | " + " | ".join(f"{BACKEND_LABELS[b]} PCK@0.1 (n)"
                                                for b in by_backend) + " |")
        out.append("|---|" + "---|" * len(by_backend))
        joints = next(iter(by_backend.values()))["per_joint"]
        for j in joints:
            cells = []
            for g in by_backend.values():
                s = g["per_joint"][j]
                cells.append(f"{pct(s.get('pck@0.1'))} ({s['n']})" if s.get("n") else "—")
            out.append(f"| {j.replace('_', ' ')} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("Errors are in P1 torso lengths from the labels. A miss (no person, or the joint "
               "below the model's own threshold) counts as a PCK failure. Occluded labels are "
               "the labeller's estimate of a hidden joint; *answered* is how often the model "
               "gave a valid point there anyway.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="data/joint_labels/labels.json")
    ap.add_argument("--manifest", default="data/joint_labels/manifest.json")
    ap.add_argument("--out", help="write the full result as JSON")
    ap.add_argument("--md", help="write the Markdown tables")
    ap.add_argument("--extract", action="store_true",
                    help="run a model on videos it has no cached poses for")
    ap.add_argument("--skip-frame-check", action="store_true")
    args = ap.parse_args(argv)

    try:
        labels = jl.load_labels(args.labels, args.manifest)
    except jl.LabelError as e:
        print("Cannot evaluate: the human labels are missing or invalid.", file=sys.stderr)
        for p in e.problems:
            print(f"  - {p}", file=sys.stderr)
        print("Label the frames in data/joint_labels/labeler.html (regenerate with "
              "scripts/make_joint_label_set.py), export, and save as "
              f"{args.labels}.", file=sys.stderr)
        return 2
    manifest = jl.load_manifest(args.manifest)
    cov = jl.coverage(labels, manifest)
    print(f"labels: {cov['frames_labelled']}/{cov['frames_expected']} frames, "
          f"{cov['joints_labelled']} joints, verified by "
          f"{sorted({v['verified_by'] for v in labels.values()})}", file=sys.stderr)
    for video, why in cov["unscorable"].items():
        print(f"  not scored: {video}: {why}", file=sys.stderr)

    if not args.skip_frame_check:
        problems = frame_alignment_problems(labels, manifest)
        if problems:
            print("Cannot evaluate: labelled frames don't match what the models saw.",
                  file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 2
        print("frame check: every labelled frame decodes to the labelled image", file=sys.stderr)

    from utils.pose_rtmpose import RTMPoseUnavailable
    try:
        sequences, identities, missing = load_sequences(labels, args.extract)
    except RTMPoseUnavailable as e:
        print(f"Cannot evaluate RTMPose here: {e}", file=sys.stderr)
        return 2
    if missing:
        print("Cannot evaluate: no cached poses for", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("Pass --extract, or run scripts/benchmark_pose.py first.", file=sys.stderr)
        return 2

    result = {
        "date": datetime.date.today().isoformat(),
        "labels_file": args.labels,
        "verified_by": ", ".join(sorted({v["verified_by"] for v in labels.values()})),
        "coverage": cov,
        "backends": identities,
        "report": jl.evaluate(labels, sequences),
    }
    md = render(result)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
