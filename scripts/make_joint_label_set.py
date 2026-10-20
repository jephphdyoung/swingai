"""Build the joint-labelling set and a local HTML labeller for it.

    python scripts/make_joint_label_set.py

Writes
    data/joint_labels/manifest.json   which frames, and why (committed)
    data/joint_labels/labeler.html    self-contained labeller with the frames
                                      embedded (git-ignored: it holds footage)

Open labeler.html in a browser, click each joint, export labels.json, and save
it as data/joint_labels/labels.json. benchmark_pose.py then scores both
backends against it.

Frame choice: P1/P4/P7 where the video has them annotated (the annotations
only choose frames; they are not ground truth), the frames where MediaPipe
and RTMPose disagree most (where accuracy actually differs), and the fastest
hand-motion frame. Nothing is pre-filled -- a pre-placed point would anchor
the labeller toward one model.
"""

import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from utils.annotations import load_annotation_record  # noqa: E402
from utils.landmarks import LANDMARK_INDEX  # noqa: E402
from utils.paths import repo_relative  # noqa: E402
from utils.pose_data import get_pose_sequence, make_backend  # noqa: E402
from utils.pose_metrics import COMMON_JOINTS, agreement, body_scale, valid_mask  # noqa: E402

VIDEOS = [
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4", "dtl"),
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4", "face_on"),
    ("videos/reference/GW_DTL.mp4", "dtl"),
    ("videos/reference/GW_faceon.mp4", "face_on"),
]
OUT_DIR = "data/joint_labels"
MAX_WIDTH = 1440


def disagreement_per_frame(a, b):
    scale = body_scale(a)
    ma, mb = valid_mask(a), valid_mask(b)
    idx = [LANDMARK_INDEX[n] for n in COMMON_JOINTS]
    d = np.linalg.norm(a.xy[:, idx] - b.xy[:, idx], axis=2) / scale
    d[~(ma & mb)] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(d, axis=1)


def pick_frames(path, window):
    mp_seq = get_pose_sequence(path, make_backend("mediapipe"), extract_if_missing=False)
    rt_seq = get_pose_sequence(path, make_backend("rtmpose"), extract_if_missing=False)
    rec = load_annotation_record(path)
    chosen = []
    for p in ("P1", "P4", "P7"):
        if rec and p in rec.frames:
            chosen.append((rec.frames[p], f"{p} (annotated frame, used only to sample)"))
    lo, hi = window
    if mp_seq is not None and rt_seq is not None:
        dis = disagreement_per_frame(mp_seq, rt_seq)
        dis[:lo] = np.nan
        dis[hi:] = np.nan
        min_gap = max(5, (hi - lo) // 10)
        for f in np.argsort(-np.nan_to_num(dis, nan=-1)):
            if np.isnan(dis[f]) or len([c for c in chosen if "disagree" in c[1]]) >= 2:
                break
            if all(abs(int(f) - c[0]) >= min_gap for c in chosen):
                chosen.append((int(f), f"backends disagree most ({dis[f]:.2f} torso)"))
        # fastest trail-wrist motion by RTMPose, if no P7 was annotated
        if not any(c[1].startswith("P7") for c in chosen):
            w = rt_seq.xy[:, LANDMARK_INDEX["right_wrist"]]
            sp = np.linalg.norm(np.diff(w, axis=0), axis=1)
            sp[:lo] = np.nan
            sp[hi:] = np.nan
            if np.isfinite(sp).any():
                f = int(np.nanargmax(sp))
                if all(abs(f - c[0]) >= 3 for c in chosen):
                    chosen.append((f, "fastest hand motion (RTMPose track)"))
    return sorted(chosen)


def swing_window(path, n):
    rec = load_annotation_record(path)
    if rec and rec.frames:
        frames = sorted(rec.frames.values())
        pad = max(5, (frames[-1] - frames[0]) // 4) if len(frames) > 1 else n // 4
        return max(0, frames[0] - pad), min(n, frames[-1] + 3 * pad)
    return 0, n


def encode(path, frame):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None, None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_WIDTH / w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode(), {"width": w, "height": h, "display_scale": scale}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest, items = [], []
    for path, view in VIDEOS:
        if not os.path.exists(path):
            continue
        n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
        rec = load_annotation_record(path)
        p1 = rec.frames.get("P1") if rec else None
        for frame, reason in pick_frames(path, swing_window(path, n)):
            b64, size = encode(path, frame)
            if b64 is None:
                continue
            item = {"id": f"{len(manifest):02d}", "video": repo_relative(path), "view": view,
                    "frame": int(frame), "reason": reason, "p1_frame": p1, **size}
            manifest.append(item)
            items.append({**item, "jpg": b64})
            print(f"{item['id']}  {os.path.basename(path)[:44]:44s} frame {frame:5d}  {reason}")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"joints": list(COMMON_JOINTS), "frames": manifest}, f, indent=1)
    html = open(os.path.join(os.path.dirname(__file__), "joint_labeler.html")).read()
    html = html.replace("/*__DATA__*/null", json.dumps({"joints": list(COMMON_JOINTS),
                                                         "frames": items}))
    with open(os.path.join(OUT_DIR, "labeler.html"), "w") as f:
        f.write(html)
    print(f"{len(manifest)} frames -> {OUT_DIR}/labeler.html")


if __name__ == "__main__":
    main()
