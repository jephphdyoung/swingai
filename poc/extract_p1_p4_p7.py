#!/usr/bin/env python3
"""
extract_p1_p4_p7.py

Given an input golf swing video, extract P1/P4/P7 frames as JPGs.

This version uses MediaPipe "Tasks" PoseLandmarker API (mp.tasks.vision),
because recent mediapipe pip packages no longer expose `mp.solutions.*`.

Heuristics (v2 step-1 baseline):
  - Swing window: main sustained motion burst based on wrist speed (waggle-resistant).
  - P1: last stable frame before takeaway (minimum stability score in a pre-window).
  - P4: top of backswing = highest hands (minimum avg wrist y in image).
  - P7: impact proxy = earliest near-max wrist-speed peak after P4 (within a window).

Dependencies:
  pip install opencv-python mediapipe numpy
External:
  ffmpeg in PATH is recommended (for CFR preprocessing).

Download the pose model (.task) once:
  curl -L -o pose_landmarker_full.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

Usage:
  python extract_p1_p4_p7.py --input swing.mp4 --outdir out --prefix user --model pose_landmarker_full.task
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe", file=sys.stderr)
    raise e


@dataclass
class Indices:
    p1: int
    p4: int
    p7: int
    swing_start: int
    swing_end: int
    fps: float
    frames: int


def run_ffmpeg_cfr(in_path: str, out_path: str, fps: int) -> None:
    """Transcode to constant frame rate with a fixed fps."""
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vf", f"fps={fps}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("ffmpeg failed:\n", p.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg CFR transcode failed")


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same")


def mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    m = float(np.median(x))
    return float(np.median(np.abs(x - m))) + 1e-9


def pose_keypoints(
    video_path: str,
    model_path: str,
    num_poses: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Extract wrists, hips, nose keypoints per frame using MediaPipe Tasks PoseLandmarker.
    Returns:
      pts: (N, 10) array [lw_x,lw_y,rw_x,rw_y, lh_x,lh_y,rh_x,rh_y, nose_x,nose_y] in pixels
      vis: (N,) average visibility/conf score (0..1)
      fps: float
      n_frames: int
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0:
        # Fallback; should not happen for CFR videos
        fps = 60.0

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Pose model not found: {model_path}\n"
                           f"Download one (example): pose_landmarker_full.task")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    pts = np.full((max(n_frames, 1), 10), np.nan, dtype=np.float32)
    vis = np.zeros((max(n_frames, 1),), dtype=np.float32)

    idx = 0
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # In VIDEO mode, timestamps must be monotonically increasing (ms).
            timestamp_ms = int(round(1000.0 * idx / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if idx >= pts.shape[0]:
                # CAP_PROP_FRAME_COUNT can lie; grow arrays
                pts = np.vstack([pts, np.full((1024, 10), np.nan, dtype=np.float32)])
                vis = np.concatenate([vis, np.zeros((1024,), dtype=np.float32)])

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm = result.pose_landmarks[0]

                def get_xy(landmark_id: int) -> Tuple[float, float, float]:
                    p = lm[landmark_id]
                    v = float(getattr(p, "visibility", 1.0))
                    return (float(p.x) * w, float(p.y) * h, v)

                # BlazePose landmark indices: 0 nose, 15 left wrist, 16 right wrist, 23 left hip, 24 right hip
                lwx, lwy, lwv = get_xy(15)
                rwx, rwy, rwv = get_xy(16)
                lhx, lhy, lhv = get_xy(23)
                rhx, rhy, rhv = get_xy(24)
                nx, ny, nv = get_xy(0)

                pts[idx, :] = [lwx, lwy, rwx, rwy, lhx, lhy, rhx, rhy, nx, ny]
                vis[idx] = float(np.clip((lwv + rwv + lhv + rhv + nv) / 5.0, 0.0, 1.0))

            idx += 1

    cap.release()

    pts = pts[:idx]
    vis = vis[:idx]
    return pts, vis, fps, idx


def compute_motion(pts: np.ndarray, fps: float) -> Dict[str, np.ndarray]:
    """
    Compute motion signals from landmark points:
      wrist_speed: mean speed of wrists (px/s)
      hip_speed: mean speed of hips (px/s)
      head_speed: nose speed (px/s)
      wrist_y: average wrist y (px) (smaller = higher in image)
    """
    dt = 1.0 / max(fps, 1e-6)

    def speed(xy: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        return np.concatenate([[0.0], d]).astype(np.float32)

    lw = pts[:, 0:2]
    rw = pts[:, 2:4]
    lh = pts[:, 4:6]
    rh = pts[:, 6:8]
    nz = pts[:, 8:10]

    wrist_speed = 0.5 * (speed(lw) + speed(rw))
    hip_speed = 0.5 * (speed(lh) + speed(rh))
    head_speed = speed(nz)
    wrist_y = 0.5 * (lw[:, 1] + rw[:, 1])

    return {
        "wrist_speed": wrist_speed,
        "hip_speed": hip_speed,
        "head_speed": head_speed,
        "wrist_y": wrist_y.astype(np.float32),
    }


def find_main_burst(motion: np.ndarray, fps: float) -> Tuple[int, int]:
    """
    Find main motion burst (swing window) using hysteresis on wrist speed.
    Returns (start_idx, end_idx) in frame indices.
    """
    m = moving_average(motion, win=7)
    N = len(m)
    if N < 5:
        return 0, max(0, N - 1)

    # Baseline from first ~1s
    n0 = int(min(N, max(30, round(fps * 1.0))))
    base = m[:n0]
    med = float(np.median(base))
    s = mad(base)

    start_thresh = med + 4.0 * s
    keep_thresh = med + 2.0 * s

    active = m > keep_thresh

    segments = []
    i = 0
    while i < N:
        if active[i]:
            j = i
            while j < N and active[j]:
                j += 1
            segments.append((i, j - 1))
            i = j
        else:
            i += 1

    if not segments:
        return 0, N - 1

    # Choose segment with highest peak
    best = segments[0]
    best_peak = -1.0
    for a, b in segments:
        peak = float(np.max(m[a:b + 1]))
        if peak > best_peak:
            best_peak = peak
            best = (a, b)

    seg_start, seg_end = best

    sustain = int(max(3, round(fps * 0.20)))  # 200ms
    refined_start = seg_start
    for k in range(seg_start, min(seg_end, N - sustain)):
        if m[k] > start_thresh and np.all(m[k:k + sustain] > keep_thresh):
            refined_start = k
            break

    tail = int(max(3, round(fps * 0.35)))  # 350ms quiet
    refined_end = seg_end
    for k in range(seg_end, N - tail):
        if np.all(m[k:k + tail] < keep_thresh):
            refined_end = k
            break

    refined_start = int(np.clip(refined_start, 0, N - 1))
    refined_end = int(np.clip(refined_end, refined_start + 1, N - 1)) if N > 1 else refined_start
    return refined_start, refined_end


def pick_p1(stability: np.ndarray, swing_start: int, fps: float) -> int:
    """
    P1: last stable frame before swing_start.
    Find minimum stability in a window before start, excluding immediate pre-start frames.
    """
    lookback = int(max(5, round(fps * 0.80)))  # 0.8s back
    guard = int(max(1, round(fps * 0.05)))     # ignore last 50ms
    a = max(0, swing_start - lookback)
    b = max(0, swing_start - guard)
    if b <= a:
        return max(0, swing_start - 1)
    window = stability[a:b]
    return int(a + int(np.argmin(window)))


def pick_p4(wrist_y: np.ndarray, swing_start: int, swing_end: int) -> int:
    """P4: top of backswing = highest hands => minimum wrist_y."""
    a = int(np.clip(swing_start, 0, len(wrist_y) - 1))
    b = int(np.clip(swing_end, a + 1, len(wrist_y)))
    return int(a + int(np.nanargmin(wrist_y[a:b])))


def pick_p7(wrist_speed: np.ndarray, p4: int, swing_end: int, fps: float) -> int:
    """
    P7 (impact proxy): earliest near-max wrist-speed peak after P4.
    Restrict search to a window after P4 to avoid follow-through peaks.
    """
    N = len(wrist_speed)
    if N < 2:
        return 0

    p4 = int(np.clip(p4, 0, N - 1))
    swing_end = int(np.clip(swing_end, p4 + 1, N - 1))

    max_win = int(max(5, round(fps * 0.90)))  # search up to 0.9s after P4
    b = min(swing_end, p4 + max_win)

    w = wrist_speed[p4:b]
    if w.size == 0:
        return min(swing_end, p4 + 1)

    w_s = moving_average(w.astype(np.float32), win=5)
    m = float(np.max(w_s))
    thresh = 0.95 * m

    candidates = np.where(w_s >= thresh)[0]
    if candidates.size > 0:
        return int(p4 + int(candidates[0]))

    return int(p4 + int(np.argmax(w_s)))


def detect_p_positions(video_path: str, model_path: str) -> Indices:
    pts, vis, fps, n_frames = pose_keypoints(video_path, model_path=model_path)
    if n_frames < 10:
        raise RuntimeError("Video too short to analyze.")

    feats = compute_motion(pts, fps)

    wrist_speed = feats["wrist_speed"]
    hip_speed = feats["hip_speed"]
    head_speed = feats["head_speed"]
    wrist_y = feats["wrist_y"]

    ws = moving_average(wrist_speed, win=7)
    hs = moving_average(hip_speed, win=7)
    ns = moving_average(head_speed, win=7)

    # Stability score (lower == more still). Wrist dominates.
    stability = (0.75 * ws + 0.20 * hs + 0.05 * ns).astype(np.float32)

    swing_start, swing_end = find_main_burst(ws, fps)

    p1 = pick_p1(stability, swing_start, fps)
    p4 = pick_p4(wrist_y, swing_start, swing_end)
    p7 = pick_p7(ws, p4, swing_end, fps)

    return Indices(
        p1=p1, p4=p4, p7=p7,
        swing_start=swing_start, swing_end=swing_end,
        fps=fps, frames=n_frames
    )


def write_frames(video_path: str, outdir: str, idx: Indices, prefix: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    targets = {
        idx.p1: f"{prefix}_P1_frame{idx.p1:05d}.jpg",
        idx.p4: f"{prefix}_P4_frame{idx.p4:05d}.jpg",
        idx.p7: f"{prefix}_P7_frame{idx.p7:05d}.jpg",
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for extraction: {video_path}")

    needed = set(targets.keys())
    frame_id = 0

    while needed:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in needed:
            outpath = os.path.join(outdir, targets[frame_id])
            if not cv2.imwrite(outpath, frame):
                raise RuntimeError(f"Failed to write: {outpath}")
            needed.remove(frame_id)
        frame_id += 1

    cap.release()

    if needed:
        raise RuntimeError(f"Failed to extract frames (out of range?): {sorted(needed)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video path (user swing).")
    ap.add_argument("--outdir", required=True, help="Output directory for JPGs.")
    ap.add_argument("--prefix", default="user", help="Prefix for output filenames.")
    ap.add_argument("--fps", type=int, default=60, help="CFR analysis fps (default 60).")
    ap.add_argument("--skip_preprocess", action="store_true", help="Analyze input directly (no CFR transcode).")
    ap.add_argument("--model", default="pose_landmarker_full.task", help="Path to pose_landmarker*.task model file.")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    analysis_path = in_path
    os.makedirs(args.outdir, exist_ok=True)

    if not args.skip_preprocess:
        analysis_path = os.path.join(args.outdir, f"{args.prefix}_analysis_cfr{args.fps}.mp4")
        print(f"[1/3] Preprocessing to CFR {args.fps}fps: {analysis_path}")
        run_ffmpeg_cfr(in_path, analysis_path, args.fps)
    else:
        print("[1/3] Skipping preprocess (using input as-is)")

    print("[2/3] Detecting P1/P4/P7 (pose + motion)")
    idx = detect_p_positions(analysis_path, model_path=args.model)
    print(f"  fps={idx.fps:.2f} frames={idx.frames}")
    print(f"  swing_start={idx.swing_start} swing_end={idx.swing_end}")
    print(f"  P1={idx.p1}  P4={idx.p4}  P7={idx.p7}")

    print("[3/3] Extracting JPG frames")
    write_frames(analysis_path, args.outdir, idx, prefix=args.prefix)

    summary = {
        "analysis_video": os.path.abspath(analysis_path),
        "fps": idx.fps,
        "frames": idx.frames,
        "swing_start": idx.swing_start,
        "swing_end": idx.swing_end,
        "P1": idx.p1,
        "P4": idx.p4,
        "P7": idx.p7,
        "P1_time_ms": int(round(1000.0 * idx.p1 / max(idx.fps, 1e-6))),
        "P4_time_ms": int(round(1000.0 * idx.p4 / max(idx.fps, 1e-6))),
        "P7_time_ms": int(round(1000.0 * idx.p7 / max(idx.fps, 1e-6))),
    }
    summary_path = os.path.join(args.outdir, f"{args.prefix}_p_positions.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Wrote: {summary_path}")


if __name__ == "__main__":
    main()
