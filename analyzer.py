import base64
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2

from utils.pose_extractor import extract_pose_with_timestamps
from utils.pose_features import detect_p_positions
from utils.annotations import load_annotations
from utils.swing_detector import detect_swing_window
from utils.video_sync import sync_on_p_positions
from utils.video_renderer import generate_comparison_video
from utils.overlay import auto_register, DEFAULT_TRANSFORM

TOTAL_STEPS = 5

# Max height (px) for the still-frame thumbnails handed to the overlay editor.
EDITOR_STILL_HEIGHT = 360


@dataclass
class AnalysisResult:
    output_path: str
    output_path_no_skeleton: str
    p_timestamps: dict[str, float]  # P-name -> seconds in output video
    # --- overlay support ---
    still_pairs: list = field(default_factory=list)   # per-P {p,user,ref,ref_ar} for the editor
    auto_transform: dict = field(default_factory=lambda: dict(DEFAULT_TRANSFORM))
    # context for rendering the overlay video on demand:
    user_video_path: str = ""
    reference_video_path: str = ""
    alignment: list = field(default_factory=list)
    user_landmarks: list = field(default_factory=list)
    ref_landmarks: list = field(default_factory=list)
    user_p: dict = field(default_factory=dict)
    ref_p: dict = field(default_factory=dict)


def _read_frame(video_path, frame_idx):
    """Read a single frame (BGR) at frame_idx, or None."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _frame_dims(video_path):
    """Return (width, height) of the video's frames."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _encode_thumb(frame):
    """Downscale a frame to EDITOR_STILL_HEIGHT and return a base64 JPEG data URI."""
    h, w = frame.shape[:2]
    if h > EDITOR_STILL_HEIGHT:
        scale = EDITOR_STILL_HEIGHT / h
        frame = cv2.resize(frame, (int(w * scale), EDITOR_STILL_HEIGHT))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _build_still_pairs(user_video_path, ref_video_path, user_p, ref_p):
    """For each shared P-position, grab the aligned user/reference still frames."""
    pairs = []
    shared = [p for p in user_p if p in ref_p and not p.startswith("_")]
    shared.sort(key=lambda x: int(x[1:]))
    for p in shared:
        uf = _read_frame(user_video_path, user_p[p])
        rf = _read_frame(ref_video_path, ref_p[p])
        if uf is None or rf is None:
            continue
        user_thumb = _encode_thumb(uf)
        ref_thumb = _encode_thumb(rf)
        if not user_thumb or not ref_thumb:
            continue
        pairs.append({
            "p": p,
            "user": user_thumb,
            "ref": ref_thumb,
            "ref_ar": rf.shape[1] / rf.shape[0],   # ref width/height
        })
    return pairs


def _compute_auto_transform(user_video_path, ref_video_path,
                            user_landmarks, ref_landmarks, user_p, ref_p):
    """Auto-register the pro onto the user using the P1 (address) frame."""
    anchor = "P1" if ("P1" in user_p and "P1" in ref_p) else None
    if anchor is None:
        return dict(DEFAULT_TRANSFORM)
    u_idx, r_idx = user_p["P1"], ref_p["P1"]
    if u_idx >= len(user_landmarks) or r_idx >= len(ref_landmarks):
        return dict(DEFAULT_TRANSFORM)
    user_dims = _frame_dims(user_video_path)
    ref_dims = _frame_dims(ref_video_path)
    return auto_register(user_landmarks[u_idx], ref_landmarks[r_idx], user_dims, ref_dims)


def analyze_swing(user_video_path, reference_video_path,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None):
    def report(step, message):
        print(message, file=sys.stderr)
        if progress_callback:
            progress_callback(step, TOTAL_STEPS, message)

    def log(msg):
        print(msg, file=sys.stderr)

    # Step 1-2: Extract poses with timestamps
    report(1, "Extracting poses from user video...")
    user_result = extract_pose_with_timestamps(user_video_path)
    log(f"  {user_result.frame_count} frames at {user_result.fps:.1f}fps")

    report(2, "Extracting poses from reference video...")
    ref_result = extract_pose_with_timestamps(reference_video_path)
    log(f"  {ref_result.frame_count} frames at {ref_result.fps:.1f}fps")

    # Step 3: Get P-positions (saved annotations or auto-detect)
    report(3, "Loading P-positions...")

    user_p = load_annotations(user_video_path)
    if user_p:
        log(f"  User: loaded saved annotations {user_p}")
    else:
        log(f"  User: no annotations found, auto-detecting...")
        user_window = detect_swing_window(user_result.landmarks, user_result.timestamps_ms)
        user_impact = user_window.impact_frame if user_window else None
        user_p = detect_p_positions(
            user_result.landmarks,
            impact_override=user_impact,
            timestamps_ms=user_result.timestamps_ms,
        )
        log(f"  User: detected {user_p}")

    ref_p = load_annotations(reference_video_path)
    if ref_p:
        log(f"  Ref:  loaded saved annotations {ref_p}")
    else:
        log(f"  Ref:  no annotations found, auto-detecting...")
        ref_window = detect_swing_window(ref_result.landmarks, ref_result.timestamps_ms)
        ref_impact = ref_window.impact_frame if ref_window else None
        ref_p = detect_p_positions(
            ref_result.landmarks,
            impact_override=ref_impact,
            timestamps_ms=ref_result.timestamps_ms,
        )
        log(f"  Ref:  detected {ref_p}")

    # Step 4: Sync
    report(4, "Synchronizing swings...")
    alignment = sync_on_p_positions(user_p, ref_p)

    # Step 5: Render (with skeleton)
    report(5, "Generating comparison video...")
    output_path, p_timestamps = generate_comparison_video(
        user_video_path, reference_video_path,
        alignment, user_result.landmarks, ref_result.landmarks,
        user_p_positions=user_p, ref_p_positions=ref_p,
        draw_skeleton=True, output_filename="comparison_output.mp4",
    )

    # Also generate version without skeleton
    log("Generating version without skeleton...")
    output_path_no_skeleton, _ = generate_comparison_video(
        user_video_path, reference_video_path,
        alignment, user_result.landmarks, ref_result.landmarks,
        user_p_positions=user_p, ref_p_positions=ref_p,
        draw_skeleton=False, output_filename="comparison_output_no_skeleton.mp4",
    )

    log(f"Saved: {output_path}, {output_path_no_skeleton}")
    log(f"P-position timestamps: {p_timestamps}")

    # Overlay support: still pairs for the editor + an auto-registration seed.
    still_pairs = _build_still_pairs(user_video_path, reference_video_path, user_p, ref_p)
    auto_transform = _compute_auto_transform(
        user_video_path, reference_video_path,
        user_result.landmarks, ref_result.landmarks, user_p, ref_p,
    )
    log(f"Auto-registration transform: {auto_transform}")

    return AnalysisResult(
        output_path=output_path,
        output_path_no_skeleton=output_path_no_skeleton,
        p_timestamps=p_timestamps,
        still_pairs=still_pairs,
        auto_transform=auto_transform,
        user_video_path=user_video_path,
        reference_video_path=reference_video_path,
        alignment=alignment,
        user_landmarks=user_result.landmarks,
        ref_landmarks=ref_result.landmarks,
        user_p=user_p,
        ref_p=ref_p,
    )


def render_overlay(result: AnalysisResult, transform: dict,
                   progress_callback: Optional[Callable[[int, int, str], None]] = None):
    """Render the ghost-overlay comparison video using a (possibly user-adjusted)
    transform. Produces skeleton + no-skeleton variants so the existing player's
    skeleton toggle keeps working.

    Returns (output_path, output_path_no_skeleton, p_timestamps).
    """
    def report(step, total, message):
        print(message, file=sys.stderr)
        if progress_callback:
            progress_callback(step, total, message)

    report(1, 2, "Rendering overlay (skeleton)...")
    output_path, p_timestamps = generate_comparison_video(
        result.user_video_path, result.reference_video_path,
        result.alignment, result.user_landmarks, result.ref_landmarks,
        user_p_positions=result.user_p, ref_p_positions=result.ref_p,
        draw_skeleton=True, render_mode="overlay", overlay_transform=transform,
        output_filename="overlay_output.mp4",
    )

    report(2, 2, "Rendering overlay (no skeleton)...")
    output_path_no_skeleton, _ = generate_comparison_video(
        result.user_video_path, result.reference_video_path,
        result.alignment, result.user_landmarks, result.ref_landmarks,
        user_p_positions=result.user_p, ref_p_positions=result.ref_p,
        draw_skeleton=False, render_mode="overlay", overlay_transform=transform,
        output_filename="overlay_output_no_skeleton.mp4",
    )

    return output_path, output_path_no_skeleton, p_timestamps
