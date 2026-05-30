import sys
from typing import Callable, Optional

from utils.pose_extractor import extract_pose_with_timestamps
from utils.pose_features import detect_p_positions
from utils.annotations import load_annotations
from utils.swing_detector import detect_swing_window
from utils.video_sync import sync_on_p_positions
from utils.video_renderer import generate_comparison_video

TOTAL_STEPS = 5


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

    # Step 5: Render
    report(5, "Generating comparison video...")
    output_path = generate_comparison_video(
        user_video_path, reference_video_path,
        alignment, user_result.landmarks, ref_result.landmarks,
        user_p_positions=user_p, ref_p_positions=ref_p,
    )

    log(f"Saved: {output_path}")
    return output_path
