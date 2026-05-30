from typing import Callable, Optional

from utils.pose_extractor import extract_pose
from utils.video_sync import sync_on_p_positions
from utils.video_renderer import generate_comparison_video
from utils.pose_features import detect_p_positions

TOTAL_STEPS = 4


def analyze_swing(user_video_path, reference_video_path,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None):
    def report(step, message):
        print(message)
        if progress_callback:
            progress_callback(step, TOTAL_STEPS, message)

    report(1, "Extracting poses from user video...")
    user_landmarks = extract_pose(user_video_path)
    print(f"  {len(user_landmarks)} frames")

    report(2, "Extracting poses from reference video...")
    ref_landmarks = extract_pose(reference_video_path)
    print(f"  {len(ref_landmarks)} frames")

    report(3, "Analyzing swings and syncing P-positions...")
    user_p = detect_p_positions(user_landmarks)
    ref_p = detect_p_positions(ref_landmarks)
    alignment = sync_on_p_positions(user_p, ref_p)

    report(4, "Generating comparison video...")
    output_path = generate_comparison_video(
        user_video_path, reference_video_path,
        alignment, user_landmarks, ref_landmarks,
        user_p_positions=user_p, ref_p_positions=ref_p
    )

    print(f"Saved: {output_path}")
    return output_path
