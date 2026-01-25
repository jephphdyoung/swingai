from utils.pose_extractor import extract_pose
from utils.video_sync import sync_on_p_positions
from utils.video_renderer import generate_comparison_video
from utils.pose_features import detect_p_positions


def analyze_swing(user_video_path, reference_video_path):
    # Step 1: Extract poses from each video
    print("Extracting poses...")
    print("  User video...")
    user_landmarks = extract_pose(user_video_path)
    print(f"    {len(user_landmarks)} frames")

    print("  Reference video...")
    ref_landmarks = extract_pose(reference_video_path)
    print(f"    {len(ref_landmarks)} frames")

    # Step 2: Analyze each video independently to find P positions
    print("\nAnalyzing user swing...")
    user_p = detect_p_positions(user_landmarks)

    print("\nAnalyzing reference swing...")
    ref_p = detect_p_positions(ref_landmarks)

    # Step 3: Sync based on P positions (aligns P1->P1, P4->P4, P6->P6, P9->P9)
    print("\nSynchronizing swings on P positions...")
    alignment = sync_on_p_positions(user_p, ref_p)

    # Step 4: Generate comparison video
    print("\nGenerating comparison video...")
    output_path = generate_comparison_video(
        user_video_path, reference_video_path,
        alignment, user_landmarks, ref_landmarks,
        user_p_positions=user_p, ref_p_positions=ref_p
    )

    print(f"\nSaved: {output_path}")
    return output_path
