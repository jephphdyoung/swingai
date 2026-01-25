#!/usr/bin/env python3
"""
Evaluation harness for P-position detection accuracy.

Loads swing videos with ground truth P-frame annotations and reports
Mean Absolute Error (MAE) in frames and milliseconds per P-position.

Usage:
    python scripts/eval_positions.py --ground-truth data/ground_truth.json
    python scripts/eval_positions.py --ground-truth data/ground_truth.json --verbose
    python scripts/eval_positions.py --ground-truth data/ground_truth.json --output results.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pose_extractor import extract_pose, extract_pose_with_timestamps
from utils.pose_features import detect_p_positions

# Try to import preprocessor (optional, for time normalization)
try:
    from utils.video_preprocessor import (
        preprocess_video,
        cleanup_temp_files,
        PreprocessedVideo,
    )
    HAS_PREPROCESSOR = True
except ImportError:
    HAS_PREPROCESSOR = False


@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    fps: float
    frame_count: int
    duration_ms: float
    width: int
    height: int


@dataclass
class PositionError:
    """Error metrics for a single P-position."""
    position: str
    predicted_frame: int
    ground_truth_frame: int
    error_frames: int
    error_ms: float


@dataclass
class VideoResult:
    """Evaluation result for a single video."""
    video_path: str
    metadata: VideoMetadata
    errors: list[PositionError] = field(default_factory=list)
    success: bool = True
    error_message: str = ""


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""
    video_results: list[VideoResult] = field(default_factory=list)

    def compute_mae_by_position(self) -> dict[str, dict[str, float]]:
        """Compute MAE per P-position across all videos."""
        position_errors: dict[str, list[PositionError]] = {}

        for result in self.video_results:
            if not result.success:
                continue
            for err in result.errors:
                if err.position not in position_errors:
                    position_errors[err.position] = []
                position_errors[err.position].append(err)

        mae_by_position = {}
        for pos, errors in sorted(position_errors.items()):
            frame_errors = [abs(e.error_frames) for e in errors]
            ms_errors = [abs(e.error_ms) for e in errors]
            mae_by_position[pos] = {
                "mae_frames": np.mean(frame_errors) if frame_errors else 0.0,
                "mae_ms": np.mean(ms_errors) if ms_errors else 0.0,
                "std_frames": np.std(frame_errors) if frame_errors else 0.0,
                "std_ms": np.std(ms_errors) if ms_errors else 0.0,
                "n_samples": len(errors),
            }

        return mae_by_position

    def compute_overall_mae(self) -> dict[str, float]:
        """Compute overall MAE across all positions and videos."""
        all_frame_errors = []
        all_ms_errors = []

        for result in self.video_results:
            if not result.success:
                continue
            for err in result.errors:
                all_frame_errors.append(abs(err.error_frames))
                all_ms_errors.append(abs(err.error_ms))

        return {
            "mae_frames": np.mean(all_frame_errors) if all_frame_errors else 0.0,
            "mae_ms": np.mean(all_ms_errors) if all_ms_errors else 0.0,
            "std_frames": np.std(all_frame_errors) if all_frame_errors else 0.0,
            "std_ms": np.std(all_ms_errors) if all_ms_errors else 0.0,
            "n_samples": len(all_frame_errors),
        }


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0

    cap.release()

    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        duration_ms=duration_ms,
        width=width,
        height=height,
    )


def load_ground_truth(json_path: str) -> dict:
    """
    Load ground truth annotations from JSON file.

    Expected format:
    {
        "videos": [
            {
                "path": "path/to/video.mp4",
                "fps": 60.0,  // optional, will be extracted from video if not provided
                "positions": {
                    "P1": {"frame": 45},         // frame number
                    "P2": {"frame": 67},
                    "P3": {"frame": 89},
                    "P4": {"frame": 112},
                    "P5": {"frame": 125},
                    "P6": {"frame": 138},
                    "P7": {"frame": 160},
                    "P8": {"frame": 185},
                    "P9": {"frame": 210}
                },
                // OR use timestamps:
                "positions": {
                    "P1": {"timestamp_ms": 750},
                    "P4": {"timestamp_ms": 1867},
                    ...
                },
                "notes": "optional notes about this video"
            }
        ],
        "metadata": {
            "annotator": "name",
            "date": "2024-01-25",
            "version": "1.0"
        }
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "videos" not in data:
        raise ValueError("Ground truth JSON must have 'videos' array")

    return data


def timestamp_to_frame(timestamp_ms: float, fps: float) -> int:
    """Convert timestamp in milliseconds to frame number."""
    return int(round(timestamp_ms * fps / 1000))


def frame_to_timestamp(frame: int, fps: float) -> float:
    """Convert frame number to timestamp in milliseconds."""
    return frame * 1000 / fps


def normalize_ground_truth_positions(
    positions: dict,
    fps: float
) -> dict[str, int]:
    """
    Normalize ground truth positions to frame numbers.

    Handles both frame-based and timestamp-based annotations.
    """
    normalized = {}

    for pos_name, pos_data in positions.items():
        if isinstance(pos_data, dict):
            if "frame" in pos_data:
                normalized[pos_name] = pos_data["frame"]
            elif "timestamp_ms" in pos_data:
                normalized[pos_name] = timestamp_to_frame(
                    pos_data["timestamp_ms"], fps
                )
            else:
                raise ValueError(
                    f"Position {pos_name} must have 'frame' or 'timestamp_ms'"
                )
        elif isinstance(pos_data, int):
            # Direct frame number
            normalized[pos_name] = pos_data
        else:
            raise ValueError(f"Invalid position data for {pos_name}: {pos_data}")

    return normalized


def evaluate_video(
    video_path: str,
    ground_truth_positions: dict,
    base_path: str = "",
    verbose: bool = False,
    use_preprocessing: bool = True,
) -> VideoResult:
    """
    Evaluate P-position detection on a single video.

    Args:
        video_path: Path to video file (relative or absolute)
        ground_truth_positions: Dict mapping position names to frame/timestamp
        base_path: Base path to prepend to relative video paths
        verbose: Print detailed progress
        use_preprocessing: Use video preprocessor for time normalization

    Returns:
        VideoResult with error metrics
    """
    # Resolve video path
    if not os.path.isabs(video_path):
        video_path = os.path.join(base_path, video_path)

    if not os.path.exists(video_path):
        return VideoResult(
            video_path=video_path,
            metadata=VideoMetadata(0, 0, 0, 0, 0),
            success=False,
            error_message=f"Video not found: {video_path}",
        )

    if verbose:
        print(f"Processing: {video_path}")

    # Get video metadata
    try:
        metadata = get_video_metadata(video_path)
    except Exception as e:
        return VideoResult(
            video_path=video_path,
            metadata=VideoMetadata(0, 0, 0, 0, 0),
            success=False,
            error_message=f"Failed to read video metadata: {e}",
        )

    if verbose:
        print(f"  FPS: {metadata.fps}, Frames: {metadata.frame_count}")

    # Extract poses and detect P-positions
    preprocessed = None
    analysis_fps = metadata.fps  # May be updated if transcoded
    try:
        # Use preprocessing if available and enabled
        if use_preprocessing and HAS_PREPROCESSOR:
            if verbose:
                print("  Preprocessing video (extracting timestamps)...")
            preprocessed = preprocess_video(video_path, target_fps=60, force_transcode=False)

            if verbose:
                vfr_status = "VFR" if preprocessed.original_info.is_vfr else "CFR"
                print(f"  Original: {preprocessed.original_info.fps:.1f}fps ({vfr_status})")
                if preprocessed.is_transcoded:
                    print(f"  Transcoded to CFR 60fps for analysis")

            # Use analysis fps for ground truth conversion
            if preprocessed.is_transcoded:
                analysis_fps = 60.0  # Target fps from transcoding

            if verbose:
                print("  Extracting poses with timestamps...")
            result = extract_pose_with_timestamps(
                preprocessed.analysis_path,
                preprocessed.frame_timestamps_ms
            )
            landmarks = result.landmarks
            timestamps_ms = result.timestamps_ms
        else:
            if verbose:
                print("  Extracting poses...")
            landmarks = extract_pose(video_path)
            timestamps_ms = None

        # Normalize ground truth to frames using analysis fps
        gt_frames = normalize_ground_truth_positions(
            ground_truth_positions, analysis_fps
        )

        if verbose:
            print(f"  Extracted {len(landmarks)} frames, detecting positions...")

        # Suppress internal print statements during evaluation
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            predicted = detect_p_positions(landmarks, timestamps_ms=timestamps_ms)

    except Exception as e:
        if preprocessed:
            cleanup_temp_files(preprocessed)
        return VideoResult(
            video_path=video_path,
            metadata=metadata,
            success=False,
            error_message=f"Failed to detect positions: {e}",
        )
    finally:
        # Cleanup temp files
        if preprocessed:
            cleanup_temp_files(preprocessed)

    # Compute errors for each position
    errors = []
    for pos_name in sorted(gt_frames.keys()):
        if pos_name not in predicted:
            continue

        gt_frame = gt_frames[pos_name]
        pred_frame = predicted[pos_name]

        error_frames = pred_frame - gt_frame
        error_ms = frame_to_timestamp(error_frames, metadata.fps)

        errors.append(PositionError(
            position=pos_name,
            predicted_frame=pred_frame,
            ground_truth_frame=gt_frame,
            error_frames=error_frames,
            error_ms=error_ms,
        ))

        if verbose:
            direction = "late" if error_frames > 0 else "early"
            print(f"  {pos_name}: pred={pred_frame}, gt={gt_frame}, "
                  f"error={abs(error_frames)} frames ({abs(error_ms):.1f}ms) {direction}")

    return VideoResult(
        video_path=video_path,
        metadata=metadata,
        errors=errors,
        success=True,
    )


def run_evaluation(
    ground_truth_path: str,
    verbose: bool = False,
    use_preprocessing: bool = True,
) -> EvaluationReport:
    """
    Run evaluation on all videos in ground truth file.

    Args:
        ground_truth_path: Path to ground truth JSON file
        verbose: Print detailed progress
        use_preprocessing: Use video preprocessor for time normalization

    Returns:
        EvaluationReport with all results
    """
    gt_data = load_ground_truth(ground_truth_path)
    base_path = os.path.dirname(os.path.abspath(ground_truth_path))

    report = EvaluationReport()

    for video_entry in gt_data["videos"]:
        video_path = video_entry["path"]
        positions = video_entry["positions"]

        result = evaluate_video(
            video_path=video_path,
            ground_truth_positions=positions,
            base_path=base_path,
            verbose=verbose,
            use_preprocessing=use_preprocessing,
        )
        report.video_results.append(result)

    return report


def print_report(report: EvaluationReport) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("P-POSITION DETECTION EVALUATION REPORT")
    print("=" * 60)

    # Count successes/failures
    n_success = sum(1 for r in report.video_results if r.success)
    n_total = len(report.video_results)

    print(f"\nVideos evaluated: {n_success}/{n_total} successful")

    # Print failures
    for result in report.video_results:
        if not result.success:
            print(f"  FAILED: {result.video_path}")
            print(f"    Error: {result.error_message}")

    # MAE by position
    mae_by_pos = report.compute_mae_by_position()

    if mae_by_pos:
        print("\nMAE by P-Position:")
        print("-" * 50)
        print(f"{'Position':<10} {'MAE (frames)':<15} {'MAE (ms)':<15} {'N':<5}")
        print("-" * 50)

        for pos, metrics in sorted(mae_by_pos.items()):
            print(f"{pos:<10} {metrics['mae_frames']:>8.2f} +/- {metrics['std_frames']:<6.2f}"
                  f"{metrics['mae_ms']:>8.1f} +/- {metrics['std_ms']:<6.1f}"
                  f"{metrics['n_samples']:<5}")

    # Overall MAE
    overall = report.compute_overall_mae()

    print("-" * 50)
    print(f"{'OVERALL':<10} {overall['mae_frames']:>8.2f} +/- {overall['std_frames']:<6.2f}"
          f"{overall['mae_ms']:>8.1f} +/- {overall['std_ms']:<6.1f}"
          f"{overall['n_samples']:<5}")
    print("=" * 60)


def save_report(report: EvaluationReport, output_path: str) -> None:
    """Save evaluation report to JSON file."""
    data = {
        "overall": report.compute_overall_mae(),
        "by_position": report.compute_mae_by_position(),
        "videos": [
            {
                "path": r.video_path,
                "success": r.success,
                "error_message": r.error_message if not r.success else None,
                "metadata": {
                    "fps": r.metadata.fps,
                    "frame_count": r.metadata.frame_count,
                    "duration_ms": r.metadata.duration_ms,
                } if r.success else None,
                "errors": [
                    {
                        "position": e.position,
                        "predicted_frame": e.predicted_frame,
                        "ground_truth_frame": e.ground_truth_frame,
                        "error_frames": e.error_frames,
                        "error_ms": e.error_ms,
                    }
                    for e in r.errors
                ] if r.success else None,
            }
            for r in report.video_results
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate P-position detection accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example ground truth JSON format:
{
    "videos": [
        {
            "path": "sample_videos/swing1.mp4",
            "positions": {
                "P1": {"frame": 45},
                "P4": {"frame": 112},
                "P6": {"frame": 138},
                "P9": {"frame": 210}
            }
        }
    ]
}
        """
    )
    parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable video preprocessing (time normalization)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    use_preprocessing = not args.no_preprocess
    if use_preprocessing and not HAS_PREPROCESSOR:
        print("Warning: Video preprocessor not available, running without time normalization")
        use_preprocessing = False

    report = run_evaluation(
        args.ground_truth,
        verbose=args.verbose,
        use_preprocessing=use_preprocessing,
    )
    print_report(report)

    if args.output:
        save_report(report, args.output)


if __name__ == "__main__":
    main()
