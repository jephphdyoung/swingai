# Step 0: Evaluation Harness

> Part of the [SwingAI P-Position Improvement Project](../README.md)

## Overview

This step adds an evaluation harness to measure P-position detection accuracy. The harness loads swing videos with ground truth annotations and reports Mean Absolute Error (MAE) in both frames and milliseconds.

## Files Created

| File | Purpose |
|------|---------|
| `scripts/eval_positions.py` | Main evaluation script |
| `data/ground_truth_example.json` | Example ground truth schema |
| `tests/test_eval_positions.py` | Unit tests (23 tests) |

## Usage

```bash
# Basic evaluation
python scripts/eval_positions.py --ground-truth data/ground_truth.json

# Verbose output (shows per-video details)
python scripts/eval_positions.py --ground-truth data/ground_truth.json --verbose

# Save results to JSON
python scripts/eval_positions.py --ground-truth data/ground_truth.json --output results.json
```

## Ground Truth JSON Format

```json
{
  "metadata": {
    "description": "Ground truth annotations",
    "version": "1.0",
    "annotator": "name",
    "date": "2025-01-25"
  },
  "videos": [
    {
      "path": "path/to/video.mp4",
      "notes": "optional notes",
      "positions": {
        "P1": {"frame": 45},
        "P4": {"frame": 112},
        "P6": {"frame": 138},
        "P9": {"frame": 210}
      }
    }
  ]
}
```

### Position Formats

You can specify positions using either frame numbers or timestamps:

```json
// Frame-based (0-indexed)
"P1": {"frame": 45}

// Timestamp-based (milliseconds)
"P1": {"timestamp_ms": 1500}

// Direct frame number
"P1": 45
```

## Sample Output

```
============================================================
P-POSITION DETECTION EVALUATION REPORT
============================================================

Videos evaluated: 2/2 successful

MAE by P-Position:
--------------------------------------------------
Position   MAE (frames)    MAE (ms)        N
--------------------------------------------------
P1             3.50 +/- 1.50     116.7 +/- 50.0   2
P4             2.00 +/- 1.00      66.7 +/- 33.3   2
P6             1.50 +/- 0.50      50.0 +/- 16.7   2
P9             4.00 +/- 2.00     133.3 +/- 66.7   2
--------------------------------------------------
OVERALL        2.75 +/- 1.25      91.7 +/- 41.7   8
============================================================
```

## Running Tests

```bash
pytest tests/test_eval_positions.py -v
```

## Creating Ground Truth

To create accurate ground truth annotations:

1. Open video in a frame-by-frame viewer (e.g., VLC with `e` key, or ffmpeg)
2. Note the 0-indexed frame number for each P-position:
   - **P1**: Address (last stable frame before takeaway)
   - **P2**: Shaft parallel to ground (backswing)
   - **P3**: Lead arm parallel to ground (backswing)
   - **P4**: Top of backswing
   - **P5**: Lead arm parallel (downswing)
   - **P6**: Impact
   - **P7**: Shaft parallel (follow-through)
   - **P8**: Lead arm parallel (follow-through)
   - **P9**: Finish position
3. Add to JSON file

### Extract frames for verification

```bash
# Extract frame 100 from video
ffmpeg -i video.mp4 -vf "select=eq(n\,100)" -vframes 1 frame_100.png

# Extract frames around suspected P-position
ffmpeg -i video.mp4 -vf "select=between(n\,95\,105)" -vsync 0 frames/frame_%04d.png
```

## Next Steps

- [Step 1: Time Normalization](step1.md) - Handle VFR videos and mixed frame rates
