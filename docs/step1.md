# Step 1: Time Normalization

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 0: Evaluation Harness](step0.md)

## Overview

This step handles Variable Frame Rate (VFR) videos and mixed frame rates by:

1. **Extracting per-frame timestamps** using ffprobe's `best_effort_timestamp_time`
2. **Transcoding to CFR 60fps** for consistent analysis
3. **Computing time-based velocities** using actual `dt` instead of assuming constant frame intervals

## Problem

Golf swings are recorded at various frame rates:
- Smartphone videos: Often VFR (variable frame rate)
- Standard video: 30fps or 60fps
- Slow-motion: 120fps or 240fps

The original code assumed constant frame rate, causing:
- Incorrect velocity calculations when dt varies
- Frame misalignment when comparing videos at different rates
- Poor P-position detection on VFR footage

## Solution

### Video Preprocessor Module

New file: `utils/video_preprocessor.py`

```python
from utils.video_preprocessor import preprocess_video, PreprocessedVideo

# Preprocess video (transcodes VFR to CFR 60fps)
result = preprocess_video("swing.mp4", target_fps=60)

# Use the analysis copy and timestamps
landmarks = extract_pose(result.analysis_path, result.frame_timestamps_ms)

# Cleanup temp files when done
cleanup_temp_files(result)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_video_info()` | Extract metadata (fps, frame count, VFR detection) |
| `extract_frame_timestamps()` | Get per-frame timestamps using ffprobe |
| `transcode_to_cfr()` | Convert VFR/other fps to CFR 60fps |
| `preprocess_video()` | Full preprocessing pipeline |
| `compute_frame_deltas()` | Calculate time between consecutive frames |

### Updated Pose Extraction

The pose extractor now accepts timestamps:

```python
from utils.pose_extractor import extract_pose_with_timestamps

# New interface with timing info
result = extract_pose_with_timestamps(video_path, timestamps_ms)
# result.landmarks, result.timestamps_ms, result.fps
```

### Updated P-Position Detection

Velocities are now computed using actual time deltas:

```python
from utils.pose_features import detect_p_positions

# Pass timestamps for accurate velocity calculation
positions = detect_p_positions(landmarks, timestamps_ms=timestamps)
```

**Before (frame-based):**
```python
velocity[i] = position[i] - position[i-1]  # Assumes constant dt
```

**After (time-based):**
```python
dt = timestamps[i] - timestamps[i-1]
velocity[i] = (position[i] - position[i-1]) / dt * 33.33  # Normalized to ~30fps
```

## Files Modified

| File | Changes |
|------|---------|
| `utils/video_preprocessor.py` | **NEW** - Video preprocessing module |
| `utils/pose_extractor.py` | Added `extract_pose_with_timestamps()`, timestamps parameter |
| `utils/pose_features.py` | Time-based velocity calculation with `timestamps_ms` parameter |
| `tests/test_video_preprocessor.py` | **NEW** - 14 unit tests |

## Usage Examples

### Basic Preprocessing

```python
from utils.video_preprocessor import preprocess_video

# Automatic VFR detection and transcoding
result = preprocess_video("smartphone_video.mp4")

print(f"Original: {result.original_info.fps}fps, VFR={result.original_info.is_vfr}")
print(f"Analysis: {result.analysis_path}")
print(f"Transcoded: {result.is_transcoded}")
```

### Full Pipeline

```python
from utils.video_preprocessor import preprocess_video, cleanup_temp_files
from utils.pose_extractor import extract_pose_with_timestamps
from utils.pose_features import detect_p_positions

# 1. Preprocess
preprocessed = preprocess_video("swing.mp4", target_fps=60)

# 2. Extract poses with timestamps
result = extract_pose_with_timestamps(
    preprocessed.analysis_path,
    preprocessed.frame_timestamps_ms
)

# 3. Detect P-positions with time-aware velocities
positions = detect_p_positions(
    result.landmarks,
    timestamps_ms=result.timestamps_ms
)

# 4. Cleanup
cleanup_temp_files(preprocessed)
```

### Command Line (ffprobe)

```bash
# Check if video is VFR
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,avg_frame_rate \
  -of json video.mp4

# Extract all frame timestamps
ffprobe -v error -select_streams v:0 \
  -show_entries frame=best_effort_timestamp_time \
  -of json video.mp4

# Transcode to CFR 60fps
ffmpeg -i input.mp4 -vf fps=60 -vsync cfr -c:v libx264 output_cfr60.mp4
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run just preprocessor tests
pytest tests/test_video_preprocessor.py -v
```

## Dependencies

- `ffprobe` (part of ffmpeg) - for metadata extraction
- `ffmpeg` - for transcoding

These are already in the Dockerfile. For local dev:
```bash
# Fedora/RHEL
sudo dnf install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Next Steps

- [Step 2: Waggle Trimming](step2.md) - Detect swing start and trim waggle/practice swings
