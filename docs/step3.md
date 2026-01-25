# Step 3: Resample Features to Fixed 60Hz Rate

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 2: Waggle Trimming](step2.md)

## Overview

This step resamples pose features to a fixed 60Hz rate using interpolation, making sequences comparable across different frame rates.

## Problem

Even after transcoding to 60fps, sequences may have:
- Slightly different frame rates
- Different durations
- Non-uniform timing from VFR sources

This makes frame-by-frame comparison and DTW alignment inconsistent.

## Solution

Resample all pose landmark sequences to exactly 60Hz using linear interpolation.

## Files Created

| File | Purpose |
|------|---------|
| `utils/feature_resampler.py` | Feature resampling utilities |
| `tests/test_feature_resampler.py` | 23 unit tests |

## Usage

### Basic Resampling

```python
from utils.feature_resampler import resample_landmarks

# Resample to 60Hz
result = resample_landmarks(landmarks, timestamps_ms, target_fps=60.0)

print(f"Resampled from {len(landmarks)} to {result.n_frames} frames")
print(f"Duration: {result.original_duration_ms:.0f}ms at {result.target_fps}Hz")
```

### Resample to Specific Length

```python
from utils.feature_resampler import resample_to_length

# Make sequence exactly 200 frames
resampled = resample_to_length(landmarks, target_length=200)
```

### Align Two Sequences

```python
from utils.feature_resampler import align_to_common_rate

# Align user and reference to same rate
user_resampled, ref_resampled = align_to_common_rate(
    user_landmarks, user_timestamps,
    ref_landmarks, ref_timestamps,
    target_fps=60.0
)
```

### Full Pipeline Integration

```python
from utils.video_preprocessor import preprocess_video
from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import trim_to_swing
from utils.feature_resampler import resample_landmarks
from utils.pose_features import detect_p_positions

# 1. Preprocess
preprocessed = preprocess_video("swing.mp4")

# 2. Extract poses
result = extract_pose_with_timestamps(
    preprocessed.analysis_path,
    preprocessed.frame_timestamps_ms
)

# 3. Trim to swing
trimmed_lm, trimmed_ts, window = trim_to_swing(
    result.landmarks,
    result.timestamps_ms
)

# 4. Resample to fixed 60Hz
resampled = resample_landmarks(trimmed_lm, trimmed_ts, target_fps=60.0)

# 5. Detect P-positions
positions = detect_p_positions(
    resampled.landmarks,
    timestamps_ms=resampled.timestamps_ms
)
```

## Key Functions

| Function | Description |
|----------|-------------|
| `resample_landmarks()` | Resample to target FPS using interpolation |
| `resample_to_length()` | Resample to specific number of frames |
| `compute_uniform_timestamps()` | Generate uniform timestamp array |
| `estimate_fps_from_timestamps()` | Estimate FPS from timestamp array |
| `align_to_common_rate()` | Resample two sequences to same rate |

## Output: ResampledFeatures

```python
@dataclass
class ResampledFeatures:
    landmarks: list           # Resampled landmarks
    timestamps_ms: list[float]  # Uniform timestamps (starting at 0)
    original_duration_ms: float # Original duration
    n_frames: int             # Number of resampled frames
    target_fps: float         # Target frame rate
```

## Interpolation Details

Uses `scipy.interpolate.interp1d` with linear interpolation:

```python
interp_func = interpolate.interp1d(
    t_original,
    values,
    kind='linear',
    fill_value='extrapolate'
)
resampled_values = interp_func(t_target)
```

Linear interpolation is chosen for:
- Speed (important for real-time processing)
- Simplicity (no oscillation artifacts)
- Adequate accuracy for pose landmarks

## Testing

```bash
# Run resampler tests
pytest tests/test_feature_resampler.py -v

# Run all tests
pytest tests/ -v
```

## Next Steps

- [Step 4: DTW Feature Improvements](step4.md) - Better features for alignment
