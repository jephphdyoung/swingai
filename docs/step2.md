# Step 2: Waggle Trimming / Swing Window Extraction

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 1: Time Normalization](step1.md)

## Overview

This step implements robust swing detection to:

1. **Detect the main swing** among waggle/practice swings
2. **Trim waggle** and pre-swing motion
3. **Find P1 (address)** as the last stable frame before takeaway

## Problem

Golf videos often contain:
- Waggle movements before the actual swing
- Practice swings
- Multiple swings in one video
- Extended setup and finish periods

The original P-position detector would find the wrong swing or misidentify P1 when these motions were present.

## Solution

### Swing Detection Algorithm

The algorithm uses **wrist speed with hysteresis** for robust detection:

1. **Baseline Estimation**: Use first ~1s of video to establish baseline motion level
2. **Threshold Calculation**: `start_threshold = median + 3.0 * MAD`, `keep_threshold = median + 1.5 * MAD`
3. **Motion Burst Detection**: Find sustained motion above threshold (≥150ms)
4. **Main Swing Selection**: Score bursts by duration, peak speed, and length
5. **Address Detection**: Find minimum-motion frame in pre-takeaway window

### Key Features

| Feature | Description |
|---------|-------------|
| **Hysteresis** | Different thresholds for starting and maintaining motion state |
| **MAD-based** | Median Absolute Deviation is robust to outliers |
| **Sustained motion** | Requires 150-250ms of continuous motion |
| **Multi-burst handling** | Correctly identifies main swing among waggles |

## Files Created

| File | Purpose |
|------|---------|
| `utils/swing_detector.py` | Swing detection and waggle trimming module |
| `tests/test_swing_detector.py` | 19 unit tests |

## Usage

### Basic Usage

```python
from utils.swing_detector import detect_swing_window, trim_to_swing

# Detect swing window
window = detect_swing_window(landmarks, timestamps_ms)
print(f"Swing: frames {window.start_frame} to {window.end_frame}")
print(f"Takeaway at frame {window.takeaway_frame}")
print(f"Impact at frame {window.impact_frame}")

# Trim to swing
trimmed_lm, trimmed_ts, window = trim_to_swing(landmarks, timestamps_ms)
```

### With Custom Parameters

```python
from utils.swing_detector import SwingDetectionParams, detect_swing_window

params = SwingDetectionParams(
    baseline_duration_ms=1500.0,    # Use 1.5s for baseline
    start_threshold_mad=4.0,        # More conservative start
    min_motion_duration_ms=200.0,   # Require 200ms sustained motion
)

window = detect_swing_window(landmarks, timestamps_ms, params)
```

### Integration with Pipeline

```python
from utils.video_preprocessor import preprocess_video
from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import trim_to_swing
from utils.pose_features import detect_p_positions

# 1. Preprocess
preprocessed = preprocess_video("swing.mp4")

# 2. Extract poses
result = extract_pose_with_timestamps(
    preprocessed.analysis_path,
    preprocessed.frame_timestamps_ms
)

# 3. Detect and trim to swing
trimmed_lm, trimmed_ts, window = trim_to_swing(
    result.landmarks,
    result.timestamps_ms
)

if window:
    print(f"Swing detected with {window.confidence:.0%} confidence")
    # 4. Detect P-positions on trimmed swing
    positions = detect_p_positions(trimmed_lm, timestamps_ms=trimmed_ts)
```

## Algorithm Details

### Baseline Estimation

```
baseline_frames = first 1000ms of video
median = median(speeds[baseline_frames])
MAD = median(|speeds - median|)
```

MAD (Median Absolute Deviation) is used instead of standard deviation because it's robust to outliers (like a waggle in the baseline period).

### Motion Burst Detection (with Hysteresis)

```
start_threshold = median + 3.0 * MAD  # High bar to start
keep_threshold = median + 1.5 * MAD   # Lower bar to maintain

State machine:
- IDLE: speed > start_threshold → IN_MOTION
- IN_MOTION: speed < keep_threshold → IDLE (if duration >= min_duration)
```

Hysteresis prevents rapid state transitions from noise.

### Main Swing Selection Scoring

```python
# Duration score (0.5-2s is ideal)
if 500ms <= duration <= 2000ms:
    duration_score = 1.0
elif 300ms <= duration <= 3000ms:
    duration_score = 0.7
else:
    duration_score = 0.3

# Combined score
score = duration_score * 0.4 + speed_score * 0.4 + length_score * 0.2
```

### Address (P1) Detection

Look back from takeaway frame within `pre_window_ms` (default 500ms) and find the frame with minimum wrist speed (most stable position).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_duration_ms` | 1000.0 | Duration to estimate baseline |
| `start_threshold_mad` | 3.0 | MAD multiplier to start motion |
| `keep_threshold_mad` | 1.5 | MAD multiplier to maintain motion |
| `min_motion_duration_ms` | 150.0 | Minimum sustained motion |
| `max_motion_duration_ms` | 3000.0 | Maximum swing duration |
| `pre_window_ms` | 500.0 | Window to search for P1 |
| `min_burst_frames` | 5 | Minimum frames in a burst |

## Output: SwingWindow

```python
@dataclass
class SwingWindow:
    start_frame: int      # P1 candidate (address)
    end_frame: int        # End of swing motion
    takeaway_frame: int   # Where takeaway begins
    impact_frame: int     # Estimated impact (max speed)
    confidence: float     # Detection confidence (0-1)
```

## Testing

```bash
# Run all swing detector tests
pytest tests/test_swing_detector.py -v

# Run all tests
pytest tests/ -v
```

## Next Steps

- [Step 3: Resample Features to 60Hz](step3.md) - Normalize feature sampling rate
