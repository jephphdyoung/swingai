# Step 5: DTW Label Transfer + Local Refinement

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 4: DTW Feature Improvements](step4.md)

## Overview

This step uses Dynamic Time Warping to:

1. **Align user swing to reference swing** using improved DTW features
2. **Transfer P-position labels** from reference to user via DTW path
3. **Locally refine** transferred positions using biomechanical cues

## Problem

Detecting P-positions from scratch is error-prone because:
- Absolute frame positions vary with swing speed
- Velocity-only detection misses subtle position markers
- No ground truth to anchor detection

## Solution

Use a reference swing with known P-positions as a template, then:
1. DTW-align user features to reference features
2. Map reference P-frames to user frames through alignment path
3. Refine key positions (P4, P6, P7) using local biomechanical rules

## Files Created

| File | Purpose |
|------|---------|
| `utils/dtw_alignment.py` | DTW alignment and label transfer |
| `tests/test_dtw_alignment.py` | 19 unit tests |

## Usage

### Main Entry Point

```python
from utils.dtw_alignment import detect_p_positions_with_dtw

# Reference swing with known positions
ref_positions = {
    "P1": 50,   # Address
    "P4": 120,  # Top of backswing
    "P6": 145,  # Impact
    "P9": 200,  # Finish
}

# Detect user positions via DTW
result = detect_p_positions_with_dtw(
    user_landmarks,
    ref_landmarks,
    ref_positions,
    user_timestamps,
    ref_timestamps,
)

print(f"User P4: frame {result.positions['P4']}")
print(f"Confidence: {result.confidences['P4']:.0%}")
print(f"Method: {result.method['P4']}")
```

### Step-by-Step Usage

```python
from utils.dtw_alignment import (
    compute_dtw_alignment,
    transfer_positions,
    refine_positions,
)

# Step 1: Compute DTW alignment
alignment = compute_dtw_alignment(
    user_landmarks, ref_landmarks,
    user_timestamps, ref_timestamps,
)
print(f"DTW distance: {alignment.distance}")

# Step 2: Transfer positions
transferred = transfer_positions(ref_positions, alignment)
print(f"Transferred P4: {transferred['P4']}")

# Step 3: Refine with local rules
refined = refine_positions(user_landmarks, user_timestamps, transferred)
print(f"Refined P4: {refined.positions['P4']}")
```

## Local Refinement Rules

### P4: Top of Backswing

Detect using **hand height peak** (minimum Y coordinate):

```python
# Search window around DTW-transferred position
# Find frame where wrist Y is minimum (highest in image)
```

### P6: Impact

Detect using **velocity spike**:

```python
# Find maximum wrist velocity in search window
# Impact has highest hand speed
```

### P7: Follow-Through

Detect using **velocity pattern**:

```python
# Look for secondary velocity peak or inflection point
# after impact
```

### Interpolated Positions (P2, P3, P5, P8)

Distributed proportionally between key positions:
- P2, P3: 1/3 and 2/3 of backswing (P1→P4)
- P5: Midpoint of downswing (P4→P6)
- P8: 2/3 of follow-through (P6→P9)

## Output: RefinedPositions

```python
@dataclass
class RefinedPositions:
    positions: dict[str, int]    # Position -> frame number
    confidences: dict[str, float]  # Position -> confidence (0-1)
    method: dict[str, str]       # Position -> detection method

# Example:
result.positions = {"P1": 50, "P4": 120, "P6": 145, ...}
result.confidences = {"P1": 0.8, "P4": 0.9, "P6": 0.85, ...}
result.method = {
    "P1": "dtw_transfer",
    "P4": "hand_height_peak",
    "P6": "velocity_peak",
    "P2": "interpolated",
    ...
}
```

## Full Pipeline Integration

```python
from utils.video_preprocessor import preprocess_video
from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import trim_to_swing
from utils.feature_resampler import resample_landmarks
from utils.dtw_alignment import detect_p_positions_with_dtw

# Preprocess both videos
user_prep = preprocess_video("user_swing.mp4")
ref_prep = preprocess_video("reference_swing.mp4")

# Extract poses
user_result = extract_pose_with_timestamps(user_prep.analysis_path, user_prep.frame_timestamps_ms)
ref_result = extract_pose_with_timestamps(ref_prep.analysis_path, ref_prep.frame_timestamps_ms)

# Trim to swing
user_lm, user_ts, _ = trim_to_swing(user_result.landmarks, user_result.timestamps_ms)
ref_lm, ref_ts, _ = trim_to_swing(ref_result.landmarks, ref_result.timestamps_ms)

# Resample to 60Hz
user_resampled = resample_landmarks(user_lm, user_ts, target_fps=60)
ref_resampled = resample_landmarks(ref_lm, ref_ts, target_fps=60)

# Known reference positions (from manual annotation)
ref_positions = {"P1": 0, "P4": 60, "P6": 90, "P9": 150}

# Detect user positions via DTW
result = detect_p_positions_with_dtw(
    user_resampled.landmarks,
    ref_resampled.landmarks,
    ref_positions,
    user_resampled.timestamps_ms,
    ref_resampled.timestamps_ms,
)

for pos in ["P1", "P4", "P6", "P9"]:
    print(f"{pos}: frame {result.positions[pos]} "
          f"({result.confidences[pos]:.0%} confidence, {result.method[pos]})")
```

## Confidence Levels

| Level | Meaning |
|-------|---------|
| 0.8+ | High confidence (clear biomechanical signal) |
| 0.6-0.8 | Medium confidence (DTW transfer or good local match) |
| 0.4-0.6 | Lower confidence (interpolated or weak signal) |
| <0.4 | Low confidence (fallback estimate) |

## Testing

```bash
# Run DTW alignment tests
pytest tests/test_dtw_alignment.py -v

# Run all tests
pytest tests/ -v
```

## Next Steps

- [Step 6: Club Tracking for P2/P6](step6.md) (Optional) - Detect shaft angle for precise P2/P6
