# Step 4: DTW Feature Improvements

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 3: Resample Features](step3.md)

## Overview

This step improves DTW alignment by:

1. **Coordinate normalization** - Pelvis-centered, scale-normalized
2. **Angle features** - Spine tilt, joint angles
3. **Smoothing** - Savitzky-Golay filter before velocity computation
4. **Wrist speed** - Time-normalized velocity feature

## Problem

Raw landmark coordinates are problematic for DTW because:
- Different body sizes produce different absolute coordinates
- Camera position/distance affects scale
- Noise in landmarks creates noisy velocities

## Solution

### Coordinate Normalization

```
1. Center at pelvis (hip midpoint)
2. Scale by torso length (shoulder-to-hip distance)
3. Track shoulder rotation for reference
```

### Angle Features

| Feature | Description |
|---------|-------------|
| `spine_tilt` | Angle from vertical |
| `shoulder_tilt` | Shoulder line angle from horizontal |
| `hip_tilt` | Hip line angle from horizontal |
| `left_elbow_angle` | Left elbow flexion (shoulder-elbow-wrist) |
| `right_elbow_angle` | Right elbow flexion |
| `left_knee_angle` | Left knee flexion (hip-knee-ankle) |
| `right_knee_angle` | Right knee flexion |

### Smoothing

Savitzky-Golay filter (window=7, polynomial=2) applied before computing velocities to reduce noise.

## Files Created

| File | Purpose |
|------|---------|
| `utils/dtw_features.py` | DTW feature extraction module |
| `tests/test_dtw_features.py` | 26 unit tests |

## Usage

### Extract Features

```python
from utils.dtw_features import extract_dtw_features, features_to_matrix

# Extract features (with smoothing)
features = extract_dtw_features(landmarks, timestamps_ms, smooth=True)

# Convert to matrix for DTW
matrix = features_to_matrix(features)
print(f"Feature matrix shape: {matrix.shape}")  # (n_frames, 71)
```

### Normalize Pose

```python
from utils.dtw_features import normalize_pose

# Normalize single frame
normalized = normalize_pose(frame_landmarks)
print(f"Scale: {normalized.scale}")
print(f"Rotation: {normalized.rotation}")
```

### Extract Angles

```python
from utils.dtw_features import extract_angles

angles = extract_angles(frame_landmarks)
print(f"Spine tilt: {np.degrees(angles['spine_tilt']):.1f}°")
print(f"Left elbow: {np.degrees(angles['left_elbow_angle']):.1f}°")
```

### Full Pipeline

```python
from utils.dtw_features import (
    extract_dtw_features,
    features_to_matrix,
    normalize_feature_matrix,
)
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Extract features from both sequences
user_features = extract_dtw_features(user_landmarks, user_timestamps)
ref_features = extract_dtw_features(ref_landmarks, ref_timestamps)

# Convert to matrices
user_matrix = features_to_matrix(user_features)
ref_matrix = features_to_matrix(ref_features)

# Optional: Z-score normalize
user_matrix = normalize_feature_matrix(user_matrix)
ref_matrix = normalize_feature_matrix(ref_matrix)

# Run DTW
distance, path = fastdtw(user_matrix, ref_matrix, dist=euclidean)
```

## Feature Vector Structure

Total feature dimension: **71**

```
[0:63]   - Normalized coordinates (21 landmarks × 3 coords)
[63:64]  - spine_tilt
[64:65]  - shoulder_tilt
[65:66]  - hip_tilt
[66:67]  - left_elbow_angle
[67:68]  - right_elbow_angle
[68:69]  - left_knee_angle
[69:70]  - right_knee_angle
[70:71]  - wrist_speed
```

## Key Functions

| Function | Description |
|----------|-------------|
| `normalize_pose()` | Pelvis-center and scale normalize |
| `extract_angles()` | Compute all angle features |
| `smooth_landmarks()` | Savitzky-Golay smoothing |
| `extract_dtw_features()` | Full feature extraction |
| `features_to_matrix()` | Convert to DTW-ready matrix |
| `normalize_feature_matrix()` | Z-score normalization |

## Normalization Details

### Pelvis Centering

```python
pelvis = (left_hip + right_hip) / 2
centered = landmarks - pelvis
```

### Scale Normalization

```python
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2
torso_length = ||shoulder_center - hip_center||
scaled = centered / torso_length
```

### Savitzky-Golay Smoothing

```python
from scipy.signal import savgol_filter

smoothed = savgol_filter(coords, window_length=7, polyorder=2)
```

Parameters chosen for:
- Window=7: ~117ms at 60fps, smooths noise without losing swing dynamics
- Polyorder=2: Preserves acceleration features

## Testing

```bash
# Run DTW feature tests
pytest tests/test_dtw_features.py -v

# Run all tests
pytest tests/ -v
```

## Next Steps

- [Step 5: DTW Label Transfer + Local Refinement](step5.md) - Use DTW for P-position alignment
