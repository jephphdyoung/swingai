# Joint accuracy vs human labels (2026-10-21)

> **Historical result (note added 2026-10-22).** The 2026-06-20 150221 user captures used here were deleted locally, so this can't be reproduced as is. See [STATUS](../STATUS.md).

Labels: `data/joint_labels/labels.json`, verified by Jeff. 20/20 frames, 238/300 joints placed. 14 of these frames include points carried over from an earlier export and kept after checking on the same image.

12 independent (blind) frames of 12 expected, 8 assisted.

- Not scored: `videos/user/Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4`: P1 (frame 395) lacks a labelled shoulder or hip, so errors can't be normalized
- Not scored: `videos/reference/GW_DTL.mp4`: P1 (frame 104) lacks a labelled shoulder or hip, so errors can't be normalized

## Independent (blind) frames: the MediaPipe vs RTMPose comparison

No model output was shown on these frames, so neither model was favoured.

### Face-on

| Model | Frames (missed) | Joints | Joint misses | Mean err | Median err | PCK@0.05 | PCK@0.1 | PCK@0.2 | Visible PCK@0.1 | Occluded: n, answered, PCK@0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MediaPipe (CPU) | 6 (0) | 76 | 0 (0.0%) | 0.166 | 0.111 | 18.4% | 44.7% | 77.6% | 44.7% | 0 |
| RTMPose GPU | 6 (0) | 76 | 0 (0.0%) | 0.074 | 0.064 | 38.2% | 69.7% | 97.4% | 69.7% | 0 |

RTMPose - MediaPipe PCK, 95% interval over 6 labelled frames: pck@0.05 +19.7% [+2.4%, +34.7%]; pck@0.1 +25.0% [+15.9%, +34.3%]; pck@0.2 +19.7% [+6.0%, +36.6%]. An interval containing 0 is no evidence of a difference.

| Joint | MediaPipe (CPU) PCK@0.1 (n) | RTMPose GPU PCK@0.1 (n) |
|---|---|---|
| nose | 50.0% (6) | 66.7% (6) |
| left ear | 33.3% (3) | 100.0% (3) |
| right ear | 40.0% (5) | 60.0% (5) |
| left shoulder | 20.0% (5) | 60.0% (5) |
| right shoulder | 20.0% (5) | 80.0% (5) |
| left elbow | 40.0% (5) | 60.0% (5) |
| right elbow | 20.0% (5) | 40.0% (5) |
| left wrist | 50.0% (4) | 75.0% (4) |
| right wrist | 16.7% (6) | 50.0% (6) |
| left hip | 33.3% (3) | 33.3% (3) |
| right hip | 66.7% (6) | 83.3% (6) |
| left knee | 80.0% (5) | 80.0% (5) |
| right knee | 66.7% (6) | 83.3% (6) |
| left ankle | 50.0% (6) | 66.7% (6) |
| right ankle | 66.7% (6) | 100.0% (6) |

## Assisted frames: reported separately, not a fair comparison

These labels started from model suggestions (the first-choice model alternated), so they lean toward whichever model was accepted. Never pool them with the independent frames.

### Face-on

| Model | Frames (missed) | Joints | Joint misses | Mean err | Median err | PCK@0.05 | PCK@0.1 | PCK@0.2 | Visible PCK@0.1 | Occluded: n, answered, PCK@0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MediaPipe (CPU) | 4 (0) | 55 | 0 (0.0%) | 0.139 | 0.109 | 9.1% | 40.0% | 81.8% | 40.0% | 0 |
| RTMPose GPU | 4 (0) | 55 | 0 (0.0%) | 0.069 | 0.055 | 41.8% | 76.4% | 98.2% | 76.4% | 0 |

RTMPose - MediaPipe PCK, 95% interval over 4 labelled frames: pck@0.05 +32.7% [+22.2%, +42.9%]; pck@0.1 +36.4% [+27.3%, +46.2%]; pck@0.2 +16.4% [+0.0%, +39.6%]. An interval containing 0 is no evidence of a difference.

| Joint | MediaPipe (CPU) PCK@0.1 (n) | RTMPose GPU PCK@0.1 (n) |
|---|---|---|
| nose | 25.0% (4) | 75.0% (4) |
| left ear | 25.0% (4) | 100.0% (4) |
| right ear | 0.0% (2) | 100.0% (2) |
| left shoulder | 0.0% (4) | 50.0% (4) |
| right shoulder | 100.0% (2) | 100.0% (2) |
| left elbow | 0.0% (4) | 75.0% (4) |
| right elbow | 66.7% (3) | 100.0% (3) |
| left wrist | 25.0% (4) | 100.0% (4) |
| right wrist | 0.0% (4) | 50.0% (4) |
| left hip | 50.0% (4) | 50.0% (4) |
| right hip | 75.0% (4) | 50.0% (4) |
| left knee | 75.0% (4) | 100.0% (4) |
| right knee | 50.0% (4) | 50.0% (4) |
| left ankle | 50.0% (4) | 100.0% (4) |
| right ankle | 75.0% (4) | 75.0% (4) |

Errors are in P1 torso lengths from the labels. A miss (no person, or the joint below the model's own threshold) counts as a PCK failure. Occluded labels are the labeller's estimate of a hidden joint; *answered* is how often the model gave a valid point there anyway.
