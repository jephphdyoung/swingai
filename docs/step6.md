# Step 6: Club Tracking for P2/P6

> Part of the [SwingAI P-Position Improvement Project](../README.md)
> Previous: [Step 5: DTW Label Transfer](step5.md)

## Overview

This step uses golf club shaft detection to refine shaft-parallel positions:

- **P2**: Club parallel to ground during backswing
- **P6**: Impact (club near vertical)
- **P7**: Club parallel during follow-through

## Problem

Pose-only detection can't precisely identify shaft-parallel positions because:
- MediaPipe doesn't track the club
- Wrist position alone doesn't indicate shaft angle
- P2 and its follow-through equivalent require knowing club orientation

## Solution

Use Hough line detection to track club shaft angle, then find frames where shaft is:
- **~80° from vertical** = Shaft parallel (P2, P7)
- **~0-15° from vertical** = Impact (P6)

## Files Created/Modified

| File | Purpose |
|------|---------|
| `utils/club_positions.py` | **NEW** - Club-based position detection |
| `tests/test_club_positions.py` | **NEW** - 16 unit tests |
| `utils/club_detector.py` | Existing Hough line detection (unchanged) |

## Usage

### Basic Usage

```python
from utils.club_positions import detect_club_positions

result = detect_club_positions(
    video_path="swing.mp4",
    landmarks=landmarks,
    p_positions={"P1": 50, "P4": 120, "P6": 145, "P9": 200}
)

print(f"P2 (backswing parallel): frame {result.p2_frame}")
print(f"P6 (impact refined): frame {result.p6_frame}")
print(f"P7 (follow-through parallel): frame {result.p2_follow_frame}")
print(f"Detection confidence: {result.confidence:.0%}")
```

### Refine Existing Positions

```python
from utils.club_positions import refine_positions_with_club

# Refine positions using club tracking
refined = refine_positions_with_club(
    video_path="swing.mp4",
    landmarks=landmarks,
    positions={"P1": 50, "P2": 80, "P4": 120, "P6": 145, "P9": 200},
)

# refined dict has updated P2, P6, P7 if club detection was reliable
```

## Club Angle Definitions

| Position | Angle from Vertical | Description |
|----------|---------------------|-------------|
| Address | ~45° | Club angled toward ball |
| P2 | ~80° (±15°) | Shaft parallel to ground |
| P4 (Top) | ~85-90° | May go past horizontal |
| Impact (P6) | ~0-15° | Club near vertical |
| P7 | ~80° (±15°) | Shaft parallel (follow-through) |
| Finish | ~45-60° | Club over shoulder |

## Detection Process

1. **Track club shaft** using existing `club_detector.py`
2. **Extract angles** for each frame (degrees from vertical)
3. **Interpolate** missing angles
4. **Find P2**: First frame reaching ~80° between P1 and P4
5. **Find P6**: Most vertical club between P4 and P9
6. **Find P7**: First frame reaching ~80° after P6

## Confidence and Reliability

The module is conservative about club detection:

| Detection Rate | Confidence | Action |
|----------------|------------|--------|
| >90% | Low (0.3) | Likely detecting body edges, not club |
| 30-90% | Good (0.5-0.8) | Reliable club tracking |
| <30% | Very low (0.2) | Insufficient data, skip club refinement |

## Output: ClubPositionResult

```python
@dataclass
class ClubPositionResult:
    p2_frame: Optional[int]       # Backswing shaft parallel
    p6_frame: Optional[int]       # Refined impact
    p2_follow_frame: Optional[int] # Follow-through parallel (~P7)
    confidence: float             # Overall confidence (0-1)
    detection_rate: float         # Fraction of frames with club detected
    angles: list[float]           # Interpolated angles per frame
```

## Integration with Full Pipeline

```python
from utils.video_preprocessor import preprocess_video
from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import trim_to_swing
from utils.dtw_alignment import detect_p_positions_with_dtw
from utils.club_positions import refine_positions_with_club

# ... preprocess, extract, trim, DTW align ...

# After DTW alignment
dtw_positions = detect_p_positions_with_dtw(
    user_landmarks, ref_landmarks, ref_positions,
    user_timestamps, ref_timestamps,
)

# Refine P2, P6, P7 with club tracking
final_positions = refine_positions_with_club(
    video_path,
    landmarks,
    dtw_positions.positions,
    timestamps,
)
```

## Limitations

- **Requires video file access** (not just landmarks)
- **Club must be visible** - obscured clubs won't be detected
- **Metallic shafts work best** - graphite may have lower contrast
- **Face-on view recommended** - DTL view may occlude shaft

## Testing

```bash
# Run club position tests
pytest tests/test_club_positions.py -v

# Run all tests
pytest tests/ -v
```

## Summary

This completes the P-position improvement pipeline:

```
Video → Preprocess → Pose Extract → Trim Waggle → Resample 60Hz
                                                        ↓
                                      DTW Features → DTW Align → Label Transfer
                                                        ↓
                                              Club Track → Refine P2/P6/P7
                                                        ↓
                                              Final P-Positions (P1-P9)
```
