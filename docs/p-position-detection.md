# P-Position Detection

How SwingAI locates P1–P10 in any down-the-line or face-on swing, and what the
body should be doing at each one.

## Source of truth

Body positions in this document follow **Bennett & Plummer, *The Stack & Tilt
Swing*** (Gotham Books, 2009). Where a checkpoint below states what the golfer
should do, that is the Stack & Tilt model, restated as something measurable from
pose landmarks.

The P-position *definitions* follow the standard P-System. That system is
normative here: **P7 is impact**; P6 is shaft-parallel in the downswing.

Handedness: the book is written for a right-handed golfer. This document uses
**lead** (left for a right-hander) and **trail** (right for a right-hander), so
detection works for either. Handedness is a per-video property alongside view
angle.

---

## Capture requirements

The book's own video guidance defines the geometry the detector can rely on:

| Requirement | Value |
|---|---|
| Face-on camera | Perpendicular to the intended start line |
| Down-the-line camera | Along the stance line |
| Camera height | **Sternum height** — the sternum is the center of the swing |
| Consistency | Same perspective and framing between sessions |
| Sampling | Record several swings; identify the pattern, discard outliers |

Camera height matters for measurement, not just aesthetics: angles measured off
a camera set at eye level or from a squat are distorted relative to the model's
prescribed angles. Treat a framing change as invalidating cross-session
comparison.

---

## The ten positions

| P | Definition | Primary signal |
|---|---|---|
| P1 | Address | Stillness before motion onset |
| P2 | Shaft parallel to ground, backswing | Shaft angle crosses horizontal |
| P3 | Lead arm parallel to ground, backswing | Lead `shoulder→wrist` crosses horizontal |
| P4 | Top of backswing | Hand-path direction reversal |
| P5 | Lead arm parallel to ground, downswing | Lead `shoulder→wrist` crosses horizontal |
| P6 | Shaft parallel to ground, downswing | Shaft angle crosses horizontal |
| P7 | Impact | Peak hand/clubhead speed |
| P8 | Shaft parallel to ground, follow-through | Shaft angle crosses horizontal |
| P9 | Lead arm parallel to ground, follow-through | Lead `shoulder→wrist` crosses horizontal |
| P10 | Finish | Motion settles |

Every geometric position is *an angle signal crossing horizontal*. Two
consequences worth building on:

- **Sub-frame precision** — interpolate the crossing between bracketing frames
  for a timestamp finer than the frame interval.
- **Confidence for free** — a steep, clean crossing is a confident detection; a
  shallow or noisy one is not. Use that to decide when to fall back to
  reference-based labeling.

Store every result as `timestamp_ms`. Frame indices are not comparable across a
240fps capture and a 30fps reference.

---

## Body checkpoints

What the golfer should be doing at each position, and the landmark measurement
that expresses it.

### P1 — Address

- **Weight favors the lead foot**, at least 55/45. A developing player should set
  60–70% forward.
- Spine is in forward flex, inclined toward the ball.
- Shoulder center sits above the sternum; hip center below it. These two centers
  define the swing's dynamic center for the whole motion.

*Measure:* lead/trail balance of the hip and shoulder centers over the stance
width; spine vector (hip center → shoulder center) against vertical; ankle
midpoint as the stance reference.

### P2 — Shaft parallel, backswing

- Hands and club are already moving **inward**, on a circular arc around the
  body — not straight back along the target line.
- Lead shoulder is turning **downward** rather than level.
- Shoulder center has not moved laterally off the ball.

*Measure:* hand-path lateral displacement toward the body relative to the target
line; lead shoulder height delta from P1; shoulder-center lateral drift (target:
no movement away from the target).

### P3 — Lead arm parallel, backswing

- Lead arm is straight, preserving the swing radius.
- Spine has **extended** from its address flex toward vertical, and **tilts
  toward the target** (lead side), keeping the head over the lead leg.
- Hips have not shifted away from the target.

*Measure:* lead elbow angle (target: near-straight); spine vector flexion delta
from P1 (target: extending); spine lateral tilt toward the lead side; hip-center
lateral position (target: unchanged from P1).

### P4 — Top of backswing

- **Shoulders turned ~90°**, made up of ~45° of shoulder rotation on top of ~45°
  of hip turn — the hips turn rather than resist.
- **Trail leg straight.** Straightening it is what frees the hips to turn.
- Lead arm ~45° across the chest; trail elbow bent ~90°. The arms stay on the
  chest rather than lifting.
- Weight is still forward. No shift to the trail side has occurred.
- Head remains over the lead leg.

*Measure:* shoulder-line rotation and hip-line rotation in the transverse plane
(DTL gives the cleaner read); trail knee angle (target: extended); lead-arm angle
across the chest plane; trail elbow angle; hip- and shoulder-center lateral
position against P1.

### P5 — Lead arm parallel, downswing

- **Hips slide toward the target** — linear motion, not just rotation.
- Head stays back; the spine begins tilting away from the target as the hips
  move under it.
- Arms remain straight; elbow separation unchanged from address.
- Hands stay inside, keeping the club approaching from the inside.

*Measure:* hip-center lateral displacement toward the target from P4; head
lateral position (target: stable); inter-elbow distance against its P1 value;
hand-path lateral position relative to the target line.

### P6 — Shaft parallel, downswing

- Legs are straightening — the body pushes off the ground.
- Belt level is **rising**.
- Wrists still hold their hinge; the club has not been thrown outward.

*Measure:* knee angles trending toward extension; hip-center vertical rise from
P4; wrist angle relative to the forearm (target: retained); club path inside the
target line.

### P7 — Impact

- **Shaft leans toward the target.** Forward lean puts the low point ahead of the
  ball, so the club strikes ball first, then ground.
- Weight is forward — more forward than at address.
- Wrists have returned to roughly their address angles. No flip.
- Arms are straight; swing radius is at full length through the ball.
- Butt is tucking under the torso; hips continue turning and rising.
- Belt has risen measurably from address.

*Measure:* shaft angle against vertical, signed toward the target; hip-center
lateral position (target: ahead of its P1 position); lead wrist angle against its
P1 value; lead elbow angle (target: straight); hip-center height delta from P1.

### P8 — Shaft parallel, follow-through

- Arms stay straight; they must not fold early.
- Hips keep turning; the spine keeps extending.
- Club continues outward, away from the body, rather than cutting across.

*Measure:* both elbow angles (target: still extended); hip rotation continuing
past its P7 value; club path relative to the target line.

### P9 — Lead arm parallel, follow-through

- Body is still rotating; the chest faces the target or beyond.
- Spine is extending past vertical.

*Measure:* shoulder-line rotation past the target line; spine vector against
vertical, extending beyond it.

### P10 — Finish

- Spine is **extended past vertical**, tilted away from the target.
- Head has stayed back — it should not have moved ahead of the lead foot.
- Weight is fully on the lead side; body has turned through.

*Measure:* spine vector against vertical (target: past it, tilted away from
target); head lateral position relative to the lead ankle (target: behind it);
hip-center over the lead foot; motion settled below threshold.

---

## Continuous checks

Some of the model is about the *path between* positions, not a single frame.
These are per-frame series worth computing across the whole swing:

| Check | Target |
|---|---|
| Shoulder-center lateral position | Fixed through the backswing — it is the axis the hands orbit |
| Hip-center lateral position | Still going back; slides toward the target coming down |
| Head lateral position | Over the lead leg going back; stays back through impact |
| Spine flexion/extension | Flexed at address → extending going back → re-flexed approaching impact → extended past vertical at the finish |
| Weight distribution | Forward at address and never moving to the trail side |
| Hand path | Inward on an arc, both directions |
| Elbow separation | Constant from address through impact |
| Belt height | Rising from the top through impact |

The spine flexion curve and the hip-center lateral curve are, between them, most
of the Stack & Tilt model. Both are single scalar time series computable from
landmarks already extracted — good candidates for direct display against a
reference.

---

## Which view measures what

The two cameras are complementary; measure each quantity where it is best
resolved.

| View | Measures well |
|---|---|
| **Face-on** | Lead-arm parallel (P3/P5/P9), spine lateral tilt, head lateral position, hip slide toward target, belt height, shaft forward lean at P7, weight distribution |
| **Down-the-line** | Shaft-parallel positions (P2/P6/P8), shoulder and hip rotation angle, hand path inward vs. along the target line, club approach from inside, spine flexion/extension |

### Cross-view fusion

Swing Catalyst writes both angles of one swing as a timestamp-matched pair:

```
Jeff Young - 2026-07-27 150416 Fox Down the line 61fc.mp4
Jeff Young - 2026-07-27 150416 Fox Face on right aaf0.mp4
```

So the pipeline should:

1. Pair clips by the filename timestamp
2. Detect arm-tier positions in the face-on clip, shaft-tier in the DTL clip
3. Merge on a shared time axis **by `timestamp_ms`**, since the cameras need not
   share a frame rate or start instant
4. Detect kinematic positions (P1/P4/P7/P10) in both and cross-check —
   disagreement beyond a threshold flags a swing for review

**Assumption to verify before relying on it:** that both filenames mark the same
instant rather than each camera's own trigger time. Confirm on a swing where
impact is visible in both views. A fixed offset is correctable; drift is not.

---

## Where reference-based labeling fits

Geometric detection is anchored to the video itself, so it is the primary path.
DTW label transfer from a hand-labeled reference (`detect_p_positions_with_dtw`)
earns its place in three specific cases:

- **Fallback** where a geometric crossing is missing or low-confidence — occluded
  club, dropped landmarks
- **Cross-check** — wide disagreement between geometry and transfer flags either a
  bad swing or a bad reference
- **Positions with no clean geometric crossing** in a given view

---

## Build order

1. **Ground truth.** Convert `data/annotations.json` into `data/ground_truth.json`
   so `scripts/eval_positions.py` runs. Record a per-position baseline.
2. **Label a real set.** Enough swings per view to be meaningful, via the Annotate
   tab. DTL first.
3. **Kinematic tier** — P1, P4, P7, P10 from pose velocity and hand-path reversal.
4. **Arm tier** — P3, P5, P9 from the lead `shoulder→wrist` angle crossing
   horizontal. Pose-only, no club tracking required.
5. **Body checkpoints** — the per-position measurements above, plus the continuous
   series. This is what turns detected positions into coaching feedback.
6. **Shaft tier** — P2, P6, P8. Tune on 240fps footage, where the club is sharp.
7. **Cross-view fusion** — merge the timestamp-matched pair.

## Implementation notes

- **Exclude carried-forward frames from velocity.** `pose_extractor.py` repeats
  the previous frame's landmarks when detection fails, to preserve frame
  alignment. Those stretches read as exactly zero velocity, which is
  indistinguishable from a still golfer — and P1 and P10 are defined by
  stillness. Track which frames were carried forward and mask them.
- **Smooth before differentiating.** Wrist landmarks are noisiest exactly where
  the hands move fastest, around P6–P7. `dtw_features.py` already has
  Savitzky-Golay.
- **Normalize by body scale**, so checkpoints are camera-distance independent.
- **View angle and handedness** are per-video properties. `_view_angle` is already
  stored in `data/annotations.json`; Swing Catalyst also encodes the camera in
  the filename.
