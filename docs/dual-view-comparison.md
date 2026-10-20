# Dual-View Comparison

Compare your down-the-line (DTL) and face-on (FO) views against a reference golfer's,
one P-position at a time: four frames, skeletons, and a table of 2D joint angles and
body-relative positions with `difference = you − reference`.

It lives in the **Dual-View Comparison** tab of the Streamlit app, next to Analyze and
Annotate, which are unchanged.

---

## Running it on your own videos (Fedora)

Your capture library does not need to be in the repo. Point `SWINGAI_USER_DIR` at a
Swing Catalyst folder; `run.sh` mounts it into the container.

```bash
cd ~/github.com/jephphdyoung/swingai

# videos already under videos/user and videos/reference:
./run.sh

# or a capture library elsewhere (searched recursively, per-session subfolders are fine):
SWINGAI_USER_DIR=~/Golf/Videos/2026-07-27/SwingCatalyst ./run.sh
```

Open <http://localhost:8501> → **Dual-View Comparison**.

`SWINGAI_REFERENCE_DIR` and `SWINGAI_OUTPUT_DIR` work the same way. Annotations are
saved to `data/annotations.json`; extracted poses are cached in
`<output dir>/pose_cache/` (default `videos/output/pose_cache/`, git-ignored).

### Workflow

1. **Your swing** — pick a recorded pair. Pairs come from `utils/swing_pairing.py`: both
   clips of one swing share a timestamp in the Swing Catalyst filename
   (`Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4` + `… Face on right aaf0.mp4`).
   The label shows how many P-positions each view has saved.
2. **Reference swing** — pick a reference DTL and a reference face-on video separately.
   They default to the clip annotated (else named) as that view, i.e. `GW_DTL.mp4` and
   `GW_faceon.mp4`. They do not need the same frame rate, length, or even to be the same
   swing.
3. Set handedness for you and for the reference (Grant Waite: right).
4. **Extract poses** — first time only, per video. MediaPipe runs over every frame and the
   result is cached (measured: ~16 s for a 1,244-frame 240fps capture, ~7 s for a
   ~550-frame reference clip, on CPU). After that, changing P-position, highlight,
   handedness or annotations never re-runs inference.
5. **P-position** — choose P1–P10 or step with Prev/Next. `(n/4)` shows how many of the
   four videos have that position annotated.
6. Read the four panels and the two measurement tables (DTL and face-on are separate).
7. **Annotations** — fix or add positions in *Add or correct P-positions* (the same
   annotator as the Annotate tab, with the canonical P definitions shown). Saving reruns
   the page, so position counts, panels and tables reflect the change at once; poses are
   not recomputed. Saving does not confirm — see below.

### Options

| Control | Effect |
|---|---|
| Skeleton | Show/hide the skeleton on all four panels. |
| Highlight | Draw the landmarks of one measurement in red on every panel. |
| Fill your missing positions from your other view | Off by default. Your two clips are co-triggered, so a position marked on one view is shown on the other at the **same timestamp** (labelled "from partner view"). Never saved, never applied to the reference. |
| Advanced → Minimum landmark confidence | Landmarks below it count as not observed (drawn hollow grey). Default 0.5. |

---

## P-positions

Saved annotations are the only source of positions. Nothing is auto-detected or
interpolated in this tab; a position that is not annotated shows no frame and no values.

Canonical definitions (`utils/annotations.py::P_DEFINITIONS`):

| | |
|---|---|
| P1 | Address |
| P2 | Takeaway: shaft parallel to ground |
| P3 | Lead arm parallel to ground (backswing) |
| P4 | Top of backswing |
| P5 | Lead arm parallel to ground (downswing) |
| P6 | Shaft parallel to ground (downswing) |
| **P7** | **Impact** |
| P8 | Shaft parallel to ground (follow-through) |
| P9 | Trail arm parallel to ground (follow-through) |
| P10 | Finish |

**Saving never verifies anything.** A brand-new entry is recorded under the canonical
system (`"_p_convention": "p7-impact"`). An existing entry keeps whatever status it had, so
re-saving a legacy entry after fixing, say, P2 does not certify its old P6/P7 (path A
called impact P6 and interpolated P7/P8). An entry is verified only by an explicit
**Confirm reviewed** (comparison tab) or **Save + confirm reviewed** (annotator), which
sets the marker and clears any `_needs_review` flag without changing frames. Until then
the UI warns. Also flagged: P6 marked without P7, and positions out of order.

`_needs_review` marks labels no person has checked yet, e.g. ones proposed by Claude. It
survives saves and is cleared only by a confirm.

`scripts/detect_p_positions.py --save-annotations` replaces the whole entry with detector
output, so it records the canonical convention explicitly.

**Frames vs. timestamps.** Within one video the saved frame index is used (it is the frame
you looked at). The saved `timestamp_ms` is used only when the frame is missing or out of
range. Across videos, nothing assumes matching frame numbers: each of the four uses its
own annotation, and the partner-view fill maps by timestamp on the shared trigger clock.
A timestamp more than half a frame outside a video's own time range is **unavailable**.
It is never clamped onto that video's first or last frame.

---

## Measurements

All values are measured in the **image plane of one camera**. They are 2D projections,
not anatomical 3D joint angles, so DTL and face-on values are separate rows in separate
tables and are never compared with each other.

| Measurement | Definition |
|---|---|
| Lead / trail elbow flexion | 180° − 2D angle shoulder–elbow–wrist. 0° = straight. |
| Lead / trail knee flexion | 180° − 2D angle hip–knee–ankle. 0° = straight. |
| Spine inclination | Hip centre → shoulder centre, from image vertical. Signed: + toward target (FO) / toward ball (DTL). |
| Shoulder tilt | Elevation of the lead shoulder over the trail shoulder. + = lead higher. Unavailable when the shoulders overlap in the projection (shoulder width < 0.2 × torso — typical DTL at address/impact). |
| Lead arm above horizontal | Elevation of lead wrist over lead shoulder, −90°…+90°. ~0° is "lead arm parallel" (P3/P5). |
| Displacement from P1 | Head centre (mid-ears), shoulder centre, hip centre, lead wrist, trail wrist, hand centre (mid-wrists): horizontal and vertical components, from **the same video's P1**, in **P1 torso lengths**. |

Conventions:

- **Scale is fixed at P1** (shoulder-centre to hip-centre distance in that video and view)
  and is not recomputed per frame.
- **Horizontal direction is body-relative**, so a reference filmed from the other side or
  mirrored still compares sign-for-sign: face-on + = the lead hip's side at P1 (toward
  target); DTL + = the side the torso bends toward at P1 (toward the ball).
- Vertical is + up.
- Lead/trail follow handedness (right-handed: lead = left).
- Missing or low-confidence landmarks, zero-length segments, or a missing P1 make a value
  **unavailable** (shown as `None` with the reason in *Note*). No value is ever filled in.

Deliberately **not** reported: weight distribution, true pelvis or shoulder rotation, or
anything else that needs depth. No coaching score; a difference is not by itself a fault.

---

## Code map

| Module | Role |
|---|---|
| `utils/landmarks.py` | Named landmarks (23, incl. ears), MediaPipe index map, lead/trail by handedness, skeleton segments by name. |
| `utils/pose_data.py` | `PoseSequence` (pixels, timestamps, per-frame `detected`, per-landmark confidence; **no carry-forward**), the `PoseBackend` protocol, `MediaPipeBackend`, the on-disk cache. |
| `utils/joint_measurements.py` | The measurement engine. Pure numpy, no I/O. |
| `utils/annotations.py` | P definitions, convention marker, `AnnotationRecord`, legacy warnings. |
| `utils/dual_view.py` | Position → frame resolution, per-panel measurement, comparison rows, panel rendering. No Streamlit. |
| `dual_view_tab.py` | The Streamlit tab. |

**Adding RTMPose (PR #4):** implement `PoseBackend` (`identity`, `open`, `detect` → a
`FrameObservation` in `LANDMARK_NAMES` order with NaN for keypoints it lacks, `close`)
and pass it to `get_pose_sequence`. The backend identity is part of the cache key, so
results from the two models never mix. Nothing downstream changes.

The legacy `utils/pose_extractor.py` (carry-forward, index-based) is untouched and still
feeds the Analyze tab.

## Tests

```bash
podman run --rm -v "$PWD":/app:Z swingai python -m pytest tests/ -q
```

New: `test_joint_measurements.py` (known angles, invariance, fixed-P1 scale, handedness,
missing/degenerate geometry), `test_pose_data.py` (no carry-forward, pixels, cache),
`test_dual_view.py` (annotation convention, frame selection across frame rates, partner
fill, end-to-end smoke on a generated four-video fixture), `test_app_smoke.py` (the
real `app.py` headless via Streamlit `AppTest`: all three tabs render, P-position switch).

## Known limitations

- **DTL lead side is often occluded.** On real 240fps DTL captures MediaPipe reports the
  lead elbow, wrist and knee at visibility 0.1–0.4 at address and impact, so those values
  (and hand centre) are unavailable there. Lowering the confidence threshold shows
  MediaPipe's guess, at your own risk.
- MediaPipe left/right labels can swap on some frames; nothing corrects this yet.
- Timestamps come from the container (`CAP_PROP_POS_MSEC`), falling back to frame/fps.
  Variable-frame-rate files are not specially handled.
- The comparison needs P1 annotated in each video for displacements and signed angles.
- Handedness is chosen in the UI each session. `save_annotations` can store
  `_handedness` (and the radio defaults to it), but the UI does not write it yet.
