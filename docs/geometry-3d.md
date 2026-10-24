# 3D geometry: calibration, reconstruction and measurements

PR #6. A validated **foundation** for measured 3D: calibrated cameras, two-view
triangulation with uncertainty, a target-line world frame, P1 alignment and clubhead
trajectory values, all gated so that nothing unsupported is published. Everything
here is validated on **synthetic** captures with known truth. **No real-camera
accuracy is claimed** until the FOX procedure below has been run on real
calibration recordings.

## Two kinds of 3D, never mixed

- **Measured stereo 3D** needs one swing recorded by two (or more) calibrated,
  synchronized cameras: a catalog shot with several streams and a documented
  same-swing association. That is what this PR reconstructs. An *asserted*
  association is enough to reconstruct experimentally; nothing is `measured` until
  it is *verified*, with its provenance (method, by, at).
- **Inferred 3D** from independent recordings — e.g. Grant Waite DTL and face-on
  clips of *different* swings, which the catalog keeps as separate single-stream
  shots — can only feed a later statistical fit with stated uncertainty.
  `reconstruct_3d` refuses such shots (`StereoInputError`).

## World frame

Right-handed, metres. **+X** toward the target along the target line, **+Y** left of
the target line, **+Z** up (normal to the ground), **origin** at the ground projection
of the ball centre at address. A right-handed golfer stands on +Y facing −Y.

Camera calibration doesn't tell where the floor or the target is, so each needs its
own evidence (`utils/world_frame.py`), recorded in the output:

| Piece | Evidence accepted |
|---|---|
| Ground plane | the calibration board, if the operator confirms it was lying on the ground (`--board-on-ground`); or `floor_a`, `floor_b`, `floor_c` labelled at address (fitted plane, RMS ≤ 10 mm) |
| Target line | a direction stored in the calibration; or `target_line_back` → `target_line_front` labelled on a marker (e.g. an alignment stick parallel to the target line, ≥ 0.3 m apart) |
| Origin | `ball_center` labelled in both views at address (P1), dropped to the ground |

Missing or degenerate evidence leaves the world frame **undefined**: coordinates stay
in the calibration's rig frame and every target-relative value is unavailable.
Each camera is stored with `camera_to_rig` and, when defined, `camera_to_world`.

## Pipeline

1. **Camera model** (`utils/camera_model.py`): OpenCV pinhole + 5-coefficient
   Brown–Conrady distortion, in the source frame's own pixels (the labels'
   coordinates). Projection matches `cv2.projectPoints`.
2. **Calibration** (`utils/calibration.py`, `scripts/geometry_3d.py`): intrinsics per
   camera (`cv2.calibrateCamera`); extrinsics from simultaneous board views
   (`cv2.stereoCalibrate`, intrinsics fixed). The rig frame is the reference board's,
   +z toward the cameras. Extrinsics need an **orientable** board (inner-corner counts
   summing to an odd number, e.g. 9×6) so both cameras number its corners from the
   same end; symmetric boards are refused.
3. **Triangulation** (`utils/triangulation.py`): DLT, then Gauss–Newton on pixel
   reprojection error through the distortion model; per-view reprojection error, ray
   parallax, cheirality, and covariance σ²(JᵀJ)⁻¹ (σ = 1 px by default).
4. **Reconstruction** (`utils/reconstruct_3d.py`) from the catalog shot, its saved
   labels and a calibration with one camera per stream id. Each camera keeps its
   own source-frame ids, timestamps, timestamp basis and image digests.
   - *static_address*: every stream's own P1 frame (the golfer is still; no sync needed).
   - *dynamic*: paired by synchronized time, t_ref = t_stream + `offset_ms`, only with
     a known offset and only within `max_pair_gap_ms` (0.5 ms on the files' timestamp
     clock). Pairing alone doesn't make a moving point trustworthy: see *Timing*.
5. **Timing** (`utils/timing_3d.py`): a timing-error bound per dynamic frame and the
   motion it allows per point.
6. **Measurements**: `utils/alignment_3d.py` (P1) and `utils/club_3d.py` (club).
7. **Artifacts** (`utils/reconstruction_io.py`).

## Statuses

| Level | Values | Meaning |
|---|---|---|
| point | `observed` | triangulated from visible labels in ≥ 2 views, passing reprojection (≤ 3 px), parallax (≥ 5°) and uncertainty (≤ 20 mm) checks |
| | `inferred` | as observed, but a contributing label was placed as hidden, or is an unreviewed model proposal |
| | `missing` | labelled in one view only, or rejected — with the reason |
| frame | `measured` | calibration verified (document **and** catalog) and same swing verified with method, by and at — for the static address too, since views of different swings don't share a pose; for dynamic frames also synchronization verified with an offset and uncertainty, every view's timestamp a verified container time, every stream's motion time known, and the frame's timing-error bound known |
| | `unverified` | anything short of that, with reasons |
| point timing (dynamic frames) | `within` / `exceeds` / `unknown` | the motion the timing bound allows at the point's speed bound, against `max_timing_position_error_m` (10 mm). Orthogonal to observed/inferred/missing, whose meaning is unchanged: an `exceeds` clubhead is still triangulated and stored, but isn't a trustworthy measured input |
| value | `measured` / `inferred` / `unavailable` | unavailable values are `null` with reasons, never a guess |

## Timing (`utils/timing_3d.py`)

Two cameras never expose at exactly the same instant; a point moving between the
exposures is seen at two places and triangulated somewhere in between. A shoulder at
1 m/s moves 1 mm in 1 ms, a clubhead at 40 m/s moves 40 mm. So timing is judged
**per landmark**, not with one threshold.

**Motion time first.** Verified presentation timestamps don't prove real swing time.
Each source's catalog `timing` gives a motion-clock scale (`catalog.motion_clock`, the
same rule as the catalog's `motion_timing` eligibility): 1 for `real_time` with a
documented basis; presented fps ÷ capture fps for `slowed` footage with a capture rate
and basis (one presented frame per captured frame; variable-rate files refused);
unknown for `retimed`, `unknown` or undocumented footage. With an unknown scale no
moving frame is measured and nothing time-dependent (speed, path, attack, swing
direction, low point) is reported. Frames carry `motion_time_ms` (real time).

**Timing-error bound** of a dynamic frame, in real ms — a worst case, not a standard
deviation:

    bound_ms = scale × (pair_gap_ms + sync_uncertainty_ms) + exposure_ms

- `pair_gap_ms`: the actual synchronized-time gap of the paired frames.
- `sync_uncertainty_ms`: the synchronization record's uncertainty (same clock as the
  offset: t_ref = t_stream + offset_ms).
- `exposure_ms`: exposure-midpoint misalignment not already in the synchronization
  uncertainty. The synchronization record says which (`exposure_alignment`):
  `included` (0 — e.g. the offset was measured from a flash both exposures caught, so
  its uncertainty covers it), `bounded` (`exposure_uncertainty_ms`, e.g. trigger
  jitter plus half the difference in exposure time), or `unknown` — the default, which
  leaves the bound unknown and the frame unverified. Equal nominal frame rates never
  imply aligned exposures.
- Streams whose motion clocks run at different rates can't be related by one offset:
  bound unknown.

**Per point:** position error bound = speed bound × bound_ms / 1000. The speed bounds
are documented **a-priori** upper bounds per landmark class (clubhead and hosel 70 m/s,
ball 105, hands and grip end 20, elbows/forearms 15, head and torso 6, legs and feet 6,
ground markers 1; unlisted points get the largest), not estimates from the
triangulations being judged, so a mistimed triangulation can't vouch for itself. They
are generous for a full swing, so slow motions are judged conservatively. A point
passes when its bound is ≤ `max_timing_position_error_m` (10 mm): a hardware-triggered
pair with 0.05 ms sync and 0.02 ms exposure uncertainty passes the clubhead (4.9 mm);
0.3 ms of sync uncertainty fails it (22 mm) while the shoulders pass (1.9 mm).

The timing bound is stored apart from the covariance and never added to it: the
covariance is a 1-sigma Gaussian model of labelling noise, the timing bound a
worst-case displacement. Consumers use `Reconstruction.problem(i, j)`, which applies
the frame status, the point state and the timing gate. Static address frames aren't
timed (the golfer is still).

## P1 alignment (`utils/alignment_3d.py`)

Lines run trail → lead and are projected onto the ground. `*_yaw_deg` is the angle
from +X (positive toward +Y); `*_open_deg` is signed by handedness, positive = open.

- `shoulder` (shoulder centres), `pelvis` (hip centres), `knee_center` (knee joint
  centres — a line through the knees, **not** kneecap direction), `stance` (toe line:
  big toes, else shoe toe-contact points, never mixed).
- `lead_foot_flare_deg`, `trail_foot_flare_deg`: heel → toe from square
  (perpendicular to the target line, toward the ball), positive = toe out.
- `shoulder_minus_pelvis_open_deg`, `shoulder_minus_stance_open_deg`,
  `pelvis_minus_stance_open_deg`.

Each needs a defined world frame and a measured address frame.

## Club and trajectory (`utils/club_3d.py`)

Club: grip end (`shaft_butt`), hosel (`shaft_hosel`), shaft (their segment), tracking
centre (`clubhead_center`), heel, toe, optional face centre. The tracking centre is
the labelled `clubhead_center`, else the heel–toe midpoint when both are observed —
**never extrapolated along the shaft**. Frames whose rigid lengths (shaft, head,
hosel–centre) deviate from the shot's median by more than max(10 mm, 5 %) are
excluded: a label error along an epipolar line passes the two-view reprojection
check but changes these lengths.

Usable samples: measured dynamic frames, tracking centre *observed*, σ ≤ 10 mm,
every point it comes from within the timing gate, consistent club geometry. Times are
the swing's real time (`motion_time_ms`). Impact = the reference stream's P7 label
time.

**Impact time is frame-level.** P7 marks the frame containing or nearest impact, so
impact lies within one frame interval of it: `impact_time_uncertainty_ms` (4.17 ms at
240 fps, real time). Nothing finer is claimed.

**Two uncertainties per value.** `uncertainty` is the 1-sigma from image localization.
`timing_uncertainty` is a separate worst-case bound: for speed, path and attack angle,
the first-order change of the value over the impact-time uncertainty (from the fit's
acceleration) plus the samples' timing position bounds through the fit weights
(Σ|wₖ|·eₖ); for the observed low point, that sample's timing bound (and the impact-time
uncertainty for its time). Near impact the path turns ~0.9°/ms, so a frame-level P7
gives path a timing uncertainty of ~4° — reported, not hidden. Swing direction and the
fitted arc don't propagate it (`null`); their samples each pass the timing gate.

| Value | Method |
|---|---|
| `clubhead_speed_m_s` / `_mph` | quadratic least-squares fit over ±10 ms of impact (≥ 2 samples each side, no gap > 1.6 frames); norm of its derivative |
| `club_path_deg` (+ = in-to-out), `path_yaw_deg` | ground direction of that velocity |
| `attack_angle_deg` | its elevation (− = descending) |
| `swing_direction_deg`, `swing_plane_inclination_deg` | total-least-squares plane over ±40 ms (≥ 7 samples, RMS ≤ 10 mm, ≥ 15° from horizontal); its horizontal line toward the target |
| observed low point (`low_point_*`) | lowest usable sample in [−30, +60] ms, bracketed on both sides; uncertainty includes half the sample spacing |
| fitted free-arc low point | circle fitted to pre-impact samples only, its lowest point; always `inferred`, distinct from the observed low point |

Plus a per-sample series of speed (central difference) and displacement.

## Artifacts

`<output>/reconstructions/<shot_id>/<reconstruction_id>/`:

- `manifest.json` — kind `reconstruction_3d` (`schemas/geometry-3d.schema.json`):
  algorithm name/version/parameters; shot record hash, each source's checksum and
  timestamp status, label revision and hash, calibration id/version/hash/status;
  association (same-swing, sync with its convention, catalog calibration); world frame
  with evidence; cameras; frames with each camera's own source frame, timestamp and
  digest; missing points with reasons; summary.
- `trajectory.npz` — `time_ms` (F), `xyz` (F,P,3), `cov` (F,P,3,3), `state` (F,P;
  0 missing, 1 observed, 2 inferred), `n_views`, `reprojection_px` (F,P,V),
  `parallax_deg`; timing: `motion_time_ms` (F), `timing_bound_ms` (F),
  `speed_bound_m_s` (P), `timing_error_m` (F,P) and `timing_status` (F,P; 0 not
  applicable, 1 within, 2 exceeds, 3 unknown). Each array's SHA-256 is in the manifest.
  The manifest's frames carry each frame's timing record, the association each
  stream's motion clock and the exposure alignment, and the summary counts moving
  points by timing status.
- `measurements.json` — kind `measurements_3d`, linked by the manifest's hash.

The reconstruction id is derived from the inputs, algorithm and parameters: the same
inputs give the same id and identical arrays. `reconstruction_io.load` re-checks every
hash.

## Real-camera validation with the two FOX cameras

Nothing below has been run yet. Until it has, all accuracy statements are synthetic.

1. **Board.** Print a flat, rigid checkerboard with an odd inner-corner sum (e.g. 9×6
   inner corners, 40 mm squares); measure a square with calipers and use that size.
2. **Intrinsics**, each camera separately, at the capture resolution: record the board
   moved through the whole image (corners and edges, tilted ±45°), then
   `scripts/geometry_3d.py calibrate-intrinsics --camera down_the_line --video ...
   --board 9x6 --square-mm 40 --out dtl.json`. Expect RMS well under 1 px.
3. **Synchronized capture.** Hardware-trigger both cameras (or genlock) so exposures
   coincide; otherwise dynamic frames will never pair. Record the frame offset between
   the two files, and establish the exposure alignment: the same exposure time on both
   cameras and the trigger's jitter, or an LED/flash visible to both (then the offset's
   uncertainty already covers it). Clubhead positions need a timing bound under
   ~0.14 ms (10 mm at 70 m/s).
4. **Extrinsics:** record the board visible to *both* cameras in several poses, ending
   with it lying flat on the floor in the hitting area. `calibrate-extrinsics ...
   --offset-frames N --reference-pair -1 --board-on-ground --calibration-id fox-YYYY-MM-DD
   --by <name>`. Expect stereo RMS under 1 px.
5. **Check a known distance** (a 1 m rule or two marks taped a measured distance apart,
   in the hitting area): read both ends' pixels in each camera's frame and run
   `check-distance ... --known-m 1.000 --tolerance-mm 5 --by <name> --out fox-v2.json`.
   Repeat at several positions and heights; only a pass writes a `verified` version.
   A pass needs the length within tolerance **and** both ends valid triangulations
   (`triangulation.check_point`): in front of both cameras, reprojection ≤ 3 px,
   rays ≥ 5° apart and 1-sigma position uncertainty no larger than the tolerance
   (`--max-reprojection-px`, `--min-parallax-deg`, `--max-sigma-mm`). A matching
   length from bad geometry proves nothing about the calibration.
6. **Scene evidence per shot:** lay an alignment stick on the ground parallel to the
   target line; in the Reference Editor label `target_line_back`/`front`, and
   `ball_center` at P1 (plus `floor_a/b/c` if the board wasn't on the ground).
7. **Catalog:** in the editor, the DTL/FO streams of the swing are one shot
   (*Add another camera view of this same swing*); once confirmed (e.g. the shared
   trigger and the same ball strike in both files), set same swing to verified with
   the method and your name; record synchronization and its exposure alignment with
   `scripts/catalog.py set-sync <shot> --status verified --method ... --by ...
   --offset-ms ... --uncertainty-ms ... --exposure bounded --exposure-uncertainty-ms ...`
   (t_ref = t_FO + offset), and calibration as verified. Register the sources with
   `--motion-time real_time --timing-basis ...`.
8. **Reconstruct:** `scripts/geometry_3d.py reconstruct --shot <id> --calibration
   fox-v2.json`; check the summary's measured frames, max reprojection error and the
   world-frame evidence and the moving points' timing summary, then compare P1 angles
   with a physical check (e.g. an alignment stick across the toes) and clubhead speed
   with a launch monitor.

## Limitations

- **No real-world accuracy has been measured.** Synthetic results: body points within
  4 mm and clubhead within 6 mm at 0.3 px label noise; alignment angles within 1°;
  speed within 2 %; path/attack/swing direction within 0.75°.
- Two views can't detect a label error **along the epipolar line**; it becomes a depth
  error. The club rigidity check catches it for club points; body points rely on
  label quality.
- Impact is the **P7 frame's time**, so it's quantized to a frame: near impact the path
  direction turns about 0.9° per millisecond at 40 m/s on a 1.6 m arc. This is now
  reported as `impact_time_uncertainty_ms` and in path/attack `timing_uncertainty`.
- The timing bound is only as good as the recorded synchronization: a wrong offset with
  a small stated uncertainty gives a small bound (reprojection error catches some of
  these, not all). The speed bounds are a priori and conservative; slower swings
  (chips) are judged as if full speed. Rolling-shutter skew within a frame isn't
  modelled.
- The observed low point is quantized by frame spacing (≈ 17 cm at 40 m/s and 240 fps);
  its stated uncertainty includes that. Heights are of the tracking centre, not the sole.
- Dynamic frames pair only within 0.5 ms, and clubhead points need a timing bound of
  ~0.14 ms. Free-running cameras would need 2D interpolation between frames, which
  isn't implemented.
- Every 2D point is a manual label; there's no automatic clubhead detection.
- Only the 5-coefficient lens model (no fisheye). Scale depends on the measured board
  square size.
- The rig frame's in-plane axes are arbitrary (board corner order); only its +z is
  normalized. The world frame never depends on them.
- No UI yet: CLI and library only.
