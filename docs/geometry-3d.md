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
  same-swing association. That is what this PR reconstructs.
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
     a known offset and only within 0.5 ms (effectively simultaneous exposures).
5. **Measurements**: `utils/alignment_3d.py` (P1) and `utils/club_3d.py` (club).
6. **Artifacts** (`utils/reconstruction_io.py`).

## Statuses

| Level | Values | Meaning |
|---|---|---|
| point | `observed` | triangulated from visible labels in ≥ 2 views, passing reprojection (≤ 3 px), parallax (≥ 5°) and uncertainty (≤ 20 mm) checks |
| | `inferred` | as observed, but a contributing label was placed as hidden, or is an unreviewed model proposal |
| | `missing` | labelled in one view only, or rejected — with the reason |
| frame | `measured` | calibration verified (document **and** catalog); for dynamic frames also synchronization verified with uncertainty ≤ 1 ms and every view's timestamp a verified container time |
| | `unverified` | anything short of that, with reasons |
| value | `measured` / `inferred` / `unavailable` | unavailable values are `null` with reasons, never a guess |

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
consistent club geometry. Impact = the reference stream's P7 label time.

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
  `parallax_deg`. Each array's SHA-256 is in the manifest.
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
   the two files.
4. **Extrinsics:** record the board visible to *both* cameras in several poses, ending
   with it lying flat on the floor in the hitting area. `calibrate-extrinsics ...
   --offset-frames N --reference-pair -1 --board-on-ground --calibration-id fox-YYYY-MM-DD
   --by <name>`. Expect stereo RMS under 1 px.
5. **Check a known distance** (a 1 m rule or two marks taped a measured distance apart,
   in the hitting area): read both ends' pixels in each camera's frame and run
   `check-distance ... --known-m 1.000 --tolerance-mm 5 --by <name> --out fox-v2.json`.
   Repeat at several positions and heights; only a pass writes a `verified` version.
6. **Scene evidence per shot:** lay an alignment stick on the ground parallel to the
   target line; in the Reference Editor label `target_line_back`/`front`, and
   `ball_center` at P1 (plus `floor_a/b/c` if the board wasn't on the ground).
7. **Catalog:** in the editor, the DTL/FO streams of the swing are one shot
   (*Add another camera view of this same swing*); record synchronization (method,
   `offset_ms` with t_ref = t_FO + offset, uncertainty) and calibration as verified.
8. **Reconstruct:** `scripts/geometry_3d.py reconstruct --shot <id> --calibration
   fox-v2.json`; check the summary's measured frames, max reprojection error and the
   world-frame evidence, then compare P1 angles with a physical check (e.g. an
   alignment stick across the toes) and clubhead speed with a launch monitor.

## Limitations

- **No real-world accuracy has been measured.** Synthetic results: body points within
  4 mm and clubhead within 6 mm at 0.3 px label noise; alignment angles within 1°;
  speed within 2 %; path/attack/swing direction within 0.75°.
- Two views can't detect a label error **along the epipolar line**; it becomes a depth
  error. The club rigidity check catches it for club points; body points rely on
  label quality.
- Impact is the **P7 frame's time**, so it's quantized to a frame: near impact the path
  direction turns about 0.9° per millisecond at 40 m/s on a 1.6 m arc.
- The observed low point is quantized by frame spacing (≈ 17 cm at 40 m/s and 240 fps);
  its stated uncertainty includes that. Heights are of the tracking centre, not the sole.
- Dynamic frames need exposures within 0.5 ms. Free-running cameras would need 2D
  interpolation between frames, which isn't implemented.
- Every 2D point is a manual label; there's no automatic clubhead detection.
- Only the 5-coefficient lens model (no fisheye). Scale depends on the measured board
  square size.
- The rig frame's in-plane axes are arbitrary (board corner order); only its +z is
  normalized. The world frame never depends on them.
- No UI yet: CLI and library only.
