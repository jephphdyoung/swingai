# Swing dataset: catalog, exact frames and the Reference Editor

PR #5. The foundation for turning reference and user footage into a curated,
versioned dataset that later 3D reconstruction and model building can trust. It
does **not** reconstruct 3D, train a model, or detect anything new.

## Concepts

| Thing | Identity | Lives in | Notes |
|---|---|---|---|
| Golfer | `golfer_id` (slug, e.g. `grant-waite`) | `data/catalog/golfers.json` | `role`: `reference` (shared, curated) or `user` (their own history) |
| Source | `source_id` + **SHA-256 of the bytes** | `data/catalog/sources.json` | One immutable video file. `location` is only where it was last found |
| Shot | `shot_id` | `data/catalog/shots.json` | **One real swing.** One stream per camera view *of that swing* |
| Stream | `stream_id` within the shot | inside the shot | Source, camera view, swing window, crop, quality |
| Collection | `collection_id` (e.g. `grant-waite-full-swing-v1`) | `data/catalog/collections.json` | A reference golfer's curated shot ids; `draft` or `frozen` |
| Collection snapshot | its SHA-256 | `data/catalog/snapshots/<collection_id>-<sha12>.json` | Write-once copy of a frozen collection's complete inputs |
| Shot labels | the shot id | `data/annotations/<shot_id>.json` | P1–P10 and body/club/ball points per stream |

All documents are validated against `schemas/swing-dataset.schema.json` (the
`kind` field selects the shape) and then against cross-file rules
(`utils/catalog.py::check_catalog`, `utils/shot_annotations.py::check`). Ids follow
the capture contract's identifier rules (usable as a directory name on Windows);
the schema copies `identifier`, `cameraView`, `rfc3339DateTime` and `relativePath`
verbatim from `capture-manifest.schema.json`, and a test keeps them identical.
Camera views use the contract's spelling (`down_the_line`), not the Python `dtl`.

## Rules the code enforces

- **Sources are never written.** Registering reads the file (checksum + frame
  index). Byte-identical copies are one source (other paths go in `also_found_at`).
  `resolve_source` reports `ok`, `relocated` (found by checksum elsewhere in the
  libraries), `changed` (bytes differ — refused, never re-registered) or `missing`.
- **Locations are portable.** Files under `SWINGAI_REFERENCE_DIR` / `SWINGAI_USER_DIR`
  are stored relative to them (`root: reference|user`), which `run.sh` mounts at the
  same container paths. Anything else is `root: absolute` and resolves only where
  that path exists.
- **One shot = one swing.** A shot with two or more streams needs `same_swing`
  `asserted` (with the basis and who says so) or `verified`. Independent Grant Waite
  DTL and face-on recordings are **separate shots**, grouped by a collection.
- **Unknown stays unknown.** Synchronization and calibration start `unknown`;
  `estimated`/`verified` need a method and a person; synchronization can't be
  `verified` while the same swing is only `asserted`.
- **Timing is explicit, in two separate facts.** (1) Are the frame times
  trustworthy? `video.timestamps.status` is `verified` only when every frame has a
  container presentation time and ffprobe's per-frame times agree within 0.5 ms;
  otherwise `unverified` (no independent check), `inconsistent` (they disagree) or
  `estimated` (some frame times were guessed from the nominal rate — counted in
  `estimated_frames`, and each label carries `timestamp_basis: container|estimated`).
  (2) Is playback the swing's real time? `timing.motion_time` (`real_time`,
  `slowed`, `retimed`, `unknown`) with `capture_fps` and a `basis`. The motion-timing
  gate needs **both**: verified frame times *and* real time with a basis (or slowed
  with a known capture rate and basis). YouTube material defaults to `unknown` and
  never qualifies, however clean its timestamps.
- **Collections hold one reference golfer's shots** and can restrict swing types.
  User shots are never put in collections. **Freezing** writes a snapshot: every
  member's shot definition (streams, windows, crops, metadata, review), its source
  records (checksums, video facts, timing) and its complete label document with
  that document's SHA-256. The snapshot is content-addressed and write-once; the
  collection records its path and hash, and a save that alters a frozen collection
  record is refused. Freezing is refused if a member's source bytes can't be found
  and checked, its labels don't validate, or any label's saved image digest differs
  from the exact decoded source frame it names (checked against the frame index). The working shots **stay editable**:
  `load_snapshot` always returns the frozen inputs (after re-checking the hash),
  and `frozen_drift` lists what has changed in the working shots since — the editor
  shows it, and `scripts/catalog.py verify` checks every snapshot.
- **Saves are safe.** Every save validates first and writes nothing on any
  problem; each file has a `revision`, and a save refuses to overwrite a file saved
  elsewhere since it was loaded; files are replaced atomically.
- **Eligibility is gates, not a score.** `catalog.eligibility(cat, shot_id)` says,
  with reasons, whether a shot can feed stereo 3D, motion timing, body/club landmark
  work, needs better resolution, or still needs human review. Unassessed ⇒ not
  eligible.

## Swing types and clubs

`swing_type` is `full`, `three_quarter` or `half` (nothing outside the full-swing
family). `club` records category, name (`6-iron`), and **physical** length in
inches with its basis — entered from a spec, never measured from the image.
Handedness and recording date (to the known precision: `2010`, `2010-10`,
`2010-10-27`) are per shot, defaulting from the golfer and source.

## Exact frames

`utils/frame_access.py`. Frame *f* is the *f*-th frame of a sequential decode —
the same numbering pose extraction and the joint-label benchmark use. OpenCV's
seek can't be trusted for that: on the 240fps Swing Catalyst captures (nominal
249.1 fps, average 249.3, first frame at 4.014 ms) a seek to *f* returns *f−1*
from about frame 700 on, and the container reports 1245 frames where 1244 decode.

A `FrameIndex` (cached by content hash and decoder under `<output>/frame_index/`)
records each frame's presentation time (relative to frame 0, plus the container's
start offset from ffprobe) and image SHA-1. `FrameReader` reads forward, or seeks
early and identifies where it landed by digest, else decodes from the start; the
returned frame's digest is always checked, and a mismatch raises
`FrameAccessError` instead of returning a neighbour. The Annotate tab, the
Dual-View panels and the Reference Editor all use it.

## Labels

Per stream, in **source** frames and pixels (`utils/shot_annotations.py`):

- `positions`: P1–P10 (P6 = shaft parallel, **P7 = impact**, fixed per file),
  each with frame, presentation time, image SHA-1, origin and who set it.
  The swing window's start/end are separate and live on the stream.
- `frames[f]`: `points` (x, y, `visible`, `origin`, model and its score for
  proposals), `unavailable` (point → `occluded`, `out_of_frame`, `motion_blur`,
  `not_present`, `cannot_determine`) and a per-frame `review`.
- Origins: `manual`, `model_proposed` (left where the model put it), `corrected`
  (a proposal a person moved). Reviewing a frame signs off every point on it; any
  later change resets the review.
- Point names reuse the PR #4 labeller's catalogue (body, feet, hands/club incl.
  shaft butt/hosel and clubhead heel/toe) plus `ball_center`. The benchmark
  catalogue and its pinned labels are unchanged.
- `identity_problems` reports any label whose image digest no longer matches its
  source frame; nothing is ever re-pointed.

Window and crop are reversible transforms (`utils/shot_edits.py`): edited frame
*k* = `start_frame + k`; edited pixel = source pixel − crop origin. No scaling,
no rotation, so changing a window or crop never moves a label.

## P1 setup measurements

`utils/setup_measurements.py` declares what each setup measurement (ball position,
ball-to-golfer distance, stance width, spine inclination, hip/knee flexion, hand
position, shoulder/pelvis alignment, shaft orientation) needs, which view, and
what an uncalibrated image supports (ratios to the golfer's own body dimensions,
image-plane angles) versus what needs calibration (real units, alignment to the
target line). `setup_readiness` reports which a stream's P1 labels already
support. **No values are computed yet**, and the future comparison should report
setup differences rather than normalize them away.

## Using it

**Command line** (run in the container so the frame index matches the app's decoder):

```bash
podman run --rm --security-opt label=disable -v "$PWD":/app -w /app swingai \
    python scripts/catalog.py bootstrap      # existing GW/TW videos, idempotent
#   ... list | verify [--no-search] (sources + frozen snapshots) | add-golfer NAME --role user
#   ... register VIDEO --golfer ID --view face_on --source-type youtube --date 2011-05
```

**Reference Editor tab** (`./run.sh`, fourth tab):

1. Enter your name (recorded on labels and reviews). Pick or add a golfer.
2. Pick a registered source, or *Register a video* (library file or path).
3. Choose *New shot*, step to the swing's first frame → *Start = this frame*,
   to its last → *End = this frame*, set view, swing type and club → *Create shot*.
   Repeat for every swing in a long video.
4. *Draw crop* and drag a rectangle around the golfer → *Save this crop*.
   *Zoom to crop* shows real pixels for labelling.
5. *P-positions*: *Set here* / *Go* / *Clear* per position.
6. *Place points*: pick group and point, click the frame (untick *Visible* for a
   hidden point you can judge; otherwise *Mark unavailable* with a reason).
   *Mark frame reviewed* when done.
7. *Shot*: details, per-view quality, shot review, multi-view statuses, *Add
   another camera view of this same swing* (co-captured partner views only).
8. *Collections*: add the shot to a draft collection; freeze a collection when it
   becomes a model input. *Readiness* shows eligibility gates and setup readiness.

## Existing data

`scripts/catalog.py bootstrap` registered, in place:

- seven Grant Waite videos under `videos/reference/`, with dates only where the
  filename gives one (`date_basis: "filename"`), timing `unknown`, URLs unrecorded;
- `videos/user/TW_face.mp4` under the **reference** golfer `tiger-woods` — the file
  is not moved; its original YouTube title comes from the byte-identical copy in
  `downloader/`;
- an empty draft `grant-waite-full-swing-v1`.

No shots were created: windows are a person's call. Five of the GW files are
untracked local videos, so another checkout's `verify` reports them missing.

Legacy `data/annotations.json` is **not converted**. A source records its legacy
key (`legacy_annotation_key`), and the editor shows those marks read-only with
their convention warnings. Legacy marks were made on seeked images, which on
high-fps files can be one frame early, and some predate P7 = impact.

## Limitations

- Model proposals are supported by the schema (`model_proposed`, model name and
  score) but the editor does not yet offer them from the pose cache.
- Crop is an axis-aligned rectangle; there is no rotation.
- Rendered (trimmed/cropped) clips are not produced; edits are metadata only.
- `capture` on a shot (link to a Rust `capture-manifest.json`) is defined but no
  importer creates it yet.
- Frame indexing decodes the whole file once (a few seconds for a 1,244-frame
  1440×1080 capture), then is cached.
- Model builds should read a frozen collection only through `load_snapshot`.
