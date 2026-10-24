# SwingAI — Current Status

**Last reviewed: 2026-10-24** (PR #6 branch `feat/geometry-3d-foundation`; PR #5 merged as `c9096d9`). This is the current implementation snapshot, not an aspirational feature list. The long-term product direction and proposed PR scope live in [ROADMAP.md](ROADMAP.md). Earlier implementation notes remain available in [the pre-update Git revision](https://github.com/jephphdyoung/swingai/blob/1fa9eee/docs/STATUS.md).

## Product goal

Automatically capture or import a golfer's synchronized down-the-line (DTL) and face-on (FO) observations, identify body and club landmarks and P1–P10, reconstruct a time-varying **3D** motion model where calibration and synchronization support it, and compare that 3D swing against a versioned reference model built from **multiple** curated Grant Waite swings. Support multiple users and additional reference golfers without duplicating the reference library.

**Current reality:** SwingAI has a working **offline, predominantly 2D** pose-analysis and comparison prototype, a swing dataset catalog and editor (PR #5), a **synthetic** multi-camera capture core, and (PR #6) a **synthetically validated** 3D geometry foundation: calibration, triangulation from labelled views, a target-line world frame, P1 alignment and clubhead trajectory values. **It has not been run on real calibration recordings yet**, so no real-world 3D accuracy is established. There is no Grant Waite aggregate 3D model and no automatic real-camera capture. MediaPipe's model-relative depth is not calibrated world-space 3D.

**Two kinds of 3D:** independent professional reference recordings (e.g. Grant Waite DTL and face-on clips of different swings) can be used for a later *statistical* 3D inference with stated uncertainty; *directly measured* stereo reconstruction needs calibrated, synchronized recordings of the same swing. The code refuses to treat the former as the latter.

## Implementation inventory

| Capability | Current state | Evidence / boundary |
|---|---|---|
| Local video discovery | Working | `videos/reference`, `videos/user` and `videos/output`; recursive discovery and `SWINGAI_{REFERENCE,USER,OUTPUT}_DIR` overrides |
| Swing Catalyst DTL/FO pairing | Working for recognized filenames | `utils/swing_pairing.py` groups by recording timestamp; matching filenames alone do **not** establish optical calibration or general multi-view sync |
| Streamlit UI | Working offline | Comparison, annotation, dual-view and **Reference Editor** tabs; not live capture |
| Body pose extraction | Working | MediaPipe CPU (default); RTMPose-m + YOLOX-m optional GPU backend, merged PR #4 |
| Pose caching | Working | Backend, model and runtime/environment-sensitive cached sequences; stores original-frame pixel coordinates |
| Offline P-position annotation | Working | Legacy Annotate tab: `data/annotations.json` (some entries need review). Reference Editor: per-shot P1–P10 on exact source frames with image digests, `data/annotations/` (PR #5) |
| 2D joint annotation | Working as a separate tool | Blind/assisted labeling, original-frame image digests, optional feet and hands/club groups; `data/joint_labels/` |
| Offline dual-view comparison | Working in 2D | Four view panels (user/reference DTL/FO), per-view 2D joint angles and body-relative displacement, not triangulated 3D |
| Side-by-side / ghost overlays | Available | Renderers and overlay editor; not a validated 3D comparison |
| Automatic P1–P10 | Experimental | `utils/p_positions.py` and `scripts/detect_p_positions.py` run on existing recordings, but accuracy and failure cases are not yet validated against a representative labeled dataset |
| Club shaft detection | Experimental | Edge/Hough method; P2/P6/P8 can be missing; **no automatic 3D shaft, clubhead or clubface tracking** (PR #6 triangulates manually labelled club points only) |
| Exact frame access | Working (PR #5) | `utils/frame_access.py`: sequential-decode frame index with presentation times and image digests; verified reads, never a neighbouring frame. Used by Annotate, Dual-View and the Reference Editor |
| Video preprocessing | Utility present | ffprobe timestamps and CFR conversion; not a calibration pipeline |
| YouTube downloader | Standalone utility | `downloader/` has yt-dlp container code; no integrated reference import/editor/catalog |
| Rust capture contracts | Implemented | Versioned capture/analysis JSON schemas, stable shot IDs, per-stream time and relative media paths |
| Rust capture simulation | Working synthetically | Independent camera ring buffers, timestamp-based retroactive extraction, per-shot PGM writer and manifest |
| Real camera/audio capture | **Not implemented** | No working MVS/FOX capture adapter, microphone trigger, hardware clock conversion or replay loop in SwingAI |
| Stereo 3D reconstruction | Foundation, synthetic-validated (PR #6) | Calibration (intrinsics, two-camera extrinsics, known-distance verification), labelled-view triangulation with reprojection/parallax/uncertainty gates, same-swing/sync/calibration/timestamp status rules, target-line world frame from explicit evidence, versioned artifacts. **Not yet run on real FOX recordings.** `docs/geometry-3d.md` |
| 3D measurements | Foundation, synthetic-validated (PR #6) | P1 shoulder/pelvis/knee-centre/stance alignment and foot flare vs. the target line; clubhead speed, path, attack angle, swing direction, observed and fitted low point. Manual 2D labels only; no automatic clubhead detection |
| Aggregated Grant Waite model | **Not implemented** | Current reference selections operate on videos, not a multi-swing 3D model |
| Automatic 3D comparison/coaching | **Not implemented** | No versioned 3D reference distributions, 3D deviation metrics or validated prioritized coaching rules |
| Swing dataset catalog | Working (PR #5) | Golfers, immutable checksummed sources, shots (one swing; windows and crops as metadata), reference collections; schema-validated, atomic, revision-checked. `docs/swing-dataset.md` |
| Reference Editor | Working (PR #5) | Register videos in place, cut shots, crop, P1–P10, body/feet/hands-club/ball points with origin and review, quality and eligibility gates |

## PR #6 — 3D geometry and measurement foundation (in review)

Branch `feat/geometry-3d-foundation`. Details: [geometry-3d.md](geometry-3d.md).

- **Contract** `schemas/geometry-3d.schema.json`: `camera_calibration`, `reconstruction_3d` (manifest + `trajectory.npz`), `measurements_3d`; shares the capture contract's id/time/path/view definitions.
- **World frame**: +X toward target, +Y left of the target line, +Z up, origin at the ball's ground projection at address; ground, target line and origin each need explicit evidence.
- **Reconstruction** from the PR #5 catalog shot and labels; refuses non-same-swing inputs; per-point observed/inferred/missing and per-frame measured/unverified with reasons; camera-specific source frames and timestamps preserved.
- **Measurements**: P1 alignment (shoulder, pelvis, knee-centre line, stance, feet, differences) and clubhead trajectory (speed, path, attack, swing direction, observed and fitted low point), each `measured` / `inferred` / `unavailable`.
- **CLI** `scripts/geometry_3d.py`: calibrate-intrinsics, calibrate-extrinsics, check-distance (the only way to a verified calibration), reconstruct.
- New shot-label points: `clubhead_center`, `floor_c`, `target_line_back`, `target_line_front`.

Tests run 2026-10-24, CPU image: Python **908 passed, 7 skipped** (PR #5 baseline on main: 856 / 7). Rust **227 passed, 0 failed** (unchanged code). **GPU image not run** (PR #6 adds no GPU code). Accuracy figures are synthetic only; the real FOX validation procedure is in geometry-3d.md and has not been run.

## PR #5 — swing dataset catalog and Reference Editor (merged)

Branch `feat/swing-catalog-reference-editor`. Details: [swing-dataset.md](swing-dataset.md).

- **Exact frames.** OpenCV seeking returned frame `f-1` past ~frame 700 on the 240fps captures (reproduced read-only on a 2026-07-27 capture: nominal 249.1 vs. average 249.3 fps, first frame at 4.014 ms, 1245 reported vs. 1244 decoded frames). `utils/frame_access.py` replaces seeking in the Annotate tab and Dual-View panels. Regression fixtures reproduce the bug (high-fps nominal/average mismatch, nonzero start offset, variable frame rate).
- **Catalog** (`data/catalog/`, `schemas/swing-dataset.schema.json`): stable ids; sources identified by SHA-256 and never written; shots = one swing; multi-view only with a same-swing assertion; synchronization/calibration `unknown` until measured; eligibility gates, no score.
- **Frozen collections are reproducible** (review round 1): freezing writes a content-addressed, write-once snapshot of every member's shot definition, sources and full labels; working shots stay editable and drift is reported, never applied. Freezing also checks every label's image digest against its exact source frame (round 2).
- **Timestamp reliability is explicit** (review round 1): each source's frame times are `verified` / `unverified` / `inconsistent` / `estimated` (decoder vs. ffprobe, fallback estimates listed); labels record `timestamp_basis`; motion timing needs verified times *and* a stated real-time basis. All 8 registered videos and a 2026-07-27 240fps capture come out `verified` (decoder and ffprobe within 0.001 ms); their motion timing is still `unknown`.
- **Per-shot labels** (`data/annotations/<shot_id>.json`): P1–P10 (P7 = impact) and points in source frames/pixels, each tied to the frame's image digest, with origin and review.
- **Existing videos registered in place** (`scripts/catalog.py bootstrap`): 7 Grant Waite sources, `TW_face.mp4` under reference golfer `tiger-woods` (file not moved), an empty draft `grant-waite-full-swing-v1`. No shots yet. Legacy `data/annotations.json` untouched (pointer only).
- **Not in PR #5:** 3D, model training, new detectors, model proposals in the editor, rendered clips, hardware capture.

Test results, **run 2026-10-24** in the CPU image (`swingai`, built 2026-10-21): Python **856 passed, 7 skipped** after review round 2 (853 after round 1, 837 at first submission; pre-PR #5 baseline the same day: 766 passed, 7 skipped). Rust workspace (`native/`, unchanged by PR #5): **227 passed, 0 failed**. The GPU image was not rerun. Manual check: the editor running in a container against `videos/reference/GW_faceon.mp4` via headless Chrome (shot window, crop drawn on the canvas, P1, a point placed by clicking the canvas). The source's checksum was unchanged.

## PR #4 — merged

PR [#4](https://github.com/jephphdyoung/swingai/pull/4) was merged on 2026-10-22 (merge commit `1fa9eee`). RTMPose + YOLOX run through the ONNX Runtime CUDA execution provider; `verify-gpu` verifies both networks independently, reports exact input video, and fails on missing/unreadable footage without selecting another video. MediaPipe remains the default.

Final **reported local** suite: CPU image **766 passed, 7 skipped** (GPU-dependent); GPU image **771 passed, 2 skipped** (CPU-image-specific). Real GPU verification passed on the committed `videos/reference/GW_faceon.mp4`. These are the final PR #4 reports, **not newly rerun results or GitHub CI results**.

Historical evaluation: 20 hand-labeled frames across four videos; on **six blind face-on frames**, RTMPose PCK@0.1 was 69.7% vs MediaPipe 44.7% (+25.0 percentage points). This is a small 2D benchmark, not 3D evidence. **DTL accuracy was not scored.** See [joint accuracy](benchmarks/joint-accuracy-2026-10-21.md) and [original behavior benchmark](benchmarks/pose-backends-2026-10-20.md).

The June 20, 2026 local user recordings were deleted as stale. Their labels, manifest and reports remain committed and **must not be silently rewritten**. Those historical comparisons are not fully reproducible without restoring the exact source footage; default benchmark scripts still reference the deleted files. Do not equate missing historical inputs with newly validated metrics.

Rust: **227 passed, 0 failed** (`cargo test --workspace`, run 2026-10-24). The earlier figure of 207 predates the final capture-core fixes.

## Known issues / limitations

1. **Frame-accurate seeking:** fixed in PR #5 for the Annotate tab, Dual-View panels and Reference Editor (`utils/frame_access.py`). Frame numbers saved in `data/annotations.json` before then were marked on seeked images, which on affected high-fps files showed `f-1`; they are not rewritten. Remaining seeks: `analyzer._read_frame` (legacy Analyze path), `make_joint_label_set.seek_digests` (deliberately reproduces the old labeller) and `verify_gpu` (sampled frames).
2. **P-position reliability:** P4, view disagreement and club-driven P2/P6/P8 need evaluation and improvement. Legacy Path A/Path B impact nomenclature must be reconciled (canonical **P7 = impact**) before unified evaluation.
3. **No calibrated 3D geometry:** DTL/FO observations alone are not proof of reconstructable 3D; views must represent the **same swing**, be synchronized, calibrated and have corresponding landmarks. For occluded points, an inference is not a measurement.
4. **Reference footage provenance:** many YouTube videos may have different swings for DTL and FO, unknown capture frame timing, edits or slow motion. Such material can inform 2D/shape priors but **cannot be falsely marked as measured stereo 3D**.
5. **Two annotation stores:** the legacy path-keyed `data/annotations.json` (Annotate / Dual-View / Analyze) and the PR #5 per-shot store. Nothing migrates between them automatically. `downloader/` duplicates are recorded as provenance, not as sources.
6. **Camera timing:** existing Rust synthetic sources already share the session clock. Real cameras/microphones require measured clock conversion and drift handling. Timestamp proximity is not synchronization proof.

## Current execution paths

```bash
./run.sh                          # Streamlit with default CPU/MediaPipe backend
SWINGAI_GPU=1 ./run.sh             # Optional GPU RTMPose backend
SWINGAI_GPU=1 ./run.sh verify-gpu  # Check both GPU networks on known reference footage
```

The launcher mounts source footage read-only without modifying host permissions or SELinux labels. User capture-library paths can be supplied through `SWINGAI_USER_DIR`. No downloaded/raw user videos need to be committed to Git.

## Next work

After PR #5 review: curate Grant Waite shots in the editor (windows, crops, P1–P10, P1 setup labels, quality and review) into `grant-waite-full-swing-v1`; decide how the Dual-View/Analyze paths read the per-shot store; offer cached-pose proposals in the editor; then the calibrated 3D proof of concept (ROADMAP).

## Source-of-truth documents

- [ROADMAP.md](ROADMAP.md): product architecture, milestones and PR #5 proposed acceptance criteria.
- [ADR 0001](adr/0001-hybrid-rust-python-runtime.md): Rust capture / Python analysis seam.
- [Dual-view comparison](dual-view-comparison.md): current 2D measurement behavior.
- [Joint labeling](joint-labelling.md): labels, frame hashes and evaluation.
- [GPU pose](gpu-pose.md): current RTMPose/YOLOX and GPU environment.
- [P-position detection](p-position-detection.md): current heuristics and definitions.
- [Swing dataset](swing-dataset.md): catalog, exact frames, labels and the Reference Editor.
- [3D geometry](geometry-3d.md): calibration, reconstruction, measurements, FOX validation procedure.
