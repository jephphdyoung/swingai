# SwingAI — Product Roadmap

**Revised 2026-10-22.** [Current verified implementation status](STATUS.md). The former [capture-first roadmap](https://github.com/jephphdyoung/swingai/blob/1fa9eee/docs/ROADMAP.md) is historical: capture/replay remains an important workflow, but the **product goal is 3D biomechanical comparison against a multi-swing reference model**.

## North star

1. Collect multiple **curated** Grant Waite reference swings, ideally with synchronized down-the-line (DTL) and face-on (FO) video of **the same swing**, then reconstruct defensible 3D body/club trajectories.
2. Build a **versioned Grant Waite model** from those swings: joint/segment positions and orientations, P1–P10, movement trajectories, relative timing, acceptable variation and uncertainty.
3. Automatically capture or import a user's DTL+FO recordings; segment the swing, label body/club landmarks and P-positions, reconstruct 3D and compare it to the selected reference model.
4. Prioritize observable, mechanically meaningful differences and show evidence through 3D trajectory visualizations, annotated source video and instructional feedback.
5. Support several users, several reference golfers (e.g. Grant Waite, Tiger Woods), reference collections and, later, overhead and impact cameras without changing the underlying shot abstraction.

**A video is an observation, not the model.** The first deliverable is auditable data from which the 3D models can be generated again, not a hard-coded average skeleton.

## Non-negotiable data and geometry rules

- **One shot = one real swing.** A shot contains multiple camera streams only when they observe that same motion. DTL/FO clips of *different* Grant Waite swings are **not** a stereo pair; they remain separate shots and may be grouped by a higher-level reference collection.
- **3D quality is explicit.** Direct triangulation requires corresponding landmarks, synchronized frames and adequate camera intrinsic/extrinsic calibration; ground/target-line geometry and reliable scale are needed for world units. Uncalibrated single-view/mismatched-view fitting may yield *inferred* 3D, never certified measured 3D. Record per-joint visibility/uncertainty and reprojection error.
- **Keep source footage immutable.** Trim windows, crop/rotation, time remapping and resolution variants are nondestructive edit instructions or regenerable derivatives. Preserve the coordinate transform and source-frame identity for annotations. Never invalidate the pinned PR #4 benchmark images or copy old labels to newly edited media without explicit mapping/verification.
- **Respect real timing.** Do not infer original motion speed from YouTube playback fps when footage was slowed or retimed. Preserve presentation/device timestamps, camera-clock conversion and sync uncertainty. Velocity/acceleration may be unavailable even when geometric P-position features are useful.
- **Canonical, view-independent representation.** Version the joint naming/side/handedness rules, anatomical landmarks, club landmarks, coordinate axes, P1–P10 convention (**P7 = impact**), source timestamps and pose backend identity. No independent per-frame resize by apparent golfer height. Normalize golfer-to-golfer comparisons using stable body-relative dimensions; claim real units only with adequate calibration.
- **Reference validity before averaging.** Do not flatten different golfers/clubs/camera setups into one 'perfect swing.' Keep individual 3D swing trajectories and model ranges; split evaluation by session/source to avoid leaking near-identical frames across train/test.
- **Build on existing contracts.** Rust's versioned `capture-manifest.json` describes a real captured shot and streams. A new library/catalog manifest should supplement, not silently redefine, this capture contract. Use portable relative media paths, stable IDs, provenance and versioned schemas.
- **Shared references, isolated users.** Reference source assets/models are read-only and shared; users' source footage, annotations, sessions and comparison history are separated by stable IDs. Authentication/remote multi-tenancy is a later capability, not a prerequisite for a multi-golfer local library.

## Architecture

```text
          Curated reference videos                     User upload / future FOX capture
       DTL + FO (same swing if stereo)                 DTL + FO (same swing, known clocks)
                       |                                          |
               Immutable source catalog + shot/stream IDs + edit provenance
                       |                                          |
               Exact-frame decoding, timing and 2D body/club annotation
                       |                                          |
                  Calibration + cross-view synchronization + association
                       |                                          |
            Validated 3D trajectories with quality/uncertainty metadata
                       |                                          |
          Versioned model builder                    Per-user reconstructed swing
       (multiple Grant Waite shots)                               |
                       +--------------- 3D compare ----------------+
                                       |
                    Deviations, sequencing, coaching priorities
                                       |
                         Video / 3D replay and feedback
```

Streams missing calibration or timing support can contribute only to the measurements they actually support, with clear labels. No fabricated 3D or motion-time measurements to make the pipeline appear complete.

**Runtime boundary:** retain Python for labeling, CV research, reconstruction and model experiments. The existing Rust workspace owns deterministic capture/session/clock contracts and is the foundation for hardware capture. File/JSON contracts are the first integration point; a server/API is an explicit future design decision, not an implemented requirement. Windows is the initial hardware integration target, with portable core logic for Fedora analysis.

## Near-term milestones

### Completed — PR #4: GPU pose foundation

- MediaPipe CPU default; optional RTMPose-m + YOLOX-m on NVIDIA GPU, separate backend caches and independent CUDA proof.
- Offline 2D comparison, P-annotation and joint-labeling tools exist.
- Synthetic Rust camera/capture pipeline exists; hardware audio/video capture and validated 3D do not.
- Small historical face-on joint benchmark only; DTL accuracy unmeasured.
- Final reported 2026-10-22 tests: CPU 766 passed/7 skipped; GPU 771 passed/2 skipped.

### Completed — PR #5: Shot catalog + reference editing foundation

Implemented on `feat/swing-catalog-reference-editor`; see [swing-dataset.md](swing-dataset.md) and [STATUS.md](STATUS.md). The catalog lives in `data/catalog/` + `data/annotations/` without moving any video (the layout below stays a logical target). Deferred from the list below: rotation, rendered derivative clips, model proposals in the editor.

**Goal:** turn raw reference or user footage into consistent, inspectable, versioned *swing records* suitable for future 3D reconstruction and model development. Keep PR #5 focused; its exact scope requires discussion before giving coding instructions.

Proposed library shape (logical target; PR #5 implements `data/catalog/` and `data/annotations/` only):

```text
videos/
  reference/<golfer-id>/<shot-id>/...         # local, mostly static shared references
  user/<golfer-id>/<session-id>/<shot-id>/... # isolated users, recording history
  sources/<source-id>/...                    # optional original downloaded long-form videos
data/
  catalog/                                   # golfer/source/shot/stream identity and metadata
  annotations/                               # P-positions, per-source-frame 2D body/club labels
  datasets/                                  # versioned reference collections, inclusion/quality
workspace/
  processed/  pose_cache/  renders/           # safely regenerable, Git-ignored
models/
  <reference-model-id>/<version>/             # later derived model artifacts
```

Retain `videos/reference`, `videos/user` and environment variable compatibility during migration. Above paths are a **proposal**, not a directive to move the pinned benchmark videos or existing recordings.

Editor requirements:
1. Import/catalog existing local sources and the already available standalone YouTube downloads. Preserve source URL/provenance, hash, original resolution/codec/fps, original capture timing *if known*, and whether the video is real time or slowed/retimed.
2. Register **each actual swing** as a stable shot ID with its golfer, full-swing/other category, club, handedness, source(s), camera view(s), and independent synchronization/calibration status. Never automatically treat DTL+FO from different swings as co-captured.
3. Accurate source-frame navigation and displayed-image identity, including high-FPS seek regression tests. Mark separate *swing-window start/stop* and P1/P10; trimming must not shift source labels silently.
4. Nondestructive crop/ROI and rotation with explicit source-to-edited pixel/frame transforms; leave originals unchanged.
5. Reuse manual P1–P10 annotation and PR #4 landmark labeler for correctable 2D body and observable club landmarks, including missing/occluded/cannot-place status and human review.
6. Show per-view quality/measurement eligibility: usable static geometry, body/club visibility, temporal validity, and whether calibrated same-swing multi-view 3D can **actually** be attempted.
7. Version metadata and exports; preserve existing PR #4 benchmark labels/manifest/reports unmodified; do not require stale deleted user videos.
8. Test round-trips, multiple users, multiple reference golfers, multiple swings from a single source video, mismatched DTL/FO views, exact frames, missing footage, and source-file immutability.

**Explicit non-goals for PR #5:** finished 3D reconstruction, automatic model training, real camera integration, guaranteed detection accuracy, sophisticated YouTube downloading service, and a production multi-account web platform.

### PR #6 — 3D geometry and measurement foundation (implemented, in review)

Calibration, triangulation from labelled same-swing views, the target-line world frame (+X target, +Y left, +Z up, origin under the ball at address), P1 alignment and clubhead trajectory values, versioned artifacts and a CLI — validated on synthetic captures only. See [geometry-3d.md](geometry-3d.md). The proof of concept below is now the **real-camera validation** of this foundation.

### Next — Calibrated 3D proof of concept (real FOX recordings)

Start with one genuinely co-captured DTL/FO FOX pair, **not** arbitrarily paired YouTube clips. Establish camera calibration and common time, triangulate visible body landmarks, reject mismatched/occluded observations, track uncertainty and reprojection error, and compare reconstructions with controlled calibration targets and physical checks. Keep 2D originals and per-view observations. Prove credible 3D before calling the product a 3D analyzer.

Reference-data bottleneck: obtaining **synchronized, calibrated** Grant Waite multi-view footage may be harder than collecting YouTube videos. **Independent professional reference swings** (separate DTL and face-on recordings, different years and clubs) are still valuable: they can feed a later *statistical* 3D inference — a prior-constrained fit with stated uncertainty — developed and validated separately. **Directly measured stereo reconstruction requires calibrated, synchronized recordings of the same swing**; PR #6 refuses independent recordings as stereo inputs, and inferred 3D must never be presented as stereo-measured.

### Next — Automatic swing understanding

- Detect swing windows and P1–P10 automatically, keeping manual edits as corrections/ground truth.
- Evaluate P timing against verified source frames: per-position error, missed detections and false positives, DTL/FO agreement. Fix legacy P6/P7 naming and questionable P4 detection.
- Track body keypoints continuously; develop dedicated grip/shaft/clubhead localization and cross-view association. Clubface orientation around impact may need additional optics/sensors.
- Compute 3D joint angles, displacement, velocity, acceleration and sequence only where geometry/timing support each metric.
- Use confidence/quality gates; an unavailable measurement is better than a fabricated number.

### Next — Grant Waite model v1 and user comparison

- Curate enough usable reference swings with distinct sessions/clubs and known or explicitly uncertain 3D provenance.
- Generate per-shot canonical 3D trajectories, align by verified P-events and normalized time *without erasing actual measured timing*.
- Build **versioned** distributions/allowed ranges and movement-sequence relationships; preserve individual exemplars and training manifest.
- Compare an imported user's reconstructed 3D trajectory against the selected model in the same coordinate system, with scale, quality and occlusion handling.
- Produce prioritized differences anchored to P-events and annotated source evidence. Do not call correlation a diagnosed swing fault without validation.
- Support other golfer-specific models and reference collections without refitting the user's body pose each time.

## Parallel track — Automated capture and replay

**Already working:** Rust synthetic multi-camera `CaptureSession`, per-camera time/byte ring buffers, retroactive triggered extraction and per-shot manifest/PGM writing.

**Not yet working:** FOX/MVS camera integration, live device timestamps, microphone impact trigger, cross-clock drift mapping, true real-time persistence/replay and 3D on a captured swing.

Sequence:
1. Validate raw synchronized recording/import and camera calibration from current FOX footage.
2. Connect real MVS/FOX cameras to Rust capture behind the existing `FrameSource` seam; measure actual fps, USB controller throughput, dropped frames, exposure and gain.
3. Build microphone transient trigger with a refractory period, measured audio-to-camera timestamp conversion, and retroactive clip extraction.
4. Start replay of the captured clips immediately; run pose/P-detection/3D analysis asynchronously after capture, and update the replay when results arrive.
5. Extend from DTL+FO to optional overhead and separate high-speed impact camera; do not assume all devices share clock rates.
6. Keep FlightScope integration and detailed launch/impact measurements as optional later capabilities.

## Acceptance gates for the eventual product

- **Input integrity:** exact decoded source frame, preserved timestamps and nondestructive edits.
- **3D validity:** paired same-swing calibrated inputs, plausible scale/geometry, measured reprojection errors, documented missing/occluded points; uncertainty attached to inferred output.
- **Automated annotation:** measured P-position/body/club accuracy on held-out swings and sessions, with recoverable failures.
- **Reference reproducibility:** model version identifies every training shot and model/processing version; rebuilding does not silently change results.
- **User experience:** imported or triggered DTL+FO swing becomes a correct 3D comparison with minimal manual intervention; untrustworthy metrics are marked unavailable rather than invented.
