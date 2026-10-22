# SwingAI — Current Status

**Last reviewed: 2026-10-22.** This is the current implementation snapshot, not an aspirational feature list. The long-term product direction and proposed PR scope live in [ROADMAP.md](ROADMAP.md). Earlier implementation notes remain available in [the pre-update Git revision](https://github.com/jephphdyoung/swingai/blob/1fa9eee/docs/STATUS.md).

## Product goal

Automatically capture or import a golfer's synchronized down-the-line (DTL) and face-on (FO) observations, identify body and club landmarks and P1–P10, reconstruct a time-varying **3D** motion model where calibration and synchronization support it, and compare that 3D swing against a versioned reference model built from **multiple** curated Grant Waite swings. Support multiple users and additional reference golfers without duplicating the reference library.

**Current reality:** SwingAI has a working **offline, predominantly 2D** pose-analysis and comparison prototype plus a **synthetic** multi-camera capture core. There is **no validated 3D reconstruction, Grant Waite aggregate 3D model, or automatic real-camera capture pipeline yet.** MediaPipe's model-relative depth is not calibrated world-space 3D.

## Implementation inventory

| Capability | Current state | Evidence / boundary |
|---|---|---|
| Local video discovery | Working | `videos/reference`, `videos/user` and `videos/output`; recursive discovery and `SWINGAI_{REFERENCE,USER,OUTPUT}_DIR` overrides |
| Swing Catalyst DTL/FO pairing | Working for recognized filenames | `utils/swing_pairing.py` groups by recording timestamp; matching filenames alone do **not** establish optical calibration or general multi-view sync |
| Streamlit UI | Working offline | Existing comparison, annotation and dual-view workflows; not a live capture or reference-dataset editor |
| Body pose extraction | Working | MediaPipe CPU (default); RTMPose-m + YOLOX-m optional GPU backend, merged PR #4 |
| Pose caching | Working | Backend, model and runtime/environment-sensitive cached sequences; stores original-frame pixel coordinates |
| Offline P-position annotation | Working | P1–P10 frame labels saved in `data/annotations.json`; canonical P7 = impact; some legacy entries need review |
| 2D joint annotation | Working as a separate tool | Blind/assisted labeling, original-frame image digests, optional feet and hands/club groups; `data/joint_labels/` |
| Offline dual-view comparison | Working in 2D | Four view panels (user/reference DTL/FO), per-view 2D joint angles and body-relative displacement, not triangulated 3D |
| Side-by-side / ghost overlays | Available | Renderers and overlay editor; not a validated 3D comparison |
| Automatic P1–P10 | Experimental | `utils/p_positions.py` and `scripts/detect_p_positions.py` run on existing recordings, but accuracy and failure cases are not yet validated against a representative labeled dataset |
| Club shaft detection | Experimental | Edge/Hough method; P2/P6/P8 can be missing; **no reliable 3D shaft, clubhead or clubface tracking** |
| Video preprocessing | Utility present | ffprobe timestamps and CFR conversion; not a nondestructive swing editor or camera calibration pipeline |
| YouTube downloader | Standalone utility | `downloader/` has yt-dlp container code; no integrated reference import/editor/catalog |
| Rust capture contracts | Implemented | Versioned capture/analysis JSON schemas, stable shot IDs, per-stream time and relative media paths |
| Rust capture simulation | Working synthetically | Independent camera ring buffers, timestamp-based retroactive extraction, per-shot PGM writer and manifest |
| Real camera/audio capture | **Not implemented** | No working MVS/FOX capture adapter, microphone trigger, hardware clock conversion or replay loop in SwingAI |
| Stereo 3D reconstruction | **Not implemented** | No camera intrinsics/extrinsics calibration, frame association, triangulation, reprojection validation or 3D body/club motion output |
| Aggregated Grant Waite model | **Not implemented** | Current reference selections operate on videos, not a multi-swing 3D model |
| Automatic 3D comparison/coaching | **Not implemented** | No versioned 3D reference distributions, 3D deviation metrics or validated prioritized coaching rules |
| Multi-user data catalog | **Not implemented** | Video directories and annotation keys are file-path-oriented rather than stable golfer/shot IDs |

## PR #4 — merged

PR [#4](https://github.com/jephphdyoung/swingai/pull/4) was merged on 2026-10-22 (merge commit `1fa9eee`). RTMPose + YOLOX run through the ONNX Runtime CUDA execution provider; `verify-gpu` verifies both networks independently, reports exact input video, and fails on missing/unreadable footage without selecting another video. MediaPipe remains the default.

Final **reported local** suite: CPU image **766 passed, 7 skipped** (GPU-dependent); GPU image **771 passed, 2 skipped** (CPU-image-specific). Real GPU verification passed on the committed `videos/reference/GW_faceon.mp4`. These are the final PR #4 reports, **not newly rerun results or GitHub CI results**.

Historical evaluation: 20 hand-labeled frames across four videos; on **six blind face-on frames**, RTMPose PCK@0.1 was 69.7% vs MediaPipe 44.7% (+25.0 percentage points). This is a small 2D benchmark, not 3D evidence. **DTL accuracy was not scored.** See [joint accuracy](benchmarks/joint-accuracy-2026-10-21.md) and [original behavior benchmark](benchmarks/pose-backends-2026-10-20.md).

The June 20, 2026 local user recordings were deleted as stale. Their labels, manifest and reports remain committed and **must not be silently rewritten**. Those historical comparisons are not fully reproducible without restoring the exact source footage; default benchmark scripts still reference the deleted files. Do not equate missing historical inputs with newly validated metrics.

The earlier Rust capture simulation reported **207 Rust tests**; this review checked the code and documented implementation, not a fresh Rust test run.

## Known issues / limitations

1. **Frame-accurate seeking:** on some high-FPS MP4s, OpenCV random seeking displays frame `f-1` while pose/labels refer to frame `f`. Sequential decoding plus image hashes protects the labeling benchmark, but the existing annotator and dual-view panel are not fixed. An editor must first establish exact source-frame identity, not trust random seek blindly.
2. **P-position reliability:** P4, view disagreement and club-driven P2/P6/P8 need evaluation and improvement. Legacy Path A/Path B impact nomenclature must be reconciled (canonical **P7 = impact**) before unified evaluation.
3. **No calibrated 3D geometry:** DTL/FO observations alone are not proof of reconstructable 3D; views must represent the **same swing**, be synchronized, calibrated and have corresponding landmarks. For occluded points, an inference is not a measurement.
4. **Reference footage provenance:** many YouTube videos may have different swings for DTL and FO, unknown capture frame timing, edits or slow motion. Such material can inform 2D/shape priors but **cannot be falsely marked as measured stereo 3D**.
5. **Source data and derivatives are conflated:** annotations use video paths; output contains pose caches; `downloader/` includes duplicate clips. There is no persistent dataset/catalog or separate raw-source-to-edit provenance.
6. **Camera timing:** existing Rust synthetic sources already share the session clock. Real cameras/microphones require measured clock conversion and drift handling. Timestamp proximity is not synchronization proof.

## Current execution paths

```bash
./run.sh                          # Streamlit with default CPU/MediaPipe backend
SWINGAI_GPU=1 ./run.sh             # Optional GPU RTMPose backend
SWINGAI_GPU=1 ./run.sh verify-gpu  # Check both GPU networks on known reference footage
```

The launcher mounts source footage read-only without modifying host permissions or SELinux labels. User capture-library paths can be supplied through `SWINGAI_USER_DIR`. No downloaded/raw user videos need to be committed to Git.

## Next work: PR #5 planning

**Proposed PR #5: multi-user/reference shot catalog and nondestructive reference editor foundation.** Final scope is still under discussion; see [ROADMAP.md](ROADMAP.md).

Required foundations:
- Stable golfer / source-video / shot / camera-stream IDs; keep current Python path-based consumers working during migration.
- Original immutable footage with checksums and provenance; edited swing windows/crops expressed as reversible source-frame/time transformations.
- Per-shot DTL/FO association with explicit `same_swing`, timing and calibration statuses (unknown != verified).
- An editor for precise trim/start/stop, crop, per-view P1–P10 and 2D body/club label correction, with provenance and review status.
- Measurement-specific quality eligibility: geometry, timing, landmark visibility and reliable 3D source eligibility.
- No moving or replacing the committed historical benchmark frames/labels; tests for migration and exact-frame reproducibility.

**Not PR #5:** fully automatic 3D reconstruction, model training, reliable clubface tracking or live FOX hardware capture. PR #5 should build the auditable inputs those stages require.

## Source-of-truth documents

- [ROADMAP.md](ROADMAP.md): product architecture, milestones and PR #5 proposed acceptance criteria.
- [ADR 0001](adr/0001-hybrid-rust-python-runtime.md): Rust capture / Python analysis seam.
- [Dual-view comparison](dual-view-comparison.md): current 2D measurement behavior.
- [Joint labeling](joint-labelling.md): labels, frame hashes and evaluation.
- [GPU pose](gpu-pose.md): current RTMPose/YOLOX and GPU environment.
- [P-position detection](p-position-detection.md): current heuristics and definitions.
