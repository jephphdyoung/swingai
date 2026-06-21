# CLAUDE.md

Guidance for working in this repo. **SwingAI** compares a user's golf swing against a
reference pro's swing: MediaPipe extracts pose landmarks, the swing is segmented into the
9 "P-positions" (P1 address → P4 top → P6 impact → P10 finish), the two swings are
time-aligned, and a synchronized side-by-side video (with optional skeleton overlay) is
rendered and served through a Streamlit UI.

## ⚠️ Read this first: there are TWO P-position pipelines

This is the single most important thing to understand before changing detection code, and
the thing that's been going back and forth.

### Pipeline V1 — "live" (what the running app actually uses)
Wired into `analyzer.analyze_swing`, which `app.py` and `main.py` call:
- `utils/pose_features.py::detect_p_positions` — heuristic detection: velocity peak = P6
  (impact), arm-height minimum = P4 (top), back-solve for P1 (address), everything else
  filled in **proportionally** between the anchors.
- `utils/swing_detector.py::detect_swing_window` — supplies an impact-frame override to V1.
- `utils/video_sync.py::sync_on_p_positions` — piecewise-**linear** time warp anchored only
  on P1→P4→P6→P9→P10. Intermediate P's are interpolated, not independently aligned.

### Pipeline V2 — "DTW" (built + fully tested, but NOT wired into the app)
The whole "P-Position Detection Pipeline (Steps 0–6, 140 tests)" section in `README.md`.
These modules exist, have docs (`docs/step0–6.md`) and tests, but **nothing outside
`tests/`, `docs/`, and `scripts/eval_positions.py` imports them.** The app does not run them.
- `utils/video_preprocessor.py` — ffprobe timestamps + transcode to CFR 60fps (handles VFR).
- `utils/feature_resampler.py` — interpolate landmarks to a uniform 60Hz grid.
- `utils/dtw_features.py` — pelvis-centered + body-scale-normalized coords, joint/spine
  angles, Savitzky-Golay smoothing → a feature matrix.
- `utils/dtw_alignment.py::detect_p_positions_with_dtw` — DTW-align user features to a
  labeled reference, **transfer** the reference's P-labels, then locally refine P4/P6/P7.
- `utils/club_positions.py` + `utils/club_detector.py` — shaft-angle / "parallel shaft"
  detection to refine P2/P6/P7.

### Why the back-and-forth (the actual tradeoff)
- **V1** is simple and self-contained but the proportional fill means P2/P3/P5/P7/P8 are
  basically guesses, and linear sync drifts when swing tempos differ. It needs no reference
  labels to produce *a* result.
- **V2** is more accurate in principle (real time-normalization, per-position refinement,
  club tracking) but **needs a labeled reference swing** to transfer labels from, is heavier
  (ffmpeg transcode, more deps/compute), and isn't connected to the renderer yet.
- The bundled demo currently sidesteps both: `data/annotations.json` holds **hand-labeled**
  P-positions for the two shipped videos, so `analyze_swing` loads saved annotations instead
  of running either detector. Delete/clear those to exercise V1's auto-detection.

**If asked to "wire in V2": the integration point is `analyzer.analyze_swing` step 3** —
replace the `detect_p_positions(...)` auto-detect branch with `detect_p_positions_with_dtw`,
feeding it preprocessed/resampled features and a labeled reference. Sync (step 4) and the
renderer (step 5) consume the resulting `{P-name: frame}` dict, so they shouldn't need
changes — but note V1's frame indices are in *original* video frames, while V2 works on a
resampled 60Hz timeline, so map frames back before rendering.

## Entry points
- `app.py` — Streamlit UI. **Analyze** tab (pick user + sample video → Run Comparison →
  custom HTML5 player with P-position jump buttons, skeleton on/off toggle, keyboard nav)
  and **Annotate** tab (`annotator.py`: frame-by-frame manual P-position marking + save).
- `main.py` — CLI: `python main.py <user_video> <reference_video>`.
- `analyzer.py::analyze_swing` — the orchestrator both entry points call (5 reported steps).
- `run.sh` / `Dockerfile` — Podman/Docker build + run on port 8501 (includes SELinux
  `chcon` handling for Fedora).

## Data & annotations
- `utils/annotations.py` ↔ `data/annotations.json`: per-video saved P-positions (frame +
  timestamp_ms) and a `_view_angle` (`face_on`/`dtl`), keyed by normalized relative path.
  `has_annotations` drives the `[P]` marker in the UI dropdowns.
- Saved annotations **take precedence over auto-detection** in `analyze_swing`.
- `data/ground_truth_example.json` — schema for the eval harness (timestamp-based, frame-rate
  independent). Ground truth is keyed by `timestamp_ms`, not frame, on purpose.

## Pose extraction conventions (important when indexing landmarks)
- `utils/pose_extractor.py` selects **23 of MediaPipe's 33** landmarks
  (`SELECTED_LANDMARK_INDICES`) and stores each frame as `[(x,y,z), ...]` in that order.
  So index into the *selected* list, not raw MediaPipe indices. In `pose_features.py`:
  `1,2`=shoulders, `3,4`=elbows, `5,6`=wrists, `13,14`=hips (these are positions in the
  selected list). `SELECTED_LANDMARK_INDICES` order also defines skeleton connections in
  `video_renderer.py` — keep them in sync.
- On a missing/!=33 detection, the extractor **carries forward the previous frame's
  landmarks** to preserve frame alignment (so landmark arrays always match frame count).

## Evaluation
`python scripts/eval_positions.py --ground-truth data/ground_truth.json --verbose`
Reports MAE in frames and ms, per-position and overall, vs. a ground-truth JSON. This is the
intended way to judge V1 vs V2 quality — use it before claiming one detector is "better."

## Tests
`pytest tests/ -v` — 140 tests, almost all covering the **V2** modules + the eval harness.
There is little/no test coverage of the V1 path actually used by the app.

## Other dirs
- `downloader/` — yt-dlp container for fetching source videos (separate Dockerfile).
- `poc/extract_p1_p4_p7.py` — standalone "v2 poc" script (recent direction).
- `docs/step0–6.md` — detailed writeups of each V2 step.
- `sample_videos/` (reference/pro) and `my_videos/` (user) are the two video-source folders
  the UI reads from and Docker mounts.

## Conventions / gotchas
- `requirements.txt` is unpinned. Core deps: streamlit, opencv-python-headless, mediapipe,
  fastdtw, scipy, numpy, dtaidistance, pytest. `pose_landmarker.task` (the MediaPipe model,
  ~5.7MB) must be present at repo root.
- `analyzer.py` logs to stderr and reports progress via an optional `progress_callback`.
- `README.md` has a couple of typos in shell snippets (`}c`, `h### 1.`) — cosmetic.
- README TODOs / desired features: clubface angle, swing cadence (backswing ms ÷ downswing
  ms), weight-distribution estimation, and explicit delta/sequencing comparison between the
  two swings.
</content>
