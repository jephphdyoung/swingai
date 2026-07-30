# CLAUDE.md

**SwingAI** compares a golf swing against a reference pro's: MediaPipe extracts pose
landmarks, the swing is segmented into the "P-positions" (P1 address → P4 top →
impact → P10 finish), the two swings are time-aligned, and a synchronized
side-by-side or ghost-overlay video is rendered and served through a Streamlit UI.

**Read [docs/STATUS.md](docs/STATUS.md) first.** It is the working state — what is built,
what is unvalidated, the environment (GPU/podman), and the ordered next steps. This file
covers only the structure and the conventions that are easy to get wrong.

## Three P-position code paths

Verified by import graph, not by intent — three separate detectors exist and only one
reaches the app.

**A. `analyzer.analyze_swing` — what the app runs.** `pose_extractor` →
`pose_features.detect_p_positions` (heuristic: velocity peak = **P6**, arm-height minimum
= P4, everything else filled in *proportionally*) with an impact override from
`swing_detector` → `video_sync.sync_on_p_positions` (piecewise-linear warp anchored on
P1→P4→**P6**→P9→P10) → `video_renderer` / `overlay`. Simple and self-contained, but the
proportional fill means P2/P3/P5/P7/P8 are essentially interpolated guesses.

⚠️ **The two paths disagree about which P is impact.** Path A calls the velocity peak
**P6** (`pose_features.py:14`, `video_sync.py:29`). Path B calls the hand-speed peak
**P7**, which matches the standard P-system where P6 is shaft-parallel in the downswing.
Path B is right; path A's naming is legacy. Anything comparing detector output across the
two paths — the eval harness especially — has to reconcile this first.

**B. `scripts/detect_p_positions.py` — the DTL/face-on pair detector.** `swing_pairing`
(parses Swing Catalyst filenames) → `utils/p_positions.py` (which uses `pose_extractor`,
`swing_detector`, `club_detector`) → a P-position JSON. Runs on real 240fps captures.
Detection quality is **not yet measured** — see STATUS.md for the specific doubts.

**The bridge between A and B already exists:** `detect_p_positions.py --save-annotations`
writes into `data/annotations.json`, and saved annotations take precedence over
auto-detection in `analyze_swing`. That is how B's output reaches the renderer today —
no code change needed.

**C. The DTW modules — imported by tests only.** `dtw_alignment`, `dtw_features`,
`feature_resampler`, `club_positions` have docs (`docs/step3–6.md`) and tests but **zero
non-test importers**. `video_preprocessor` is imported only by `scripts/eval_positions.py`.
Treat these as a documented, tested shelf — not part of any running path. Wiring path C in
would mean replacing the detector in `analyze_swing` step 3, and mapping frames back from
its resampled 60Hz timeline to original video frames before rendering.

## Entry points
- `app.py` — Streamlit UI. **Analyze** tab (pick user + reference video → custom HTML5
  player with P-position jump buttons, skeleton toggle, side-by-side / overlay switch) and
  **Annotate** tab (`annotator.py`: frame-by-frame manual P-marking + save).
- `main.py` — CLI: `python main.py <user_video> <reference_video>`.
- `analyzer.py::analyze_swing` — the orchestrator both call (5 reported steps; logs to
  stderr, reports progress via an optional `progress_callback`).
- `scripts/detect_p_positions.py` — `--swing <clip>` (partner view found automatically) or
  `--session <dir>` for a whole capture session.
- `scripts/eval_positions.py` — MAE vs. a ground-truth JSON. **The intended way to judge
  any detector change.** Its input `data/ground_truth.json` does not exist yet.

## Paths and data
- `utils/paths.py` resolves `videos/{reference,user,output}`, each overridable with
  `SWINGAI_REFERENCE_DIR` / `SWINGAI_USER_DIR` / `SWINGAI_OUTPUT_DIR`. `run.sh` mounts
  whatever they resolve to. `list_videos()` recurses, so pointing `SWINGAI_USER_DIR` at a
  capture session finds clips in per-session subfolders.
- `utils/annotations.py` ↔ `data/annotations.json`: per-video P-positions (frame +
  `timestamp_ms`) and a `_view_angle` (`face_on` / `dtl`), keyed by normalized relative
  path. Ground truth is keyed by `timestamp_ms`, not frame, on purpose — frame-rate
  independence. Schema example: `data/ground_truth_example.json`.

## Pose extraction conventions (easy to get wrong)
- `pose_extractor.py` selects **23 of MediaPipe's 33** landmarks
  (`SELECTED_LANDMARK_INDICES`) and stores each frame as `[(x,y,z), ...]` in that order.
  **Index into the selected list, not raw MediaPipe indices.** In `pose_features.py`:
  `1,2`=shoulders, `3,4`=elbows, `5,6`=wrists, `13,14`=hips. `p_positions.py` defines its
  own named constants against the same selected list. `SELECTED_LANDMARK_INDICES` order
  also defines skeleton connections in `video_renderer.py` — keep them in sync.
- On a missing or `!= 33` detection the extractor **carries forward the previous frame's
  landmarks**, so landmark arrays always match frame count.
- Handedness is assumed right-handed and passed as a flag; it is not yet stored per-video
  alongside `_view_angle`.

## Running things
There is no local pytest — the container has the deps.
```bash
podman run --rm -v "$PWD":/app:Z swingai python -m pytest tests/ -q   # 207 tests
./run.sh                                                             # app on :8501
```
GPU containers need `--device nvidia.com/gpu=all --security-opt=label=disable`, and
Blackwell (sm_120) needs cu128 or newer. Details and the CDI-regeneration fix: STATUS.md.

## Gotchas
- `requirements.txt` is unpinned. `pose_landmarker.task` (~5.7MB) must be at repo root.
- Test coverage is lopsided: almost all of it is on paths B and C. Path A — the one the
  app actually runs — has essentially none.
- `data/annotations.json.pre-videos-move` and `.host-backup` are stale restructure
  backups.
