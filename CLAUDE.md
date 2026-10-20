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

## Dual-view comparison (PR #3)
`dual_view_tab.py` → `utils/dual_view.py` → `utils/pose_data.py` + `utils/joint_measurements.py`.
Four panels (your DTL/FO vs. reference DTL/FO) at one P-position, from **saved annotations
only**, with 2D joint measurements. Uses the canonical P-system (P7 = impact) and its own
pose layer: named landmarks (`utils/landmarks.py`), pixel coordinates, per-frame `detected`,
**no carry-forward**, cached on disk under `<output>/pose_cache/`. Path A's extractor is
untouched. Full doc: [docs/dual-view-comparison.md](docs/dual-view-comparison.md).

## Pose backends (PR #4)
`utils/pose_data.py` holds the `PoseBackend` protocol, `make_backend("mediapipe"|"rtmpose")`
and the cache. RTMPose (`utils/pose_rtmpose.py`: YOLOX-m + RTMPose-m Halpe26, ONNX Runtime
CUDA) runs only in the GPU image: `SWINGAI_GPU=1 ./run.sh`. **MediaPipe stays the default**:
accuracy vs. human labels is not measured yet (labelling set in `data/joint_labels/`).
Gotchas:
- Confidence is **per backend** (MediaPipe visibility 0.5, RTMPose SimCC 0.3), carried on
  each `PoseSequence`; never compare the scores.
- Cache identity includes model hashes, preprocessing config, ORT version, device and a
  **runtime fingerprint**. The same MediaPipe gives different landmarks in the Debian CPU
  image vs. the Ubuntu GPU image, so entries are per environment.
- `onnxruntime-gpu` 1.31 needs **CUDA 13**; on CUDA 12 its provider silently fails to load
  and ORT falls back to CPU (the backend refuses that).
- `/etc/cdi/nvidia.yaml` goes stale on driver updates; `run.sh` uses a per-user spec in
  `~/.local/share/swingai/cdi` via `podman --cdi-spec-dir` instead (no sudo).
- Both images install with `-c constraints.txt` (frozen CPU-image versions).
Docs: [docs/gpu-pose.md](docs/gpu-pose.md), [benchmark](docs/benchmarks/pose-backends-2026-10-20.md).

## Entry points
- `app.py` — Streamlit UI. **Analyze** tab (pick user + reference video → custom HTML5
  player with P-position jump buttons, skeleton toggle, side-by-side / overlay switch) and
  **Annotate** tab (`annotator.py`: frame-by-frame manual P-marking + save), and
  **Dual-View Comparison** tab (`dual_view_tab.py`).
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
  independence. Schema example: `data/ground_truth_example.json`. Saves now also write
  `_p_convention: "p7-impact"`; entries without it are legacy (may call impact P6) and
  are flagged in the UI until confirmed.

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
podman run --rm --security-opt label=disable -v "$PWD":/app swingai python -m pytest tests/ -q   # 600 pass, 4 GPU-only skip
./run.sh                                                             # app on :8501
SWINGAI_GPU=1 ./run.sh             # + RTMPose on the GPU; `verify-gpu` to prove it
```
GPU containers need `--device nvidia.com/gpu=all --security-opt=label=disable`, and
Blackwell (sm_120) needs cu128 or newer. Details and the CDI-regeneration fix: STATUS.md.

## Gotchas
- `requirements.txt` is unpinned. `pose_landmarker.task` (~5.7MB) must be at repo root.
- Test coverage is lopsided: almost all of it is on paths B and C and the dual-view tab.
  Path A (the Analyze tab) has essentially none beyond the app smoke test.
- **Never mount footage with `:z`/`:Z`** (even `:ro,Z`): it relabels the host files. Use
  `--security-opt label=disable` and plain `:ro` mounts, as `run.sh` and the commands above
  do. `run.sh` mounts user/reference read-only and never chmods or chcons anything
  (`tests/test_run_sh.py`). `:Z` on the repo also locks out any other container using it.
- `data/annotations.json.pre-videos-move` and `.host-backup` are stale restructure
  backups.
