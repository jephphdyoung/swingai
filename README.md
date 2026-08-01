# SwingAI: Golf Swing Comparison Tool

SwingAI is a containerized tool that uses MediaPipe and DTW (Dynamic Time Warping) to compare a golfer’s swing against a reference video. It extracts body landmarks, aligns pose sequences, and generates a side-by-side comparison video.

## Features

- Extracts body pose landmarks using MediaPipe
- Aligns videos with FastDTW
- Generates side-by-side comparison video
- Streamlit UI for selecting videos
- Fully containerized (via Docker or Podman)

---

## 📁 Project Structure

swingai/
│
├── main.py
├── streamlit_app.py
├── pose_landmarker.task   # MediaPipe model (required)
├── requirements.txt
├── Dockerfile
├── sample_videos/
│   ├── GW_faceon.mp4
│   └── GW_DTL.mp4
├── my_videos/
│   └── TW_face.mp4
└── utils/
    ├── pose_extractor.py
    ├── video_sync.py
    └── video_renderer.py

---

## 🐳 Run in a Container

### 1. Build the container

podman build -t swingai .
# or use Docker:
# docker build -t swingai .

### 2. Run the app

Mount your video folders and run:

podman run --rm -p 8501:8501 \
  -v "$PWD/sample_videos:/app/sample_videos" \
  -v "$PWD/my_videos:/app/my_videos" \
  -v "$PWD:/app/output" \
  swingai

Then visit: http://localhost:8501

---

## ▶️ Run Without UI (CLI Mode)
```bash
podman run --rm \
  -v "$PWD/sample_videos:/app/sample_videos" \
  -v "$PWD/my_videos:/app/my_videos" \
  swingai python main.py my_videos/TW_face.mp4 sample_videos/GW_faceon.mp4
```c
---

## 💻 Local Dev (No Container)

h### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the analyzer
```bash
python main.py my_videos/TW_face.mp4 sample_videos/GW_faceon.mp4
```

### 3. Run the UI

streamlit run streamlit_app.py

---

## 📦 Sample Videos

Use these sample links to populate your folders:

- Grant Waite Caddie View: https://www.youtube.com/watch?v=6sD5asBLNrw
- Tiger Woods Iron Shot: https://www.youtube.com/watch?v=pccwKQsEeO4

Download them using `yt-dlp`, or use the downloader container if you've built one.

---

## 📂 Output

The comparison video will be saved as:

comparison_output.mp4

If running in a container, the file appears in your mounted volume.

---

## ✅ Requirements

- Python 3.8+
- ffmpeg (only required for local dev with yt-dlp)
- Streamlit (for UI)
- MediaPipe
- OpenCV
- fastdtw

---

## 🎯 P-Position Detection Pipeline

This project includes a comprehensive P-position detection improvement pipeline. See the step-by-step implementation guides:

### Pipeline Overview

```
Input Video
    │
    ▼
┌─────────────────────────┐
│  Step 1: Preprocess     │  Extract timestamps, transcode to CFR 60fps
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Step 2: Trim Waggle    │  Detect swing window, remove practice swings
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Step 3: Resample 60Hz  │  Uniform sampling for comparison
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Step 4: DTW Features   │  Normalize coords, extract angles, smooth
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Step 5: DTW Alignment  │  Align to reference, transfer labels, refine
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Step 6: Club Tracking  │  Refine P2/P6/P7 with shaft angle
└───────────┬─────────────┘
            ▼
      P-Positions (P1-P9)
```

### Implementation Steps

| Step | Description | Files | Tests |
|------|-------------|-------|-------|
| [Step 0](docs/step0.md) | **Evaluation Harness** - MAE metrics for ground truth comparison | `scripts/eval_positions.py` | 23 |
| [Step 1](docs/step1.md) | **Time Normalization** - ffprobe timestamps, CFR 60fps transcode | `utils/video_preprocessor.py` | 14 |
| [Step 2](docs/step2.md) | **Waggle Trimming** - Detect swing start with wrist speed hysteresis | `utils/swing_detector.py` | 19 |
| [Step 3](docs/step3.md) | **Feature Resampling** - Interpolate to uniform 60Hz | `utils/feature_resampler.py` | 23 |
| [Step 4](docs/step4.md) | **DTW Features** - Pelvis-centered coords, angles, Savitzky-Golay smoothing | `utils/dtw_features.py` | 26 |
| [Step 5](docs/step5.md) | **DTW Alignment** - Label transfer from reference + local refinement | `utils/dtw_alignment.py` | 19 |
| [Step 6](docs/step6.md) | **Club Tracking** - Shaft angle detection for P2/P6/P7 | `utils/club_positions.py` | 16 |

**Total: 140 tests**

### Quick Start

```bash
# Run all tests
pytest tests/ -v

# Evaluate P-position accuracy
python scripts/eval_positions.py --ground-truth data/ground_truth.json --verbose
```

### Key Modules

```python
from utils.video_preprocessor import preprocess_video
from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import trim_to_swing
from utils.feature_resampler import resample_landmarks
from utils.dtw_alignment import detect_p_positions_with_dtw
from utils.club_positions import refine_positions_with_club
```

---

## 🗺️ Roadmap

Where SwingAI is headed — from a comparison tool into a full **capture + coaching booth**.
Full detail (vision, architecture direction, technical notes) in
**[docs/ROADMAP.md](docs/ROADMAP.md)**.

**The headline feature: a mic-triggered capture + replay booth.** You swing, the microphone
hears impact, and your swing is looping on screen from both angles with the P-positions
marked — until you hit the next one.

```
2× Fox cameras ──▶ ring buffer (continuous, ~5s)
   microphone  ──▶ impact detected ──▶ extract the 5s BEFORE the trigger
                                            ├──▶ replay loop starts immediately
                                            └──▶ P-detection in background,
                                                 markers painted on when ready
```

**Priority order**
1. **Capture + replay** — webcam/laptop-mic prototype first (buildable now), then the Fox
   cameras via the MVS SDK, then P-markers, then the pro ghost.
2. **Detection quality** *(parallel)* — ground truth + eval baseline, label a real set,
   fix P4, club detection for P2/P6/P8.
3. **GPU pose estimation** *(accelerant, not a blocker)*.

✅ **Shipped:** pro ghost overlay + draggable/resizable overlay editor; DTL/face-on pair
P-position detection running on real 240fps captures.

**Deferred:** the FlightScope Mevo+ integration (the mic replaces it as the trigger; club/
ball data is still wanted later), the 2×2 live-views grid, and the live mirror — none are
prerequisites now that the replay is full-screen and the live feed is never shown.

**Architecture direction:** evolve with clean seams; the FastAPI backend hub is now
load-bearing rather than directional, since a continuous capture loop and an
always-listening audio thread can't live in Streamlit's rerun model. Defer Rust to a proven
hot path. See [docs/ROADMAP.md](docs/ROADMAP.md).

## 🦀 Native runtime (`native/`)

Everything above is the **Python research implementation** — pose extraction, P-position
detection, evaluation, the Streamlit UI. It stays as it is: the baseline, the regression
harness, and where the model work happens.

Native runtime work is now beginning under **`native/`**, a Rust workspace that will own
camera capture, frame timing, synchronization, ring buffers and session storage — the
soft-real-time parts a 240fps two-camera capture booth needs and Python is a poor fit for.
The reasoning, and what is deliberately *not* decided yet, is in
**[docs/adr/0001-hybrid-rust-python-runtime.md](docs/adr/0001-hybrid-rust-python-runtime.md)**.

**Direct Fox camera support is not implemented.** There is no camera code, no ring buffer
and no MVS SDK binding in the workspace today — only the contracts the two sides will
speak. Captures still come from Swing Catalyst.

### The seam: versioned JSON

Rust and Python exchange two documents per shot, both in [`schemas/`](schemas/):

| Document | Written by | Says |
|---|---|---|
| [`capture-manifest`](schemas/capture-manifest.schema.json) | Rust capture runtime | which cameras recorded, where the frames are, and when every frame happened |
| [`analysis-result`](schemas/analysis-result.schema.json) | Python analysis pipeline | the detected swing events, with confidences, warnings and artifacts |

Four rules the contracts enforce, each because the alternative fails quietly:

- **Times are integer nanoseconds since the capture session's monotonic origin**, and the
  persisted origin is zero. Frame indexes are stream-local bookkeeping only — two cameras
  that drop different frames diverge, and the microphone trigger has no frame index at
  all. This matches `data/ground_truth.json` already keying on `timestamp_ms` rather than
  frame number. Timestamps are unsigned, so a negative one cannot be represented.
  Camera and audio devices report on *their own* clocks, possibly with a different epoch
  or tick frequency; the capture runtime must convert into the session clock before
  writing, and may keep the raw device values in a stream's `metadata` for drift
  diagnosis only.
- **Wall-clock `created_at` is validated RFC 3339** and is a distinct type from session
  timestamps, so the two cannot be swapped at a call site. It is for filing, never for
  correlating streams.
- **Paths are forward-slash and host-independent.** A backslash is rejected on every
  platform rather than normalised — on Linux `clips\shot.mkv` is one legal filename, so
  reinterpreting it would invent a path nobody wrote. That also rules out `C:\...`, UNC
  and `\\?\...` spellings for free.
- **The schema version must match exactly.** There is no minor-version compatibility yet,
  so writer and reader move together and a mismatch is a clear error rather than a silent
  partial read. `metadata` and `context` maps are the open extension points; unknown keys
  elsewhere are ignored, not preserved.

Note the contract spells down-the-line `down_the_line`, where `utils/swing_pairing.py` and
`data/annotations.json` say `dtl`. Whichever component bridges the two maps between them;
the Rust side rejects `dtl` rather than guessing.

### Build and test the Rust workspace

Needs a Rust toolchain (1.85+, edition 2024) — [rustup.rs](https://rustup.rs). No camera
SDK, no system libraries, nothing platform-specific; it builds on Windows 11 and Fedora
alike.

```bash
cd native
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

From Linux, you can confirm the Windows build without a Windows machine — `cargo check`
does not link, so this needs only the target's standard library:

```bash
rustup target add x86_64-pc-windows-msvc
cargo check --workspace --all-targets --target x86_64-pc-windows-msvc
```

### Validate the example contracts

```bash
cd native
cargo run -p swingai-contract-check -- capture  ../schemas/examples/capture-manifest.example.json
cargo run -p swingai-contract-check -- analysis ../schemas/examples/analysis-result.example.json
```

The checker deserializes into the Rust types and applies the semantic constraints
deserialization cannot express — timestamp ordering, duplicate camera ids, gaps that
exceed the dropped-frame count. It prints a summary and exits 0, or prints every problem
it found and exits nonzero.

The same examples are validated against the JSON Schemas from the Python side:

```bash
podman run --rm -v "$PWD":/app:Z swingai python -m pytest tests/test_schema_examples.py -q
```

### Layout

```
native/
├── Cargo.toml                        # workspace
├── crates/
│   ├── swingai-core/                 # ShotId, CameraId, CameraView, Timestamp,
│   │                                 #   FrameSequence, PixelFormat, validation errors
│   └── swingai-contracts/            # the two JSON contracts as Serde types
└── apps/
    └── swingai-contract-check/       # CLI validator
```

`swingai-core` and `swingai-contracts` are platform-neutral by rule, not by accident —
a test scans them for `cfg(windows)` and friends. Platform-specific code will live in a
capture crate behind a trait when it arrives.

## TODOs

*in the user video. p1 is around the 2-2.5 second mark
in the ref video , p1 is around the 2sec mark

* ✅ highlight club shaft and club head
* ✅ show joint angles
* show clubface angle
* pause as p1- p8 positions
* show delta between videos
* show sequencing differences
* playback view that will allow users for frame/step by step playback. with p1-p8 markers
* calculate swing cadence. time from start to top of swing (ms) divided by top of swing to impact (ms)
* is it possible to calculate/estimate weight distrubition? And the diretion, and accleration of it?

---

## 📜 License

MIT License – do whatever you want with this, just don't blame us if it slices.
