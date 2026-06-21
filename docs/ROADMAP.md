# SwingAI Roadmap

Where SwingAI is headed: from a swing-comparison tool into a full **capture + coaching
booth**. This doc holds the vision, the architecture direction, and the major initiatives.
Granular analysis-feature TODOs live in the README.

---

## Vision

A golf capture + coaching system (booth) that:

1. **Captures everything for a swing**, shot-triggered:
   - DTL + face-on video (two Fox 240fps cameras)
   - Club + ball flight data from a FlightScope Mevo+
   - Triggered by the Mevo+ shot event so video + data are captured together.
2. **Plays back** the captured swing — both angle videos + the data.
3. **Overlays a pro on playback** — ghost the pro's swing onto the user's captured swing
   (the overlay editor already shipped; extend it from the bundled demo to captured swings).
4. **Live mirror with the pro** — real-time: the user sees their live (mirror-flipped)
   camera with a pro overlaid so they can match positions live.

How the pieces interlock: **Mevo+ fires → triggers the two Fox cameras → captures video +
data → playback with pro overlay**; separately, the live camera feed powers the live mirror.

The capture stack is Windows-native (cameras / Mevo+ / MVS / FS Golf), so the live rig likely
runs SwingAI on Windows too (one machine) rather than splitting across machines.

---

## Architecture direction

Verdict: **evolve with clean seams; do not big-bang rewrite. Defer Rust to a proven hot
path. Outgrow Streamlit via a backend, not a frontend swap (yet).**

- **Keep:** the CV/analysis core (MediaPipe, OpenCV, `utils/overlay.py`, P-detection, DTW).
  Product IP + fastest iteration path.
- **Strain points the roadmap exposes:**
  - Streamlit's rerun model fights live / multi-stream video (already worked around with
    custom HTML/JS twice).
  - Python on the hot capture path — sustained 2×240fps capture + rolling buffer + encode is
    where the GIL bites.
- **Plan:**
  1. Introduce a **FastAPI backend hub** — home for the Mevo+ OpenConnect listener, the
     MJPEG/WebRTC live streams, and a REST/WebSocket API over stored swings + events.
     Streamlit becomes one *client*, not the whole app.
  2. Three separable pieces: **capture/ingest → core (CV/analysis) → UI client(s)**.
  3. UI: keep Streamlit as the dev/analysis tool; a real booth/live UI (web on FastAPI, or
     desktop) comes later — the backend makes that choice cheap.
  4. Rust: only if a benchmark proves the capture/encode loop is the bottleneck, and only for
     that loop (via PyO3 / native bindings). Everything else stays Python.
- **Suggested sequence:** webcam capture-loop prototype → FastAPI backend wrapping core +
  Mevo+ listener → migrate live features onto it → then evaluate frontend + any Rust hot path.

---

## Major initiatives

### 1. Pro overlay on captured swings  *(overlay engine shipped)*
Ghost a pro's swing onto the user's swing, registered on the body, with manual drag/resize.
**Done:** `utils/overlay.py`, `render_mode="overlay"`, the `overlay_editor` component, the
Side-by-side / Overlay view switch. **Next:** extend from the bundled demo pair to arbitrary
captured swings (needs reliable P-labels to sync on — see #2).

### 2. Auto-label P-positions from a reference library
Turn hand-labeled swings into an automated labeler.
- Wire in **V2 DTW label-transfer** (`detect_p_positions_with_dtw`) at the
  `analyzer.analyze_swing` step-3 seam: pick the best-matching labeled reference per input,
  transfer labels along the DTW warp, refine locally; median across several references for
  robustness.
- Use the hand labels as **ground truth** for `scripts/eval_positions.py` (MAE) to prove
  V2 > V1 and tune.
- Per view angle (face_on / dtl). **Bootstrap:** hand-label a small clean set → auto-label
  the rest → correct misses in the Annotate tab → corrections become new references.
- (Later) a learned per-frame / temporal P-detector trained on the corpus.

### 3. Live Views page
New page, **2×2 grid**: live DTL, live face-on, looped DTL reference, looped face-on
reference — so the user watches their two live angles next to the matching looped pro.
Needs the live-capture transport and (in-container) camera device passthrough.

### 4. Live mirror with the pro
Real-time, mirror-flipped camera with a pro **skeleton + faint ghost** auto-fit onto the
user's body every frame, with **step-through** (match & hold P-positions) and **continuous
loop** modes.
- Reuses the overlay module: live auto-fit = `auto_register` run per live frame.
- Architecture: server-side capture→pose→composite loop streamed as MJPEG/WebRTC (not
  Streamlit `st.image` in a loop); MediaPipe LIVE_STREAM mode; MVS SDK frames.
- **Phase 0 (buildable now, no cameras):** webcam prototype to de-risk real-time framerate
  and whether auto-fit *feels* right.

### 5. Capture hardware integration

**Fox cameras @ 240fps.** Two USB machine-vision cameras (model **U3-C-CS-16**), captured on
Windows via **MVS** (reads as Hikrobot Machine Vision Software). Not UVC webcams, so likely
need the **MVS SDK** (`MvCameraControl`); check for a Linux build. Investigate USB3 bandwidth
for 2 cams @ 240fps and a rolling buffer for retroactive clipping.

**FlightScope Mevo+ trigger + data.** Goal: shot-event → trigger capture, and ingest
club/ball metrics.
- Best path: **impersonate GSPro via the OpenConnect v1 protocol**
  ([spec](https://gsprogolf.com/GSProConnectV1.html)). GSPro is the TCP **server** on
  `127.0.0.1:921`; FlightScope's connector is the **client**. Run our own server on `:921`,
  point FlightScope's connector at it → it streams every shot's full **BallData + ClubData**,
  reply `200`. The shot message (`ContainsBallData:true`, `IsHeartBeat:false`,
  `LaunchMonitorBallDetected`) is the camera trigger. **Note:** GSPro is *not* in the data
  path — we replace it; data comes from FlightScope's connector (which reads the Mevo+
  directly). Reference impl: `springbok/MLM2PRO-GSPro-Connector`, `kenjdavidson/gspro-connector`.
- Alternatives: licensed FlightScope API/SDK (sanctioned, gated); or read FS Golf PC's
  exported session files and correlate by timestamp (post-hoc, no live trigger).

---

## Recently shipped
- Ghost-overlay comparison mode + draggable/resizable overlay editor.
- Fixed `data/` volume mount so annotations persist across container restarts.
