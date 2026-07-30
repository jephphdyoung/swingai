# SwingAI Roadmap

Where SwingAI is headed: from a swing-comparison tool into a **mic-triggered capture and
replay booth**. This doc holds the vision, the architecture direction, and the initiatives
in priority order. Current working state is in [STATUS.md](STATUS.md); granular
analysis-feature TODOs are in the README.

Revised 2026-07-30: live capture + replay moved to the top; the microphone replaces the
Mevo+ as the shot trigger.

---

## Vision

A bay you hit in. You swing, the mic hears impact, and your swing is looping on the screen
in front of you before you've finished your follow-through — from both angles, with the
P-positions marked, until you hit the next one.

The loop:

```
2× Fox cameras ──▶ ring buffer (continuous, ~5s)
                        │
   microphone ──▶ impact transient detected
                        │
                        ▼
        extract the 5s preceding the trigger
                        │
                        ├──▶ replay loop starts IMMEDIATELY (raw clip)
                        │
                        └──▶ pose + P-detection in the background
                                     │
                                     ▼
                        markers painted onto the running loop
                                     │
                        (loops until the next shot triggers)
```

---

## Architecture direction

**Evolve with clean seams; do not big-bang rewrite. Defer Rust to a proven hot path.**

Three separable pieces: **capture/ingest → core (CV/analysis) → UI client(s)**.

- **Keep:** the CV/analysis core (MediaPipe, OpenCV, `utils/overlay.py`, P-detection).
  Product IP and the fastest iteration path.
- **The FastAPI backend is now load-bearing, not directional.** A continuous capture loop,
  an always-listening audio thread, and a ring buffer are not things Streamlit's rerun
  model can host. The backend is the home for the capture service and a REST/WebSocket API
  over captured swings; Streamlit stays as the dev/analysis client.
- **Rust:** only if a benchmark proves the capture/encode loop is the bottleneck, and only
  for that loop. Everything else stays Python.

**Two design rules that fall out of the mic trigger:**

1. **Capture is retroactive.** The trigger fires *after* the swing. Nothing can be
   "started on trigger" — the cameras run continuously into a ring buffer and the clip is
   extracted backwards from the trigger timestamp.
2. **Replay and analysis are decoupled.** The clip is in RAM at trigger time, so the loop
   starts instantly with no analysis at all. P-markers are progressive enhancement,
   applied to the already-running loop whenever detection finishes. Detection latency is
   therefore not on the critical path for the feature to feel live.

The capture stack is Windows-native (cameras/MVS), so the booth rig likely runs SwingAI on
Windows on one machine rather than splitting across machines.

---

## Initiative 1 — Mic-triggered capture + replay loop  ← **the priority**

### Phase 0: webcam + laptop mic prototype *(buildable now, no hardware needed)*

De-risks everything except the MVS SDK and USB3 bandwidth. Build this first.

- Continuous capture into an in-memory ring buffer; retroactive extraction of the N
  seconds before a timestamp.
- Impact detection on the audio stream: onset/transient detection on the amplitude
  envelope, a threshold, and a **refractory period** (~2s) so ball bounce, the club
  hitting the mat, or a dropped club don't double-trigger.
- **Audio→frame timestamp mapping.** The mic timestamp has to resolve to a frame index in
  the video ring buffer. This is the single riskiest unknown in the whole design — both
  streams need a shared monotonic clock, and drift between the audio device clock and the
  camera clock has to be measured, not assumed.
- Replay loop in the UI: full-screen, both views, looping until the next trigger.

Open questions Phase 0 answers: how reliable is the trigger in a noisy bay? How much
pre-trigger buffer is actually needed to catch address? Does replay-then-enhance feel
right, or does the marker pop-in read as broken?

### Phase 1: Fox cameras

Two USB machine-vision cameras (**U3-C-CS-16**), captured on Windows via **MVS**
(Hikrobot Machine Vision Software). Not UVC webcams — likely needs the **MVS SDK**
(`MvCameraControl`); a Linux build is an open question.

**Check the bandwidth arithmetic before designing around it.** At 240fps, per camera,
frame size × 240 is the sustained rate. If the sensor is ~1.6MP mono that's ~384 MB/s per
camera and ~768 MB/s for the pair — past what a single USB3 5Gbps controller delivers in
practice (~350–400 MB/s). Likely implications: separate host controllers, a reduced ROI,
or the discovery that 240fps is only available at reduced ROI anyway. A 5s two-camera
ring buffer at that frame size is ~3.8GB of RAM — sized, but not free.

### Phase 2: P-positions on the replay

Path B (`scripts/detect_p_positions.py` → `utils/p_positions.py`) already takes a
DTL/face-on pair and emits P-positions — that is exactly the shape a captured swing has.
The adaptation is small: with live capture *we* name the files and know which camera is
which, so `swing_pairing`'s Swing Catalyst filename parsing is bypassed entirely.

Two replay modes, toggleable:
- **Continuous** (default) — swing loops start-to-finish, P1–P10 as clickable ticks on a
  scrubber. Degrades gracefully: if detection is off, the video is still correct.
- **Step-and-hold** — pauses at each detected P-position in turn. Much more sensitive to
  detection accuracy, which makes it a useful informal check on the detector.

### Phase 3: pro ghost on the replay

Ghost the reference pro onto the looping capture, synced on P-positions. Reuses
`utils/overlay.py`, `render_mode="overlay"`, and the shipped overlay editor. Depends on
P-labels being trustworthy — see Initiative 2.

---

## Initiative 2 — Detection quality *(parallel track)*

No longer blocks Initiative 1: the capture/replay plumbing doesn't care whether P4 is
30ms off. It does gate Phase 2 markers being *useful* and Phase 3 working at all. Run it
alongside, not before.

1. **Make detection measurable.** Convert hand labels in `data/annotations.json` into
   `data/ground_truth.json` and get `scripts/eval_positions.py` running for a per-position
   MAE baseline. The harness is written and tested; it has never had an input.
2. **Label a real set.** Three labeled videos is not enough to tune anything. Prioritise
   DTL. Correcting auto-detected positions is faster than labelling from scratch.
3. **Reconcile the impact-frame naming.** Path A calls the velocity peak P6; path B calls
   it P7. Fix before any cross-path evaluation (see CLAUDE.md).
4. **Fix P4.** Establish whether the 814ms P4→P7 span is a real slow transition or a
   mis-detected top. Candidate replacement for the current global-speed-minimum rule:
   hand-path direction reversal.
5. **Club detection for P2/P6/P8.** These fail outright today; `club_detector.py` uses
   Canny + `HoughLinesP` and picks up body edges. A small trained shaft/clubhead detector
   would replace it — 240fps footage has low motion blur and there are 5,132 clips to
   build a dataset from.
6. **Cross-view fusion.** Take each position from the view that resolves it best — arm
   tier from face-on, shaft tier from DTL — rather than preferring face-on wholesale.

---

## Initiative 3 — GPU pose estimation *(accelerant)*

Not a blocker under progressive enhancement, but it shortens the gap between trigger and
markers, and it's the fix for the DTL disagreement and wrist jitter near impact that
MediaPipe BlazePose causes. A stronger top-down model (RTMPose / ViTPose / YOLO-pose) is
markedly better on fast motion and on the self-occluding DTL orientation.

Keep it behind the existing extractor interface — `p_positions.py` consumes landmarks per
frame and needs no changes. Compare against MediaPipe on swing `2026-07-27 135006`, where
the current failure is already characterised.

Cheap wins available first, without any model swap: trim to the swing window before
extracting pose (a 2s swing at 240fps is ~480 frames, not 1,244), and run detection on a
decimated stream (every 4th frame = 60Hz) with local refinement at full rate around P4 and
impact.

The running `swingai` image is CPU-only (`python:3.12-slim`); GPU work needs a CUDA base.

---

## Deferred

Explicitly not on the critical path any more.

- **FlightScope Mevo+ integration.** The microphone replaces it as the trigger, which was
  the only reason it was urgent. Club/ball metrics remain desirable later — the best path
  is still impersonating GSPro over the OpenConnect v1 protocol
  ([spec](https://gsprogolf.com/GSProConnectV1.html)): GSPro is the TCP server on
  `127.0.0.1:921` and FlightScope's connector is the client, so running our own server on
  `:921` gets full BallData + ClubData per shot. Reference impls:
  `springbok/MLM2PRO-GSPro-Connector`, `kenjdavidson/gspro-connector`.
- **Live Views 2×2 grid.** Since the replay is full-screen and the live feed is never
  shown, no live video transport is needed for v1. This becomes a separate feature with
  its own MJPEG/WebRTC work, not a prerequisite.
- **Live mirror with the pro.** Real-time mirror-flipped camera with a pro auto-fit onto
  the user every frame. Still wanted; needs the live transport above.

---

## Recently shipped
- Ghost-overlay comparison mode + draggable/resizable overlay editor.
- DTL/face-on pair P-position detection (`utils/p_positions.py`), running end-to-end on
  real 240fps captures — quality not yet validated.
- `videos/{reference,user,output}` layout with env-var overrides; swing pairing from
  Swing Catalyst filenames.
