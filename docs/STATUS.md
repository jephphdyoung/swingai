# Status & Next Steps

Working state as of **2026-07-27**. Read this first when picking the work back up.

Design rationale lives in [p-position-detection.md](p-position-detection.md);
the longer-range vision is in [ROADMAP.md](ROADMAP.md). This file is just where
things stand and what to do next.

---

## Where things stand

### Built and verified

**Video layout** — `videos/{reference,user,output}`, resolved through
`utils/paths.py` and overridable with `SWINGAI_REFERENCE_DIR`,
`SWINGAI_USER_DIR`, `SWINGAI_OUTPUT_DIR`. `run.sh` mounts whatever they resolve
to. `list_videos()` recurses, so pointing `SWINGAI_USER_DIR` at a capture
session finds clips nested in per-session folders.

**Swing pairing** (`utils/swing_pairing.py`) — parses Swing Catalyst filenames:

```
Jeff Young - 2026-07-27 135006 Fox Down the line 61fc.mp4
<golfer>   - <date>     <time> <camera label>     <camera id>
```

The timestamp identifies the swing, the camera id identifies the view. Verified
against the 2026-07-27 library: 93 swings, all pairing 1:1, camera ids stable
(`61fc` down-the-line, `aaf0` face-on). Unpaired clips are reported, never
silently dropped.

**P-position detection** (`utils/p_positions.py`, `scripts/detect_p_positions.py`)
— consumes a DTL/face-on pair and writes a P-position JSON file. Runs
end-to-end on real 240fps captures.

207 tests pass (140 pre-existing, 67 new).

### Detection quality: not yet validated

The pipeline runs. Whether its numbers are *right* is unmeasured, and there are
specific reasons for doubt on a real swing (2026-07-27 135006):

| Symptom | Reading |
|---|---|
| P4→P7 spans 814ms | A downswing is ~250–300ms. Either P4 is catching a hesitation partway up the backswing, or P7 is early. P5 landing 72ms after P4 points the same way. |
| DTL disagrees with face-on: P1 by 1063ms, P10 by 1368ms, P7 by 140ms | `swing_detector`'s window is unreliable on the DTL view. Face-on wins on confidence; DTL currently contributes only a cross-check. |
| P2/P6/P8 absent | Club tracking failed. Reported as absent rather than guessed. |
| Confidence is 1.00 for most positions | Only P7's (0.94) discriminates. The rest saturate and carry no information. |

**These are plausibility readings, not error measurements.** Nothing can be
scored until ground truth exists — the first item under Next steps.

### Two bugs found and fixed by end-to-end testing

Worth knowing, because both were invisible to unit tests:

- **Impact was delegated to `swing_detector.impact_frame`**, which is derived
  from motion-burst boundaries, not peak speed. It landed 353ms late at less
  than half peak hand speed. P7 is now the actual hand-speed peak, and the late
  P7 had been dragging P9 to one frame after impact.
- **View alignment was computed from detected impacts**, which propagated a
  detection error into a bogus −489ms clock offset. Matched pairs are
  co-triggered and frame-identical (both 249.3fps, 1244 frames, 4.99s), so
  alignment now trusts the shared trigger and uses impact agreement purely as a
  validation check.

---

## Environment

### GPU is available and working

```
GPU        NVIDIA GeForce RTX 5070 Ti, 16GB
Driver     610.43.03, CUDA UMD 13.3
Capability 12.0 (sm_120, Blackwell)
Verified   torch 2.11.0+cu128, sm_120 in build, CUDA matmul executes
```

Two things had to be fixed to get containers onto the GPU, and both will bite
again on a driver update:

1. **The CDI spec goes stale.** It referenced `/dev/dri/card0`, which no longer
   exists on this machine (the nodes are `card1`/`card2`). Regenerate with:
   ```bash
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   ```
2. **Rootless podman needs SELinux relabelling disabled** or NVML fails with
   "Insufficient Permissions".

The invocation that works:

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable <image>
```

**Blackwell needs cu128 or newer.** Any image built against cu121 or older will
fail with "no kernel image available" — this rules out a lot of older prebuilt
CV containers.

The running `swingai` image is CPU-only (`python:3.12-slim`). GPU work needs a
CUDA base image.

Also present: Ollama with `qwen2.5-coder:32b`, `qwen2.5:14b`, `llama3.2:3b`,
`nomic-embed-text`.

### Running things

```bash
# tests (no local pytest; the container has the deps)
podman run --rm -v "$PWD":/app:Z swingai python -m pytest tests/ -q

# detect P-positions for one swing (partner view found automatically)
podman run --rm -v "$PWD":/app:Z -v "<session>":/swing:ro,Z swingai \
  python scripts/detect_p_positions.py --swing "/swing/<clip>.mp4" \
  --out-dir /app/videos/output

# whole session
... --session /swing --out-dir /app/videos/output

# the app
./run.sh          # http://localhost:8501
```

Capture library lives at `~/Golf/Videos/2026-07-27/` (SwingCatalyst + FSGolf);
`~/Golf/Claude.md` documents how to remount the Windows partition for more.

---

## Next steps

**Reprioritised 2026-07-30.** Mic-triggered capture + replay is now the headline
feature; the microphone replaces the Mevo+ as the shot trigger. Full reasoning
in [ROADMAP.md](ROADMAP.md). Two tracks run in parallel — the capture plumbing
no longer waits on detection quality, because the replay loop starts on the raw
clip and P-markers are painted on when detection finishes.

### Track A — capture + replay

**A1. Webcam + laptop mic prototype.** Buildable now, no hardware needed.
Continuous capture into a ring buffer, retroactive extraction of the N seconds
*before* a trigger timestamp, transient detection on the audio envelope with a
~2s refractory period, and a full-screen looping replay.

The riskiest unknown is **audio→frame timestamp mapping**: the mic timestamp has
to resolve to a frame index in the video ring buffer, which needs a shared
monotonic clock and measured (not assumed) drift between the audio device clock
and the camera clock. Settle this in the prototype.

**A2. Fox cameras via the MVS SDK.** Check the USB3 bandwidth arithmetic before
designing around it — 2 cameras × 240fps may exceed one host controller, and a
5s two-camera ring buffer is multiple GB of RAM. See ROADMAP.md Phase 1.

**A3. P-markers on the replay.** `scripts/detect_p_positions.py` already takes a
DTL/face-on pair, which is exactly the shape a captured swing has; with live
capture we name the files ourselves, so the Swing Catalyst filename parsing in
`swing_pairing` is bypassed. Continuous mode (scrubber ticks) and step-and-hold
mode, toggleable.

**A4. Pro ghost on the replay.** Reuses `utils/overlay.py` and the shipped
overlay editor. Needs Track B to be trustworthy first.

### Track B — detection quality

**B1. Make detection measurable.** Convert the hand-labeled swings in
`data/annotations.json` into `data/ground_truth.json` and get
`scripts/eval_positions.py` running. The harness is written and tested; it has
no input. Record a per-position MAE baseline.

Note `data/ground_truth.json` is referenced by `README.md` and `docs/step0.md`
but has never existed.

**B2. Label a real set.** Three labeled videos today (two face-on, one DTL) is
not enough to tune anything. Use the Annotate tab; prioritise DTL. Correcting
auto-detected positions is faster than labelling from scratch, so run the
detector first and fix its output.

**B3. Reconcile the impact-frame naming.** Path A (`pose_features.py`,
`video_sync.py`) calls the velocity peak **P6**; path B (`p_positions.py`) calls
the hand-speed peak **P7**. P7 is correct per the standard P-system. Fix before
any cross-path evaluation, or the eval harness will compare different positions.

**B4. Fix P4.** Once ground truth exists, establish whether the 814ms P4→P7 span
is a real slow transition or a mis-detected top. The current rule takes the
global hand-speed minimum between takeaway and impact, which a hesitant
backswing will fool. Candidate replacement: hand-path direction reversal (the
frame where the hand velocity vector flips sign) rather than a speed minimum.

**B5. Club detection for P2/P6/P8.** Three positions fail outright today.
`club_detector.py` uses Canny + `HoughLinesP` and warns in its own comments that
it picks up body edges. A small trained shaft/clubhead detector would replace
it. 240fps footage has low motion blur, and there are 5,132 clips to build a
dataset from.

**B6. Cross-view fusion.** Once per-view detection is measured, take each
position from the view that resolves it best — arm tier from face-on, shaft tier
from DTL — rather than preferring face-on wholesale as the code does today.

### Accelerant — GPU pose estimation

Not a blocker (progressive enhancement means detection latency is off the
critical path), but it shortens trigger→markers and fixes the DTL disagreement
and wrist jitter that MediaPipe BlazePose causes. Keep it behind the existing
extractor interface. Compare against MediaPipe on swing 2026-07-27 135006.

Cheap wins first, no model swap needed: trim to the swing window before
extracting pose (a 2s swing at 240fps is ~480 frames, not 1,244), and decimate
to 60Hz for detection with local refinement at full rate around P4 and impact.

Scale note: 5,132 clips × ~1,200 frames ≈ 6M frames. Seconds per swing on GPU;
an overnight job for the whole library. Batch by session.

### Later

**Body checkpoints.** Turn detected positions into coaching feedback: the
per-position measurements in [p-position-detection.md](p-position-detection.md)
— weight distribution, spine flexion and lateral tilt, shoulder/hip rotation,
hand path, elbow separation, belt height. The spine-flexion and hip-centre
lateral curves are between them most of the Stack & Tilt model, and both are
single scalar series over landmarks already extracted.

---

## Smaller loose ends

- `data/annotations.json.pre-videos-move` is a backup from the video
  restructure. Delete once the migrated labels are confirmed good.
- `has_annotations()` re-reads and re-parses `data/annotations.json` once per
  video per render. With 131 clips that's 131 parses per page load, and it
  showed as a visible lag. Cache `_load_all()`.
- `README.md` and `docs/step0.md` document
  `--ground-truth data/ground_truth.json`, which does not exist (see step 1).
- Handedness is assumed right-handed (a CLI flag). It should be a stored
  per-video property alongside `_view_angle`.
