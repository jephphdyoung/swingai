# Status & Next Steps

Working state as of **2026-10-22**. Read this first when picking the work back up.

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

**Native runtime foundation** (added 2026-07-31) — `native/` is a Rust workspace holding
the domain types (`swingai-core`), the two JSON contracts between capture and analysis
(`swingai-contracts`, mirroring `schemas/`), and a `swingai-contract-check` CLI. 116 Rust
tests; `cargo fmt`/`clippy` clean; type-checks for `x86_64-pc-windows-msvc`. **No camera
code, no ring buffer, no MVS binding** — this is the seam only. Rationale and the list of
deliberately deferred decisions: [adr/0001-hybrid-rust-python-runtime.md](adr/0001-hybrid-rust-python-runtime.md).
The Python suite is now 282 tests (207 + 75 schema-example checks).

**Deterministic capture core** (added 2026-07-31) — `native/crates/swingai-capture` adds the
capture model itself: a `FrameSource` trait with a deterministic synthetic implementation,
per-camera timestamp-based ring buffers (time *and* byte limits, strict ordering, sequence-gap
detection), a multi-camera `CaptureSession` whose trigger extracts `[T - pre_roll, T]` from
each stream independently, and a shot-directory writer that emits PGM image sequences plus a
real `capture-manifest.json`. `native/apps/swingai-capture-sim` runs a two-camera scenario
end to end. 207 Rust tests. **Still no cameras, no microphone, no MVS binding, no ring-buffer
threading and no device-clock conversion** — the synthetic sources are already in the session
clock domain, which is precisely the assumption real hardware will not satisfy (A1 below).

Two things the contracts make the capture work owe: timestamps are nanoseconds since the
session's monotonic origin (persisted origin zero, unsigned), so **each device clock must
be converted with a measured offset and rate** before anything can be written — the
audio→frame mapping risk in A1 below, made unavoidable rather than assumable. And the
schema version matches exactly for now, so a contract change is a lockstep deploy of both
sides.

**Offline dual-view comparison** (PR #3, 2026-10-20) — new **Dual-View Comparison** tab:
your DTL + face-on pair vs. a reference DTL + face-on, four frames at one P-position with
skeletons, plus 2D joint-angle / body-relative displacement tables (`you − reference`, DTL
and face-on kept separate). Positions come only from saved annotations, under the
canonical P-system (P7 = impact); legacy entries are flagged until confirmed. New pose
layer with named landmarks, no carry-forward, and an on-disk cache (~16 s per 240fps
capture, once). Verified on real footage (2026-06-20 150221 vs. `GW_DTL`/`GW_faceon`).
How to run it and what each number means: [dual-view-comparison.md](dual-view-comparison.md).

**Human verification still owed** (2026-10-20). A full four-view P1/P4 comparison now runs on
real footage (150221 pair vs. `GW_DTL`/`GW_faceon`), but every label it uses needs a person:

| Video | P1 / P4 | Status |
|---|---|---|
| your 150221 DTL | 395 / 652 | yours, legacy (pre-P7-impact): confirm P6/P7 too |
| your 150221 face-on | 395 / 652 | **proposed by Claude** (shared clock from your DTL, eyeballed) |
| `GW_DTL.mp4` | 104 / 280 | **proposed by Claude** from a contact sheet (slow-mo clip) |
| `GW_faceon.mp4` | 53 / 278 | yours, legacy: confirm P6/P7 too |

Use **Confirm reviewed** in the Dual-View tab after checking each. The 2026-07-27 library has
no annotations. The two 150221 clips were deleted locally on 2026-10-22; their entries in
`data/annotations.json` are kept but point at missing files.

**PR #4: final state (2026-10-22).** Ready for final review. Not merged.

*Done and verified 2026-10-22:*
- RTMPose-m + YOLOX-m on the GPU. `verify-gpu` passes, with each network shown executing CUDA
  kernels on its own.
- Backend switching in the Dual-View tab, with MediaPipe as the default.
- A per-backend, per-environment pose cache. On one video, MediaPipe in the CPU image, MediaPipe in
  the GPU image and RTMPose each got their own entry, and each fresh extraction matched the existing
  cache exactly.
- The joint-labelling tools: blind/assisted split, Feet and Hands/Club groups, image-digest checks.
- Tests: CPU image 759 passed and 7 skipped (GPU/ORT-only). GPU image 764 passed and 2 skipped
  (CPU-image-only checks).
- The app renders all three tabs on the current local data without errors.

*Deferred: accuracy evaluation.*
- Face-on is scored (below). DTL is not: no P1 torso scale.
- No further labelling is planned in PR #4.

**The 2026-06-20 150221 user captures were deleted locally on 2026-10-22** and are no longer part of
the video library. Ten of the 20 benchmark frames and two of the seven benchmarked videos come from
them. Kept unchanged as historical record:
- `data/joint_labels/{labels.json, manifest.json, labels-old-labeller-2026-10-21.json}`
- `docs/benchmarks/pose-backends-2026-10-20.*` and `docs/benchmarks/joint-accuracy-2026-10-21.*`

They can't be reproduced now. The tools refuse rather than produce partial numbers:
- `evaluate_joint_labels.py` stops with "video not found", including for the Grant Waite frames,
  because it checks every labelled frame first.
- `make_joint_label_set.py` stops with "no longer decodes" before writing anything.
- `benchmark_pose.py` names each missing video, then runs on the rest.

`verify-gpu` and `tests/test_gpu_rtmpose.py` used one of those clips and would have skipped silently.
They now use the committed `videos/reference/GW_faceon.mp4`, and a missing file fails the test.
`scripts/make_joint_label_set.py` and `benchmark_pose.py` still list the deleted clips as defaults;
the video-library rework replaces them.

**GPU pose estimation** (PR #4, 2026-10-20). RTMPose-m (Halpe26) + YOLOX-m on ONNX Runtime
CUDA, selectable in the Dual-View tab under `SWINGAI_GPU=1 ./run.sh`. Both networks (YOLOX
detector and RTMPose) are verified separately to execute CUDA kernels on the RTX 5070 Ti
(`verify-gpu`: per-node profile, NVML kernel activity with a CPU control, CUDA/CPU parity).
That's 16 ms/frame end to end, ~3× the CPU path. Benchmarked
against MediaPipe on 7 videos ([report](benchmarks/pose-backends-2026-10-20.md)): MediaPipe
swaps left/right labels (16 events) where RTMPose doesn't, and the two disagree by tens of
degrees on some P4 measurements. **Accuracy against human labels is measured for face-on only**
(below); DTL is not scored yet. MediaPipe stays the default.

A first full pass of labels (2026-10-21) is kept as
`data/joint_labels/labels-old-labeller-2026-10-21.json`. It was exported from a labeller
still open from before the digest change, so the evaluator refuses it (no `image_sha1`).
Checked by decoding: 16 of its 20 frames were labelled on the right image. DTL 730/1238 and
face-on 736/774 were labelled on the frame before. **Decided:** the rebuilt labeller offers
the 16 for checking (hollow rings: keep or move each), and the 4 are relabelled from
scratch. Nothing is certified without that check.

**Assisted labelling (2026-10-21).** 12 of the 20 benchmark frames are fixed as **blind**:
each video's P1 plus two chosen by image hash, before any suggestion existed. They show no
model output and are the fair comparison. The other 8 get suggestions from the cached poses
of both models. Disagreements over 0.1 torso lengths are flagged with no first choice, and
the first-choice model alternates. Assisted frames are scored and reported separately, never
pooled. Accepted model points need the labeller's sign-off.

**First joint-accuracy results (2026-10-21)** ([report](benchmarks/joint-accuracy-2026-10-21.md),
labels in `data/joint_labels/labels.json`, all 20 frames, 238/300 joints):
- **Face-on, 6 blind frames (76 joints, both golfers):** RTMPose is closer to the labels.
  Within 0.1 torso lengths: RTMPose 69.7% vs MediaPipe 44.7%, a difference of +25.0 points
  (95% interval +15.9 to +34.3). Mean error 0.074 vs 0.166 torso lengths. That's good evidence,
  but from 6 frames of 2 golfers.
- **DTL: not scored.** Errors are scaled by the P1 torso length, which needs both shoulders and both hips.
  On both DTL P1 frames (Jeff 395, GW 104) the far-side shoulder and hip are hidden and were marked
  can't-place. Two ways to score DTL: place them as occluded estimates, or change the scale to fall back
  to the near-side shoulder→hip. Neither is decided. A scratch preview with the near-side fallback gave
  RTMPose 71.9% vs MediaPipe 63.2% (+8.8, interval −7.4 to +24.6: no clear difference). It is not in the
  report.
- **The 8 assisted frames were labelled without suggestions.** Suggestions unlock only when every blind
  frame's Body group is complete, which needs hidden joints marked occluded or can't-place, and most
  blind frames have blank hidden joints. No label came from a model. The 8 frames are still reported as
  assisted, because the export can't prove that nothing was shown.
- Hidden joints were left blank or marked can't-place, never estimated, so the occluded-joint numbers are
  empty: every scored joint is one the labeller could see.

The labeller now also has optional **Feet** and **Hands/Club** groups, and the manifest can
hold **sequences** for P2–P3, P5–P8 and heel-lift work, all outside the benchmark score
([joint labelling](joint-labelling.md)). Not done yet:
- RTMPose heel/small toe and MediaPipe heel aren't kept by the pose layer, so foot points
  can't be scored against models.
- Sequence scoring.

**Known issue (pre-existing, not fixed in PR #4): OpenCV seeking lands one frame early on the
240fps captures** from about frame 700 on (the stream's average frame rate, 249.30, differs
from its nominal one). Pose extraction decodes sequentially, so wherever the UI *seeks*
(`utils/dual_view.read_frame`, the annotator), the image shown is frame *f*−1 while the
skeleton, measurements and saved P-position index are frame *f*. That's 4 ms, but the hands
move far in 4 ms near impact. Measured on the 150221 pair (since deleted locally): frames 730/736/774/1238 are off by
one; 395–652 are exact; Grant Waite's 30fps clips are exact. The joint-label set was fixed for
this (sequential decode plus per-frame pixel digests). The annotator and panels still need it.

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
podman run --rm --security-opt label=disable -v "$PWD":/app swingai python -m pytest tests/ -q

# detect P-positions for one swing (partner view found automatically)
podman run --rm --security-opt label=disable -v "$PWD":/app -v "<session>":/swing:ro swingai \
  python scripts/detect_p_positions.py --swing "/swing/<clip>.mp4" \
  --out-dir /app/videos/output

# whole session
... --session /swing --out-dir /app/videos/output

# the app
./run.sh          # http://localhost:8501
```

`run.sh` mounts user/reference footage **read-only** with no relabelling and never
chmods or chcons (see dual-view-comparison.md). Never add `:z`/`:Z` to a footage mount.

**Files modified by the old launcher (found 2026-10-20, not yet restored):** the clips under
`~/Golf/Videos/2026-07-27/SwingCatalyst/` are `-rwxrwxrwx` and labelled
`container_file_t:s0:c262,c689` (the old `run.sh` ran `chmod -R a+rw` + `chcon -Rt`). Their
contents and mtimes are intact. To restore, after checking the intended modes:

```bash
restorecon -Rv ~/Golf/Videos                         # SELinux labels back to policy default
find ~/Golf/Videos -type f -exec chmod 644 {} +      # if 644 was the original mode
find ~/Golf/Videos -type d -exec chmod 755 {} +
```

The repo's own `videos/` and `data/` also carry container labels from `:Z` test runs;
`restorecon -Rv videos data` resets those.

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
