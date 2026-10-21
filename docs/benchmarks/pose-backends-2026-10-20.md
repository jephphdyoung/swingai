# Pose backends: MediaPipe vs RTMPose (2026-10-20)

**Bottom line:** RTMPose runs on the RTX 5070 Ti and is fully integrated, but **this report
does not show it to be more accurate, because no human-verified joint labels exist yet.**
The default stays MediaPipe. What the data does show:

- **The choice of model changes the numbers you see**, sometimes by tens of degrees at P4
  (table below). The measurement engine is the same; only the landmarks differ. So model
  accuracy matters, and it has to be measured, not assumed.
- **MediaPipe swaps left and right labels; RTMPose didn't.** There were 12 swap events on
  `GW_DTL` and 4 on `GW_2009_faceon` for MediaPipe, and 0 on all seven videos for RTMPose.
  One was confirmed visually: at `GW_DTL` frame 486→487 MediaPipe's left shoulder and hip
  jump to the right side of the body. Mislabelled lead/trail joints corrupt every lead/trail
  measurement.
- **On the occluded DTL lead side, MediaPipe abstains and RTMPose always answers.** MediaPipe
  marks your lead elbow/wrist/knee valid in only 34–67% of DTL frames; RTMPose scores them
  above its threshold 100% of the time. That is more coverage only if those points are
  right, which is what the labelled frames include.
- **At 240fps RTMPose jitters more** (no temporal model): 99th-percentile frame-to-frame
  deviation is 0.018–0.060 torso lengths vs MediaPipe's ~0.007. At 30fps (Grant Waite) it's
  the other way round.
- **It's not faster end to end.** It runs at 60–74 fps against MediaPipe's 75–87 fps on the same
  frames, even though GPU inference itself is ~15 ms/frame. Decode/convert on the CPU and a
  detector on every frame dominate. Cached results load in 1–4 ms for either model.

Reproduce:

```bash
podman --cdi-spec-dir ~/.local/share/swingai/cdi run --rm --device nvidia.com/gpu=all \
  --security-opt label=disable -v "$PWD":/app \
  -v "$PWD/videos/user":/app/videos/user:ro -v "$PWD/videos/reference":/app/videos/reference:ro \
  -v "$PWD/models":/app/models:ro -v ~/Golf/Videos/2026-07-27/SwingCatalyst:/app/videos/extra:ro \
  swingai-gpu python scripts/benchmark_pose.py --out docs/benchmarks/pose-backends-2026-10-20.json \
  "videos/extra/2026-07-27 135033/Jeff Young - 2026-07-27 135006 Fox Down the line 61fc.mp4" \
  "videos/extra/2026-07-27 135033/Jeff Young - 2026-07-27 135006 Fox Face on right aaf0.mp4" \
  --view dtl --view face_on
python3 scripts/benchmark_report.py docs/benchmarks/pose-backends-2026-10-20.json
```

Raw results: [pose-backends-2026-10-20.json](pose-backends-2026-10-20.json). Videos: your
2026-06-20 150221 DTL/face-on pair; `GW_DTL`, `GW_faceon`; plus `GW_2009_faceon` (old,
low-resolution footage with bystanders) and your 2026-07-27 135006 pair, which STATUS.md
flags as hard for DTL tracking. Every frame of every video was processed by both models
(7 videos, 7,243 frames each).

## Ground truth (pending: needs a person)

Accuracy has to be measured against joints a person has placed and verified. Claude can't
supply that, and the Claude-proposed P-position labels are not used as truth for anything.
The labelling set is ready:

- **20 frames**, 5 from each of the four main videos: P1/P4/P7 where annotated (the
  annotations only *choose* frames), the two frames where the models disagree most, and the
  fastest hand-motion frame. The list is in [`data/joint_labels/manifest.json`](../../data/joint_labels/manifest.json).
- **15 joints per frame** (both models' common set: nose, ears, shoulders, elbows, wrists,
  hips, knees, ankles), each marked visible or occluded (position estimated), or skipped.

```bash
xdg-open data/joint_labels/labeler.html       # label, enter your name, Export
mv ~/Downloads/labels.json data/joint_labels/labels.json
podman --cdi-spec-dir ~/.local/share/swingai/cdi run --rm --device nvidia.com/gpu=all \
  --security-opt label=disable -v "$PWD":/app \
  -v "$PWD/videos/user":/app/videos/user:ro -v "$PWD/videos/reference":/app/videos/reference:ro \
  -v "$PWD/models":/app/models:ro swingai-gpu \
  python scripts/evaluate_joint_labels.py --out docs/benchmarks/joint-accuracy-<date>.json \
  --md docs/benchmarks/joint-accuracy-<date>.md
```

The labeller has no pre-placed points, since they would anchor you toward one model. It
stores coordinates in original-video pixels. `evaluate_joint_labels.py` then:

1. **Validates** `labels.json` against the manifest and refuses it, listing every problem,
   if it has no `verified_by`, has frames or joints outside the set, has coordinates off the
   frame or non-boolean visibility, or was labelled on a different image than the manifest's.
   Nothing is repaired or filled in.
2. **Checks the pixels.** Each labelled frame is decoded *sequentially*, as pose extraction
   reads it, and its SHA-1 must match the digest the labeller exported. This matters: on the
   240fps captures OpenCV's seek lands one frame early past ~frame 700, and the first
   labeller (2026-10-20) used seeking, so frames 03, 04, 08 and 09 showed frame *f*−1. The
   set was regenerated on 2026-10-21 with sequential decoding. The frames are the same and
   only those four images changed. Its progress key now depends on the images, so it starts
   clean, and exports from the old labeller are refused.
3. **Scores both models, separately for DTL and face-on**: error in P1 torso lengths (from
   your P1 labels; mean and median), frame misses (no person found) and joint misses (no
   point above the model's own threshold), PCK@0.05/0.1/0.2 with a miss counted as a
   failure, all split into visible vs occluded labels, per joint, plus how often each model
   answered on joints you marked occluded. RTMPose − MediaPipe PCK comes with a 95%
   bootstrap interval over labelled frames. With 10 frames per view, an interval containing
   0 is no evidence of a difference.

It reads both models' poses from the pose cache (run in the GPU image, where this report's
caches were made; `--extract` computes missing ones). The benchmark script uses the same
validation and scoring for its per-video accuracy table.

## How much the model changes the measurements

Same P1/P4 frames, same geometry engine; only the landmarks differ:

| P | View | Measurement | You: MediaPipe | You: RTMPose | Ref: MediaPipe | Ref: RTMPose |
|---|---|---|---|---|---|---|
| P1 | dtl | Lead elbow flexion | — | 13.6 | — | 1.1 |
| P1 | dtl | Trail elbow flexion | 25.2 | 17.7 | 29.7 | 15.4 |
| P1 | dtl | Lead knee flexion | — | 25.2 | — | 28.5 |
| P1 | dtl | Trail knee flexion | 26.4 | 32.3 | 15.3 | 15.8 |
| P1 | dtl | Spine inclination from vertical | 34.3 | 33.6 | 34.0 | 31.9 |
| P1 | dtl | Shoulder tilt | — | — | -39.4 | — |
| P1 | dtl | Lead arm above horizontal | — | -87.1 | — | -88.6 |
| P1 | face_on | Lead elbow flexion | 13.1 | 13.2 | 10.0 | 7.7 |
| P1 | face_on | Trail elbow flexion | 10.8 | 7.0 | 16.3 | 10.3 |
| P1 | face_on | Lead knee flexion | 13.6 | 14.4 | 19.1 | 11.2 |
| P1 | face_on | Trail knee flexion | 15.0 | 9.9 | 11.3 | 6.9 |
| P1 | face_on | Spine inclination from vertical | -0.1 | -0.3 | -0.6 | -0.6 |
| P1 | face_on | Shoulder tilt | 3.2 | 8.7 | 3.5 | 11.0 |
| P1 | face_on | Lead arm above horizontal | -81.3 | -80.0 | -81.0 | -82.2 |
| P4 | dtl | Lead elbow flexion | 42.8 | 29.5 | 32.9 | 19.7 |
| P4 | dtl | Trail elbow flexion | 109.8 | 127.1 | 102.4 | 135.5 |
| P4 | dtl | Lead knee flexion | 42.9 | 50.3 | 50.9 | 51.6 |
| P4 | dtl | Trail knee flexion | 1.9 | 7.2 | 39.5 | 10.8 |
| P4 | dtl | Spine inclination from vertical | 33.3 | 31.6 | 30.6 | 33.4 |
| P4 | dtl | Shoulder tilt | -30.4 | -37.6 | -31.3 | -36.0 |
| P4 | dtl | Lead arm above horizontal | 34.3 | 40.2 | 42.0 | 36.4 |
| P4 | face_on | Lead elbow flexion | 82.0 | 76.7 | 63.3 | 36.5 |
| P4 | face_on | Trail elbow flexion | 98.1 | 83.5 | 80.6 | 40.7 |
| P4 | face_on | Lead knee flexion | 13.6 | 12.9 | 5.0 | 5.4 |
| P4 | face_on | Trail knee flexion | 1.5 | 3.8 | 8.9 | 4.6 |
| P4 | face_on | Spine inclination from vertical | -2.0 | 5.3 | -4.4 | -0.9 |
| P4 | face_on | Shoulder tilt | -3.0 | -59.3 | 8.7 | -45.8 |
| P4 | face_on | Lead arm above horizontal | 16.4 | 40.0 | 18.7 | 34.3 |

`—` = unavailable (MediaPipe's occluded DTL lead arm at address; shoulders overlapping in
DTL projection). The P4 face-on shoulder tilt is the extreme case: at the top the shoulders
are turned ~90°, so their horizontal separation is small and the angle is ill-conditioned.
At your frame 652 MediaPipe places them almost level (confidence 1.0), while RTMPose puts
the lead shoulder clearly lower (confidence 0.51/0.57). These are exactly the frames the
labelling set includes.

### Accuracy against human-verified joints

**Not computed: no human-verified joint labels exist yet.** See *Ground truth* above.

### Performance

Model initialisation: MediaPipe 0.17 s, RTMPose 4.16 s (session creation + warm-up).

| Video | Frames | Model | Process (s) | fps | GPU peak (MiB) | Cache load (ms) |
|---|---|---|---|---|---|---|
| 2026-06-20 150221 Fox Down the line 61fc | 1244 | MediaPipe | 16.6 | 74.8 | — | 4 |
| 2026-06-20 150221 Fox Down the line 61fc | 1244 | RTMPose | 20.8 | 59.7 | 904 | 2 |
| 2026-06-20 150221 Fox Face on right aaf0 | 1244 | MediaPipe | 15.6 | 79.6 | — | 3 |
| 2026-06-20 150221 Fox Face on right aaf0 | 1244 | RTMPose | 20.8 | 59.9 | 908 | 2 |
| GW_DTL | 540 | MediaPipe | 6.4 | 84.8 | — | 2 |
| GW_DTL | 540 | RTMPose | 7.7 | 70.4 | 902 | 1 |
| GW_faceon | 571 | MediaPipe | 6.6 | 86.6 | — | 2 |
| GW_faceon | 571 | RTMPose | 7.8 | 72.7 | 902 | 1 |
| GW_2009_faceon | 1156 | MediaPipe | 13.5 | 85.7 | — | 3 |
| GW_2009_faceon | 1156 | RTMPose | 15.6 | 74.2 | 904 | 2 |
| 2026-07-27 135006 Fox Down the line 61fc | 1244 | MediaPipe | 15.6 | 79.8 | — | 3 |
| 2026-07-27 135006 Fox Down the line 61fc | 1244 | RTMPose | 20.7 | 60.0 | 902 | 2 |
| 2026-07-27 135006 Fox Face on right aaf0 | 1244 | MediaPipe | 15.3 | 81.3 | — | 3 |
| 2026-07-27 135006 Fox Face on right aaf0 | 1244 | RTMPose | 20.9 | 59.4 | 902 | 2 |

MediaPipe runs on the CPU, so it has no GPU figure. RTMPose's figure is the whole process (CUDA context + both sessions).

### Behaviour without ground truth

| Video | View | Model | Detected | Valid (mean) | DTL lead elbow/wrist/knee valid | Spike rate | Spike p99 | L/R swaps |
|---|---|---|---|---|---|---|---|---|
| 2026-06-20 150221 Fox Down the line 61fc | dtl | MediaPipe | 100.0% | 85.7% | 41% / 38% / 54% | 0.00% | 0.007 | 0 |
| 2026-06-20 150221 Fox Down the line 61fc | dtl | RTMPose | 100.0% | 100.0% | 100% / 100% / 100% | 0.01% | 0.018 | 0 |
| 2026-06-20 150221 Fox Face on right aaf0 | face_on | MediaPipe | 100.0% | 96.5% | — | 0.00% | 0.008 | 1 |
| 2026-06-20 150221 Fox Face on right aaf0 | face_on | RTMPose | 100.0% | 100.0% | — | 0.01% | 0.021 | 0 |
| GW_DTL | dtl | MediaPipe | 100.0% | 90.2% | 67% / 62% / 72% | 0.01% | 0.050 | 12 |
| GW_DTL | dtl | RTMPose | 100.0% | 100.0% | 100% / 100% / 100% | 0.00% | 0.020 | 0 |
| GW_faceon | face_on | MediaPipe | 100.0% | 98.0% | — | 0.05% | 0.047 | 0 |
| GW_faceon | face_on | RTMPose | 100.0% | 100.0% | — | 0.00% | 0.033 | 0 |
| GW_2009_faceon | face_on | MediaPipe | 99.7% | 96.5% | — | 0.48% | 0.076 | 4 |
| GW_2009_faceon | face_on | RTMPose | 100.0% | 100.0% | — | 0.05% | 0.033 | 0 |
| 2026-07-27 135006 Fox Down the line 61fc | dtl | MediaPipe | 100.0% | 82.9% | 51% / 34% / 59% | 0.00% | 0.007 | 0 |
| 2026-07-27 135006 Fox Down the line 61fc | dtl | RTMPose | 100.0% | 100.0% | 100% / 100% / 100% | 0.04% | 0.060 | 0 |
| 2026-07-27 135006 Fox Face on right aaf0 | face_on | MediaPipe | 100.0% | 95.8% | — | 0.00% | 0.006 | 0 |
| 2026-07-27 135006 Fox Face on right aaf0 | face_on | RTMPose | 100.0% | 100.0% | — | 0.09% | 0.060 | 0 |

Valid = landmark above the backend's own threshold (MediaPipe 0.5, RTMPose 0.3); the two are not comparable as confidence. Spike = a point more than 0.2 torso lengths off the midpoint of its neighbouring frames; p99 is the median over joints of each joint's 99th-percentile deviation. L/R swap = relabelling the left/right pair explains the frame-to-frame motion at least twice as well.

### Cross-model agreement (not accuracy)

| Video | View | nose | left shoulder | left elbow | left wrist | right wrist | left hip | left knee | left ankle |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-20 150221 Fox Down the line 61fc | dtl | 0.059 | 0.061 | 0.055 | 0.057 | 0.063 | 0.056 | 0.072 | 0.057 |
| 2026-06-20 150221 Fox Face on right aaf0 | face_on | 0.033 | 0.067 | 0.045 | 0.051 | 0.055 | 0.049 | 0.034 | 0.047 |
| GW_DTL | dtl | 0.051 | 0.065 | 0.097 | 0.103 | 0.074 | 0.098 | 0.079 | 0.136 |
| GW_faceon | face_on | 0.084 | 0.193 | 0.139 | 0.097 | 0.069 | 0.052 | 0.043 | 0.033 |
| GW_2009_faceon | face_on | 0.053 | 0.088 | 0.066 | 0.089 | 0.095 | 0.089 | 0.126 | 0.035 |
| 2026-07-27 135006 Fox Down the line 61fc | dtl | 0.080 | 0.060 | 0.045 | 0.086 | 0.042 | 0.064 | 0.037 | 0.053 |
| 2026-07-27 135006 Fox Face on right aaf0 | face_on | 0.069 | 0.152 | 0.023 | 0.050 | 0.054 | 0.048 | 0.052 | 0.037 |

Median distance between the two models' points, in torso lengths, on frames where both are valid.


## Notes on the method

- Both models ran on identical decoded frames, in the GPU image with the CPU image's exact
  package versions. MediaPipe runs on the CPU there too. Each model's **own** confidence
  threshold is used (0.5 / 0.3); they are not comparable as confidence.
- Spike and swap detectors are heuristics. The swap detector was spot-checked on a real
  event (above); a "0 swaps" result means no clear swap, not proven consistency.
- Cross-model agreement is not accuracy: two models can agree and both be wrong, or differ
  by a constant offset because their keypoint definitions differ (e.g. hips).
- MediaPipe output depends on the runtime environment: same version and model, CPU image vs
  GPU image, median 0.3 px but up to 28 px apart (tracking divergence). Cache keys include an
  environment fingerprint for this reason; these MediaPipe numbers are from the GPU image.
