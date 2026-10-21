# Joint labelling

Human-placed joint labels are the only ground truth for pose accuracy. This page covers what
can be labelled, which models can be compared on each point, how earlier work is carried
over, and how sequences fit in. Scoring is described in the
[pose benchmark report](benchmarks/pose-backends-2026-10-20.md#ground-truth-pending-needs-a-person).

Code: `utils/label_points.py` (the catalogue), `utils/joint_labels.py` (validation and
scoring), `scripts/make_joint_label_set.py` (the set and the labeller),
`scripts/joint_labeler.html` (the labeller page), `scripts/evaluate_joint_labels.py`.

## What the benchmark scores (unchanged)

The MediaPipe vs RTMPose benchmark scores **the 15 body joints on the 20 benchmark frames,
and nothing else**. Those are `COMMON_JOINTS`, in the manifest's `joints` list. The frames
and their image SHA-1s are pinned by `test_benchmark_frames_and_images_are_pinned`. Feet,
hands/club points and sequence frames are validated, but they never enter that score.
`test_benchmark_results_are_unchanged_by_extra_points_and_frames` checks this.

## Groups and points

The labeller has three tabs. A frame can have Body only. Feet and Hands/Club are optional
on any frame. Every point is placed, marked *occluded* (the position is your estimate), or
marked *can't place*.

**Kinds.** *Anatomical* points are body landmarks a pose model may predict. Everything else
is a manual label that no model predicts:

| Kind | Meaning |
|---|---|
| contact | where the shoe meets, or would meet, the floor |
| reference | a fixed scene point (the floor line) |
| marker | something worn: a glove logo, tape |
| equipment | a point on the club |

**Model support.** *Same* means the model's keypoint has the label's definition.
*Approximate* means the model has a similar point with a looser or undocumented
definition. *Not kept* means the model outputs the point, but SwingAI's pose layer
(`utils/landmarks.py`) drops it today, so it can't be scored until that layer is extended.

### Body (benchmark)

| Point | Kind | MediaPipe | RTMPose (Halpe26) |
|---|---|---|---|
| nose, left/right ear, shoulder, elbow, wrist, hip, knee, ankle | anatomical | same | same |

Hips differ in definition between the two models (COCO-family hips sit on the joint), so
compare hip-based values within one model.

### Feet

| Point | Kind | MediaPipe | RTMPose (Halpe26) |
|---|---|---|---|
| left/right_heel | anatomical | heel 29/30, approximate, not kept | heel 24/25, approximate, not kept |
| left/right_big_toe | anatomical | **none** | big_toe 20/21, same |
| left/right_little_toe | anatomical | none | small_toe 22/23, same, not kept |
| left/right_shoe_heel_contact | contact | none | none |
| left/right_shoe_toe_contact | contact | none | none |
| floor_a, floor_b | reference | none | none |

**MediaPipe's `foot_index` is not the big toe.** It marks the front of the foot. It is not a
label point and is never scored against one. RTMPose's big toe sits in the `foot_index`
column of a `PoseSequence` only so the skeleton draws a foot.

These points are for future **heel clearance** (the shoe-heel contact point against the
floor line, or against its own address position), **heel-lift timing** (the first frame of a
sequence where the heel clears) and **foot pivot** (the heel–toe line's angle in the image).
All of them are 2D image measurements. They measure no ground pressure or weight shift, and
no 3D foot rotation: one camera's 2D foot line confounds rotation with viewing angle.

### Hands / Club

| Point | Kind | MediaPipe | RTMPose (Halpe26) |
|---|---|---|---|
| left/right_thumb, index, pinky | anatomical | Pose hand points 17–22, approximate | **none** |
| left/right_hand_back | marker | none | none |
| left/right_forearm_marker | marker | none | none |
| shaft_butt, shaft_hosel | equipment | none | none |
| clubhead_heel, clubhead_toe, clubface_center | equipment | none | none |

RTMPose's Body26/Halpe26 model has **no finger points**, and none are estimated for it.
MediaPipe Pose's thumb, index and pinky are coarse points from the body model, not the
21-point hand model, and the model doesn't define them anatomically. The labels are thumb
tip and the index and little-finger base knuckles, so the match is only approximate.
`utils/club_detector.py` fits a shaft *line* by edge detection. It is not a keypoint model,
and it is unvalidated.

**Forearm rotation.** True pronation/supination cannot be inferred reliably from 2D elbow
and wrist points. Rotation about the forearm's long axis barely moves either point. It shows
only in something off that axis: the back of the hand, the clubface, or a worn marker. Even
then, one 2D view mixes rotation with arm and camera angle. The hand-back and forearm
markers are there to *collect* that evidence, not because rotation can be read from them
yet.

## The labeller

```bash
podman run --rm --security-opt label=disable -v "$PWD":/app swingai \
  python scripts/make_joint_label_set.py \
  --review-prior data/joint_labels/labels-old-labeller-2026-10-21.json
xdg-open data/joint_labels/labeler.html
```

Rebuilding keeps the manifest's frames. Each one is decoded again in order and must match
its recorded SHA-1, or the build stops. For each frame, the labeller shows:
- the full video path
- the frame index as decoded in order from 0 (the same frames the pose models read)
- the image's digest prefix
- the approximate P-position from saved annotations. It is flagged when a person hasn't
  confirmed the annotation; the annotator seeks, so it can be one frame off past ~frame 700
  on the 240fps captures.
- the set (benchmark or a sequence) and why the frame was chosen

Each frame shows its progress per group. The header shows overall progress: Body against
the 20 benchmark frames, and Feet and Hands/Club as complete or started.

Progress is saved in the browser **per image digest**. Adding frames never resets it, and a
label can't move to a different image. The export (schema 2) contains:
- per video, `image_sha1` for every exported frame
- per frame, `frame_status`: which groups are complete, which points couldn't be placed,
  and any carried-over points kept unchanged

## Earlier labels (2026-10-21)

`data/joint_labels/labels-old-labeller-2026-10-21.json` is kept byte-for-byte; a test pins
its SHA-1. It came from the pre-digest labeller, which loaded images by seeking.
`--review-prior` re-creates each labelled image the same way and compares it with the
current one:

- **16 frames were labelled on the identical image.** *Load 16 earlier frames for checking*
  puts their points on those frames as hollow rings. They are **not labels yet**: click to
  move one, press Enter to keep one, or use *Keep all* after checking every ring. Only kept
  or moved points are exported. Kept points are listed in `frame_status.review` with who
  confirmed them, and the validator refuses a review entry without `confirmed_by`. Points
  still unchecked are left out of the export. The earlier frames have 8 to 15 of the 15
  body joints, so the rest still need placing (or marking can't-place).
- **4 frames were labelled on frame N−1** (DTL 730 and 1238, face-on 736 and 774). They are
  not offered. The labeller says why on those frames, and they are labelled from scratch.

## Sequences

The benchmark compares single frames. Swing analysis needs runs of frames: P2–P3, P5–P8 and
heel-lift onset. Add one as a separate, non-benchmark set:

```bash
python scripts/make_joint_label_set.py --add-sequence p5-p8 \
  "videos/user/Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4" 700 760 \
  --step 2 --purpose "P5-P8 lead arm and trail heel"
```

Sequence frames are decoded in order and get their own SHA-1. They are recorded with
`"set": "sequence"` and `"sequence": {name, start, end, step, purpose}`. The benchmark
ignores them. They appear in the labeller under their own heading, and existing progress is
kept. Each sequence video needs its P1 frame labelled (body torso) for errors to be
normalised; for the four benchmark videos that's already a benchmark frame. Scoring sequences
(temporal error, heel-lift onset error) is future work. The schema needs no change for it.
