"""Human-verified joint labels: loading, validation and scoring.

Labels are placed by a person in `data/joint_labels/labeler.html` and exported
as `labels.json`. Nothing here creates, fills in or repairs a label: a file
that fails validation is refused with every problem listed.

Scoring is per model and separately per camera view (DTL, face-on):

- **normalized error**: distance from the model's point to the labelled
  point, in units of that video's P1 torso length taken from the labels
  (shoulder centre to hip centre).
- **detection misses**: labelled frames where the model found no person
  (frame misses), and labelled joints with no valid point under the model's
  own threshold (joint misses; these include every joint of a missed frame).
- **PCK@t**: fraction of labelled joints within t torso lengths. A miss counts
  as a failure, so a model can't raise PCK by abstaining.

Each is also split by the labeller's visible/occluded flag. On occluded
labels `answered` is how often the model gave a valid point anyway -- the
case where a confident score can hide a guessed position.

Schema 2 (the current labeller) adds, without changing what is scored:

- feet and hands/club points (`utils/label_points.py`), validated but never
  part of the benchmark;
- frames outside the benchmark set (targeted sequences, e.g. P5-P8). The
  benchmark scores only manifest frames whose set is "benchmark";
- `frame_status` per frame: which groups the labeller finished, which points
  they could not place, and, for points carried over from an earlier export,
  who confirmed them on the current image. Unconfirmed carried-over points
  are refused.

Schema-1 exports stay valid: body joints only, benchmark frames only.
"""

import hashlib
import json
import math

import numpy as np

from utils import label_points as lp
from utils import pose_metrics as pm

SCHEMA = 2
SCHEMAS = (1, 2)
BENCHMARK = "benchmark"
VIEWS = ("dtl", "face_on")
PCK_THRESHOLDS = (0.05, 0.1, 0.2)
TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


class LabelError(ValueError):
    """labels.json is missing, malformed, unverified or doesn't match the manifest."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def frame_digest(bgr):
    """SHA-1 of a decoded frame's pixels: proves which image was labelled."""
    return hashlib.sha1(np.ascontiguousarray(bgr).tobytes()).hexdigest()


def decode_frames(video, wanted):
    """{frame: BGR} for `wanted` frame indexes, decoded sequentially.

    Never by seeking: on the 240fps captures OpenCV's seek lands one frame
    early past ~frame 700 (the stream's average and nominal frame rates
    differ), while pose extraction decodes sequentially.
    """
    import cv2
    wanted, out = set(wanted), {}
    cap = cv2.VideoCapture(video)
    try:
        idx = 0
        while wanted - set(out):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out


def load_manifest(path):
    with open(path) as f:
        return load_manifest_dict(json.load(f))


def load_manifest_dict(m):
    """joints: the benchmark joints; points: every labellable point name."""
    cat = m.get("catalogue")
    points = [p["name"] for p in cat["points"]] if cat else list(m["joints"])
    frames = {}
    for f in m["frames"]:
        frames[(f["video"], int(f["frame"]))] = {"set": BENCHMARK, **f}
    return {"joints": list(m["joints"]), "points": points, "frames": frames}


def frame_set(manifest, video, frame):
    return manifest["frames"][(video, frame)]["set"]


def validate(data, manifest):
    """Labels as {video: {"view", "p1_frame", "verified_by", "frames": {int: {joint:
    {"x", "y", "visible"}}}}}. Raises LabelError listing every problem."""
    problems = []
    if not isinstance(data, dict):
        raise LabelError(["labels file is not a JSON object"])
    schema = data.get("schema")
    if schema not in SCHEMAS:
        problems.append(f"schema is {schema!r}, expected one of {SCHEMAS}")
    if not str(data.get("verified_by") or "").strip():
        problems.append("no verified_by: labels must name the person who placed them")
    videos = data.get("videos")
    if not isinstance(videos, dict) or not videos:
        problems.append("no labelled videos")
        raise LabelError(problems)

    # Schema 1 predates the feet/hands groups and sequence frames.
    joints_allowed = set(manifest["points"] if schema == 2 else manifest["joints"])
    by_video = {}
    for (video, frame), item in manifest["frames"].items():
        by_video.setdefault(video, {})[frame] = item

    out = {}
    for video, v in videos.items():
        where = f"{video}"
        if video not in by_video:
            problems.append(f"{where}: not in the labelling manifest")
            continue
        items = by_video[video]
        some = next(iter(items.values()))
        if not str(v.get("verified_by") or "").strip():
            problems.append(f"{where}: no verified_by")
        if v.get("view") != some["view"]:
            problems.append(f"{where}: view {v.get('view')!r}, manifest says {some['view']!r}")
        if v.get("p1_frame") != some["p1_frame"]:
            problems.append(f"{where}: p1_frame {v.get('p1_frame')!r}, "
                            f"manifest says {some['p1_frame']!r}")
        digests = v.get("image_sha1") or {}
        frames = {}
        for fkey, joints in (v.get("frames") or {}).items():
            try:
                frame = int(fkey)
            except (TypeError, ValueError):
                problems.append(f"{where}: frame key {fkey!r} is not an integer")
                continue
            if frame not in items:
                problems.append(f"{where} frame {frame}: not a frame of the labelling set")
                continue
            if schema != 2 and items[frame]["set"] != BENCHMARK:
                problems.append(f"{where} frame {frame}: a schema-1 export can't label "
                                "frames outside the benchmark set")
                continue
            expected = items[frame].get("image_sha1")
            if not expected:
                problems.append(f"{where} frame {frame}: the manifest has no image digest; "
                                "regenerate it with scripts/make_joint_label_set.py")
                continue
            if digests.get(str(frame)) != expected:
                problems.append(
                    f"{where} frame {frame}: labelled on a different image than the "
                    "manifest's (exported from an older labeller?); relabel this frame in "
                    "the current labeler.html")
                continue
            w, h = items[frame]["width"], items[frame]["height"]
            clean = {}
            for name, lab in (joints or {}).items():
                at = f"{where} frame {frame} {name}"
                if name not in joints_allowed:
                    problems.append(f"{at}: not a labelled joint")
                    continue
                x, y, vis = lab.get("x"), lab.get("y"), lab.get("visible")
                if not all(isinstance(c, (int, float)) and not isinstance(c, bool)
                           and math.isfinite(c) for c in (x, y)):
                    problems.append(f"{at}: x/y are not finite numbers")
                    continue
                if not (0 <= x <= w and 0 <= y <= h):
                    problems.append(f"{at}: ({x}, {y}) is outside the {w}x{h} frame")
                    continue
                if not isinstance(vis, bool):
                    problems.append(f"{at}: visible must be true or false")
                    continue
                clean[name] = {"x": float(x), "y": float(y), "visible": vis}
            if clean:
                frames[frame] = clean
        status = _frame_status(v.get("frame_status"), frames, items, joints_allowed, where,
                               problems) if schema == 2 else {}
        out[video] = {"view": some["view"], "p1_frame": some["p1_frame"],
                      "verified_by": str(v.get("verified_by") or "").strip(), "frames": frames,
                      "sets": {f: items[f]["set"] for f in frames}, "frame_status": status}
    if problems:
        raise LabelError(problems)
    return out


def _frame_status(raw, frames, items, points, where, problems):
    """Per-frame groups finished, points that couldn't be placed, and provenance."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        problems.append(f"{where}: frame_status is not an object")
        return {}
    out = {}
    for fkey, st in raw.items():
        try:
            frame = int(fkey)
        except (TypeError, ValueError):
            problems.append(f"{where}: frame_status key {fkey!r} is not an integer")
            continue
        at = f"{where} frame {frame}"
        if frame not in items:
            problems.append(f"{at}: frame_status for a frame outside the labelling set")
            continue
        if not isinstance(st, dict):
            problems.append(f"{at}: frame_status entry is not an object")
            continue
        groups = st.get("groups_complete", [])
        if not isinstance(groups, list) or not set(groups) <= set(lp.GROUPS):
            problems.append(f"{at}: groups_complete must be a list of {lp.GROUPS}")
            groups = []
        cannot = st.get("cannot_place", [])
        if not isinstance(cannot, list) or not set(cannot) <= points:
            problems.append(f"{at}: cannot_place must list known points")
            cannot = []
        placed = set(frames.get(frame, {}))
        if placed & set(cannot):
            problems.append(f"{at}: {sorted(placed & set(cannot))} both placed and "
                            "marked as impossible to place")
        clean = {"groups_complete": list(groups), "cannot_place": list(cannot)}
        review = st.get("review")
        if review is not None:
            ok = (isinstance(review, dict) and str(review.get("source") or "").strip()
                  and str(review.get("confirmed_by") or "").strip()
                  and str(review.get("confirmed_at") or "").strip()
                  and isinstance(review.get("points_unchanged", []), list))
            if not ok:
                problems.append(f"{at}: points carried over from an earlier export were not "
                                "confirmed on this image (review needs source, confirmed_by, "
                                "confirmed_at)")
            else:
                clean["review"] = {k: review[k] for k in
                                   ("source", "confirmed_by", "confirmed_at")}
                clean["review"]["points_unchanged"] = list(review.get("points_unchanged", []))
        out[frame] = clean
    return out


def benchmark_labels(labels):
    """Only the benchmark frames: what the MediaPipe vs RTMPose benchmark scores."""
    out = {}
    for video, v in labels.items():
        sets = v.get("sets", {})
        keep = {f: j for f, j in v["frames"].items() if sets.get(f, BENCHMARK) == BENCHMARK}
        out[video] = {**v, "frames": keep}
    return out


def load_labels(path, manifest_path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise LabelError([f"no labels at {path}"]) from None
    except json.JSONDecodeError as e:
        raise LabelError([f"{path} is not valid JSON: {e}"]) from None
    return validate(data, load_manifest(manifest_path))


def coverage(labels, manifest):
    """How much of the benchmark set the file covers, and what can't be scored."""
    expected = {}
    for (video, _), item in manifest["frames"].items():
        if item["set"] == BENCHMARK:
            expected[video] = expected.get(video, 0) + 1
    bench = benchmark_labels(labels)
    joints = set(manifest["joints"])
    out = {"frames_expected": sum(expected.values()),
           "frames_labelled": sum(len(v["frames"]) for v in bench.values()),
           "joints_labelled": sum(len(set(j) & joints) for v in bench.values()
                                  for j in v["frames"].values()),
           "joints_possible": sum(expected.values()) * len(manifest["joints"]),
           "frames_confirmed_from_earlier_export": sum(
               1 for v in bench.values() for f in v["frames"]
               if "review" in v.get("frame_status", {}).get(f, {})),
           "frames_outside_benchmark": sum(len(v["frames"]) for v in labels.values())
                                       - sum(len(v["frames"]) for v in bench.values()),
           "videos_unlabelled": sorted(set(expected) - set(labels)),
           "unscorable": {}}
    for video, v in labels.items():
        if video_scale(v) is None:
            out["unscorable"][video] = (f"P1 (frame {v['p1_frame']}) lacks a labelled "
                                        "shoulder or hip, so errors can't be normalized")
    return out


def video_scale(video_labels):
    """P1 torso length from the labels, or None."""
    return pm.label_torso(video_labels["frames"].get(video_labels["p1_frame"], {}))


# --- scoring ----------------------------------------------------------------

def score_video(seq, video, video_labels):
    """One row per labelled joint: error in P1 torso lengths, None = missed."""
    scale = video_scale(video_labels)
    if scale is None:
        return []
    rows = pm.label_errors(seq, video_labels["frames"], scale)
    for r in rows:
        r["video"] = video
        r["view"] = video_labels["view"]
        r["detected"] = bool(r["frame"] < seq.frame_count and seq.detected[r["frame"]])
    return rows


def summarize(rows, thresholds=PCK_THRESHOLDS):
    """Errors, misses and PCK for a set of rows (see module docstring)."""
    frames = {}
    for r in rows:
        frames[(r["video"], r["frame"])] = r["detected"]
    out = pm.summarize_errors(rows, thresholds)
    out["frames"] = len(frames)
    out["frames_missed"] = sum(not d for d in frames.values())
    out["joints_missed"] = sum(r["error"] is None for r in rows)
    return out


def group_summary(rows, joints):
    occluded = [r for r in rows if not r["visible"]]
    out = {"all": summarize(rows),
           "visible": summarize([r for r in rows if r["visible"]]),
           "occluded": summarize(occluded),
           "per_joint": {j: summarize([r for r in rows if r["joint"] == j]) for j in joints}}
    if occluded:
        out["occluded"]["answered"] = 1 - out["occluded"]["missing"]
    return out


def _pck_by_frame(rows, t):
    """{(video, frame): (hits, n)} for clustered resampling."""
    out = {}
    for r in rows:
        k = (r["video"], r["frame"])
        hit = r["error"] is not None and r["error"] <= t
        h, n = out.get(k, (0, 0))
        out[k] = (h + hit, n + 1)
    return out


def paired_bootstrap(rows_a, rows_b, t=0.1, n_boot=4000, seed=0):
    """PCK@t of b minus a, with a 95% interval from resampling labelled frames.

    Both row sets must come from the same labels. Frames, not joints, are
    resampled: joints of one frame share its blur, pose and difficulty.
    """
    a, b = _pck_by_frame(rows_a, t), _pck_by_frame(rows_b, t)
    if set(a) != set(b):
        raise ValueError("the two models were scored on different labelled frames")
    keys = sorted(a)
    if not keys:
        return None
    ha, na = np.array([a[k] for k in keys]).T
    hb, nb = np.array([b[k] for k in keys]).T
    diff = hb.sum() / nb.sum() - ha.sum() / na.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = hb[idx].sum(1) / nb[idx].sum(1) - ha[idx].sum(1) / na[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"threshold": t, "difference": float(diff), "ci95": [float(lo), float(hi)],
            "frames": len(keys)}


def evaluate(labels, sequences, joints=pm.COMMON_JOINTS):
    """sequences: {backend: {video: PoseSequence}} for every scorable video.

    Returns {"by_view": {view: {backend: group}}, "by_video": {video: {backend:
    group}}, "comparison": {view: {"<b> - <a>": bootstrap}}}.
    """
    labels = benchmark_labels(labels)
    backends = list(sequences)
    rows = {b: [] for b in backends}
    for video, vl in labels.items():
        if video_scale(vl) is None:
            continue
        for b in backends:
            if video not in sequences[b]:
                raise KeyError(f"{b} has no pose sequence for {video}")
            rows[b] += score_video(sequences[b][video], video, vl)

    report = {"by_view": {}, "by_video": {}, "comparison": {}}
    for view in VIEWS:
        per = {b: [r for r in rows[b] if r["view"] == view] for b in backends}
        if not any(per.values()):
            continue
        report["by_view"][view] = {b: group_summary(per[b], joints) for b in backends}
        if len(backends) == 2:
            a, b = backends
            report["comparison"][view] = {
                f"{b} - {a}": {f"pck@{t}": paired_bootstrap(per[a], per[b], t)
                               for t in PCK_THRESHOLDS}}
    for video in labels:
        per = {b: [r for r in rows[b] if r["video"] == video] for b in backends}
        if any(per.values()):
            report["by_video"][video] = {b: group_summary(per[b], joints) for b in backends}
    return report


# --- carrying an earlier export over for review ------------------------------

def review_prior(prior, manifest, shown, previous=None, source=""):
    """Which frames of an earlier export may be offered again for review.

    prior: the earlier export (any schema). shown: {(video, frame): SHA-1 of
    the image the earlier labeller displayed}, from the export's own digests
    or by re-decoding the way that labeller did. previous: optional
    {(video, frame): SHA-1 of the sequentially decoded frame - 1}, used only
    to explain a mismatch.

    A frame is offered only when the image it was labelled on is identical
    to the current one. Offered points are copied unchanged, as candidates:
    the labeller must show them and a person must confirm or move each one
    before they can be exported. Nothing here certifies a label.
    Returns {"source", "verified_by", "exported_at", "frames": [...],
    "refused": [{"video", "frame", "why"}]}.
    """
    previous = previous or {}
    out = {"source": source, "verified_by": str(prior.get("verified_by") or "").strip(),
           "exported_at": prior.get("exported_at"), "frames": [], "refused": []}
    for video, v in (prior.get("videos") or {}).items():
        for fkey, joints in (v.get("frames") or {}).items():
            frame = int(fkey)
            item = manifest["frames"].get((video, frame))

            def refuse(why):
                out["refused"].append({"video": video, "frame": frame, "why": why})

            if item is None:
                refuse("not in the current labelling set")
                continue
            digest = shown.get((video, frame))
            if digest is None:
                refuse("no record of which image it was labelled on")
                continue
            if digest != item["image_sha1"]:
                if previous.get((video, frame)) == digest:
                    refuse(f"labelled on the image of frame {frame - 1} (the earlier labeller's "
                           "seek landed one frame early); relabel it")
                else:
                    refuse("labelled on a different image; relabel it")
                continue
            w, h = item["width"], item["height"]
            points, bad = {}, []
            for name, lab in (joints or {}).items():
                x, y, vis = lab.get("x"), lab.get("y"), lab.get("visible")
                ok = (name in lp.POINT and isinstance(vis, bool)
                      and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                              and math.isfinite(c) for c in (x, y))
                      and 0 <= x <= w and 0 <= y <= h)
                if ok:
                    points[name] = {"x": x, "y": y, "visible": vis}
                else:
                    bad.append(name)
            if bad:
                refuse(f"invalid points {sorted(bad)}; relabel it")
                continue
            out["frames"].append({"video": video, "frame": frame,
                                  "image_sha1": item["image_sha1"], "points": points})
    return out
