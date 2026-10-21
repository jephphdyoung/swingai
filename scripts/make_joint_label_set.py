"""Build the joint-labelling set and a local HTML labeller for it.

    python scripts/make_joint_label_set.py [--review-prior OLD_EXPORT.json]
        [--add-sequence NAME VIDEO START END [--step N] [--purpose TEXT]] [--repick]

Writes
    data/joint_labels/manifest.json   which frames, and why (committed)
    data/joint_labels/labeler.html    self-contained labeller with the frames
                                      embedded (git-ignored: it holds footage)

Open labeler.html in a browser, label the Body group (and Feet / Hands-Club
where wanted), export labels.json, and save it as data/joint_labels/labels.json.
evaluate_joint_labels.py then scores both backends on the benchmark frames.

By default the frames already in the manifest are kept, and each is decoded
again and checked against its recorded SHA-1: the benchmark set never changes
silently. --repick chooses benchmark frames afresh (only before anyone has
labelled). --add-sequence appends a run of consecutive frames (e.g. P5-P8, or
around heel-lift onset) as a separate, non-benchmark set.

Assisted labelling: a fixed, model-blind subset of the benchmark frames gets no
model output at all (assign_blind). Every other frame gets proposals read from
the pose cache (never re-inferred unless --extract): both models' points where
both have the point, flagged where they disagree, with the first-choice model
alternating so neither is favoured. Run it in the GPU image, where both
models' caches live. --no-assist builds without proposals.

--review-prior offers an earlier export's points again, as unconfirmed
candidates, on frames whose labelled image is identical to the current one
(utils.joint_labels.review_prior). An export without digests came from the
seek-based labeller, so its images are re-created by seeking the same way.

Frame choice for --repick: P1/P4/P7 where the video has them annotated (the annotations
only choose frames; they are not ground truth), the frames where MediaPipe
and RTMPose disagree most (where accuracy actually differs), and the fastest
hand-motion frame. Nothing is pre-filled -- a pre-placed point would anchor
the labeller toward one model.

Frames are decoded sequentially, exactly as pose extraction reads them, and
each one's pixel SHA-1 goes into the manifest and the exported labels, so the
evaluation can prove labels and poses refer to the same image.
"""

import argparse
import base64
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from utils import joint_labels as jl  # noqa: E402
from utils import label_points as lp  # noqa: E402
from utils.annotations import is_confirmed, load_annotation_record  # noqa: E402
from utils.joint_labels import decode_frames, frame_digest  # noqa: E402
from utils.landmarks import LANDMARK_INDEX  # noqa: E402
from utils.paths import repo_relative  # noqa: E402
from utils.pose_data import get_pose_sequence, make_backend  # noqa: E402
from utils.pose_metrics import COMMON_JOINTS, body_scale, valid_mask  # noqa: E402

VIDEOS = [
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Down the line 61fc.mp4", "dtl"),
    ("videos/user/Jeff Young - 2026-06-20 150221 Fox Face on right aaf0.mp4", "face_on"),
    ("videos/reference/GW_DTL.mp4", "dtl"),
    ("videos/reference/GW_faceon.mp4", "face_on"),
]
OUT_DIR = "data/joint_labels"
MAX_WIDTH = 1440


def disagreement_per_frame(a, b):
    scale = body_scale(a)
    ma, mb = valid_mask(a), valid_mask(b)
    idx = [LANDMARK_INDEX[n] for n in COMMON_JOINTS]
    d = np.linalg.norm(a.xy[:, idx] - b.xy[:, idx], axis=2) / scale
    d[~(ma & mb)] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(d, axis=1)


def pick_frames(path, window):
    mp_seq = get_pose_sequence(path, make_backend("mediapipe"), extract_if_missing=False)
    rt_seq = get_pose_sequence(path, make_backend("rtmpose"), extract_if_missing=False)
    rec = load_annotation_record(path)
    chosen = []
    for p in ("P1", "P4", "P7"):
        if rec and p in rec.frames:
            chosen.append((rec.frames[p], f"{p} (annotated frame, used only to sample)"))
    lo, hi = window
    if mp_seq is not None and rt_seq is not None:
        dis = disagreement_per_frame(mp_seq, rt_seq)
        dis[:lo] = np.nan
        dis[hi:] = np.nan
        min_gap = max(5, (hi - lo) // 10)
        for f in np.argsort(-np.nan_to_num(dis, nan=-1)):
            if np.isnan(dis[f]) or len([c for c in chosen if "disagree" in c[1]]) >= 2:
                break
            if all(abs(int(f) - c[0]) >= min_gap for c in chosen):
                chosen.append((int(f), f"backends disagree most ({dis[f]:.2f} torso)"))
        # fastest trail-wrist motion by RTMPose, if no P7 was annotated
        if not any(c[1].startswith("P7") for c in chosen):
            w = rt_seq.xy[:, LANDMARK_INDEX["right_wrist"]]
            sp = np.linalg.norm(np.diff(w, axis=0), axis=1)
            sp[:lo] = np.nan
            sp[hi:] = np.nan
            if np.isfinite(sp).any():
                f = int(np.nanargmax(sp))
                if all(abs(f - c[0]) >= 3 for c in chosen):
                    chosen.append((f, "fastest hand motion (RTMPose track)"))
    return sorted(chosen)


def swing_window(path, n):
    rec = load_annotation_record(path)
    if rec and rec.frames:
        frames = sorted(rec.frames.values())
        pad = max(5, (frames[-1] - frames[0]) // 4) if len(frames) > 1 else n // 4
        return max(0, frames[0] - pad), min(n, frames[-1] + 3 * pad)
    return 0, n


def size_of(img):
    h, w = img.shape[:2]
    return {"width": w, "height": h, "display_scale": min(1.0, MAX_WIDTH / w)}


def encode(img):
    h, w = img.shape[:2]
    scale = size_of(img)["display_scale"]
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode(), {"width": w, "height": h, "display_scale": scale}


BLIND_PER_VIDEO = 3
BLIND_SALT = "swingai-blind-v1"
BLIND_RULE = (f"per video: the P1 frame, then the other benchmark frames in order of "
              f"sha1('{BLIND_SALT}' + image_sha1) until {BLIND_PER_VIDEO} are blind; "
              "chosen before any model proposal was generated, never from model output")
ASSIST_MODELS = ("rtmpose", "mediapipe")      # first-choice model alternates in this order
# Models disagree on a point when it's this far apart (torso lengths): the
# PCK@0.1 radius, so a disagreement means at least one of them would fail it.
DISAGREE_TORSO = 0.1


def _blind_order(f):
    return hashlib.sha1((BLIND_SALT + f["image_sha1"]).encode()).hexdigest()


def assign_blind(frames, per_video=BLIND_PER_VIDEO):
    """Mark which benchmark frames are blind. Once set, flags never change.

    Uses only the frame list and image digests, never model output, so the
    blind subset can't be chosen by where the models do well or badly.
    """
    bench = [f for f in frames if f["set"] == jl.BENCHMARK]
    if any("blind" in f for f in bench):
        for f in frames:
            f.setdefault("blind", False)
        return frames
    by_video = {}
    for f in bench:
        by_video.setdefault(f["video"], []).append(f)
    for fs in by_video.values():
        p1 = [f for f in fs if f["frame"] == f["p1_frame"]]
        rest = sorted((f for f in fs if f not in p1), key=_blind_order)
        for f in (p1 + rest)[:per_video]:
            f["blind"] = True
    for f in frames:
        f.setdefault("blind", False)
    return frames


def assign_primary(frames):
    """First-choice model per assisted frame, alternating within each video."""
    by_video = {}
    for f in frames:
        if f["blind"]:
            f.pop("assist_primary", None)
        elif "assist_primary" not in f:
            by_video.setdefault((f["video"], f["set"]), []).append(f)
    for fs in by_video.values():
        for i, f in enumerate(sorted(fs, key=lambda f: (_blind_order(f), f["frame"]))):
            f["assist_primary"] = ASSIST_MODELS[i % len(ASSIST_MODELS)]
    return frames


def frame_proposals(frame, seqs, scale):
    """{point: {model: [x, y], ..., "disagree": torso lengths | None}} from cached poses.

    Only points a model really produces (label_points.proposal_columns), each
    under that model's own confidence threshold. Nothing for contact, marker,
    equipment or floor points.
    """
    out = {}
    for name, cols in lp.proposal_columns().items():
        pts = {}
        for model, col in cols.items():
            seq = seqs.get(model)
            p = seq.point(frame, col) if seq is not None and frame < seq.frame_count else None
            if p is not None:
                pts[model] = [round(float(p[0]), 1), round(float(p[1]), 1)]
        if not pts:
            continue
        if len(pts) == 2 and scale and np.isfinite(scale):
            a, b = (np.array(pts[m]) for m in sorted(pts))
            pts["disagree"] = round(float(np.linalg.norm(a - b) / scale), 3)
        out[name] = pts
    return out


def assist_data(frames, extract=False):
    """Proposals for every non-blind frame, from the pose cache; and model identities."""
    from utils.pose_rtmpose import RTMPoseUnavailable
    backends, identities = {}, {}
    for model in ASSIST_MODELS:
        try:
            backends[model] = make_backend(model)
            identities[model] = backends[model].identity()
        except RTMPoseUnavailable as e:
            raise SystemExit(f"assisted labelling needs both models' caches; {model}: {e}\n"
                             "Run this in the GPU image, or pass --no-assist.")
    proposals = {}
    for video in sorted({f["video"] for f in frames if not f["blind"]}):
        seqs = {m: get_pose_sequence(video, b, extract_if_missing=extract)
                for m, b in backends.items()}
        missing = [m for m, q in seqs.items() if q is None]
        if missing:
            print(f"  no proposals for {os.path.basename(video)}: no cached {missing} poses "
                  "(pass --extract to compute them)")
            continue
        scales = [body_scale(q) for q in seqs.values()]
        scale = float(np.nanmean(scales)) if np.isfinite(scales).any() else None
        for f in frames:
            if f["video"] == video and not f["blind"]:
                proposals[(video, f["frame"])] = frame_proposals(f["frame"], seqs, scale)
    return proposals, identities


def p_position(frame, rec):
    """Approximate P-position of a frame from saved annotations, or None.

    Annotations were placed in the seek-based annotator (one frame early past
    ~frame 700 on the 240fps captures) and some are unconfirmed, so this is a
    guide for the labeller, never a measurement.
    """
    marks = sorted(((f, p) for p, f in (rec.frames.items() if rec else ())),
                   key=lambda fp: (fp[0], int(fp[1][1:])))
    if not marks:
        return None
    exact = [p for f, p in marks if f == frame]
    if exact:
        label = exact[0]
    elif frame < marks[0][0]:
        label = f"before {marks[0][1]}"
    elif frame > marks[-1][0]:
        label = f"after {marks[-1][1]}"
    else:
        lo = max((fp for fp in marks if fp[0] < frame), key=lambda fp: fp[0])
        hi = min((fp for fp in marks if fp[0] > frame), key=lambda fp: fp[0])
        label = f"{lo[1]}–{hi[1]}"
    note = "approximate, from saved annotations"
    if not is_confirmed(rec):
        note += " (not confirmed by a person, or legacy P6/P7 naming)"
    return {"label": label, "note": note}


def item_for(video, view, frame, image, reason, p1, set_name=jl.BENCHMARK, sequence=None):
    item = {"video": video, "view": view, "frame": int(frame), "set": set_name,
            "reason": reason, "p1_frame": p1, "image_sha1": frame_digest(image),
            **size_of(image)}
    if sequence:
        item["sequence"] = sequence
    return item


def kept_frames(frames):
    """Re-decode the manifest's frames sequentially; refuse if any image changed."""
    images, problems = {}, []
    by_video = {}
    for f in frames:
        by_video.setdefault(f["video"], []).append(f)
    for video, fs in by_video.items():
        decoded = decode_frames(video, [f["frame"] for f in fs])
        for f in fs:
            img = decoded.get(f["frame"])
            if img is None:
                problems.append(f"{video}: frame {f['frame']} no longer decodes")
            elif frame_digest(img) != f.get("image_sha1"):
                problems.append(f"{video}: frame {f['frame']} decodes to a different image "
                                "than the manifest records")
            else:
                images[(video, f["frame"])] = img
    if problems:
        raise SystemExit("The labelling set's images changed; refusing to rebuild over them:\n  - "
                         + "\n  - ".join(problems))
    return [{"set": jl.BENCHMARK, **f} for f in frames], images


def sequence_frames(video, view, name, start, end, step=1, purpose=None, existing=()):
    """Consecutive frames start..end (inclusive), decoded sequentially."""
    if end < start or step < 1:
        raise SystemExit(f"bad sequence range {start}..{end} step {step}")
    wanted = [f for f in range(start, end + 1, step) if (video, f) not in set(existing)]
    decoded = decode_frames(video, wanted)
    missing = sorted(set(wanted) - set(decoded))
    if missing:
        raise SystemExit(f"{video} ends before frame {missing[0]}")
    rec = load_annotation_record(video)
    p1 = rec.frames.get("P1") if rec else None
    seq = {"name": name, "start": start, "end": end, "step": step}
    if purpose:
        seq["purpose"] = purpose
    reason = f"sequence {name}" + (f": {purpose}" if purpose else "")
    items = [item_for(video, view, f, decoded[f], reason, p1, "sequence", seq) for f in wanted]
    return items, {(video, f): decoded[f] for f in wanted}


def seek_digests(keys):
    """{(video, frame): SHA-1} decoded by seeking, as the pre-digest labeller did."""
    out = {}
    for video in sorted({v for v, _ in keys}):
        cap = cv2.VideoCapture(video)
        try:
            for v, f in sorted(k for k in keys if k[0] == video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = cap.read()
                if ok:
                    out[(v, f)] = frame_digest(img)
        finally:
            cap.release()
    return out


def prior_review(path, manifest):
    """Candidates from an earlier export, offered only on identical images."""
    with open(path) as f:
        prior = json.load(f)
    keys, own = [], {}
    for video, v in (prior.get("videos") or {}).items():
        for fkey in (v.get("frames") or {}):
            keys.append((video, int(fkey)))
            if (v.get("image_sha1") or {}).get(fkey):
                own[(video, int(fkey))] = v["image_sha1"][fkey]
    existing = [k for k in keys if os.path.exists(k[0])]
    shown = {**seek_digests([k for k in existing if k not in own]), **own}
    previous = {}
    for video in sorted({v for v, _ in existing}):
        frames = [f for v, f in existing if v == video]
        decoded = decode_frames(video, [f - 1 for f in frames if f > 0])
        previous.update({(video, f): frame_digest(decoded[f - 1])
                         for f in frames if f - 1 in decoded})
    how = ("its own image digests" if len(own) == len(keys) else
           "re-decoding by seeking, as the pre-digest labeller did")
    out = jl.review_prior(prior, manifest, shown, previous, source=os.path.basename(path))
    out["image_check"] = how
    return out


def labeler_data(items, review, assist):
    return {"schema": jl.SCHEMA, "benchmark_joints": list(lp.BENCHMARK_POINTS),
            "catalogue": lp.catalogue(), "frames": items, "review": review, "assist": assist}


def assert_blind_frames_have_no_model_output(data):
    """The page itself must hold nothing a blind frame could be anchored by."""
    for f in data["frames"]:
        if f["blind"] and ("proposals" in f or "assist_primary" in f):
            raise SystemExit(f"internal error: blind frame {f['id']} carries model output")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repick", action="store_true",
                    help="choose benchmark frames afresh instead of keeping the manifest's")
    ap.add_argument("--add-sequence", nargs=4, metavar=("NAME", "VIDEO", "START", "END"),
                    action="append", default=[])
    ap.add_argument("--step", type=int, default=1, help="frame step for --add-sequence")
    ap.add_argument("--purpose", help="what an added sequence is for, e.g. 'P5-P8'")
    ap.add_argument("--review-prior", help="earlier labels.json to offer again for review")
    ap.add_argument("--no-assist", action="store_true", help="no model proposals at all")
    ap.add_argument("--extract", action="store_true",
                    help="run a model on videos it has no cached poses for")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    if args.repick or not os.path.exists(manifest_path):
        frames, images = [], {}
        for path, view in VIDEOS:
            if not os.path.exists(path):
                continue
            n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
            rec = load_annotation_record(path)
            p1 = rec.frames.get("P1") if rec else None
            picks = pick_frames(path, swing_window(path, n))
            # Decoded sequentially, as pose extraction reads them -- not by seeking.
            decoded = decode_frames(path, [f for f, _ in picks])
            for frame, reason in picks:
                if frame in decoded:
                    video = repo_relative(path)
                    frames.append(item_for(video, view, frame, decoded[frame], reason, p1))
                    images[(video, frame)] = decoded[frame]
    else:
        with open(manifest_path) as f:
            frames, images = kept_frames(json.load(f)["frames"])
        print(f"kept {len(frames)} frames; every one decodes to its recorded image")

    views = {f["video"]: f["view"] for f in frames}
    for name, video, start, end in args.add_sequence:
        video = repo_relative(video)
        view = views.get(video)
        if view is None:
            rec = load_annotation_record(video)
            view = rec.view_angle if rec else None
        if view not in jl.VIEWS:
            raise SystemExit(f"{video}: unknown view; annotate its view angle first")
        new, imgs = sequence_frames(video, view, name, int(start), int(end), args.step,
                                    args.purpose, [(f["video"], f["frame"]) for f in frames])
        frames += new
        images.update(imgs)
        print(f"added sequence {name}: {len(new)} frames of {video}")

    assign_primary(assign_blind(frames))
    for i, f in enumerate(frames):
        f["id"] = f"{i:02d}"
        pp = p_position(f["frame"], load_annotation_record(f["video"]))
        f.pop("p_position", None)
        if pp:
            f["p_position"] = pp
        mode = "blind" if f["blind"] else f"assist:{f['assist_primary']}"
        print(f"{f['id']}  {os.path.basename(f['video'])[:44]:44s} frame {f['frame']:5d}  "
              f"{f['set']:9s} {mode:16s} {pp['label'] if pp else '':10s} {f['reason']}")
    manifest = {"schema": jl.SCHEMA, "joints": list(lp.BENCHMARK_POINTS),
                "blind_rule": BLIND_RULE, "catalogue": lp.catalogue(), "frames": frames}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    review = None
    if args.review_prior:
        review = prior_review(args.review_prior, jl.load_manifest_dict(manifest))
        print(f"review: {len(review['frames'])} frames of {review['source']} can be offered "
              f"for confirmation (image check: {review['image_check']})")
        for r in review["refused"]:
            print(f"  not offered: {os.path.basename(r['video'])} frame {r['frame']}: {r['why']}")

    assist = None
    proposals = {}
    if not args.no_assist:
        proposals, identities = assist_data(frames, args.extract)
        assist = {"models": identities, "disagree_torso": DISAGREE_TORSO,
                  "blind_rule": BLIND_RULE}
        print(f"assist: proposals for {len(proposals)} frames; "
              f"{sum(f['blind'] for f in frames)} blind frames have none")
    items = []
    for f in frames:
        item = {**f, "jpg": encode(images[(f["video"], f["frame"])])[0]}
        if not f["blind"] and (f["video"], f["frame"]) in proposals:
            item["proposals"] = proposals[(f["video"], f["frame"])]
        items.append(item)
    html = open(os.path.join(os.path.dirname(__file__), "joint_labeler.html")).read()
    data = labeler_data(items, review, assist)
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    assert_blind_frames_have_no_model_output(data)
    with open(os.path.join(OUT_DIR, "labeler.html"), "w") as f:
        f.write(html)
    print(f"{len(frames)} frames -> {OUT_DIR}/labeler.html")


if __name__ == "__main__":
    main()
