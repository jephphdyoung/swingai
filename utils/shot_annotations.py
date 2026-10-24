"""Per-shot labels: P1-P10 and body/club/ball points, tied to exact source frames.

One file per shot, `data/annotations/<shot_id>.json`, validated against
`schemas/swing-dataset.schema.json` (kind "shot_annotations"). Keyed by the
shot's stream ids. Each label records:

- the **source frame index** and its **presentation time**, from the
  sequential frame index (`utils/frame_access.py`), with `timestamp_basis`
  "container" or "estimated" (the decoder gave no time for that frame). The
  source's `video.timestamps.status` says whether container times were
  verified;
- the **SHA-1 of the decoded image** it was placed on, the same digest the
  joint-label benchmark pins. A label whose digest no longer matches the
  source's frame is reported by `identity_problems`, never silently moved;
- its **origin**: manual (placed by a person), model_proposed (a model's
  point, kept where the model put it) or corrected (a model's point a person
  moved); and, per frame, a human **review** status. Reviewing a frame is the
  sign-off on every point on it, model proposals included; any later change
  sends the frame back to unreviewed.

Points are in source pixels (crops never move them; see `utils/shot_edits.py`).
A point a person looked for and couldn't place is recorded as unavailable
with a reason, never guessed.

P-positions use the canonical convention (P6 = shaft parallel, P7 = impact),
fixed in every file. The swing window's start/end are separate and live on
the shot stream. Legacy `data/annotations.json` entries are not read or
converted here.

Point names reuse the PR #4 labeller's catalogue (`utils/label_points.py`) --
the 15 body joints, feet and hands/club groups -- plus a ball group. The
benchmark catalogue itself is untouched.
"""

import json
import os
from typing import Optional

from utils import catalog as cat_mod
from utils import label_points as lp
from utils import paths
from utils.annotations import P_DEFINITIONS, P_POSITIONS

P_CONVENTION = "p7-impact"
ORIGINS = ("manual", "model_proposed", "corrected")
UNAVAILABLE_REASONS = ("occluded", "out_of_frame", "motion_blur", "not_present",
                       "cannot_determine")
UNAVAILABLE_LABELS = {"occluded": "Hidden (occluded)", "out_of_frame": "Out of frame",
                      "motion_blur": "Motion blur", "not_present": "Not present",
                      "cannot_determine": "Can't determine"}

BALL_POINTS = (
    {"name": "ball_center", "group": "ball", "kind": "equipment",
     "help": "Centre of the ball at address. Skip once it has left the frame.",
     "models": {m: {"point": None, "match": None, "extracted": False, "column": None}
                for m in lp.MODELS}},
)
GROUPS = lp.GROUPS + ("ball",)
GROUP_TITLES = {**lp.GROUP_TITLES, "ball": "Ball"}
POINTS = lp.POINTS + BALL_POINTS
POINT = {p["name"]: p for p in POINTS}
GROUP_POINTS = {g: tuple(p["name"] for p in POINTS if p["group"] == g) for g in GROUPS}


def annotations_path(shot_id, directory=None):
    return os.path.join(directory or paths.get_shot_annotations_dir(), f"{shot_id}.json")


def empty(shot_id):
    return {"schema_version": cat_mod.SCHEMA_VERSION, "kind": "shot_annotations",
            "revision": 0, "shot_id": shot_id, "p_convention": P_CONVENTION, "streams": {}}


def load(shot_id, directory=None):
    """The shot's labels (an empty document if none are saved yet)."""
    path = annotations_path(shot_id, directory)
    if not os.path.exists(path):
        doc = empty(shot_id)
        doc["_loaded_revision"] = None
        return doc
    with open(path) as f:
        doc = json.load(f)
    problems = cat_mod.schema_problems(doc, f"{os.path.basename(path)}: ")
    if not problems and doc["shot_id"] != shot_id:
        problems = [f"{path}: holds shot {doc['shot_id']!r}"]
    if problems:
        raise cat_mod.CatalogError(problems)
    doc["_loaded_revision"] = doc["revision"]
    return doc


def save(doc, catalog=None, directory=None):
    """Validate (schema, and against the catalog when given) and write
    atomically. Refuses if the file was saved elsewhere since it was loaded."""
    out = {k: v for k, v in doc.items() if not k.startswith("_")}
    loaded = doc.get("_loaded_revision")
    out["revision"] = (loaded or 0) + 1
    problems = cat_mod.schema_problems(out, f"{doc['shot_id']}.json: ")
    if catalog is not None:
        problems += check(out, catalog)
    if problems:
        raise cat_mod.CatalogError(problems)
    path = annotations_path(doc["shot_id"], directory)
    on_disk = cat_mod.read_revision(path)
    if on_disk != loaded:
        raise cat_mod.ConflictError(
            f"{os.path.basename(path)} was saved elsewhere (revision {on_disk}, loaded "
            f"{loaded}); reload and redo the change")
    cat_mod.atomic_write_json(path, out)
    doc["revision"] = out["revision"]
    doc["_loaded_revision"] = out["revision"]
    return path


def check(doc, catalog):
    """Cross-checks against the catalog: the shot, its streams and sources,
    frame ranges, image bounds, point names, origins and reviews."""
    problems = []
    shot = catalog.shots.get(doc["shot_id"])
    if shot is None:
        return [f"labels for unknown shot {doc['shot_id']!r}"]
    streams = {s["stream_id"]: s for s in shot["streams"]}
    for sid, sa in doc["streams"].items():
        where = f"{doc['shot_id']} {sid}"
        st = streams.get(sid)
        if st is None:
            problems.append(f"{where}: the shot has no such stream")
            continue
        if sa["source_id"] != st["source_id"]:
            problems.append(f"{where}: labelled on source {sa['source_id']}, the stream "
                            f"uses {st['source_id']}")
            continue
        src = catalog.sources[st["source_id"]]
        if sa["source_sha256"] != src["sha256"]:
            problems.append(f"{where}: labelled on different bytes than the registered source")
        n, w, h = (src["video"]["frame_count"], src["video"]["width"],
                   src["video"]["height"])
        for p, lab in sa["positions"].items():
            if not 0 <= lab["frame"] < n:
                problems.append(f"{where} {p}: frame {lab['frame']} outside 0..{n - 1}")
        for fkey, fl in sa["frames"].items():
            at = f"{where} frame {fkey}"
            if not 0 <= int(fkey) < n:
                problems.append(f"{at}: outside 0..{n - 1}")
            for name, pt in fl["points"].items():
                if name not in POINT:
                    problems.append(f"{at} {name}: not a known point")
                elif not (0 <= pt["x"] <= w and 0 <= pt["y"] <= h):
                    problems.append(f"{at} {name}: ({pt['x']}, {pt['y']}) outside {w}x{h}")
                elif pt["origin"] != "manual" and not pt.get("model"):
                    problems.append(f"{at} {name}: {pt['origin']} without the model's name")
            for name in fl.get("unavailable", {}):
                if name not in POINT:
                    problems.append(f"{at} {name}: not a known point")
                elif name in fl["points"]:
                    problems.append(f"{at} {name}: both placed and marked unavailable")
            review = fl.get("review") or {"status": "unreviewed"}
            if review["status"] != "unreviewed" and not review.get("by"):
                problems.append(f"{at}: review {review['status']} without a reviewer")
    return problems


def window_warnings(doc, shot):
    """Labels outside their stream's swing window: allowed (the window may be
    narrowed later) but worth a look."""
    out = []
    streams = {s["stream_id"]: s for s in shot["streams"]}
    for sid, sa in doc["streams"].items():
        st = streams.get(sid)
        if st is None:
            continue
        start, end = st["window"]["start_frame"], st["window"]["end_frame"]
        for p in P_POSITIONS:
            lab = sa["positions"].get(p)
            if lab and not start <= lab["frame"] <= end:
                out.append(f"{sid} {p} (frame {lab['frame']}) is outside the swing window "
                           f"{start}–{end}")
        ordered = [(p, sa["positions"][p]["frame"]) for p in P_POSITIONS if p in sa["positions"]]
        for (pa, fa), (pb, fb) in zip(ordered, ordered[1:]):
            if fb < fa:
                out.append(f"{sid}: {pb} (frame {fb}) is earlier than {pa} (frame {fa})")
    return out


def identity_problems(doc, stream_id, index):
    """Labels whose image digest differs from the source frame they name.

    `index` is the source's `FrameIndex`. A mismatch means the label was
    placed on another image (a different decode or a changed file): it is
    reported, never re-pointed to another frame.
    """
    sa = doc["streams"].get(stream_id)
    if sa is None:
        return []
    out = []
    for p, lab in sa["positions"].items():
        f = lab["frame"]
        if f >= index.frame_count or index.digests[f] != lab["image_sha1"]:
            out.append(f"{p} at frame {f}: placed on a different image than this frame")
    for fkey, fl in sa["frames"].items():
        f = int(fkey)
        if f >= index.frame_count or index.digests[f] != fl["image_sha1"]:
            out.append(f"labels at frame {f}: placed on a different image than this frame")
    return out


# --- edits -------------------------------------------------------------------

def _stream_doc(doc, shot, stream_id):
    st = next((s for s in shot["streams"] if s["stream_id"] == stream_id), None)
    if st is None:
        raise cat_mod.CatalogError(f"shot {shot['shot_id']} has no stream {stream_id!r}")
    sa = doc["streams"].get(stream_id)
    if sa is None:
        sa = doc["streams"][stream_id] = {"source_id": st["source_id"], "source_sha256": None,
                                         "positions": {}, "frames": {}}
    return sa


def _identity(sa, source, index, frame):
    if index.file_sha256 != source["sha256"]:
        raise cat_mod.CatalogError("the frame index is for different bytes than the source")
    if not 0 <= frame < index.frame_count:
        raise cat_mod.CatalogError(f"frame {frame} outside 0..{index.frame_count - 1}")
    sa["source_sha256"] = source["sha256"]
    return {"timestamp_ms": round(index.timestamps_ms[frame], 3),
            "timestamp_basis": "estimated" if index.timestamp_is_estimated(frame)
            else "container",
            "image_sha1": index.digests[frame]}


def set_position(doc, shot, source, index, stream_id, position, frame, by=None,
                 origin="manual"):
    if position not in P_DEFINITIONS:
        raise cat_mod.CatalogError(f"{position!r} is not one of P1-P10")
    if origin not in ORIGINS:
        raise cat_mod.CatalogError(f"origin must be one of {ORIGINS}")
    sa = _stream_doc(doc, shot, stream_id)
    sa["positions"][position] = {"frame": int(frame), **_identity(sa, source, index, frame),
                                 "origin": origin, "set_by": by, "set_at": cat_mod.now_iso()}


def clear_position(doc, stream_id, position):
    sa = doc["streams"].get(stream_id)
    if sa is not None:
        sa["positions"].pop(position, None)


def _frame_doc(sa, source, index, frame):
    ident = _identity(sa, source, index, frame)
    fl = sa["frames"].get(str(frame))
    if fl is None:
        fl = sa["frames"][str(frame)] = {**ident, "points": {}, "unavailable": {},
                                         "review": {"status": "unreviewed", "by": None,
                                                    "at": None, "notes": None}}
    elif fl["image_sha1"] != ident["image_sha1"]:
        raise cat_mod.CatalogError(
            f"frame {frame}'s saved labels were placed on a different image; resolve that "
            "before adding more")
    return fl


def _unreview(fl):
    fl["review"] = {"status": "unreviewed", "by": None, "at": None, "notes": None}


def set_point(doc, shot, source, index, stream_id, frame, name, x, y, visible=True,
              by=None, origin="manual", model=None, confidence=None):
    """Place (or move) a point. Moving a model proposal makes it 'corrected'.
    Any change sends the frame back to unreviewed."""
    if name not in POINT:
        raise cat_mod.CatalogError(f"{name!r} is not a known point")
    w, h = source["video"]["width"], source["video"]["height"]
    if not (0 <= x <= w and 0 <= y <= h):
        raise cat_mod.CatalogError(f"({x}, {y}) is outside the {w}x{h} frame")
    sa = _stream_doc(doc, shot, stream_id)
    fl = _frame_doc(sa, source, index, frame)
    prev = fl["points"].get(name)
    if origin == "manual" and prev and prev["origin"] in ("model_proposed", "corrected"):
        origin, model = "corrected", prev.get("model")
    fl["points"][name] = {"x": float(x), "y": float(y), "visible": bool(visible),
                          "origin": origin, "model": model, "confidence": confidence,
                          "set_by": by, "set_at": cat_mod.now_iso()}
    fl["unavailable"].pop(name, None)
    _unreview(fl)


def mark_unavailable(doc, shot, source, index, stream_id, frame, name, reason):
    if name not in POINT:
        raise cat_mod.CatalogError(f"{name!r} is not a known point")
    if reason not in UNAVAILABLE_REASONS:
        raise cat_mod.CatalogError(f"reason must be one of {UNAVAILABLE_REASONS}")
    sa = _stream_doc(doc, shot, stream_id)
    fl = _frame_doc(sa, source, index, frame)
    fl["points"].pop(name, None)
    fl["unavailable"][name] = reason
    _unreview(fl)


def remove_point(doc, stream_id, frame, name):
    sa = doc["streams"].get(stream_id)
    fl = sa and sa["frames"].get(str(frame))
    if not fl:
        return
    fl["points"].pop(name, None)
    fl["unavailable"].pop(name, None)
    _unreview(fl)
    if not fl["points"] and not fl["unavailable"]:
        del sa["frames"][str(frame)]


def set_frame_review(doc, stream_id, frame, status, by, notes=None):
    sa = doc["streams"].get(stream_id)
    fl = sa and sa["frames"].get(str(frame))
    if not fl:
        raise cat_mod.CatalogError(f"no labels at frame {frame} to review")
    if status not in cat_mod.REVIEW_STATUSES:
        raise cat_mod.CatalogError(f"review status must be one of {cat_mod.REVIEW_STATUSES}")
    if status != "unreviewed" and not str(by or "").strip():
        raise cat_mod.CatalogError("a review needs the reviewer's name")
    fl["review"] = {"status": status, "by": by if status != "unreviewed" else None,
                    "at": cat_mod.now_iso() if status != "unreviewed" else None,
                    "notes": notes}


def frame_labels(doc, stream_id, frame) -> Optional[dict]:
    sa = doc["streams"].get(stream_id)
    return sa["frames"].get(str(frame)) if sa else None


def positions(doc, stream_id):
    sa = doc["streams"].get(stream_id)
    return {p: lab["frame"] for p, lab in (sa["positions"] if sa else {}).items()}
