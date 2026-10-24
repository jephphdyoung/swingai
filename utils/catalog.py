"""The swing dataset catalog: golfers, source videos, shots and collections.

Four JSON files under `data/catalog/` (`utils.paths.get_catalog_dir`), each
validated against `schemas/swing-dataset.schema.json` and then against the
cross-file rules in `check_catalog`. See docs/swing-dataset.md.

Identity is by stable id, never by path:

- a **golfer** is a reference golfer (shared, curated) or a user (their own
  recording history);
- a **source** is one immutable video file, identified by its SHA-256. Its
  `location` is only where it was last found; registering never moves, copies
  or rewrites it, and a file that no longer matches its checksum is reported,
  not re-registered;
- a **shot** is one real swing. It has one stream per camera view of *that*
  swing. Streams of different swings are never one shot, so independent
  Grant Waite DTL and face-on recordings are two shots, which a collection
  can group. A shot with several streams needs an explicit same-swing
  assertion; synchronization and calibration stay "unknown" until measured;
- a **collection** is a curated list of a reference golfer's shot ids, e.g.
  `grant-waite-full-swing-v1`. Freezing writes a **snapshot** -- a write-once
  copy of every member shot's definition (streams, windows, crops, metadata),
  its sources (checksums, video facts, timing) and its full label document --
  named and referenced by its content hash. The frozen collection *is* that
  snapshot: the working shots stay editable, and `frozen_drift` reports how
  they have moved on, but nothing changes what the collection represents.

A stream's swing window (start/end) and crop are edit metadata in source
frames and pixels; `utils/shot_edits.py` converts. Labels live per shot in
`utils/shot_annotations.py`.

Saving validates everything first and writes nothing if any rule fails. Each
file carries a revision; a save refuses to overwrite a file someone else
saved since it was loaded (`ConflictError`). Files are replaced atomically.
"""

import copy
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from utils import paths

SCHEMA_VERSION = "1.0"
KINDS = ("golfers", "sources", "shots", "collections")
ID_KEY = {"golfers": "golfer_id", "sources": "source_id", "shots": "shot_id",
          "collections": "collection_id"}
SCHEMA_PATH = os.path.join(paths.REPO_ROOT, "schemas", "swing-dataset.schema.json")

ROLES = ("reference", "user")
HANDEDNESS = ("right", "left", "unknown")
CAMERA_VIEWS = ("face_on", "down_the_line", "overhead", "impact", "unknown")
SWING_TYPES = ("full", "three_quarter", "half")
SWING_TYPE_LABELS = {"full": "Full swing", "three_quarter": "Three-quarter swing",
                     "half": "Half swing"}
CLUB_CATEGORIES = ("driver", "fairway_wood", "hybrid", "iron", "wedge", "unknown")
SOURCE_TYPES = ("youtube", "swing_catalyst", "fox_capture", "local_file", "other", "unknown")
MOTION_TIME = ("real_time", "slowed", "retimed", "unknown")
VISIBILITY = ("unassessed", "good", "partial", "poor")
REVIEW_STATUSES = ("unreviewed", "reviewed", "rejected")

# Python's view names (annotations.json, swing_pairing) <-> the contract's.
VIEW_TO_CONTRACT = {"dtl": "down_the_line", "face_on": "face_on"}
VIEW_FROM_CONTRACT = {v: k for k, v in VIEW_TO_CONTRACT.items()}


class CatalogError(ValueError):
    """The catalog (or a requested change) breaks a rule. Lists every problem."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


class ConflictError(RuntimeError):
    """A catalog file changed on disk since it was loaded."""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --- identifiers -------------------------------------------------------------
# The capture contract's rules (schemas/capture-manifest.schema.json
# #/$defs/identifier, swingai-core's ShotId): usable verbatim as a directory
# name on Windows and Linux.
_BAD_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f\x7f-\x9f]')
_RESERVED = re.compile(r"^(?:con|prn|aux|nul|(?:com|lpt)[1-9])(?:\.|$)", re.IGNORECASE)


def is_identifier(value):
    return (isinstance(value, str) and bool(value) and not _BAD_CHARS.search(value)
            and value not in (".", "..") and not value.endswith((".", " "))
            and not _RESERVED.match(value))


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "item"


def new_id(prefix):
    return f"{prefix}-{secrets.token_hex(5)}"


# --- schema and files -------------------------------------------------------

@lru_cache(maxsize=1)
def _validator():
    import jsonschema
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def schema_problems(doc, where=""):
    """Every JSON Schema violation in a dataset document, as readable lines."""
    out = []
    for e in sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path)):
        at = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"{where}{at}: {e.message}")
    return out


def atomic_write_json(path, data):
    """Write to a temp file in the same directory, fsync, then rename over."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_revision(path):
    """The revision of the file on disk, or None if there is no file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("revision")


@dataclass
class Catalog:
    root: str
    golfers: dict = field(default_factory=dict)       # id -> record, in file order
    sources: dict = field(default_factory=dict)
    shots: dict = field(default_factory=dict)
    collections: dict = field(default_factory=dict)
    revisions: dict = field(default_factory=dict)     # kind -> revision loaded (None: no file)
    _loaded: dict = field(default_factory=dict, repr=False)
    _pending_snapshots: dict = field(default_factory=dict, repr=False)  # path -> document

    def records(self, kind):
        return getattr(self, kind)

    def path(self, kind):
        return os.path.join(self.root, f"{kind}.json")

    def document(self, kind, revision):
        return {"schema_version": SCHEMA_VERSION, "kind": kind, "revision": revision,
                kind: list(self.records(kind).values())}

    def golfer(self, golfer_id):
        return _get(self.golfers, golfer_id, "golfer")

    def source(self, source_id):
        return _get(self.sources, source_id, "source")

    def shot(self, shot_id):
        return _get(self.shots, shot_id, "shot")

    def collection(self, collection_id):
        return _get(self.collections, collection_id, "collection")


def _get(records, key, what):
    if key not in records:
        raise CatalogError(f"no {what} {key!r} in the catalog")
    return records[key]


def load_catalog(root=None) -> Catalog:
    """Load and validate all four files. A missing file is an empty list."""
    root = root or paths.get_catalog_dir()
    cat = Catalog(root=root)
    problems = []
    for kind in KINDS:
        path = cat.path(kind)
        if not os.path.exists(path):
            cat.revisions[kind] = None
            cat._loaded[kind] = json.dumps([])
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{kind}.json: unreadable ({e})")
            continue
        found = schema_problems(doc, f"{kind}.json: ")
        if not found and doc.get("kind") != kind:
            found = [f"{kind}.json: holds kind {doc.get('kind')!r}"]
        if found:
            problems += found
            continue
        records = getattr(cat, kind)
        for rec in doc[kind]:
            key = rec[ID_KEY[kind]]
            if key in records:
                problems.append(f"{kind}.json: duplicate id {key!r}")
            records[key] = rec
        cat.revisions[kind] = doc["revision"]
        cat._loaded[kind] = json.dumps(doc[kind], sort_keys=True)
    if problems:
        raise CatalogError(problems)
    problems = check_catalog(cat)
    if problems:
        raise CatalogError(problems)
    return cat


def save_catalog(cat: Catalog):
    """Validate everything, then write the files that changed. Writes nothing
    if any rule fails or any changed file was saved by someone else since load."""
    problems = check_catalog(cat)
    changed = []
    for kind in KINDS:
        current = json.dumps(list(cat.records(kind).values()), sort_keys=True)
        if current == cat._loaded.get(kind):
            continue
        rev = (cat.revisions.get(kind) or 0) + 1
        doc = cat.document(kind, rev)
        problems += schema_problems(doc, f"{kind}.json: ")
        changed.append((kind, doc, current))
    problems += _frozen_changes(cat)
    if problems:
        raise CatalogError(problems)
    for kind, _, _ in changed:
        on_disk = read_revision(cat.path(kind))
        if on_disk != cat.revisions.get(kind):
            raise ConflictError(
                f"{kind}.json was saved elsewhere (revision {on_disk}, loaded "
                f"{cat.revisions.get(kind)}); reload and redo the change")
    # Snapshots before the collections that point at them. Content-addressed
    # names make them write-once: an existing file must already be identical.
    for rel, doc in list(cat._pending_snapshots.items()):
        path = os.path.join(cat.root, *rel.split("/"))
        if os.path.exists(path):
            with open(path) as f:
                if canonical_sha256(json.load(f)) != canonical_sha256(doc):
                    raise CatalogError(f"{rel} exists with different content")
        else:
            atomic_write_json(path, doc)
        del cat._pending_snapshots[rel]
    # Referenced records first, so a reader never sees a dangling id.
    for kind, doc, current in changed:
        atomic_write_json(cat.path(kind), doc)
        cat.revisions[kind] = doc["revision"]
        cat._loaded[kind] = current
    return [kind for kind, _, _ in changed]


def _frozen_changes(cat):
    """A collection frozen on disk must be saved unchanged."""
    path = cat.path("collections")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        disk = {c["collection_id"]: c for c in json.load(f).get("collections", [])}
    out = []
    for cid, old in disk.items():
        if old.get("status") == "frozen" and cat.collections.get(cid) != old:
            out.append(f"collection {cid!r} is frozen and can't be changed or removed")
    return out


# --- cross-file rules ---------------------------------------------------------

def _decided(rec, what, where, problems):
    """A verified (or asserted) status must say how and by whom."""
    if rec["status"] in ("verified", "asserted", "estimated"):
        missing = [k for k in ("method", "by") if not str(rec.get(k) or "").strip()]
        if missing:
            problems.append(f"{where}: {what} is {rec['status']} but has no "
                            f"{' or '.join(missing)}")


def check_catalog(cat: Catalog):
    problems = []
    for gid, g in cat.golfers.items():
        if not is_identifier(gid):
            problems.append(f"golfer {gid!r}: not a valid identifier")
    shas = {}
    for sid, s in cat.sources.items():
        where = f"source {sid}"
        if s.get("golfer_id") is not None and s["golfer_id"] not in cat.golfers:
            problems.append(f"{where}: unknown golfer {s['golfer_id']!r}")
        if s["sha256"] in shas:
            problems.append(f"{where}: same bytes (sha256) as source {shas[s['sha256']]}")
        shas[s["sha256"]] = sid
    for shot_id, shot in cat.shots.items():
        problems += _check_shot(cat, shot)
    for cid, c in cat.collections.items():
        where = f"collection {cid}"
        golfer = cat.golfers.get(c["golfer_id"])
        if golfer is None:
            problems.append(f"{where}: unknown golfer {c['golfer_id']!r}")
        elif golfer["role"] != "reference":
            problems.append(f"{where}: {c['golfer_id']} is a user, and collections hold "
                            "reference swings; a user's shots stay in their own history")
        if c["status"] == "frozen" and not (c.get("frozen_at") and c.get("frozen_by")
                                            and c.get("snapshot")):
            problems.append(f"{where}: frozen without frozen_at / frozen_by / snapshot")
        if c["status"] == "draft" and (c.get("frozen_at") or c.get("frozen_by")
                                       or c.get("snapshot")):
            problems.append(f"{where}: a draft can't carry frozen_at / frozen_by / snapshot")
        for shot_id in c["shot_ids"]:
            shot = cat.shots.get(shot_id)
            if shot is None:
                problems.append(f"{where}: unknown shot {shot_id!r}")
                continue
            if shot["golfer_id"] != c["golfer_id"]:
                problems.append(f"{where}: shot {shot_id} is {shot['golfer_id']}'s, not "
                                f"{c['golfer_id']}'s")
            # A frozen collection was checked when it was frozen; its snapshot,
            # not the (editable) working shot, is what it holds.
            if (c["status"] == "draft" and c["swing_types"]
                    and shot["swing_type"] not in c["swing_types"]):
                problems.append(f"{where}: shot {shot_id} is a {shot['swing_type']} swing; "
                                f"the collection holds {c['swing_types']}")
    return problems


def _check_shot(cat, shot):
    problems = []
    where = f"shot {shot['shot_id']}"
    if shot["golfer_id"] not in cat.golfers:
        problems.append(f"{where}: unknown golfer {shot['golfer_id']!r}")
    seen = set()
    for st in shot["streams"]:
        at = f"{where} stream {st['stream_id']}"
        if st["stream_id"] in seen:
            problems.append(f"{at}: duplicate stream id")
        seen.add(st["stream_id"])
        src = cat.sources.get(st["source_id"])
        if src is None:
            problems.append(f"{at}: unknown source {st['source_id']!r}")
            continue
        if src.get("golfer_id") not in (None, shot["golfer_id"]):
            problems.append(f"{at}: source {src['source_id']} is {src['golfer_id']}'s, "
                            f"not {shot['golfer_id']}'s")
        n = src["video"]["frame_count"]
        w = st["window"]
        if not 0 <= w["start_frame"] <= w["end_frame"] < n:
            problems.append(f"{at}: window {w['start_frame']}..{w['end_frame']} is not within "
                            f"the source's frames 0..{n - 1} with start <= end")
        c = st.get("crop")
        if c is not None:
            W, H = src["video"]["width"], src["video"]["height"]
            if min(c["x"], c["y"]) < 0 or min(c["width"], c["height"]) < 1:
                problems.append(f"{at}: crop {c} needs x, y >= 0 and a positive size")
            elif c["x"] + c["width"] > W or c["y"] + c["height"] > H:
                problems.append(f"{at}: crop {c} extends past the {W}x{H} frame")
    mv = shot["multi_view"]
    same, sync, cal = mv["same_swing"], mv["synchronization"], mv["calibration"]
    if len(shot["streams"]) == 1:
        if same["status"] != "not_applicable" or sync["status"] != "not_applicable":
            problems.append(f"{where}: one stream, so same_swing and synchronization are "
                            "not_applicable")
    else:
        if same["status"] not in ("asserted", "verified"):
            problems.append(
                f"{where}: {len(shot['streams'])} streams, but same_swing is "
                f"{same['status']}. Streams of different swings are separate shots "
                "(group them in a collection instead)")
        if sync["status"] == "not_applicable" or cal["status"] == "not_applicable":
            problems.append(f"{where}: several streams, so synchronization and calibration "
                            "must be unknown, estimated or verified")
        if sync["status"] == "verified" and same["status"] != "verified":
            problems.append(f"{where}: synchronization can't be verified while same_swing "
                            "is only asserted")
    _decided(same, "same_swing", where, problems)
    _decided(sync, "synchronization", where, problems)
    _decided(cal, "calibration", where, problems)
    if shot["review"]["status"] != "unreviewed" and not shot["review"].get("by"):
        problems.append(f"{where}: review status {shot['review']['status']} needs a reviewer")
    return problems


# --- locations ---------------------------------------------------------------

def _under(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == \
            os.path.abspath(root)
    except ValueError:
        return False


def location_for(path):
    """reference/user-relative when inside those libraries (portable between
    host and container), else absolute."""
    path = os.path.abspath(path)
    for root, base in (("reference", paths.get_reference_dir()), ("user", paths.get_user_dir())):
        if _under(path, base):
            return {"root": root, "path": os.path.relpath(path, base).replace(os.sep, "/")}
    return {"root": "absolute", "path": path}


def resolve_location(loc):
    if loc["root"] == "reference":
        return os.path.join(paths.get_reference_dir(), *loc["path"].split("/"))
    if loc["root"] == "user":
        return os.path.join(paths.get_user_dir(), *loc["path"].split("/"))
    return loc["path"]


_sha_memo = {}


def _sha256_cached(path):
    from utils.frame_access import file_sha256
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _sha_memo:
        _sha_memo[key] = file_sha256(path)
    return _sha_memo[key]


@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    state: str            # ok | relocated | changed | missing
    path: Optional[str]   # usable path when ok/relocated
    detail: str = ""

    @property
    def usable(self):
        return self.state in ("ok", "relocated")


def resolve_source(cat: Catalog, source_id, search=True) -> SourceStatus:
    """Find the source's bytes. Its location is tried first, then its other
    known copies, then (search=True) any same-size video in the libraries. A
    file at the location with different bytes is `changed`, never accepted."""
    src = cat.source(source_id)
    primary = resolve_location(src["location"])
    changed = None
    candidates = [primary] + [resolve_location(l) for l in src.get("also_found_at", [])]
    for i, p in enumerate(candidates):
        if not os.path.isfile(p):
            continue
        if os.path.getsize(p) == src["size_bytes"] and _sha256_cached(p) == src["sha256"]:
            return SourceStatus(source_id, "ok" if i == 0 else "relocated", p,
                                "" if i == 0 else f"found at a recorded copy: {p}")
        if i == 0:
            changed = p
    if search:
        for base in (paths.get_reference_dir(), paths.get_user_dir()):
            for rel in paths.list_videos(base):
                p = os.path.join(base, rel)
                if (os.path.getsize(p) == src["size_bytes"]
                        and _sha256_cached(p) == src["sha256"]):
                    return SourceStatus(source_id, "relocated", p,
                                        f"found by checksum at {p}")
    if changed:
        return SourceStatus(source_id, "changed", None,
                            f"{changed} no longer matches the registered sha256; the source "
                            "is immutable, so its labels can't be trusted against this file")
    return SourceStatus(source_id, "missing", None, f"not found at {primary}")


def verify_sources(cat: Catalog, search=True):
    return [resolve_source(cat, sid, search=search) for sid in cat.sources]


# --- golfers and sources -----------------------------------------------------

def add_golfer(cat, display_name, role, handedness="unknown", golfer_id=None, notes=None):
    golfer_id = golfer_id or slugify(display_name)
    if golfer_id in cat.golfers:
        raise CatalogError(f"golfer {golfer_id!r} already exists")
    if not is_identifier(golfer_id):
        raise CatalogError(f"{golfer_id!r} is not a valid identifier")
    if role not in ROLES or handedness not in HANDEDNESS:
        raise CatalogError(f"role must be one of {ROLES}, handedness one of {HANDEDNESS}")
    rec = {"golfer_id": golfer_id, "display_name": display_name, "role": role,
           "handedness": handedness, "notes": notes}
    cat.golfers[golfer_id] = rec
    return rec


def find_source_by_sha(cat, sha):
    return next((s for s in cat.sources.values() if s["sha256"] == sha), None)


def register_source(cat, path, *, golfer_id, camera_view="unknown", source_type="unknown",
                    url=None, title=None, original_filename=None, provenance_notes=None,
                    recording_date=None, date_basis=None, recording_notes=None,
                    motion_time="unknown", capture_fps=None, timing_basis=None,
                    source_id=None, progress=None):
    """Register a video file in place. Returns (record, created).

    The file is read (checksum, sequential frame index) and never written.
    Registering bytes that are already registered returns that source and
    records this path as another copy.
    """
    from utils.annotations import _normalize_path, has_annotations
    from utils.frame_access import FrameAccessError, get_frame_index

    if not os.path.isfile(path):
        raise CatalogError(f"no video file at {path}")
    if golfer_id is not None and golfer_id not in cat.golfers:
        raise CatalogError(f"unknown golfer {golfer_id!r}")
    sha = _sha256_cached(path)
    loc = location_for(path)
    existing = find_source_by_sha(cat, sha)
    if existing is not None:
        known = [existing["location"]] + existing.get("also_found_at", [])
        if loc not in known:
            existing.setdefault("also_found_at", []).append(loc)
        return existing, False
    try:
        index = get_frame_index(path, progress=progress)
    except FrameAccessError as e:
        raise CatalogError(f"{path}: {e}") from None
    source_id = source_id or new_id("src")
    if source_id in cat.sources or not is_identifier(source_id):
        raise CatalogError(f"source id {source_id!r} is taken or invalid")
    rec = {
        "source_id": source_id,
        "golfer_id": golfer_id,
        "sha256": sha,
        "size_bytes": os.path.getsize(path),
        "location": loc,
        "also_found_at": [],
        "registered_at": now_iso(),
        "legacy_annotation_key": _normalize_path(path) if has_annotations(path) else None,
        "provenance": {"source_type": source_type, "url": url, "title": title,
                       "original_filename": original_filename or os.path.basename(path),
                       "notes": provenance_notes},
        "recording": {"date": recording_date, "date_basis": date_basis,
                      "camera_view": camera_view, "notes": recording_notes},
        "video": {**index.summary(), "decoder": index.decoder},
        "timing": {"motion_time": motion_time, "capture_fps": capture_fps,
                   "basis": timing_basis},
    }
    cat.sources[source_id] = rec
    return rec, True


SOURCE_EDITABLE = {"golfer_id", "provenance", "recording", "timing"}


def update_source(cat, source_id, **fields):
    """Change descriptive metadata. Identity, bytes and video facts are fixed."""
    src = cat.source(source_id)
    bad = set(fields) - SOURCE_EDITABLE
    if bad:
        raise CatalogError(f"source fields {sorted(bad)} can't be edited")
    for k, v in fields.items():
        src[k] = {**src[k], **v} if isinstance(v, dict) and isinstance(src.get(k), dict) else v
    return src


# --- shots -------------------------------------------------------------------

def _status(status, method=None, by=None, at=None, notes=None, **extra):
    rec = {"status": status, "method": method, "by": by, "at": at, "notes": notes}
    rec.update(extra)
    return rec


def single_view_status():
    return {"same_swing": _status("not_applicable"),
            "synchronization": _status("not_applicable", offset_ms=None, uncertainty_ms=None),
            "calibration": _status("unknown")}


def _new_stream(cat, shot, source_id, camera_view, start_frame, end_frame, crop=None,
                stream_id=None):
    src = cat.source(source_id)
    if camera_view not in CAMERA_VIEWS:
        raise CatalogError(f"camera_view must be one of {CAMERA_VIEWS}")
    taken = {s["stream_id"] for s in shot["streams"]}
    stream_id = stream_id or (camera_view if camera_view not in taken else
                              f"{camera_view}-{len(taken) + 1}")
    return {"stream_id": stream_id, "source_id": src["source_id"], "camera_view": camera_view,
            "window": {"start_frame": int(start_frame), "end_frame": int(end_frame)},
            "crop": crop,
            "quality": {"body_visibility": "unassessed", "club_visibility": "unassessed",
                        "resolution_usable": "unassessed", "notes": None}}


def add_shot(cat, *, golfer_id, source_id, camera_view, start_frame, end_frame,
             swing_type, club=None, handedness=None, recording_date=None, crop=None,
             setup_notes=None, notes=None, shot_id=None, stream_id=None):
    """A new single-view shot: one swing seen by one camera."""
    golfer = cat.golfer(golfer_id)
    src = cat.source(source_id)
    if swing_type not in SWING_TYPES:
        raise CatalogError(f"swing_type must be one of {SWING_TYPES}")
    shot_id = shot_id or new_id("shot")
    if shot_id in cat.shots or not is_identifier(shot_id):
        raise CatalogError(f"shot id {shot_id!r} is taken or invalid")
    club = {"category": "unknown", "name": None, "length_in": None, "length_basis": None,
            **(club or {})}
    shot = {"shot_id": shot_id, "golfer_id": golfer_id, "created_at": now_iso(),
            "swing_type": swing_type, "club": club,
            "handedness": handedness or golfer["handedness"],
            "recording_date": recording_date if recording_date is not None
            else src["recording"]["date"],
            "setup_notes": setup_notes, "streams": [], "multi_view": single_view_status(),
            "review": {"status": "unreviewed", "by": None, "at": None, "notes": None},
            "capture": None, "notes": notes}
    shot["streams"].append(_new_stream(cat, shot, source_id, camera_view, start_frame,
                                       end_frame, crop, stream_id))
    problems = _check_shot(cat, shot)
    if problems:
        raise CatalogError(problems)
    cat.shots[shot_id] = shot
    return shot


def add_stream(cat, shot_id, *, source_id, camera_view, start_frame, end_frame,
               same_swing_method, asserted_by, crop=None, stream_id=None):
    """Another camera's view of the *same* swing. Requires saying why they are
    the same swing; synchronization and calibration start unknown."""
    shot = cat.shot(shot_id)
    if not str(same_swing_method or "").strip() or not str(asserted_by or "").strip():
        raise CatalogError("adding a view to a shot asserts it records the same swing: give "
                           "the basis (e.g. 'co-triggered Swing Catalyst pair') and who says so")
    updated = copy.deepcopy(shot)
    updated["streams"].append(_new_stream(cat, updated, source_id, camera_view, start_frame,
                                          end_frame, crop, stream_id))
    mv = updated["multi_view"]
    if mv["same_swing"]["status"] == "not_applicable":
        mv["same_swing"] = _status("asserted", same_swing_method, asserted_by, now_iso())
    if mv["synchronization"]["status"] == "not_applicable":
        mv["synchronization"] = _status("unknown", offset_ms=None, uncertainty_ms=None)
    if mv["calibration"]["status"] == "not_applicable":
        mv["calibration"] = _status("unknown")
    problems = _check_shot(cat, updated)
    if problems:
        raise CatalogError(problems)
    cat.shots[shot_id] = updated
    return updated


def _stream(shot, stream_id):
    for st in shot["streams"]:
        if st["stream_id"] == stream_id:
            return st
    raise CatalogError(f"shot {shot['shot_id']} has no stream {stream_id!r}")


def _apply(cat, shot_id, change):
    """Apply `change(copy)` to a copy of the shot; keep it only if valid."""
    updated = copy.deepcopy(cat.shot(shot_id))
    change(updated)
    problems = _check_shot(cat, updated)
    if problems:
        raise CatalogError(problems)
    cat.shots[shot_id] = updated
    return updated


def set_window(cat, shot_id, stream_id, start_frame, end_frame):
    def change(shot):
        _stream(shot, stream_id)["window"] = {"start_frame": int(start_frame),
                                              "end_frame": int(end_frame)}
    return _apply(cat, shot_id, change)


def set_crop(cat, shot_id, stream_id, crop):
    if crop is not None:
        crop = {k: int(crop[k]) for k in ("x", "y", "width", "height")}
        if crop["width"] < 1 or crop["height"] < 1 or crop["x"] < 0 or crop["y"] < 0:
            raise CatalogError(f"crop {crop} must have x, y >= 0 and a positive size")

    def change(shot):
        _stream(shot, stream_id)["crop"] = crop
    return _apply(cat, shot_id, change)


def set_stream_quality(cat, shot_id, stream_id, **quality):
    def change(shot):
        q = _stream(shot, stream_id)["quality"]
        for k, v in quality.items():
            if k not in q:
                raise CatalogError(f"unknown quality field {k!r}")
            q[k] = v
    return _apply(cat, shot_id, change)


SHOT_EDITABLE = {"swing_type", "club", "handedness", "recording_date", "setup_notes", "notes"}


def update_shot(cat, shot_id, **fields):
    bad = set(fields) - SHOT_EDITABLE
    if bad:
        raise CatalogError(f"shot fields {sorted(bad)} can't be edited here")
    def change(shot):
        for k, v in fields.items():
            shot[k] = {**shot[k], **v} if k == "club" else v
    return _apply(cat, shot_id, change)


def set_review(cat, shot_id, status, by=None, notes=None):
    if status not in REVIEW_STATUSES:
        raise CatalogError(f"review status must be one of {REVIEW_STATUSES}")

    def change(shot):
        shot["review"] = {"status": status, "by": by if status != "unreviewed" else None,
                          "at": now_iso() if status != "unreviewed" else None, "notes": notes}
    return _apply(cat, shot_id, change)


def set_multi_view(cat, shot_id, which, status, method=None, by=None, notes=None, **extra):
    """Change same_swing, synchronization or calibration status."""
    if which not in ("same_swing", "synchronization", "calibration"):
        raise CatalogError(f"unknown multi-view status {which!r}")

    def change(shot):
        rec = _status(status, method, by, now_iso() if status not in
                      ("unknown", "not_applicable") else None, notes)
        if which == "synchronization":
            rec["offset_ms"] = extra.get("offset_ms")
            rec["uncertainty_ms"] = extra.get("uncertainty_ms")
        shot["multi_view"][which] = rec
    return _apply(cat, shot_id, change)


def remove_shot(cat, shot_id, annotations_dir=None):
    """Delete a shot record (never its source). Refused while it is in a
    collection or has saved labels."""
    cat.shot(shot_id)
    used = [c["collection_id"] for c in cat.collections.values() if shot_id in c["shot_ids"]]
    if used:
        raise CatalogError(f"shot {shot_id} is in collection(s) {used}; remove it there first")
    labels = os.path.join(annotations_dir or paths.get_shot_annotations_dir(),
                          f"{shot_id}.json")
    if os.path.exists(labels):
        raise CatalogError(f"shot {shot_id} has saved labels ({labels}); they are kept, so "
                           "the shot can't be removed")
    del cat.shots[shot_id]


def shots_for_source(cat, source_id):
    return [s for s in cat.shots.values()
            if any(st["source_id"] == source_id for st in s["streams"])]


def shots_for_golfer(cat, golfer_id):
    return [s for s in cat.shots.values() if s["golfer_id"] == golfer_id]


# --- collections -------------------------------------------------------------

def create_collection(cat, collection_id, golfer_id, title, description=None,
                      swing_types=("full",)):
    if collection_id in cat.collections or not is_identifier(collection_id):
        raise CatalogError(f"collection id {collection_id!r} is taken or invalid")
    rec = {"collection_id": collection_id, "golfer_id": golfer_id, "title": title,
           "description": description, "swing_types": list(swing_types), "status": "draft",
           "frozen_at": None, "frozen_by": None, "snapshot": None, "shot_ids": []}
    cat.collections[collection_id] = rec
    problems = [p for p in check_catalog(cat) if f"collection {collection_id}" in p]
    if problems:
        del cat.collections[collection_id]
        raise CatalogError(problems)
    return rec


def _draft(cat, collection_id):
    c = cat.collection(collection_id)
    if c["status"] == "frozen":
        raise CatalogError(f"collection {collection_id} is frozen; create a new version "
                           "instead (e.g. ...-v2)")
    return c


def add_to_collection(cat, collection_id, shot_id):
    c = _draft(cat, collection_id)
    cat.shot(shot_id)
    if shot_id in c["shot_ids"]:
        return c
    c["shot_ids"].append(shot_id)
    problems = [p for p in check_catalog(cat) if f"collection {collection_id}" in p]
    if problems:
        c["shot_ids"].remove(shot_id)
        raise CatalogError(problems)
    return c


def remove_from_collection(cat, collection_id, shot_id):
    c = _draft(cat, collection_id)
    if shot_id in c["shot_ids"]:
        c["shot_ids"].remove(shot_id)
    return c


def canonical_sha256(doc):
    """SHA-256 of a JSON document's canonical form (sorted keys, no spaces)."""
    import hashlib
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_collection(cat, collection_id, by, annotations_dir=None):
    """Freeze a draft: snapshot every member's complete inputs.

    Refused when a member's source bytes can't be found and checked, or its
    labels don't validate against the catalog -- a frozen input has to be
    reproducible. The snapshot is written by the next `save_catalog`.
    """
    from utils import shot_annotations as sa
    c = _draft(cat, collection_id)
    if not str(by or "").strip():
        raise CatalogError("freezing needs the name of who froze it")
    if not c["shot_ids"]:
        raise CatalogError(f"collection {collection_id} has no shots to freeze")
    problems, shots = [], []
    for shot_id in c["shot_ids"]:
        shot = cat.shot(shot_id)
        source_ids = list(dict.fromkeys(st["source_id"] for st in shot["streams"]))
        for sid in source_ids:
            status = resolve_source(cat, sid)
            if not status.usable:
                problems.append(f"shot {shot_id}: source {sid} is {status.state} "
                                f"({status.detail})")
        doc = None
        if os.path.exists(sa.annotations_path(shot_id, annotations_dir)):
            doc = sa.load(shot_id, annotations_dir)
            doc = {k: v for k, v in doc.items() if not k.startswith("_")}
            problems += sa.check(doc, cat)
        shots.append({"shot": copy.deepcopy(shot),
                      "sources": [copy.deepcopy(cat.source(s)) for s in source_ids],
                      "annotations": doc,
                      "annotations_sha256": None if doc is None else canonical_sha256(doc)})
    if problems:
        raise CatalogError([f"can't freeze {collection_id}: {p}" for p in problems])
    frozen_at = now_iso()
    snapshot = {"schema_version": SCHEMA_VERSION, "kind": "collection_snapshot", "revision": 0,
                "collection_id": collection_id, "golfer_id": c["golfer_id"],
                "title": c["title"], "swing_types": list(c["swing_types"]),
                "frozen_at": frozen_at, "frozen_by": by, "shots": shots}
    problems = schema_problems(snapshot, "snapshot: ")
    if problems:
        raise CatalogError(problems)
    sha = canonical_sha256(snapshot)
    rel = f"snapshots/{collection_id}-{sha[:12]}.json"
    cat._pending_snapshots[rel] = snapshot
    c.update(status="frozen", frozen_at=frozen_at, frozen_by=by,
             snapshot={"path": rel, "sha256": sha, "shot_count": len(shots)})
    return c


def load_snapshot(cat, collection_id):
    """The frozen collection's inputs, after proving they are the ones frozen.

    Raises CatalogError when the collection isn't frozen, or its snapshot is
    missing, invalid, or not byte-for-byte (canonically) what was recorded.
    """
    c = cat.collection(collection_id)
    ref = c.get("snapshot")
    if c["status"] != "frozen" or not ref:
        raise CatalogError(f"collection {collection_id} is not frozen")
    if ref["path"] in cat._pending_snapshots:
        return copy.deepcopy(cat._pending_snapshots[ref["path"]])
    path = os.path.join(cat.root, *ref["path"].split("/"))
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise CatalogError(f"{collection_id}: snapshot {ref['path']} unreadable ({e})") from None
    if canonical_sha256(doc) != ref["sha256"]:
        raise CatalogError(f"{collection_id}: snapshot {ref['path']} does not match its "
                           "recorded sha256; the frozen inputs have been altered")
    problems = schema_problems(doc, f"{ref['path']}: ")
    if problems:
        raise CatalogError(problems)
    if [s["shot"]["shot_id"] for s in doc["shots"]] != c["shot_ids"]:
        raise CatalogError(f"{collection_id}: snapshot shots differ from the collection's")
    return doc


def frozen_drift(cat, collection_id, annotations_dir=None):
    """{shot_id: [what changed]} between a frozen snapshot and the working
    shots. Informational: the collection keeps its snapshot regardless."""
    from utils import shot_annotations as sa
    doc = load_snapshot(cat, collection_id)
    out = {}
    for entry in doc["shots"]:
        shot_id = entry["shot"]["shot_id"]
        changes = []
        current = cat.shots.get(shot_id)
        if current is None:
            changes.append("the working shot no longer exists")
        else:
            frozen = entry["shot"]
            for st_old in frozen["streams"]:
                st_new = next((s for s in current["streams"]
                               if s["stream_id"] == st_old["stream_id"]), None)
                if st_new is None:
                    changes.append(f"stream {st_old['stream_id']} removed")
                    continue
                for key in ("window", "crop", "camera_view", "source_id", "quality"):
                    if st_new.get(key) != st_old.get(key):
                        changes.append(f"{st_old['stream_id']} {key} changed")
            if len(current["streams"]) != len(frozen["streams"]):
                changes.append("streams added or removed")
            for key in ("swing_type", "club", "handedness", "recording_date", "multi_view",
                        "review", "setup_notes"):
                if current.get(key) != frozen.get(key):
                    changes.append(f"{key} changed")
        path = sa.annotations_path(shot_id, annotations_dir)
        now = None
        if os.path.exists(path):
            now = sa.load(shot_id, annotations_dir)
            now = canonical_sha256({k: v for k, v in now.items() if not k.startswith("_")})
        if now != entry["annotations_sha256"]:
            changes.append("labels changed")
        for src in entry["sources"]:
            live = cat.sources.get(src["source_id"])
            if live is None or live["sha256"] != src["sha256"]:
                changes.append(f"source {src['source_id']} record changed")
        if changes:
            out[shot_id] = changes
    return out


def frozen_collections_with(cat, shot_id):
    return [c for c in cat.collections.values()
            if c["status"] == "frozen" and shot_id in c["shot_ids"]]


# --- eligibility -------------------------------------------------------------

def _gate(reasons):
    return {"eligible": not reasons, "reasons": reasons}


def eligibility(cat, shot_id):
    """What each later stage may use this shot for, with reasons when not.

    Gates, not a score: every "eligible" follows from recorded statuses, and
    anything unassessed or unknown is not eligible.
    """
    shot = cat.shot(shot_id)
    streams = shot["streams"]
    mv = shot["multi_view"]
    out = {}

    reasons = []
    if len(streams) < 2:
        reasons.append("a single camera view: usable as an independent 2D observation, "
                       "not a stereo measurement")
    else:
        if mv["same_swing"]["status"] != "verified":
            reasons.append(f"same swing is {mv['same_swing']['status']}, not verified")
        if mv["synchronization"]["status"] != "verified":
            reasons.append(f"synchronization is {mv['synchronization']['status']}")
        if mv["calibration"]["status"] != "verified":
            reasons.append(f"calibration is {mv['calibration']['status']}")
        views = [s["camera_view"] for s in streams]
        if "unknown" in views or len(set(views)) < len(views):
            reasons.append(f"camera views {views} are not distinct, known views")
    out["stereo_3d"] = _gate(reasons)

    reasons = []
    for st in streams:
        src = cat.sources[st["source_id"]]
        t = src["timing"]
        ts = src["video"]["timestamps"]
        if ts["status"] != "verified":
            reasons.append(f"{st['stream_id']}: frame timestamps are {ts['status']} "
                           f"({ts['detail']}), not verified presentation times")
        if t["motion_time"] == "real_time" and t.get("basis"):
            continue
        if t["motion_time"] == "slowed" and t.get("capture_fps") and t.get("basis"):
            continue
        if t["motion_time"] == "real_time":
            reasons.append(f"{st['stream_id']}: real time is claimed without a basis")
            continue
        reasons.append(f"{st['stream_id']}: motion time is {t['motion_time']}"
                       + (" without a known capture rate and basis"
                          if t["motion_time"] == "slowed" else "")
                       + "; playback fps is not real swing time")
    if len(streams) > 1 and mv["synchronization"]["status"] in ("unknown", "not_applicable"):
        reasons.append("cross-view timing needs synchronization")
    out["motion_timing"] = _gate(reasons)

    for key, field_, what in (("body_landmarks", "body_visibility", "body"),
                              ("club_landmarks", "club_visibility", "club")):
        reasons = [f"{st['stream_id']}: {what} visibility {st['quality'][field_]}"
                   for st in streams if st["quality"][field_] not in ("good", "partial")]
        out[key] = _gate(reasons)
    out["resolution"] = _gate([f"{st['stream_id']}: resolution usable is "
                               f"{st['quality']['resolution_usable']}"
                               for st in streams if st["quality"]["resolution_usable"] != "yes"])
    out["human_review"] = _gate([] if shot["review"]["status"] == "reviewed" else
                                [f"review status is {shot['review']['status']}"])
    return out
