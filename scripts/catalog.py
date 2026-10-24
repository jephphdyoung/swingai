"""Swing dataset catalog from the command line.

    python scripts/catalog.py bootstrap         # register the existing GW / TW videos
    python scripts/catalog.py list
    python scripts/catalog.py verify            # re-check every source's checksum
    python scripts/catalog.py add-golfer "Jane Doe" --role user --handedness right
    python scripts/catalog.py register <video> --golfer grant-waite --view face_on \\
        --source-type youtube --url https://... --date 2011-05 --motion-time unknown

Registering reads a video (checksum + sequential frame index) and never moves,
copies or writes it. Run inside the container so the frame index uses the
same decoder as the app:

    podman run --rm --security-opt label=disable -v "$PWD":/app swingai \\
        python scripts/catalog.py bootstrap

Shots (one swing each: window, crop, P-positions, labels) are made in the
app's Reference Editor tab. See docs/swing-dataset.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import catalog as cm  # noqa: E402
from utils import paths  # noqa: E402

GW = "grant-waite"
TW = "tiger-woods"
FILENAME = "filename"
DOWNLOADER = ("Downloaded with downloader/ (yt-dlp); a byte-identical copy is in downloader/. "
              "Its URL was not recorded. downloader/README.md lists "
              "https://www.youtube.com/watch?v=6sD5asBLNrw as 'Grant Waite (slow motion)', "
              "but which file came from it is unconfirmed.")

# Known facts about the videos already in the repo, keyed by content so they
# apply only to these exact bytes. Anything not known stays null/unknown.
EXISTING = {
    "1572418fa34d0613dc44a0078832a4302f6f4025c39c1f37bbad9fb0f39f9919": dict(
        golfer=GW, view="face_on", source_type="youtube", provenance_notes=DOWNLOADER),
    "4c12cf38a6b3c9c34c6f06741b08c9f9455d808226c6d91ce7695a4b3c547673": dict(
        golfer=GW, view="down_the_line", source_type="youtube", provenance_notes=DOWNLOADER),
    "148e80d1fad577cae67257f94394c301f2eee95f9035d5a35a9215b5b51bcdd1": dict(
        golfer=GW, view="face_on", date="2010-10-27", date_basis=FILENAME,
        recording_notes="Filename says 6-iron; unverified."),
    "f2bdf579b1b8abafcebf70686dacf73897f1d290c58128d29d72d1aa920896c3": dict(
        golfer=GW, view="face_on", date="2009", date_basis=FILENAME),
    "f6d19af5ea9ed424212b687a82a93a7b5e5b9036cc8cf7009ebfd08dbce0a949": dict(
        golfer=GW, view="face_on", date="2010", date_basis=FILENAME,
        recording_notes="Filename says Las Vegas; unverified."),
    "7417c26b66d894197c0438b617af6b4f51aeaf4a2868198149c26e4347988f94": dict(
        golfer=GW, view="face_on", recording_notes="Filename mentions a caddie; unreviewed."),
    "abfd2ab268f91d034ed5631fbef3085101d0cea3a00d99043fb6e207722525dc": dict(
        golfer=GW, view="unknown",
        recording_notes="Content unreviewed: may hold several swings and camera views, so "
                        "each shot states its own view."),
    "b3fd238757d239b2164e8b90683f950590ccf31d08276def12b206b80b2945b3": dict(
        golfer=TW, view="face_on", source_type="youtube",
        title="Tiger Woods Face On Iron Swing At The Presidents Cup",
        original_filename="Tiger Woods Face On Iron Swing At The Presidents Cup.mp4",
        provenance_notes="Downloaded with downloader/ (yt-dlp); the original file, with its "
                         "YouTube title as the name, is a byte-identical copy in downloader/. "
                         "URL not recorded.",
        recording_notes="Stored under videos/user/ for historical reasons, but this is Tiger "
                        "Woods reference footage. The catalog records it under the reference "
                        "golfer tiger-woods; the file is not moved."),
}
COLLECTION = "grant-waite-full-swing-v1"


def _load():
    return cm.load_catalog()


def bootstrap(args):
    cat = _load()
    for gid, name in ((GW, "Grant Waite"), (TW, "Tiger Woods")):
        if gid not in cat.golfers:
            cm.add_golfer(cat, name, "reference", "right", golfer_id=gid)
            print(f"+ golfer {gid}")
    found = {}
    for base in (paths.get_reference_dir(), paths.get_user_dir()):
        for rel in paths.list_videos(base):
            p = os.path.join(base, rel)
            sha = cm._sha256_cached(p)
            if sha in EXISTING and sha not in found:
                found[sha] = p
    for sha, meta in EXISTING.items():
        if sha not in found:
            print(f"- not found (skipped): {sha[:12]}…")
            continue
        path = found[sha]
        existing = cm.find_source_by_sha(cat, sha)
        if existing is not None:
            print(f"= {existing['source_id']} {os.path.basename(path)} (already registered)")
            continue
        rec, _ = cm.register_source(
            cat, path, golfer_id=meta["golfer"], camera_view=meta["view"],
            source_type=meta.get("source_type", "unknown"), title=meta.get("title"),
            original_filename=meta.get("original_filename"),
            provenance_notes=meta.get("provenance_notes"), recording_date=meta.get("date"),
            date_basis=meta.get("date_basis"), recording_notes=meta.get("recording_notes"),
            motion_time="unknown")
        print(f"+ {rec['source_id']} {os.path.basename(path)} → {meta['golfer']}, "
              f"{rec['video']['frame_count']} frames")
    if COLLECTION not in cat.collections:
        cm.create_collection(cat, COLLECTION, GW, "Grant Waite full swings, v1",
                             description="Curated in the Reference Editor. Independent DTL "
                                         "and face-on recordings are separate shots.")
        print(f"+ collection {COLLECTION} (draft, empty)")
    written = cm.save_catalog(cat)
    print(f"saved: {', '.join(written) or 'nothing changed'}  ({cat.root})")


def list_(args):
    cat = _load()
    for g in cat.golfers.values():
        shots = cm.shots_for_golfer(cat, g["golfer_id"])
        srcs = [s for s in cat.sources.values() if s["golfer_id"] == g["golfer_id"]]
        print(f"{g['golfer_id']} ({g['role']}, {g['handedness']}): {len(srcs)} source(s), "
              f"{len(shots)} shot(s)")
        for s in srcs:
            v = s["video"]
            print(f"   {s['source_id']}  {s['location']['root']}:{s['location']['path']}  "
                  f"{v['frame_count']} fr @ {v['nominal_fps']:.2f} fps  "
                  f"view={s['recording']['camera_view']}  date={s['recording']['date']}  "
                  f"motion_time={s['timing']['motion_time']}")
    for c in cat.collections.values():
        print(f"collection {c['collection_id']} [{c['status']}]: {len(c['shot_ids'])} shot(s)")


def verify(args):
    cat = _load()
    bad = 0
    for st in cm.verify_sources(cat, search=not args.no_search):
        print(f"{st.state:9s} {st.source_id}  {st.path or ''} {st.detail}")
        bad += st.state == "changed"
    return 1 if bad else 0


def add_golfer(args):
    cat = _load()
    rec = cm.add_golfer(cat, args.name, args.role, args.handedness, golfer_id=args.id)
    cm.save_catalog(cat)
    print(f"+ golfer {rec['golfer_id']}")


def register(args):
    cat = _load()
    rec, created = cm.register_source(
        cat, args.video, golfer_id=args.golfer, camera_view=args.view,
        source_type=args.source_type, url=args.url, title=args.title,
        recording_date=args.date, date_basis=args.date_basis, motion_time=args.motion_time,
        capture_fps=args.capture_fps, timing_basis=args.timing_basis)
    cm.save_catalog(cat)
    print(("+ " if created else "= already registered as ") + rec["source_id"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap", help="register the existing GW / TW reference videos")
    sub.add_parser("list")
    v = sub.add_parser("verify")
    v.add_argument("--no-search", action="store_true",
                   help="don't look for moved files by checksum")
    g = sub.add_parser("add-golfer")
    g.add_argument("name")
    g.add_argument("--role", choices=cm.ROLES, required=True)
    g.add_argument("--handedness", choices=cm.HANDEDNESS, default="unknown")
    g.add_argument("--id")
    r = sub.add_parser("register")
    r.add_argument("video")
    r.add_argument("--golfer", required=True)
    r.add_argument("--view", choices=cm.CAMERA_VIEWS, default="unknown")
    r.add_argument("--source-type", choices=cm.SOURCE_TYPES, default="unknown")
    r.add_argument("--url")
    r.add_argument("--title")
    r.add_argument("--date", help="YYYY, YYYY-MM or YYYY-MM-DD")
    r.add_argument("--date-basis")
    r.add_argument("--motion-time", choices=cm.MOTION_TIME, default="unknown")
    r.add_argument("--capture-fps", type=float)
    r.add_argument("--timing-basis")
    args = p.parse_args(argv)
    try:
        return {"bootstrap": bootstrap, "list": list_, "verify": verify,
                "add-golfer": add_golfer, "register": register}[args.cmd](args) or 0
    except (cm.CatalogError, cm.ConflictError) as e:
        for problem in getattr(e, "problems", [str(e)]):
            print(f"error: {problem}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
