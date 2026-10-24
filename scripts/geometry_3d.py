"""Calibrate two cameras, check the calibration, and reconstruct a shot in 3D.

Real-world validation workflow for the two FOX cameras (docs/geometry-3d.md):

    # 1. intrinsics, per camera, from a checkerboard waved through the whole image
    python scripts/geometry_3d.py calibrate-intrinsics --camera down_the_line \\
        --video board_dtl.mp4 --every 10 --board 9x6 --square-mm 40 --out dtl.json
    python scripts/geometry_3d.py calibrate-intrinsics --camera face_on ... --out fo.json

    # 2. extrinsics from the same board seen by both cameras at the same moments;
    #    end with the board lying flat on the ground (the reference pose)
    python scripts/geometry_3d.py calibrate-extrinsics --intrinsics dtl.json fo.json \\
        --video-a both_dtl.mp4 --video-b both_fo.mp4 --every 5 --board 9x6 \\
        --square-mm 40 --reference-pair -1 --board-on-ground \\
        --calibration-id fox-2026-11-01 --version 1 --by "Jeff" --out fox.json

    # 3. check a known distance (e.g. a 1.000 m stick): pixel ends in each view
    python scripts/geometry_3d.py check-distance --calibration fox.json \\
        --a 812,640 1033,598 --b 402,700 655,690 --known-m 1.000 --tolerance-mm 5

    # 4. reconstruct a labelled, catalogued same-swing shot
    python scripts/geometry_3d.py reconstruct --shot <shot_id> --calibration fox.json

A calibration is written "estimated". Only check-distance with --by and a
passing result writes a "verified" version (version + 1) -- and the shot's
catalog calibration status must be verified separately in the editor.
Nothing here claims real-world accuracy that hasn't been checked this way.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from utils import calibration as cal  # noqa: E402
from utils.camera_model import Camera  # noqa: E402


def _board(text):
    cols, rows = (int(v) for v in text.lower().split("x"))
    return cols, rows


def _frames(video, every, offset=0):
    """(index, BGR) for every Nth frame, decoded sequentially (exact indexes)."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            if i >= offset and (i - offset) % every == 0:
                yield i, img
            i += 1
    finally:
        cap.release()


def _images(patterns):
    import cv2
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        img = cv2.imread(path)
        if img is not None:
            yield path, img


def calibrate_intrinsics(args):
    cols, rows = _board(args.board)
    obj = cal.board_object_points(cols, rows, args.square_mm / 1000.0)
    found, size = [], None
    source = _frames(args.video, args.every) if args.video else _images(args.images)
    for _, img in source:
        size = (img.shape[1], img.shape[0])
        corners = cal.detect_board(img, cols, rows)
        if corners is not None:
            found.append(corners)
    if size is None:
        raise SystemExit("no images were read")
    cam, rms = cal.calibrate_intrinsics(args.camera, size, found, [obj] * len(found))
    out = {**cam.to_json("camera_to_rig"), "intrinsic_rms_px": rms, "n_views": len(found)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{args.camera}: {len(found)} board views, RMS {rms:.3f} px, fx {cam.fx:.1f}, "
          f"fy {cam.fy:.1f}, k1 {cam.dist[0]:.4f} -> {args.out}")


def _load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    return Camera.from_json(d, "camera_to_rig"), d.get("intrinsic_rms_px")


def calibrate_extrinsics(args):
    (a, rms_a), (b, rms_b) = (_load_intrinsics(p) for p in args.intrinsics)
    cols, rows = _board(args.board)
    obj = cal.board_object_points(cols, rows, args.square_mm / 1000.0)
    pairs = []
    if args.video_a:
        fa = _frames(args.video_a, args.every)
        fb = _frames(args.video_b, args.every, args.offset_frames)
        for (ia, img_a), (ib, img_b) in zip(fa, fb):
            ca = cal.detect_board(img_a, cols, rows, canonical=True)
            cb = cal.detect_board(img_b, cols, rows, canonical=True)
            if ca is not None and cb is not None:
                pairs.append((ca, cb))
    else:
        imgs_a, imgs_b = list(_images([args.images_a])), list(_images([args.images_b]))
        for (_, ia), (_, ib) in zip(imgs_a, imgs_b):
            ca = cal.detect_board(ia, cols, rows, canonical=True)
            cb = cal.detect_board(ib, cols, rows, canonical=True)
            if ca is not None and cb is not None:
                pairs.append((ca, cb))
    if not pairs:
        raise SystemExit("the board was never found in both views at once")
    ref = args.reference_pair % len(pairs)
    a, b, info = cal.calibrate_extrinsics(a, b, obj, pairs, reference_pair=ref)
    doc = cal.calibration_document(
        args.calibration_id, args.version, [a, b], "estimated",
        f"checkerboard {cols}x{rows}, {args.square_mm} mm squares; cv2.calibrateCamera "
        "per camera, cv2.stereoCalibrate with intrinsics fixed", args.by, None,
        intrinsic_rms={a.camera_id: rms_a, b.camera_id: rms_b},
        extrinsic={"method": "cv2.stereoCalibrate (intrinsics fixed); rig = reference board",
                   **info},
        ground=cal.ground_from_board() if args.board_on_ground else None)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    base = np.linalg.norm(a.center - b.center)
    print(f"{len(pairs)} shared board views, stereo RMS {info['stereo_rms_px']:.3f} px, "
          f"baseline {base:.3f} m, ground: "
          f"{'reference board (operator-confirmed)' if args.board_on_ground else 'none'}"
          f" -> {args.out} (status estimated)")


def check_distance(args):
    from utils.triangulation import triangulate
    doc, cams, _ = cal.load_calibration(args.calibration)
    ids = list(cams)
    if len(ids) != 2:
        raise SystemExit("check-distance needs a two-camera calibration")
    pa = [tuple(float(v) for v in p.split(",")) for p in args.a]
    pb = [tuple(float(v) for v in p.split(",")) for p in args.b]
    ends = []
    for k in range(2):
        p = triangulate([cams[ids[0]], cams[ids[1]]], [pa[k], pb[k]])
        ends.append(p)
        print(f"end {k + 1}: reprojection {p.reprojection_px.round(2).tolist()} px, "
              f"1-sigma {p.sigma * 1000:.1f} mm")
    d = float(np.linalg.norm(ends[0].xyz - ends[1].xyz))
    err_mm = (d - args.known_m) * 1000
    ok = abs(err_mm) <= args.tolerance_mm
    print(f"measured {d:.4f} m vs known {args.known_m:.4f} m: error {err_mm:+.1f} mm "
          f"({'within' if ok else 'OUTSIDE'} ±{args.tolerance_mm} mm)")
    if ok and args.by and args.out:
        from utils.catalog import now_iso
        doc = {**doc, "version": doc["version"] + 1,
               "status": {"status": "verified", "by": args.by, "at": now_iso(),
                          "method": f"known distance {args.known_m} m measured as {d:.4f} m "
                                    f"({err_mm:+.1f} mm, tolerance {args.tolerance_mm} mm)",
                          "notes": None}}
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote verified calibration version {doc['version']} -> {args.out}")
    return 0 if ok else 1


def _frame_indexes(cat, shot):
    """{stream_id: FrameIndex} for the shot's sources, refusing changed bytes."""
    from utils import catalog as cm
    from utils.frame_access import get_frame_index
    out = {}
    for st in shot["streams"]:
        status = cm.resolve_source(cat, st["source_id"])
        if not status.usable:
            raise SystemExit(f"source {st['source_id']} is {status.state}: {status.detail}")
        out[st["stream_id"]] = get_frame_index(status.path)
    return out


def reconstruct(args):
    from utils import catalog as cm
    from utils import club_3d
    from utils import reconstruct_3d as r3
    from utils import reconstruction_io as rio
    from utils import shot_annotations as sa
    from utils.alignment_3d import p1_alignment
    cat = cm.load_catalog()
    shot = cat.shot(args.shot)
    annotations = sa.load(args.shot)
    calibration = cal.load_calibration(args.calibration)
    try:
        rec = r3.reconstruct_shot(cat, args.shot, annotations, calibration,
                                  indexes=_frame_indexes(cat, shot))
    except r3.StereoInputError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    d = rio.write(rec, args.out)
    m, _ = rio.load(d)
    hand = shot["handedness"] if shot["handedness"] in ("right", "left") else None
    if hand is None:
        print("handedness unknown: alignment and club measurements need it", file=sys.stderr)
    else:
        impact = club_3d.impact_time_ms(annotations, rec.streams[0].stream_id)
        club = club_3d.club_metrics(rec, impact, hand)
        doc = club_3d.measurements_document(rec, rio.manifest_sha256(m),
                                            m["reconstruction_id"], hand, impact,
                                            p1_alignment(rec, hand), club)
        from utils.catalog import atomic_write_json
        atomic_write_json(os.path.join(d, "measurements.json"), doc)
    s = m["summary"]
    print(f"{m['reconstruction_id']}: {s['frames']} frames ({s['frames_measured']} measured), "
          f"points {s['points']}, world frame {m['world_frame']['status']} -> {d}")
    for r in m["world_frame"]["reasons"]:
        print(f"  world frame: {r}")
    for n in m["association"]["notes"]:
        print(f"  {n}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("calibrate-intrinsics")
    i.add_argument("--camera", required=True, help="the shot stream id, e.g. down_the_line")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", nargs="+")
    i.add_argument("--every", type=int, default=10)
    i.add_argument("--board", required=True, help="inner corners, e.g. 9x6")
    i.add_argument("--square-mm", type=float, required=True)
    i.add_argument("--out", required=True)
    e = sub.add_parser("calibrate-extrinsics")
    e.add_argument("--intrinsics", nargs=2, required=True)
    e.add_argument("--video-a")
    e.add_argument("--video-b")
    e.add_argument("--offset-frames", type=int, default=0,
                   help="camera B frame index of camera A's frame 0 (synchronized capture)")
    e.add_argument("--images-a")
    e.add_argument("--images-b")
    e.add_argument("--every", type=int, default=5)
    e.add_argument("--board", required=True)
    e.add_argument("--square-mm", type=float, required=True)
    e.add_argument("--reference-pair", type=int, default=0,
                   help="which shared view defines the rig frame (-1 = last)")
    e.add_argument("--board-on-ground", action="store_true",
                   help="the reference board was lying flat on the ground (operator-confirmed)")
    e.add_argument("--calibration-id", required=True)
    e.add_argument("--version", type=int, default=1)
    e.add_argument("--by", required=True)
    e.add_argument("--out", required=True)
    c = sub.add_parser("check-distance")
    c.add_argument("--calibration", required=True)
    c.add_argument("--a", nargs=2, required=True, help="x,y of both ends in camera A")
    c.add_argument("--b", nargs=2, required=True, help="x,y of both ends in camera B")
    c.add_argument("--known-m", type=float, required=True)
    c.add_argument("--tolerance-mm", type=float, default=5.0)
    c.add_argument("--by", help="with --out: write a verified version if within tolerance")
    c.add_argument("--out")
    r = sub.add_parser("reconstruct")
    r.add_argument("--shot", required=True)
    r.add_argument("--calibration", required=True)
    r.add_argument("--out", help="artifact root (default <output>/reconstructions)")
    args = p.parse_args(argv)
    if args.cmd == "calibrate-extrinsics" and not (args.video_a and args.video_b) and not (
            args.images_a and args.images_b):
        p.error("give --video-a/--video-b or --images-a/--images-b")
    try:
        return {"calibrate-intrinsics": calibrate_intrinsics,
                "calibrate-extrinsics": calibrate_extrinsics,
                "check-distance": check_distance, "reconstruct": reconstruct}[args.cmd](args) or 0
    except cal.CalibrationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
