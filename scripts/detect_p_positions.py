#!/usr/bin/env python3
"""Detect P1-P10 from a down-the-line / face-on video pair.

Swing Catalyst filenames carry a timestamp (identifying the swing) and a camera
id (identifying the view), so a pair can usually be resolved automatically:

    # one clip -- its partner is found alongside it
    python scripts/detect_p_positions.py --swing "Jeff Young - 2026-07-27 135006 Fox Down the line 61fc.mp4"

    # every swing in a session
    python scripts/detect_p_positions.py --session "~/Golf/Videos/2026-07-27/SwingCatalyst"

    # explicit pair, for footage that isn't named that way
    python scripts/detect_p_positions.py --dtl DTL.mp4 --face-on FACE.mp4

Writes a P-position file (JSON) per swing. With --save-annotations the positions
are also registered in data/annotations.json so the Streamlit UI picks them up.

See docs/p-position-detection.md for what is being detected and why.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.p_positions import detect_from_pair, ORDER
from utils.paths import get_output_dir
from utils.swing_pairing import SwingPair, find_pairs, find_partner, parse_capture


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect P-positions from a DTL + face-on video pair.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--swing", metavar="VIDEO",
                        help="One clip of a swing; its partner view is found alongside it")
    source.add_argument("--session", metavar="DIR",
                        help="Directory of captures; every complete pair is processed")
    source.add_argument("--dtl", help="Down-the-line video (use with --face-on)")

    parser.add_argument("--face-on", dest="face_on",
                        help="Face-on video of the same swing (use with --dtl)")
    parser.add_argument("--out", help="Output JSON path (single swing only)")
    parser.add_argument("--out-dir", help="Directory for output files "
                                          "(default: the configured output dir)")
    parser.add_argument("--handedness", choices=["right", "left"], default="right",
                        help="Golfer's handedness (default: right)")
    parser.add_argument("--save-annotations", action="store_true",
                        help="Also write positions into data/annotations.json")
    parser.add_argument("--limit", type=int,
                        help="Process at most N swings (with --session)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-swing summary table")

    args = parser.parse_args(argv)
    if args.dtl and not args.face_on:
        parser.error("--dtl requires --face-on")
    if args.face_on and not args.dtl:
        parser.error("--face-on requires --dtl")
    if args.out and args.session:
        parser.error("--out applies to a single swing; use --out-dir with --session")
    return args


def resolve_pairs(args):
    """Work out which swings to process. Returns (pairs, error_message)."""
    if args.dtl:
        for label, path in (("--dtl", args.dtl), ("--face-on", args.face_on)):
            if not os.path.isfile(path):
                return [], f"{label} not found: {path}"
        dtl = parse_capture(args.dtl)
        face_on = parse_capture(args.face_on)
        if dtl and face_on and dtl.swing_key != face_on.swing_key:
            print(f"warning: {args.dtl} and {args.face_on} carry different capture "
                  f"timestamps -- they may not be the same swing", file=sys.stderr)
        # Explicit paths win regardless of naming.
        return [(args.dtl, args.face_on, None)], None

    if args.swing:
        path = os.path.expanduser(args.swing)
        if not os.path.isfile(path):
            return [], f"--swing not found: {path}"
        pair = find_partner(path)
        if pair is None:
            return [], (f"could not find the partner view for {path}. "
                        f"Pass --dtl and --face-on explicitly.")
        return [(pair.dtl.path, pair.face_on.path, pair)], None

    directory = os.path.expanduser(args.session)
    if not os.path.isdir(directory):
        return [], f"--session not a directory: {directory}"
    pairs, unpaired = find_pairs(directory)
    if unpaired:
        print(f"warning: {len(unpaired)} clip(s) had no matching partner view "
              f"and were skipped", file=sys.stderr)
    if not pairs:
        return [], f"no complete DTL/face-on pairs found under {directory}"
    if args.limit:
        pairs = pairs[:args.limit]
    return [(p.dtl.path, p.face_on.path, p) for p in pairs], None


def output_path_for(args, face_on_path, pair):
    if args.out:
        return args.out
    directory = args.out_dir or get_output_dir()
    if pair is not None:
        stem = f"{pair.golfer} - {pair.captured_at}".replace(" ", "_")
    else:
        stem = Path(face_on_path).stem
    return os.path.join(directory, f"{stem}.p-positions.json")


def print_summary(result):
    positions = result["positions"]
    print()
    print(f"  {'':5}{'timestamp':>12}  {'conf':>5}  {'view':<8} method")
    print(f"  {'-' * 53}")
    for name in ORDER:
        if name not in positions:
            print(f"  {name:5}{'not detected':>12}")
            continue
        p = positions[name]
        print(f"  {name:5}{p['timestamp_ms']:>10.1f}ms  {p['confidence']:>5.2f}  "
              f"{p['detected_in']:<8} {p['method']}")

    offset = result["views"]["dtl"]["offset_ms"]
    print()
    if offset is None:
        print("  views not aligned (impact missing in one view)")
    else:
        print(f"  impact-alignment check: down-the-line clock differs by "
              f"{offset:+.1f}ms from face-on")

    for w in result["warnings"]:
        print(f"  ! {w}")


def save_to_annotations(result):
    from utils.annotations import save_annotations

    views = result["views"]
    written = []
    for view_name in ("face_on", "dtl"):
        frames = {
            name: p["frames"][view_name]
            for name, p in result["positions"].items()
            if p["frames"].get(view_name) is not None
        }
        if not frames:
            continue
        save_annotations(views[view_name]["path"], frames,
                         views[view_name]["fps"], view_angle=view_name)
        written.append(views[view_name]["path"])
    return written


def main(argv=None):
    args = parse_args(argv)

    jobs, error = resolve_pairs(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"{len(jobs)} swing(s) to process ({args.handedness}-handed)")
    succeeded = 0

    for index, (dtl_path, face_on_path, pair) in enumerate(jobs, start=1):
        label = pair.captured_at if pair else Path(face_on_path).stem
        print(f"\n[{index}/{len(jobs)}] {label}")

        try:
            result = detect_from_pair(dtl_path, face_on_path,
                                      handedness=args.handedness)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        if pair is not None:
            result["swing"] = {
                "golfer": pair.golfer,
                "captured_at": pair.captured_at,
                "cameras": {
                    "dtl": pair.dtl.camera_id,
                    "face_on": pair.face_on.camera_id,
                },
            }

        out_path = output_path_for(args, face_on_path, pair)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        if not args.quiet:
            print_summary(result)
        print(f"  wrote {out_path}")

        if args.save_annotations:
            for path in save_to_annotations(result):
                print(f"  annotated {os.path.basename(path)}")

        if result["positions"]:
            succeeded += 1

    print(f"\n{succeeded}/{len(jobs)} swing(s) produced positions")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
