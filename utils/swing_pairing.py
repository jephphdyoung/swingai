"""Pair Swing Catalyst captures into down-the-line / face-on swings.

Swing Catalyst names every clip:

    Jeff Young - 2026-07-27 135006 Fox Down the line 61fc.mp4
    <golfer>   - <date>     <time> <camera label>     <camera id>

The **timestamp identifies the swing** and the **camera id identifies the
view**, so the two clips of one swing share a timestamp and differ only in
camera. Both cameras stamp the same trigger time, which is what makes the two
views directly comparable on a shared clock.

The camera id is the reliable view discriminator -- it is a fixed per-camera
value, where the human-readable label is free text that can be renamed.
"""

from dataclasses import dataclass
from typing import Optional
import os
import re

# <golfer> - <YYYY-MM-DD> <HHMMSS> <camera label> <camera id>
CAPTURE_RE = re.compile(
    r"^(?P<golfer>.+?) - "
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{6}) "
    r"(?P<camera>.+?) "
    r"(?P<camera_id>[0-9a-fA-F]{4})$"
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# Matched against the camera label, lowercased.
VIEW_BY_LABEL = (
    ("down the line", "dtl"),
    ("downtheline", "dtl"),
    ("dtl", "dtl"),
    ("face on", "face_on"),
    ("faceon", "face_on"),
    ("front", "face_on"),
)


@dataclass(frozen=True)
class Capture:
    """One clip, as described by its filename."""
    path: str
    golfer: str
    date: str
    time: str
    camera: str
    camera_id: str
    view: Optional[str]        # "dtl" | "face_on" | None when the label is unfamiliar

    @property
    def swing_key(self):
        """Identifies the swing: same golfer, same trigger instant."""
        return (self.golfer, self.date, self.time)

    @property
    def captured_at(self):
        return f"{self.date} {self.time}"


@dataclass(frozen=True)
class SwingPair:
    """Both views of a single swing."""
    dtl: Capture
    face_on: Capture

    @property
    def captured_at(self):
        return self.dtl.captured_at

    @property
    def golfer(self):
        return self.dtl.golfer


def view_from_label(camera_label):
    """Map a camera label to a view angle, or None if unrecognised."""
    lowered = camera_label.lower()
    for needle, view in VIEW_BY_LABEL:
        if needle in lowered:
            return view
    return None


def parse_capture(path):
    """Parse a capture filename, or return None if it doesn't match."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = CAPTURE_RE.match(stem)
    if not match:
        return None
    return Capture(
        path=path,
        golfer=match.group("golfer"),
        date=match.group("date"),
        time=match.group("time"),
        camera=match.group("camera"),
        camera_id=match.group("camera_id").lower(),
        view=view_from_label(match.group("camera")),
    )


def list_captures(directory, recursive=True):
    """Every parseable capture under a directory."""
    captures = []
    if not os.path.isdir(directory):
        return captures
    walker = os.walk(directory) if recursive else [
        (directory, [], os.listdir(directory))
    ]
    for root, dirs, files in walker:
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if not name.lower().endswith(VIDEO_EXTENSIONS):
                continue
            capture = parse_capture(os.path.join(root, name))
            if capture is not None:
                captures.append(capture)
    return captures


def pair_captures(captures):
    """Group captures into complete DTL + face-on pairs.

    Returns (pairs, unpaired). A swing missing a view, or with an unrecognised
    camera label, lands in `unpaired` rather than being silently dropped.
    """
    by_swing = {}
    for capture in captures:
        by_swing.setdefault(capture.swing_key, []).append(capture)

    pairs, unpaired = [], []
    for key in sorted(by_swing):
        group = by_swing[key]
        dtl = next((c for c in group if c.view == "dtl"), None)
        face_on = next((c for c in group if c.view == "face_on"), None)
        if dtl is not None and face_on is not None:
            pairs.append(SwingPair(dtl=dtl, face_on=face_on))
        else:
            unpaired.extend(group)
    return pairs, unpaired


def find_pairs(directory, recursive=True):
    """Convenience: list and pair every capture under a directory."""
    return pair_captures(list_captures(directory, recursive))


def find_partner(path):
    """Given one clip, find the other view of the same swing beside it.

    Returns a SwingPair, or None when the file is unparseable or its partner is
    missing from the same directory.
    """
    capture = parse_capture(path)
    if capture is None or capture.view is None:
        return None

    directory = os.path.dirname(os.path.abspath(path))
    siblings = list_captures(directory, recursive=False)
    candidates = [
        c for c in siblings
        if c.swing_key == capture.swing_key and c.view and c.view != capture.view
    ]
    if not candidates:
        return None

    partner = candidates[0]
    if capture.view == "dtl":
        return SwingPair(dtl=capture, face_on=partner)
    return SwingPair(dtl=partner, face_on=capture)
