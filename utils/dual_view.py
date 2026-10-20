"""Offline dual-view comparison: your DTL + face-on against a reference's.

Four videos, one P-position at a time. Each video carries its own saved
annotations, frame rate and timeline; nothing assumes the four share frame
numbers. The reference DTL and face-on may even be different swings.

Kept free of Streamlit so the whole pipeline is testable; `dual_view_tab.py`
is the UI over it.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from utils.annotations import AnnotationRecord, convention_warnings
from utils.joint_measurements import (
    BodyReference, ComparisonRow, Measurement, body_reference, compare, joints_for, measure,
)
from utils.landmarks import SKELETON_SEGMENTS, VIEW_LABELS
from utils.pose_data import DEFAULT_MIN_CONFIDENCE, PoseSequence, nearest_frame

ROLES = ("user", "reference")
ROLE_LABELS = {"user": "Your swing", "reference": "Reference swing"}


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    def timestamps_ms(self):
        return np.arange(self.frame_count) * 1000.0 / self.fps


def probe_video(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        return VideoInfo(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                         frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


@dataclass
class Slot:
    """One of the four panels."""
    role: str                    # "user" | "reference"
    view: str                    # "dtl" | "face_on"
    path: str
    info: Optional[VideoInfo]
    record: Optional[AnnotationRecord]
    handedness: str = "right"
    pose: Optional[PoseSequence] = None

    @property
    def label(self):
        return f"{ROLE_LABELS[self.role]} — {VIEW_LABELS[self.view]}"

    def timestamps_ms(self):
        if self.pose is not None:
            return self.pose.timestamps_ms
        return self.info.timestamps_ms() if self.info else np.array([])

    @property
    def frame_count(self):
        if self.pose is not None:
            return self.pose.frame_count
        return self.info.frame_count if self.info else 0


@dataclass
class PositionFrame:
    position: str
    frame: Optional[int]
    timestamp_ms: Optional[float]
    source: Optional[str]        # "annotation" | "annotation timestamp" | "partner view"
    note: Optional[str] = None


def resolve_position(slot: Slot, position: str, partner: Optional[Slot] = None):
    """Frame of `position` in this slot's own video, from its own annotations.

    Within a video the saved frame index is exact (it is the frame that was
    looked at when marking). The saved timestamp is used only when there is no
    usable frame. With `partner` -- the other view of a co-triggered capture --
    a position missing here is taken at the same instant on the shared clock.
    Nothing else fills a gap.
    """
    ts = slot.timestamps_ms()
    n = len(ts)
    rec = slot.record
    if rec is not None and position in rec.frames:
        f = rec.frames[position]
        if 0 <= f < n:
            return PositionFrame(position, f, float(ts[f]), "annotation")
        note = f"saved frame {f} is outside this video (0–{n - 1})"
        if position in rec.timestamps_ms and n:
            f = nearest_frame(ts, rec.timestamps_ms[position])
            return PositionFrame(position, f, float(ts[f]), "annotation timestamp", note)
        return PositionFrame(position, None, None, None, note)
    if rec is not None and position in rec.timestamps_ms and n:
        f = nearest_frame(ts, rec.timestamps_ms[position])
        return PositionFrame(position, f, float(ts[f]), "annotation timestamp")

    if partner is not None:
        pf = resolve_position(partner, position)
        if pf.frame is not None and pf.source == "annotation" and n:
            f = nearest_frame(ts, pf.timestamp_ms)
            return PositionFrame(position, f, float(ts[f]), "partner view",
                                 f"{position} not annotated here; taken from the "
                                 f"{VIEW_LABELS[partner.view]} annotation at "
                                 f"{pf.timestamp_ms:.1f} ms on the shared trigger clock")
    return PositionFrame(position, None, None, None, f"{position} not annotated")


def available_positions(slot: Slot):
    rec = slot.record
    if rec is None:
        return set()
    return set(rec.frames) | set(rec.timestamps_ms)


@dataclass
class SlotResult:
    slot: Slot
    position: PositionFrame
    p1: PositionFrame
    measurements: list = field(default_factory=list)
    reference: Optional[BodyReference] = None
    warnings: list = field(default_factory=list)


def measure_slot(slot: Slot, position: str, partner: Optional[Slot] = None,
                 min_confidence=DEFAULT_MIN_CONFIDENCE):
    """Resolve the frames and compute this slot's measurements."""
    pf = resolve_position(slot, position, partner)
    p1 = resolve_position(slot, "P1", partner)
    result = SlotResult(slot=slot, position=pf, p1=p1)
    result.warnings.extend(convention_warnings(slot.record))
    if slot.record and slot.record.view_angle and slot.record.view_angle != slot.view:
        result.warnings.append(
            f"Annotations say this clip is {slot.record.view_angle}, "
            f"but it is in the {VIEW_LABELS[slot.view]} slot.")

    if slot.pose is None:
        result.measurements = _unavailable(slot, "pose not extracted")
        return result
    if pf.frame is None:
        result.measurements = _unavailable(slot, pf.note or f"{position} not annotated")
        return result

    ref, why = None, "P1 not annotated"
    if p1.frame is not None:
        ref, why = body_reference(slot.pose.points(p1.frame, min_confidence),
                                  slot.view, slot.handedness)
    if not slot.pose.detected[pf.frame]:
        result.measurements = _unavailable(slot, "no pose detected in this frame")
        result.reference = ref
        return result
    result.reference = ref
    result.measurements = measure(slot.pose.points(pf.frame, min_confidence),
                                  slot.view, slot.handedness, ref, why)
    return result


def _unavailable(slot, reason):
    return [Measurement(m.key, m.label, m.view, m.unit, None, reason)
            for m in measure({}, slot.view, slot.handedness, None, reason)]


@dataclass
class Comparison:
    position: str
    slots: dict                  # (role, view) -> SlotResult
    rows: dict                   # view -> [ComparisonRow]


def build_comparison(slots: dict, position: str, use_partner_view=False,
                     min_confidence=DEFAULT_MIN_CONFIDENCE):
    """slots: {(role, view): Slot} for the four panels.

    `use_partner_view` lets *your* two views fill each other's missing
    positions through the shared capture clock. It is never applied to the
    reference, whose two views need not be the same swing.
    """
    results = {}
    for (role, view), slot in slots.items():
        partner = None
        if use_partner_view and role == "user":
            other = "face_on" if view == "dtl" else "dtl"
            partner = slots.get((role, other))
        results[(role, view)] = measure_slot(slot, position, partner, min_confidence)

    rows = {}
    for view in ("dtl", "face_on"):
        u, r = results.get(("user", view)), results.get(("reference", view))
        if u is not None and r is not None:
            rows[view] = compare(u.measurements, r.measurements)
    return Comparison(position=position, slots=results, rows=rows)


# --- frames and drawing ---------------------------------------------------

def read_frame(path, frame_idx):
    """BGR frame by index, or None. Same seek the annotator uses."""
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


SKELETON_COLOR = (230, 230, 230)
JOINT_COLOR = (0, 200, 255)
LOW_CONF_COLOR = (90, 90, 90)
HIGHLIGHT_COLOR = (60, 60, 255)


def draw_pose(frame_bgr, pose: PoseSequence, frame_idx, highlight=(), show_skeleton=True,
              min_confidence=DEFAULT_MIN_CONFIDENCE):
    """Skeleton over a frame. Only validly observed landmarks are drawn solid;
    low-confidence ones are drawn hollow and grey, undetected frames get none.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    t = max(1, round(min(w, h) / 360))
    if not show_skeleton or pose is None or not 0 <= frame_idx < pose.frame_count:
        return out
    if not pose.detected[frame_idx]:
        cv2.putText(out, "no pose detected", (10 * t, 24 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * t, HIGHLIGHT_COLOR, t, cv2.LINE_AA)
        return out

    valid = pose.points(frame_idx, min_confidence)
    highlight = set(highlight)

    def px(p):
        return int(round(p[0])), int(round(p[1]))

    for a, b in SKELETON_SEGMENTS:
        if a in valid and b in valid:
            hot = a in highlight and b in highlight
            cv2.line(out, px(valid[a]), px(valid[b]),
                     HIGHLIGHT_COLOR if hot else SKELETON_COLOR, 3 * t if hot else t,
                     cv2.LINE_AA)
    from utils.landmarks import LANDMARK_INDEX
    for name, j in LANDMARK_INDEX.items():
        if name in valid:
            hot = name in highlight
            cv2.circle(out, px(valid[name]), (5 if hot else 3) * t,
                       HIGHLIGHT_COLOR if hot else JOINT_COLOR, -1, cv2.LINE_AA)
        elif pose.detected[frame_idx] and not np.isnan(pose.xy[frame_idx, j]).any():
            cv2.circle(out, px(pose.xy[frame_idx, j]), 3 * t, LOW_CONF_COLOR, t, cv2.LINE_AA)
    return out


def render_panel(result: SlotResult, highlight_key=None, show_skeleton=True,
                 max_width=960, min_confidence=DEFAULT_MIN_CONFIDENCE):
    """RGB image for one panel at the resolved frame, aspect ratio preserved."""
    slot = result.slot
    if result.position.frame is None:
        return None
    frame = read_frame(slot.path, result.position.frame)
    if frame is None:
        return None
    highlight = joints_for(highlight_key, slot.view, slot.handedness) if highlight_key else ()
    frame = draw_pose(frame, slot.pose, result.position.frame, highlight,
                      show_skeleton, min_confidence)
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rows_as_table(rows: list[ComparisonRow], decimals_angle=1, decimals_disp=3):
    """Rows as plain dicts for display. Unavailable values are None, not 0."""
    out = []
    for r in rows:
        d = decimals_angle if r.unit.startswith("deg") else decimals_disp
        rnd = lambda v: None if v is None else round(v, d)  # noqa: E731
        out.append({
            "Measurement": r.label,
            "Your value": rnd(r.user),
            "Reference value": rnd(r.reference),
            "Difference (you − ref)": rnd(r.difference),
            "Unit": r.unit,
            "Note": r.note or "",
        })
    return out
