"""2D joint angles and body-relative displacements from one camera view.

Everything here is measured **in the image plane** of a single camera. These
are projected quantities, not anatomical 3D joint angles: a 2D elbow angle
from face-on and the same elbow from down-the-line are different numbers, and
neither is the true elbow flexion. Results are therefore always tagged with
their view, and only like-for-like views are ever compared.

Not measured, because two 2D cameras can't support them: weight distribution,
true pelvis/shoulder rotation, anything needing depth.

Conventions
-----------
- Input points are pixel (x, y) with y pointing down, keyed by landmark name.
- "Up" is image-up. Vertical components are reported up-positive.
- Horizontal components are made body-relative, so a reference filmed from the
  opposite side (or mirrored) still compares sign-for-sign:
    face-on: + = toward the target (the lead hip's side at P1)
    DTL:     + = toward the ball   (the side the torso bends toward at P1, so
                                    DTL spine inclination is positive at address)
- Displacements are from the same video's P1 and divided by the P1 torso
  length (shoulder-centre to hip-centre). The scale is fixed at P1 -- it is
  not recomputed per frame.
- Missing landmarks or degenerate geometry give `value=None` with a reason.
  No value is ever filled in.
"""

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from utils.landmarks import sided

# A segment shorter than this many pixels has no direction.
MIN_SEGMENT_PX = 1e-6
# Shoulder tilt is undefined when the shoulders nearly overlap in this
# projection (typical down-the-line): width below this fraction of torso length.
MIN_SHOULDER_WIDTH_RATIO = 0.2
# The P1 cue that orients the horizontal axis must be at least this far apart,
# as a fraction of torso length, or the direction is not trusted.
MIN_ORIENTATION_RATIO = 0.05

ANGLE_UNIT = "deg (2D)"
DISPLACEMENT_UNIT = "torso lengths"


@dataclass(frozen=True)
class Measurement:
    key: str
    label: str
    view: str
    unit: str
    value: Optional[float]
    reason: Optional[str] = None        # why value is None
    joints: tuple = ()                  # landmark names involved, for highlighting


@dataclass(frozen=True)
class BodyReference:
    """Per-video, per-view constants established at P1."""
    view: str
    handedness: str
    scale_px: float                         # P1 torso length
    horizontal_sign: Optional[float]        # multiplies image dx; None if unknown
    anchors: dict = field(default_factory=dict)   # P1 positions of tracked centres
    orientation_reason: Optional[str] = None


@dataclass(frozen=True)
class ComparisonRow:
    key: str
    label: str
    view: str
    unit: str
    user: Optional[float]
    reference: Optional[float]
    difference: Optional[float]   # user - reference
    note: Optional[str] = None


# --- geometry -------------------------------------------------------------

def _pt(points, name):
    p = points.get(name)
    if p is None:
        return None
    p = np.asarray(p, dtype=float)
    return None if np.isnan(p).any() else p


def _mid(points, a, b):
    pa, pb = _pt(points, a), _pt(points, b)
    if pa is None or pb is None:
        return None
    return (pa + pb) / 2.0


def included_angle(a, vertex, b):
    """Angle a-vertex-b in degrees, or None if either arm has zero length."""
    u = np.asarray(a, float) - np.asarray(vertex, float)
    v = np.asarray(b, float) - np.asarray(vertex, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < MIN_SEGMENT_PX or nv < MIN_SEGMENT_PX:
        return None
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def elevation(frm, to):
    """Angle of frm->to above image-horizontal, in [-90, 90]. None if zero-length."""
    d = np.asarray(to, float) - np.asarray(frm, float)
    if np.linalg.norm(d) < MIN_SEGMENT_PX:
        return None
    return math.degrees(math.atan2(-d[1], abs(d[0])))


def _centres(points, handedness):
    lead_wrist = _pt(points, sided("lead", "wrist", handedness))
    trail_wrist = _pt(points, sided("trail", "wrist", handedness))
    hands = None
    if lead_wrist is not None and trail_wrist is not None:
        hands = (lead_wrist + trail_wrist) / 2.0
    return {
        "head": _mid(points, "left_ear", "right_ear"),
        "shoulders": _mid(points, "left_shoulder", "right_shoulder"),
        "hips": _mid(points, "left_hip", "right_hip"),
        "lead_wrist": lead_wrist,
        "trail_wrist": trail_wrist,
        "hands": hands,
    }


def _torso(points):
    s = _mid(points, "left_shoulder", "right_shoulder")
    h = _mid(points, "left_hip", "right_hip")
    if s is None or h is None:
        return None
    length = float(np.linalg.norm(s - h))
    return length if length >= MIN_SEGMENT_PX else None


# --- P1 reference ---------------------------------------------------------

def body_reference(p1_points, view, handedness):
    """Establish scale and orientation from the P1 frame.

    Returns (BodyReference, None) or (None, reason).
    """
    scale = _torso(p1_points)
    if scale is None:
        return None, "P1 torso (both shoulders and both hips) not detected"

    sign, why = None, None
    if view == "face_on":
        lead = _pt(p1_points, sided("lead", "hip", handedness))
        trail = _pt(p1_points, sided("trail", "hip", handedness))
        if lead is None or trail is None:
            why = "P1 hips not detected"
        elif abs(lead[0] - trail[0]) < MIN_ORIENTATION_RATIO * scale:
            why = "P1 hips overlap horizontally; target direction unknown"
        else:
            sign = 1.0 if lead[0] > trail[0] else -1.0
    elif view == "dtl":
        # Address forward bend: the shoulders sit over the ball side of the hips.
        # Uses only the torso (already required for scale) -- the lead arm is
        # usually occluded from down-the-line at P1.
        s = _mid(p1_points, "left_shoulder", "right_shoulder")
        h = _mid(p1_points, "left_hip", "right_hip")
        if abs(s[0] - h[0]) < MIN_ORIENTATION_RATIO * scale:
            why = "no forward bend at P1; ball direction unknown"
        else:
            sign = 1.0 if s[0] > h[0] else -1.0
    else:
        raise ValueError(f"unknown view {view!r}")

    anchors = {k: v for k, v in _centres(p1_points, handedness).items() if v is not None}
    return BodyReference(view=view, handedness=handedness, scale_px=scale,
                         horizontal_sign=sign, anchors=anchors,
                         orientation_reason=why), None


# --- measurements ---------------------------------------------------------

def _horizontal_word(view):
    return "toward target" if view == "face_on" else "toward ball"


def measurement_catalog(view, handedness="right"):
    """(key, label, joints) for every measurement, in display order."""
    lead = lambda j: sided("lead", j, handedness)    # noqa: E731
    trail = lambda j: sided("trail", j, handedness)  # noqa: E731
    h = _horizontal_word(view)
    torso = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    cat = [
        ("lead_elbow_flexion", "Lead elbow flexion",
         (lead("shoulder"), lead("elbow"), lead("wrist"))),
        ("trail_elbow_flexion", "Trail elbow flexion",
         (trail("shoulder"), trail("elbow"), trail("wrist"))),
        ("lead_knee_flexion", "Lead knee flexion",
         (lead("hip"), lead("knee"), lead("ankle"))),
        ("trail_knee_flexion", "Trail knee flexion",
         (trail("hip"), trail("knee"), trail("ankle"))),
        ("spine_inclination", f"Spine inclination from vertical (+ {h})", torso),
        ("shoulder_tilt", "Shoulder tilt (+ lead shoulder higher)",
         ("left_shoulder", "right_shoulder")),
        ("lead_arm_inclination", "Lead arm above horizontal",
         (lead("shoulder"), lead("wrist"))),
    ]
    disp = [
        ("head", "Head centre", ("left_ear", "right_ear")),
        ("shoulders", "Shoulder centre", ("left_shoulder", "right_shoulder")),
        ("hips", "Hip centre", ("left_hip", "right_hip")),
        ("lead_wrist", "Lead wrist", (lead("wrist"),)),
        ("trail_wrist", "Trail wrist", (trail("wrist"),)),
        ("hands", "Hand centre", (lead("wrist"), trail("wrist"))),
    ]
    for key, label, joints in disp:
        cat.append((f"{key}_dx", f"{label} Δ from P1, horizontal (+ {h})", joints))
        cat.append((f"{key}_dy", f"{label} Δ from P1, vertical (+ up)", joints))
    return cat


def joints_for(key, view, handedness):
    """Landmarks a measurement uses, for highlighting in a given panel."""
    for k, _, joints in measurement_catalog(view, handedness):
        if k == key:
            return joints
    raise KeyError(key)


def measure(points, view, handedness, reference: Optional[BodyReference],
            reference_reason: Optional[str] = None):
    """All measurements for one frame of one view."""
    out = []
    catalog = {k: (label, joints) for k, label, joints in measurement_catalog(view, handedness)}

    def emit(key, value, reason=None, unit=ANGLE_UNIT):
        label, joints = catalog[key]
        if value is not None and not math.isfinite(value):
            value, reason = None, "non-finite result"
        out.append(Measurement(key=key, label=label, view=view, unit=unit,
                               value=value, reason=None if value is not None else reason,
                               joints=joints))

    def flexion(key):
        a, v, b = (_pt(points, j) for j in catalog[key][1])
        if a is None or v is None or b is None:
            return emit(key, None, "landmark not detected")
        ang = included_angle(a, v, b)
        if ang is None:
            return emit(key, None, "zero-length segment")
        emit(key, 180.0 - ang)

    for key in ("lead_elbow_flexion", "trail_elbow_flexion",
                "lead_knee_flexion", "trail_knee_flexion"):
        flexion(key)

    # Spine inclination: hip centre -> shoulder centre, signed from vertical.
    s = _mid(points, "left_shoulder", "right_shoulder")
    hc = _mid(points, "left_hip", "right_hip")
    if s is None or hc is None:
        emit("spine_inclination", None, "landmark not detected")
    elif reference is None or reference.horizontal_sign is None:
        emit("spine_inclination", None,
             (reference.orientation_reason if reference else reference_reason)
             or "no P1 reference")
    else:
        d = s - hc
        if np.linalg.norm(d) < MIN_SEGMENT_PX:
            emit("spine_inclination", None, "zero-length segment")
        else:
            emit("spine_inclination",
                 math.degrees(math.atan2(reference.horizontal_sign * d[0], -d[1])))

    # Shoulder tilt: elevation of the lead shoulder over the trail shoulder.
    ls = _pt(points, sided("lead", "shoulder", handedness))
    ts = _pt(points, sided("trail", "shoulder", handedness))
    torso = _torso(points)
    if ls is None or ts is None:
        emit("shoulder_tilt", None, "landmark not detected")
    elif torso is not None and np.linalg.norm(ls - ts) < MIN_SHOULDER_WIDTH_RATIO * torso:
        emit("shoulder_tilt", None, "shoulders overlap in this projection")
    else:
        emit("shoulder_tilt", elevation(ts, ls), "zero-length segment")

    # Lead arm: elevation of lead wrist over lead shoulder.
    lw = _pt(points, sided("lead", "wrist", handedness))
    if ls is None or lw is None:
        emit("lead_arm_inclination", None, "landmark not detected")
    else:
        emit("lead_arm_inclination", elevation(ls, lw), "zero-length segment")

    # Displacements from P1, in P1 torso lengths.
    centres = _centres(points, handedness)
    for name in ("head", "shoulders", "hips", "lead_wrist", "trail_wrist", "hands"):
        cur = centres[name]
        if reference is None:
            why = reference_reason or "no P1 reference"
            emit(f"{name}_dx", None, why, DISPLACEMENT_UNIT)
            emit(f"{name}_dy", None, why, DISPLACEMENT_UNIT)
            continue
        origin = reference.anchors.get(name)
        if cur is None or origin is None:
            why = "landmark not detected" if cur is None else "not detected at P1"
            emit(f"{name}_dx", None, why, DISPLACEMENT_UNIT)
            emit(f"{name}_dy", None, why, DISPLACEMENT_UNIT)
            continue
        d = (cur - origin) / reference.scale_px
        if reference.horizontal_sign is None:
            emit(f"{name}_dx", None, reference.orientation_reason, DISPLACEMENT_UNIT)
        else:
            emit(f"{name}_dx", float(reference.horizontal_sign * d[0]), unit=DISPLACEMENT_UNIT)
        emit(f"{name}_dy", float(-d[1]), unit=DISPLACEMENT_UNIT)

    return out


def compare(user, reference):
    """Pair two measurement lists from the SAME view. difference = user - reference."""
    ref_by_key = {m.key: m for m in reference}
    rows = []
    for u in user:
        r = ref_by_key.get(u.key)
        if r is None:
            continue
        if u.view != r.view:
            raise ValueError(f"refusing to compare {u.view} against {r.view} for {u.key}")
        diff = None
        if u.value is not None and r.value is not None:
            diff = u.value - r.value
        notes = []
        if u.value is None:
            notes.append(f"yours: {u.reason}")
        if r.value is None:
            notes.append(f"reference: {r.reason}")
        rows.append(ComparisonRow(key=u.key, label=u.label, view=u.view, unit=u.unit,
                                  user=u.value, reference=r.value, difference=diff,
                                  note="; ".join(notes) or None))
    return rows
