"""Automatic P-position detection from a down-the-line / face-on video pair.

See docs/p-position-detection.md for the model this implements. In short:

- P1, P4, P7, P10 are kinematic — motion onset, hand-path reversal, peak hand
  speed, settle. Detectable in either view.
- P3, P5, P9 are lead-arm geometry — the lead shoulder->wrist vector crossing
  horizontal. Best resolved face-on.
- P2, P6, P8 are shaft geometry — the club shaft crossing horizontal. Best
  resolved down-the-line, and only as good as club tracking on the footage.

Every geometric position is a zero crossing of an angle signal, so crossings are
interpolated between bracketing frames for sub-frame precision, and the slope at
the crossing gives a confidence measure.

The two views are merged by aligning on impact rather than on wall-clock, so the
result does not depend on the two cameras sharing a trigger instant.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import math

from utils.pose_extractor import extract_pose_with_timestamps
from utils.swing_detector import detect_swing_window

# Indices into the selected-landmark list produced by pose_extractor.
NOSE = 0
L_SHOULDER, R_SHOULDER = 1, 2
L_ELBOW, R_ELBOW = 3, 4
L_WRIST, R_WRIST = 5, 6
L_HIP, R_HIP = 13, 14

# A right-handed golfer leads with the left side.
LEAD_SHOULDER = {"right": L_SHOULDER, "left": R_SHOULDER}
LEAD_WRIST = {"right": L_WRIST, "left": R_WRIST}

ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]

# A crossing whose signal barely moves is noise, not a position. Slope is in
# body-scale units per second.
MIN_CROSSING_SLOPE = 0.15

# The lead arm is far from horizontal at the top and at impact, so an
# arm-crossing found right beside either anchor is jitter. Keep this clear of
# the anchors by a margin shorter than any real span between positions.
ARM_SPAN_GUARD_MS = 40.0


def _frames_within(timestamps_ms, span_ms):
    """How many frames span `span_ms`, given a timestamp series."""
    if len(timestamps_ms) < 2:
        return 0
    per_frame = (timestamps_ms[-1] - timestamps_ms[0]) / (len(timestamps_ms) - 1)
    if per_frame <= 0:
        return 0
    return max(1, int(round(span_ms / per_frame)))


@dataclass
class Detection:
    """One located P-position."""
    name: str
    timestamp_ms: float
    frame: int
    view: str
    method: str
    confidence: float


@dataclass
class ViewSignals:
    """Pose-derived signals for a single video."""
    path: str
    view: str
    fps: float
    frame_count: int
    timestamps_ms: list
    landmarks: list
    valid: list                      # False where the extractor carried a frame forward
    lead_arm_elevation: list         # wrist_y - shoulder_y, body-scale units; 0 = arm horizontal
    hand_speed: list                 # body-scale units per second
    swing: object = None             # SwingWindow or None
    warnings: list = field(default_factory=list)


def _valid_frame_mask(landmarks):
    """Flag frames the extractor carried forward from the previous frame.

    pose_extractor repeats the previous frame's landmarks when detection fails,
    to keep frame alignment. Those runs read as exactly zero velocity, which is
    indistinguishable from a genuinely still golfer -- and P1 and P10 are defined
    by stillness. Real pose output is never bit-identical between frames, so an
    exact repeat identifies a carried-forward frame.
    """
    valid = [True] * len(landmarks)
    for i in range(1, len(landmarks)):
        if landmarks[i] == landmarks[i - 1]:
            valid[i] = False
    return valid


def _midpoint(frame, a, b):
    return ((frame[a][0] + frame[b][0]) / 2.0, (frame[a][1] + frame[b][1]) / 2.0)


def _body_scale(frame):
    """Shoulder-centre to hip-centre distance: the natural per-frame length unit.

    Normalising by this makes every threshold independent of how far the golfer
    stands from the camera.
    """
    sx, sy = _midpoint(frame, L_SHOULDER, R_SHOULDER)
    hx, hy = _midpoint(frame, L_HIP, R_HIP)
    scale = math.hypot(sx - hx, sy - hy)
    return scale if scale > 1e-6 else 1e-6


def _lead_arm_elevation(landmarks, handedness):
    """Lead shoulder->wrist height difference, in body-scale units.

    Zero means the lead arm is parallel to the ground. Positive means the wrist
    is below the shoulder (image y grows downward), which is the address side of
    the crossing.
    """
    sh = LEAD_SHOULDER[handedness]
    wr = LEAD_WRIST[handedness]
    return [(f[wr][1] - f[sh][1]) / _body_scale(f) for f in landmarks]


def _hand_speed(landmarks, timestamps_ms, valid):
    """Speed of the hand midpoint, in body-scale units per second.

    Frames the extractor carried forward are skipped rather than reported as
    zero speed.
    """
    n = len(landmarks)
    speed = [0.0] * n
    for i in range(1, n):
        if not valid[i] or not valid[i - 1]:
            speed[i] = speed[i - 1]
            continue
        dt = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        if dt <= 0:
            speed[i] = speed[i - 1]
            continue
        x0, y0 = _midpoint(landmarks[i - 1], L_WRIST, R_WRIST)
        x1, y1 = _midpoint(landmarks[i], L_WRIST, R_WRIST)
        speed[i] = math.hypot(x1 - x0, y1 - y0) / _body_scale(landmarks[i]) / dt
    if n > 1:
        speed[0] = speed[1]
    return speed


def _interpolate_crossing(values, timestamps_ms, i):
    """Sub-frame crossing between frames i-1 and i.

    Returns (timestamp_ms, nearest_frame, slope_per_second).
    """
    v0, v1 = values[i - 1], values[i]
    t0, t1 = timestamps_ms[i - 1], timestamps_ms[i]
    if v1 == v0:
        return t1, i, 0.0
    frac = v0 / (v0 - v1)                       # 0 at frame i-1, 1 at frame i
    ts = t0 + frac * (t1 - t0)
    dt = (t1 - t0) / 1000.0
    slope = abs(v1 - v0) / dt if dt > 0 else 0.0
    return ts, (i if frac >= 0.5 else i - 1), slope


def _find_crossings(values, valid, timestamps_ms, lo, hi):
    """Sign changes of `values` within [lo, hi], strongest-slope first."""
    found = []
    for i in range(max(lo + 1, 1), min(hi + 1, len(values))):
        if not valid[i] or not valid[i - 1]:
            continue
        if (values[i - 1] > 0) != (values[i] > 0):
            found.append(_interpolate_crossing(values, timestamps_ms, i))
    return found


def _pick_crossing(crossings, prefer="first"):
    """Choose one crossing, ignoring any too shallow to be a real position."""
    usable = [c for c in crossings if c[2] >= MIN_CROSSING_SLOPE]
    if not usable:
        return None
    return usable[0] if prefer == "first" else usable[-1]


def _confidence_from_slope(slope):
    """Map crossing steepness to 0-1. A steep crossing is unambiguous in time."""
    return max(0.0, min(1.0, slope / 2.0))


def load_view(path, view, handedness="right"):
    """Extract pose and derive the signals used for detection."""
    result = extract_pose_with_timestamps(path)
    valid = _valid_frame_mask(result.landmarks)

    signals = ViewSignals(
        path=path,
        view=view,
        fps=result.fps,
        frame_count=result.frame_count,
        timestamps_ms=result.timestamps_ms,
        landmarks=result.landmarks,
        valid=valid,
        lead_arm_elevation=_lead_arm_elevation(result.landmarks, handedness),
        hand_speed=_hand_speed(result.landmarks, result.timestamps_ms, valid),
    )

    dropped = valid.count(False)
    if dropped:
        pct = 100.0 * dropped / max(1, len(valid))
        signals.warnings.append(
            f"{view}: {dropped} of {len(valid)} frames ({pct:.0f}%) had no pose "
            f"detection and were carried forward; excluded from velocity"
        )

    signals.swing = detect_swing_window(result.landmarks, result.timestamps_ms)
    if signals.swing is None:
        signals.warnings.append(f"{view}: no swing window detected")

    return signals


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def detect_kinematic(signals):
    """P1 address, P4 top, P7 impact, P10 finish.

    The swing window supplies the search bounds; the positions themselves are
    located from the hand-speed profile. The window's own impact estimate is not
    used -- it is derived from motion-burst boundaries rather than peak speed,
    and lands materially late.
    """
    out = {}
    swing = signals.swing
    if swing is None:
        return out

    ts = signals.timestamps_ms
    speed = signals.hand_speed
    n = len(ts)

    def add(name, frame, method, confidence):
        frame = max(0, min(frame, n - 1))
        out[name] = Detection(name, ts[frame], frame, signals.view, method,
                              max(0.0, min(1.0, confidence)))

    start = max(0, min(swing.start_frame, n - 1))
    end = max(start, min(swing.end_frame, n - 1))
    takeaway = max(start, min(swing.takeaway_frame, n - 1))

    # P7 is the peak of the hand-speed profile. Confidence comes from how far
    # that peak stands above the swing's typical speed -- a swing with no clear
    # peak is one we should not claim to have located impact in.
    moving = [i for i in range(takeaway, end + 1) if signals.valid[i]]
    if not moving:
        return out
    impact = max(moving, key=lambda i: speed[i])
    typical = _median([speed[i] for i in moving])
    prominence = (speed[impact] - typical) / speed[impact] if speed[impact] > 0 else 0.0
    add("P7", impact, "peak-hand-speed", prominence)

    # P1 and P10 bound the motion. Confidence reflects how still the golfer
    # actually is there, relative to peak speed.
    def stillness(frame):
        return 1.0 - (speed[frame] / speed[impact]) if speed[impact] > 0 else 0.0

    add("P1", start, "motion-onset", stillness(start))
    add("P10", end, "motion-settle", stillness(end))

    # P4 is the hand-path reversal: hand speed dips to a minimum at the
    # transition between backswing and downswing.
    window = [i for i in range(takeaway + 1, impact) if signals.valid[i]]
    if window:
        top = min(window, key=lambda i: speed[i])
        surrounding = max(speed[takeaway:impact + 1] or [0.0])
        dip = 1.0 - (speed[top] / surrounding) if surrounding > 0 else 0.0
        add("P4", top, "hand-path-reversal", dip)

    return out


def detect_lead_arm(signals, kinematic):
    """P3, P5, P9 -- the lead arm crossing horizontal."""
    out = {}
    if not {"P1", "P4", "P7", "P10"} <= set(kinematic):
        return out

    elev = signals.lead_arm_elevation
    ts = signals.timestamps_ms
    p1, p4 = kinematic["P1"].frame, kinematic["P4"].frame
    p7, p10 = kinematic["P7"].frame, kinematic["P10"].frame

    # Backswing: last crossing before the top. Downswing: first after it.
    # Follow-through: first after impact.
    #
    # The downswing and follow-through spans start a short way past their
    # anchor. The lead arm is nowhere near horizontal at the top or at impact,
    # so a crossing found immediately beside either anchor is landmark jitter,
    # not a position.
    guard = _frames_within(ts, ARM_SPAN_GUARD_MS)
    spans = [
        ("P3", p1, p4 - guard, "last"),
        ("P5", p4 + guard, p7 - guard, "first"),
        ("P9", p7 + guard, p10, "first"),
    ]
    for name, lo, hi, prefer in spans:
        if hi <= lo:
            continue
        picked = _pick_crossing(_find_crossings(elev, signals.valid, ts, lo, hi), prefer)
        if picked is None:
            continue
        timestamp, frame, slope = picked
        out[name] = Detection(
            name, timestamp, frame, signals.view, "lead-arm-horizontal",
            _confidence_from_slope(slope),
        )
    return out


def detect_shaft(signals, kinematic):
    """P2, P6, P8 -- the club shaft crossing horizontal.

    Returns {} when the club cannot be tracked, which is common on low-frame-rate
    footage where the shaft is motion-blurred. Callers should treat these
    positions as genuinely absent rather than substituting a guess.
    """
    try:
        from utils.club_detector import track_club_in_video, get_club_angle
    except Exception:
        return {}

    if not {"P1", "P4", "P7", "P10"} <= set(kinematic):
        return {}

    try:
        club_lines = track_club_in_video(signals.path, signals.landmarks)
    except Exception:
        return {}

    n = len(signals.timestamps_ms)
    # get_club_angle measures from vertical; 90 deg is parallel to the ground.
    offset, tracked = [], [False] * n
    for i in range(n):
        line = club_lines[i] if i < len(club_lines) else None
        angle = get_club_angle(line) if line is not None else None
        if angle is None:
            offset.append(offset[-1] if offset else 0.0)
        else:
            offset.append(abs(angle) - 90.0)
            tracked[i] = True

    if sum(tracked) < 0.5 * max(1, n):
        return {}

    ts = signals.timestamps_ms
    p1, p4 = kinematic["P1"].frame, kinematic["P4"].frame
    p7, p10 = kinematic["P7"].frame, kinematic["P10"].frame

    out = {}
    spans = [
        ("P2", p1, p4, "first"),
        ("P6", p4, p7, "last"),
        ("P8", p7, p10, "first"),
    ]
    for name, lo, hi, prefer in spans:
        crossings = _find_crossings(offset, tracked, ts, lo, hi)
        usable = [c for c in crossings if c[2] > 0]
        if not usable:
            continue
        timestamp, frame, slope = usable[0] if prefer == "first" else usable[-1]
        out[name] = Detection(
            name, timestamp, frame, signals.view, "shaft-horizontal",
            # Shaft tracking is the least reliable signal here; cap accordingly.
            min(0.6, _confidence_from_slope(slope / 60.0)),
        )
    return out


def _frame_at(signals, timestamp_ms):
    """Nearest frame in a view to a timestamp on that view's own clock."""
    ts = signals.timestamps_ms
    if not ts:
        return None
    return min(range(len(ts)), key=lambda i: abs(ts[i] - timestamp_ms))


def detect_from_pair(dtl_path, face_on_path, handedness="right"):
    """Locate P1-P10 from a down-the-line / face-on pair.

    The two views are aligned on impact, not on wall-clock, so the result does
    not assume the cameras share a trigger instant. Timestamps in the result are
    on the face-on clock.
    """
    warnings = []

    face_on = load_view(face_on_path, "face_on", handedness)
    dtl = load_view(dtl_path, "dtl", handedness)
    warnings.extend(face_on.warnings)
    warnings.extend(dtl.warnings)

    kin_face = detect_kinematic(face_on)
    kin_dtl = detect_kinematic(dtl)

    # Swing Catalyst triggers both cameras together and stamps both clips with
    # the same capture time, so a matched pair is already frame-aligned. When
    # the two clips agree on frame rate and length, take that alignment and use
    # the independently-detected impacts only to check it. Otherwise the clips
    # were not co-triggered, and impact is the one event reliable enough to
    # align on.
    have_impact = "P7" in kin_face and "P7" in kin_dtl
    co_triggered = (
        abs(face_on.fps - dtl.fps) < 0.5
        and face_on.frame_count == dtl.frame_count
    )

    if co_triggered:
        offset_ms = 0.0
        alignment = "shared-trigger"
        if have_impact:
            drift = kin_dtl["P7"].timestamp_ms - kin_face["P7"].timestamp_ms
            # Both views watch the same instant, so a large gap means one of the
            # two impact detections is wrong -- not that the clips are offset.
            if abs(drift) > 40.0:
                warnings.append(
                    f"impact detected {drift:+.0f}ms apart between views on "
                    f"frame-aligned clips; one view's impact is unreliable"
                )
    elif have_impact:
        offset_ms = kin_dtl["P7"].timestamp_ms - kin_face["P7"].timestamp_ms
        alignment = "impact"
        warnings.append(
            f"clips are not frame-aligned ({face_on.fps:.1f}fps/"
            f"{face_on.frame_count}f vs {dtl.fps:.1f}fps/{dtl.frame_count}f); "
            f"aligned on impact instead"
        )
    else:
        missing = "face-on" if "P7" not in kin_face else "down-the-line"
        warnings.append(
            f"impact not found in the {missing} view and clips are not "
            f"frame-aligned; falling back to face-on only"
        )
        offset_ms = None
        alignment = "none"

    def to_face_on_clock(detection):
        """Re-express a DTL detection on the face-on clock."""
        if offset_ms is None:
            return None
        return Detection(
            detection.name, detection.timestamp_ms - offset_ms, detection.frame,
            detection.view, detection.method, detection.confidence,
        )

    positions = {}

    # Kinematic tier: both views see it. Keep the more confident, and flag
    # disagreement rather than silently averaging it away.
    for name in ("P1", "P4", "P7", "P10"):
        candidates = []
        if name in kin_face:
            candidates.append(kin_face[name])
        if name in kin_dtl and offset_ms is not None:
            candidates.append(to_face_on_clock(kin_dtl[name]))
        if not candidates:
            continue
        best = max(candidates, key=lambda d: d.confidence)
        # Impact disagreement is reported separately when aligning on impact,
        # where it is zero by construction rather than by agreement.
        if len(candidates) == 2 and not (alignment == "impact" and name == "P7"):
            spread = abs(candidates[0].timestamp_ms - candidates[1].timestamp_ms)
            if spread > 50.0:
                warnings.append(
                    f"{name}: views disagree by {spread:.0f}ms "
                    f"(face-on vs down-the-line) -- swing may need review"
                )
        positions[name] = best

    # Arm tier from face-on, where the lead arm sweeps through the image plane.
    positions.update(detect_lead_arm(face_on, kin_face))

    # Shaft tier from down-the-line, where the swing plane is edge-on.
    shaft = detect_shaft(dtl, kin_dtl)
    if shaft and offset_ms is not None:
        for name, detection in shaft.items():
            positions[name] = to_face_on_clock(detection)
    elif not shaft:
        warnings.append(
            "club shaft could not be tracked in the down-the-line view; "
            "P2/P6/P8 are absent from this result"
        )

    missing = [p for p in ORDER if p not in positions]
    if missing:
        warnings.append(f"not detected: {', '.join(missing)}")

    return {
        "schema": "swingai.p-positions/1",
        "handedness": handedness,
        "time_base": "face_on",
        "alignment": alignment,
        "views": {
            "face_on": {
                "path": face_on.path,
                "fps": face_on.fps,
                "frame_count": face_on.frame_count,
            },
            "dtl": {
                "path": dtl.path,
                "fps": dtl.fps,
                "frame_count": dtl.frame_count,
                "offset_ms": offset_ms,
            },
        },
        "positions": {
            name: {
                "timestamp_ms": round(positions[name].timestamp_ms, 1),
                "confidence": round(positions[name].confidence, 3),
                "method": positions[name].method,
                "detected_in": positions[name].view,
                "frames": {
                    "face_on": _frame_at(face_on, positions[name].timestamp_ms),
                    "dtl": (
                        _frame_at(dtl, positions[name].timestamp_ms + offset_ms)
                        if offset_ms is not None else None
                    ),
                },
            }
            for name in ORDER if name in positions
        },
        "warnings": warnings,
    }
