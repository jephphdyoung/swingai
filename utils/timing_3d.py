"""Timing reliability of dynamic stereo observations.

Two cameras never expose at exactly the same instant. A point that moves
between the exposures is seen at two places, and triangulation joins them
into a position that is neither. This module bounds that error, separately
from the triangulation covariance (which models image-localization noise
only).

**Clocks.** Each stream's labels carry timestamps on that file's own clock.
Synchronization relates them: t_reference = t_stream + offset_ms, with
offset_ms and uncertainty_ms on those timestamp clocks. A source's motion
clock (`catalog.motion_clock`) scales timestamp ms to the swing's real ms:
1 for real-time footage, presented fps / capture fps for documented slowed
footage, unknown otherwise.

**Timing-error bound** of a dynamic observation, in real ms (a worst case,
not a standard deviation)::

    bound_ms = scale * (pair_gap_ms + sync_uncertainty_ms) + exposure_ms

- pair_gap_ms: how far apart the paired frames' synchronized timestamps
  actually are (the largest over the frame's views);
- sync_uncertainty_ms: the synchronization record's uncertainty;
- exposure_ms: exposure-midpoint misalignment not already covered by the
  synchronization uncertainty -- 0 when the record says it is "included",
  its exposure_uncertainty_ms when "bounded", and **unknown** otherwise
  (never 0 by default: equal nominal frame rates don't align exposures).

Every stream must share one motion-clock scale; otherwise the linear offset
convention can't relate them and the bound is unknown.

**Position error from timing.** A point moving at speed v for bound_ms is
displaced by at most v * bound_ms / 1000 between the two exposures. v is a
documented *a priori* upper bound per landmark (`SPEED_BOUNDS_M_S`), not
estimated from the triangulations being judged: a mistimed triangulation
can't vouch for its own timing. The bounds are deliberately generous for a
full swing; slower motions are judged conservatively.

The timing error is reported apart from the covariance and never added to
it: one is a worst-case displacement, the other a 1-sigma Gaussian model of
labelling noise. A point is a trustworthy dynamic input only when both pass
their own gates.
"""

import numpy as np

# Conservative upper bounds on the speed of each landmark during a full swing,
# in m/s, with the reasoning. Measured peaks are well under these.
SPEED_CLASSES = (
    ("clubhead", 70.0, "long-drive clubhead speeds reach ~67 m/s; the hosel moves "
                       "almost as fast",
     ("clubhead_center", "clubhead_heel", "clubhead_toe", "clubface_center", "shaft_hosel")),
    ("ball", 105.0, "long-drive ball speeds reach ~100 m/s after impact",
     ("ball_center",)),
    ("hands_grip", 20.0, "lead-hand peaks are ~10 m/s even for long hitters",
     ("left_wrist", "right_wrist", "left_thumb", "right_thumb", "left_index", "right_index",
      "left_pinky", "right_pinky", "left_hand_back", "right_hand_back", "shaft_butt")),
    ("forearm_elbow", 15.0, "slower than the hands they lead to",
     ("left_elbow", "right_elbow", "left_forearm_marker", "right_forearm_marker")),
    ("head_torso", 6.0, "shoulder and hip joint centres peak at ~3 m/s",
     ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip",
      "right_hip")),
    ("legs_feet", 6.0, "knees and feet move least; heel lift and trail-foot release "
                       "stay under ~3 m/s",
     ("left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
      "left_big_toe", "right_big_toe", "left_little_toe", "right_little_toe",
      "left_shoe_heel_contact", "right_shoe_heel_contact", "left_shoe_toe_contact",
      "right_shoe_toe_contact")),
    ("ground_marker", 1.0, "floor and target-line markers lie still on the ground",
     ("floor_a", "floor_b", "floor_c", "target_line_back", "target_line_front")),
)
SPEED_BOUNDS_M_S = {c: v for c, v, _, _ in SPEED_CLASSES}
UNKNOWN_SPEED_CLASS = "ball"            # an unlisted point gets the largest bound
_CLASS_OF = {name: c for c, _, _, names in SPEED_CLASSES for name in names}

TIMING_NOT_APPLICABLE, TIMING_WITHIN, TIMING_EXCEEDS, TIMING_UNKNOWN = 0, 1, 2, 3
TIMING_NAMES = {TIMING_NOT_APPLICABLE: "not_applicable", TIMING_WITHIN: "within",
                TIMING_EXCEEDS: "exceeds", TIMING_UNKNOWN: "unknown"}
MOTION_UNKNOWN = "a stream's motion time is unknown, so its timestamps aren't real time"
BOUND_FORMULA = ("bound_ms = scale * (pair_gap_ms + sync_uncertainty_ms) + exposure_ms; "
                 "position error = speed bound * bound_ms / 1000 (worst case, kept apart "
                 "from the 1-sigma triangulation covariance)")


def speed_class(name):
    return _CLASS_OF.get(name, UNKNOWN_SPEED_CLASS)


def speed_bound(name, bounds=None):
    return float((bounds or SPEED_BOUNDS_M_S)[speed_class(name)])


def exposure_term(sync):
    """(exposure ms, description) from a synchronization record; ms is None when unknown."""
    alignment = sync.get("exposure_alignment") or "unknown"
    if alignment == "included":
        return 0.0, "included in the synchronization uncertainty"
    if alignment == "bounded" and sync.get("exposure_uncertainty_ms") is not None:
        return float(sync["exposure_uncertainty_ms"]), "bounded separately"
    return None, "exposure alignment unknown"


def frame_timing(pair_gap_ms, sync, scales):
    """The timing record of one dynamic frame.

    pair_gap_ms: the frame's actual synchronized-time gap (timestamp ms);
    sync: the shot's synchronization record; scales: each stream's motion-clock
    scale (None = unknown).
    """
    unknown = []
    u = sync.get("uncertainty_ms")
    if u is None:
        unknown.append("synchronization uncertainty is unknown")
    exposure, how = exposure_term(sync)
    if exposure is None:
        unknown.append("exposure alignment between the cameras is unknown (equal frame "
                       "rates don't make exposures simultaneous)")
    known = [s for s in scales if s is not None]
    scale = None
    if len(known) < len(scales):
        unknown.append(MOTION_UNKNOWN)
    elif max(known) - min(known) > 1e-9 * max(known):
        unknown.append(f"the streams' clocks run at different rates relative to real time "
                       f"({', '.join(f'{s:g}' for s in known)})")
    else:
        scale = known[0]
    bound = None
    if not unknown:
        bound = scale * (pair_gap_ms + u) + exposure
    return {"bound_ms": None if bound is None else round(bound, 6),
            "pair_gap_ms": pair_gap_ms, "sync_uncertainty_ms": u,
            "exposure_ms": exposure, "exposure": how, "motion_scale": scale,
            "unknown": unknown}


def point_timing(bound_ms, names, triangulated, max_error_m, bounds=None):
    """Per point: (position-error bound in m, timing status) for one frame.

    triangulated: (P,) bool, which points have a 3D position. bound_ms None =
    the frame's timing bound is unknown.
    """
    err = np.full(len(names), np.nan)
    status = np.zeros(len(names), np.int8)
    for j, name in enumerate(names):
        if not triangulated[j]:
            continue
        if bound_ms is None:
            status[j] = TIMING_UNKNOWN
            continue
        err[j] = speed_bound(name, bounds) * bound_ms / 1000.0
        status[j] = TIMING_WITHIN if err[j] <= max_error_m else TIMING_EXCEEDS
    return err, status


def describe(name, frame_timing_rec, error_m, status, max_error_m, bounds=None):
    """Why a point fails the timing gate (None when it passes)."""
    if status == TIMING_WITHIN:
        return None
    if status == TIMING_UNKNOWN or frame_timing_rec is None or \
            frame_timing_rec.get("bound_ms") is None:
        unknown = (frame_timing_rec or {}).get("unknown") or ["no timing bound"]
        return "timing-error bound unknown: " + "; ".join(unknown)
    return (f"timing: {name} may move {error_m * 1000:.1f} mm between the exposures "
            f"({speed_bound(name, bounds):g} m/s bound × {frame_timing_rec['bound_ms']:.3f} "
            f"ms) > {max_error_m * 1000:.1f} mm")
