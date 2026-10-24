"""P1 alignment relative to the target line, from a 3D reconstruction.

Every line is taken from the golfer's **trail** side to the **lead** side
(lead = left for a right-handed golfer; `utils/landmarks.lead_side`), and
projected onto the ground plane. In the world frame (+X toward the target,
+Y left of it, +Z up):

- ``<line>_yaw_deg``: the projected direction's angle from +X, positive
  toward +Y. 0 = parallel to the target line.
- ``<line>_open_deg``: the same angle signed by handedness so that positive
  is **open** (aligned left of the target for a right-handed golfer, right
  of it for a left-handed one) and negative closed.

Lines (the point names are the shot-label catalogue's):

- shoulder: shoulder joint centres;
- pelvis: hip joint centres;
- knee_center: knee joint centres -- a line through the two knee centres,
  **not** the direction either kneecap faces, which needs more than one
  point per knee and is not measured;
- stance: the toe line (big toes; else the shoe toe-contact points, never
  one of each).

Feet: ``lead_foot_flare_deg`` / ``trail_foot_flare_deg`` are each foot's
heel-to-toe direction on the ground, measured from "square" (perpendicular
to the target line, toward the ball) and positive when the toe is turned
out: the lead toe toward the target, the trail toe away from it.

Differences: shoulder - pelvis, shoulder - stance and pelvis - stance, of
the open angles.

A value is "measured" only from observed points on a measured address frame
in a defined world frame; "inferred" when a contributing point was placed as
hidden (its value is given, flagged); otherwise "unavailable" with the
reason and no number. Uncertainty is first-order from the points'
covariances.
"""

import math

import numpy as np

from utils.landmarks import lead_side, trail_side
from utils.reconstruct_3d import STATE_INFERRED, STATE_OBSERVED

ALGORITHM = {"name": "swingai-p1-alignment", "version": "1.0.0"}
MIN_HORIZONTAL_M = 0.02
LINES = {"shoulder": ("shoulder",), "pelvis": ("hip",), "knee_center": ("knee",),
         "stance": ("big_toe", "shoe_toe_contact")}
LINE_METHOD = {
    "shoulder": "trail->lead shoulder joint centres, projected to the ground",
    "pelvis": "trail->lead hip joint centres, projected to the ground",
    "knee_center": "trail->lead knee joint centres, projected to the ground (a line through "
                   "the knee centres, not kneecap direction)",
    "stance": "trail->lead toe line (big toes, else shoe toe-contact points), projected to "
              "the ground"}
FOOT_HEEL = ("heel", "shoe_heel_contact")
FOOT_TOE = ("big_toe", "shoe_toe_contact")


def measurement(value, unit, method, uncertainty=None, status="measured", reasons=()):
    return {"status": status, "value": None if value is None else float(value), "unit": unit,
            "uncertainty": None if uncertainty is None else float(uncertainty),
            "method": method, "reasons": list(reasons)}


def unavailable(unit, method, *reasons):
    return measurement(None, unit, method, status="unavailable", reasons=reasons)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _angle(a, b, cov_a, cov_b):
    """Ground-plane yaw of a->b and its 1-sigma, in degrees."""
    d = b[:2] - a[:2]
    L2 = float(d @ d)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    g = np.array([-d[1], d[0]]) / L2
    var = g @ (cov_a[:2, :2] + cov_b[:2, :2]) @ g
    return yaw, math.degrees(math.sqrt(max(var, 0.0))), math.sqrt(L2)


def _sign(handedness):
    return 1.0 if handedness == "right" else -1.0


class _Frame:
    """The static address frame's points, with their states."""

    def __init__(self, rec):
        self.rec = rec
        self.i = rec.frame_with("static_address")

    def problem(self):
        if self.rec.world["status"] != "defined":
            return ("the world frame (target line, ground, ball) is undefined: "
                    + "; ".join(self.rec.world["reasons"]))
        if self.i is None:
            return "no address (P1) frame was reconstructed"
        fr = self.rec.frames[self.i]
        if fr["status"] != "measured":
            return "the address frame is not measured: " + "; ".join(fr["reasons"])
        return None

    def get(self, name):
        if name not in self.rec.point_names:
            return None
        j = self.rec.point_index(name)
        state = int(self.rec.state[self.i, j])
        if state not in (STATE_OBSERVED, STATE_INFERRED):
            return None
        return self.rec.xyz[self.i, j], self.rec.cov[self.i, j], state

    def why_missing(self, name):
        for m in self.rec.missing:
            if m["frame"] == self.i and m["point"] == name:
                return f"{name}: {m['reason']}"
        return f"{name} not observed in two views at address"


def _pair(frame, parts, side_a, side_b):
    """First landmark kind with both sides present: (name_a, name_b, pa, pb)."""
    reasons = []
    for part in parts:
        a, b = f"{side_a}_{part}", f"{side_b}_{part}"
        pa, pb = frame.get(a), frame.get(b)
        if pa and pb:
            return a, b, pa, pb, []
        reasons += [frame.why_missing(n) for n, p in ((a, pa), (b, pb)) if p is None]
    return None, None, None, None, reasons


def _status(states, names):
    hidden = [n for n, s in zip(names, states) if s == STATE_INFERRED]
    if hidden:
        return "inferred", [f"{', '.join(hidden)} placed as hidden: an estimate, not observed"]
    return "measured", []


def p1_alignment(rec, handedness):
    """{name: measurement} for the address frame. See the module docstring."""
    out = {}
    frame = _Frame(rec)
    problem = frame.problem()
    lead, trail = lead_side(handedness), trail_side(handedness)
    sign = _sign(handedness)
    open_by_line = {}
    for line, parts in LINES.items():
        method = LINE_METHOD[line]
        if problem:
            for key in ("yaw", "open"):
                out[f"{line}_{key}_deg"] = unavailable("deg", method, problem)
            continue
        a, b, pa, pb, reasons = _pair(frame, parts, trail, lead)
        if a is None:
            for key in ("yaw", "open"):
                out[f"{line}_{key}_deg"] = unavailable("deg", method, *reasons)
            continue
        yaw, sigma, length = _angle(pa[0], pb[0], pa[1], pb[1])
        if length < MIN_HORIZONTAL_M:
            r = f"{a}->{b} is only {length * 1000:.0f} mm long on the ground: no direction"
            for key in ("yaw", "open"):
                out[f"{line}_{key}_deg"] = unavailable("deg", method, r)
            continue
        status, why = _status((pa[2], pb[2]), (a, b))
        method = f"{method} [{a} -> {b}]"
        out[f"{line}_yaw_deg"] = measurement(yaw, "deg", method, sigma, status, why)
        out[f"{line}_open_deg"] = measurement(sign * yaw, "deg", method + "; positive = open",
                                              sigma, status, why)
        open_by_line[line] = out[f"{line}_open_deg"]
    for role, side in (("lead", lead), ("trail", trail)):
        key = f"{role}_foot_flare_deg"
        method = (f"{role} foot heel->toe on the ground, from square (perpendicular to the "
                  "target line, toward the ball); positive = toe turned out")
        if problem:
            out[key] = unavailable("deg", method, problem)
            continue
        found = None
        reasons = []
        for heel, toe in zip(FOOT_HEEL, FOOT_TOE):
            h, t = frame.get(f"{side}_{heel}"), frame.get(f"{side}_{toe}")
            if h and t:
                found = (f"{side}_{heel}", f"{side}_{toe}", h, t)
                break
            reasons += [frame.why_missing(f"{side}_{n}") for n, p in ((heel, h), (toe, t))
                        if p is None]
        if found is None:
            out[key] = unavailable("deg", method, *reasons)
            continue
        hn, tn, h, t = found
        yaw, sigma, length = _angle(h[0], t[0], h[1], t[1])
        if length < MIN_HORIZONTAL_M:
            out[key] = unavailable("deg", method, f"{hn}->{tn} too short on the ground")
            continue
        square = -90.0 * sign                   # facing the ball: -Y (RH) / +Y (LH)
        delta = wrap_deg(yaw - square)
        out_turn = 1.0 if role == "lead" else -1.0
        flare = sign * out_turn * delta
        status, why = _status((h[2], t[2]), (hn, tn))
        out[key] = measurement(flare, "deg", f"{method} [{hn} -> {tn}]", sigma, status, why)
    for a, b in (("shoulder", "pelvis"), ("shoulder", "stance"), ("pelvis", "stance")):
        key = f"{a}_minus_{b}_open_deg"
        method = f"{a}_open_deg - {b}_open_deg"
        if a not in open_by_line or b not in open_by_line:
            out[key] = unavailable("deg", method, *(r for k in (a, b)
                                                    for r in out[f"{k}_open_deg"]["reasons"]))
            continue
        va, vb = open_by_line[a], open_by_line[b]
        statuses = {va["status"], vb["status"]}
        out[key] = measurement(va["value"] - vb["value"], "deg", method,
                               math.hypot(va["uncertainty"], vb["uncertainty"]),
                               "inferred" if "inferred" in statuses else "measured",
                               va["reasons"] + vb["reasons"])
    return out
