"""The club as a 3D structure, and clubhead trajectory measurements.

Club landmarks (shot-label names):

=================  ==================  =======================================
part               label               notes
=================  ==================  =======================================
grip end           shaft_butt          butt end of the grip
hosel              shaft_hosel         where the shaft enters the hosel
shaft              shaft_butt->hosel   a segment, not a point
tracking centre    clubhead_center     the point trajectories follow
heel, toe          clubhead_heel/toe   ends of the leading edge
face centre        clubface_center     optional; only when the face is visible
=================  ==================  =======================================

**Tracking centre.** The labelled `clubhead_center`; failing that, the
midpoint of the heel and toe when *both* are observed. It is never
extrapolated along the shaft: the head's position relative to the shaft
depends on lie, loft, shaft bend and face rotation, which the shaft line
doesn't determine.

**Rigidity.** Shaft (butt-hosel), head (heel-toe) and hosel-centre lengths
are fixed for a real club. A frame whose lengths deviate from the shot's
median by more than max(10 mm, 5 %) has inconsistent observations
(mislabelled point, wrong frame pairing) and is excluded from trajectories.

**Trajectory values** (world frame, metres, seconds; impact = the reference
stream's P7 time). A frame is usable only if it is a measured dynamic frame,
the tracking centre is *observed* (not inferred), its 1-sigma position
uncertainty is <= `max_sigma_m`, and the club geometry is consistent.

- velocity at impact: per-axis quadratic least-squares fit of usable frames
  within +/- `fit_window_ms` of impact, needing at least two on each side and
  no gap over 1.6 frame intervals; speed is its norm.
- club path: the velocity's direction on the ground. ``club_path_deg`` is
  signed by handedness so positive = in-to-out (right of the target line
  for a right-handed golfer).
- attack angle: the velocity's elevation; negative = descending.
- swing direction: total-least-squares plane through the tracking centre
  over +/- `plane_window_ms` (>= 7 points, RMS <= `max_plane_rms_m`); its
  horizontal line (plane normal x up), toward the target, as an angle from
  the target line signed like club path. Not reported for a near-horizontal
  plane.
- observed low point: the lowest usable sample in [-30, +60] ms around
  impact, only if samples on both sides bracket it. Its uncertainty includes
  half the distance to the neighbouring samples (frame spacing at 40 m/s and
  240 fps is ~17 cm). Distance ahead of the ball is its X (ball at X = 0).
- fitted free-swing-arc low point (optional): a circle fitted in its plane
  to pre-impact samples only (the free arc, before contact) and its lowest
  point. A model extrapolation, so always "inferred" and never the
  observed low point.

Anything that doesn't meet its gate is "unavailable" with the reasons; no
value is guessed.
"""

import math

import numpy as np

from utils.alignment_3d import measurement, unavailable
from utils.reconstruct_3d import STATE_OBSERVED

ALGORITHM = {"name": "swingai-club-trajectory", "version": "1.0.0"}
MPS_TO_MPH = 2.2369362920544
SEGMENTS = (("shaft_butt", "shaft_hosel"), ("clubhead_heel", "clubhead_toe"),
            ("shaft_hosel", "clubhead_center"))
CLUB_PARTS = {"grip_end": "shaft_butt", "hosel": "shaft_hosel",
              "tracking_center": "clubhead_center", "heel": "clubhead_heel",
              "toe": "clubhead_toe", "face_center": "clubface_center"}


def _sign(handedness):
    return 1.0 if handedness == "right" else -1.0


def _pt(rec, i, name, observed_only=True):
    if name not in rec.point_names:
        return None
    j = rec.point_index(name)
    if observed_only and rec.state[i, j] != STATE_OBSERVED:
        return None
    if rec.state[i, j] == 0:
        return None
    return rec.xyz[i, j], rec.cov[i, j]


def club_structure(rec, i):
    """The club's observed parts at frame i, and derived shaft geometry."""
    parts = {part: _pt(rec, i, name) for part, name in CLUB_PARTS.items()}
    out = {part: (None if p is None else p[0]) for part, p in parts.items()}
    if out["grip_end"] is not None and out["hosel"] is not None:
        shaft = out["hosel"] - out["grip_end"]
        out["shaft_vector"] = shaft
        out["shaft_length_m"] = float(np.linalg.norm(shaft))
    return out


def tracking_center(rec, i):
    """(xyz, cov, basis) of the clubhead tracking centre at frame i, or None."""
    c = _pt(rec, i, "clubhead_center")
    if c is not None:
        return c[0], c[1], "clubhead_center"
    h, t = _pt(rec, i, "clubhead_heel"), _pt(rec, i, "clubhead_toe")
    if h is not None and t is not None:
        return (h[0] + t[0]) / 2, (h[1] + t[1]) / 4, "heel-toe midpoint"
    return None


def rigidity(rec, frames, rel=0.05, abs_m=0.010):
    """{frame: [inconsistent segment descriptions]} for the given frames."""
    lengths = {seg: {} for seg in SEGMENTS}
    for i in frames:
        for a, b in SEGMENTS:
            pa, pb = _pt(rec, i, a), _pt(rec, i, b)
            if pa is not None and pb is not None:
                lengths[(a, b)][i] = float(np.linalg.norm(pa[0] - pb[0]))
    bad = {}
    for (a, b), by in lengths.items():
        if len(by) < 3:
            continue
        med = float(np.median(list(by.values())))
        for i, L in by.items():
            if abs(L - med) > max(abs_m, rel * med):
                bad.setdefault(i, []).append(f"{a}-{b} {L * 1000:.0f} mm vs. shot median "
                                             f"{med * 1000:.0f} mm")
    return bad


class _Samples:
    """Usable tracking-centre samples, with the reasons others were dropped."""

    def __init__(self, rec, impact_ms, max_sigma_m):
        self.t, self.x, self.sig, self.dropped = [], [], [], []
        dyn = [i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic"]
        bad = rigidity(rec, dyn)
        for i in dyn:
            f = rec.frames[i]
            why = None
            tc = tracking_center(rec, i)
            if f["status"] != "measured":
                why = "frame not measured: " + "; ".join(f["reasons"])
            elif tc is None:
                why = "tracking centre not observed in two views"
            elif i in bad:
                why = "inconsistent club geometry: " + "; ".join(bad[i])
            else:
                s = math.sqrt(max(np.linalg.eigvalsh(tc[1]).max(), 0.0))
                if s > max_sigma_m:
                    why = f"tracking-centre uncertainty {s * 1000:.1f} mm"
            if why:
                self.dropped.append((f["time_ms"] - impact_ms, why))
                continue
            self.t.append((f["time_ms"] - impact_ms) / 1000.0)
            self.x.append(tc[0])
            self.sig.append(math.sqrt(max(np.linalg.eigvalsh(tc[1]).max(), 0.0)))
        order = np.argsort(self.t)
        self.t = np.asarray(self.t, float)[order]
        self.x = np.asarray(self.x, float).reshape(-1, 3)[order]
        self.sig = np.asarray(self.sig, float)[order]
        d = np.diff(self.t)
        self.dt = float(np.median(d)) if len(d) else None

    def window(self, lo_s, hi_s):
        m = (self.t >= lo_s) & (self.t <= hi_s)
        return self.t[m], self.x[m], self.sig[m]

    def dropped_near(self, lo_s, hi_s):
        return [why for t, why in self.dropped if lo_s * 1000 <= t <= hi_s * 1000]


def _gap_ok(t, dt):
    return dt is not None and len(t) > 1 and float(np.max(np.diff(t))) <= 1.6 * dt


def _velocity(samples, window_s):
    t, x, sig = samples.window(-window_s, window_s)
    reasons = []
    if (t < 0).sum() < 2 or (t > 0).sum() < 2:
        reasons.append(f"needs two usable samples each side of impact within "
                       f"±{window_s * 1000:.0f} ms (have {(t < 0).sum()} before, "
                       f"{(t > 0).sum()} after)")
    elif not _gap_ok(t, samples.dt):
        reasons.append("a gap in the samples around impact")
    if reasons:
        return None, None, reasons + samples.dropped_near(-window_s, window_s)
    A = np.column_stack([np.ones_like(t), t, t ** 2])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    v = coef[1]
    s = float(np.max(sig))
    cov_coef = s ** 2 * np.linalg.inv(A.T @ A)
    return v, math.sqrt(cov_coef[1, 1]), []


def _fit_plane(x):
    c = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - c)
    n = vt[-1]
    rms = float(np.sqrt(np.mean(((x - c) @ n) ** 2)))
    return c, n, vt[0], vt[1], rms


def _circle_low_point(x):
    """Fit a circle in the points' plane; (low point, radius, rms, centre)."""
    c, n, e1, e2, plane_rms = _fit_plane(x)
    uv = np.column_stack([(x - c) @ e1, (x - c) @ e2])
    A = np.column_stack([2 * uv, np.ones(len(uv))])
    b = (uv ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv = sol[:2]
    r = math.sqrt(max(sol[2] + cu ** 2 + cv ** 2, 0.0))
    center = c + cu * e1 + cv * e2
    rms = float(np.sqrt(np.mean((np.linalg.norm(uv - [cu, cv], axis=1) - r) ** 2)))
    down = np.array([0, 0, -1.0]) - (np.array([0, 0, -1.0]) @ n) * n
    if np.linalg.norm(down) < 1e-6:
        return None, r, rms, center
    low = center + r * down / np.linalg.norm(down)
    return low, r, max(rms, plane_rms), center


def club_metrics(rec, impact_ms, handedness, *, max_sigma_m=0.01, fit_window_ms=10.0,
                 plane_window_ms=40.0, max_plane_rms_m=0.01, low_window_ms=(-30.0, 60.0),
                 arc_window_ms=(-40.0, -3.0), max_arc_rms_m=0.005, min_arc_points=5):
    """{"impact_time_ms", "values": {...}, "points": {...}, "series": {...}}."""
    sign = _sign(handedness)
    values, points = {}, {}
    names = {"clubhead_speed_m_s": "m/s", "clubhead_speed_mph": "mph", "club_path_deg": "deg",
             "path_yaw_deg": "deg", "attack_angle_deg": "deg", "swing_direction_deg": "deg",
             "swing_plane_inclination_deg": "deg", "low_point_ahead_of_ball_m": "m",
             "low_point_height_m": "m", "low_point_time_ms": "ms",
             "fitted_arc_low_point_ahead_of_ball_m": "m"}
    gate = None
    if rec.world["status"] != "defined":
        gate = "the world frame is undefined: " + "; ".join(rec.world["reasons"])
    elif impact_ms is None:
        gate = "no impact (P7) time on the reference stream"
    if gate:
        out = {k: unavailable(u, "", gate) for k, u in names.items()}
        pts = {k: {**unavailable("m", "", gate), "value": None, "time_ms": None, "frame": None}
               for k in ("observed_low_point", "fitted_free_arc_low_point")}
        return {"impact_time_ms": impact_ms, "values": out, "points": pts,
                "series": {"time_ms": [], "speed_m_s": [], "displacement_m": []}}
    s = _Samples(rec, impact_ms, max_sigma_m)

    # velocity-derived values
    v, sv, why = _velocity(s, fit_window_ms / 1000.0)
    vm = (f"quadratic least-squares fit of the tracking centre over ±{fit_window_ms:g} ms of "
          "impact; derivative at impact")
    if v is None:
        for k in ("clubhead_speed_m_s", "clubhead_speed_mph", "club_path_deg", "path_yaw_deg",
                  "attack_angle_deg"):
            values[k] = unavailable(names[k], vm, *why)
    else:
        speed = float(np.linalg.norm(v))
        horiz = math.hypot(v[0], v[1])
        yaw = math.degrees(math.atan2(v[1], v[0]))
        ang_sigma = math.degrees(sv / speed)
        values["clubhead_speed_m_s"] = measurement(speed, "m/s", vm, sv)
        values["clubhead_speed_mph"] = measurement(speed * MPS_TO_MPH, "mph", vm,
                                                   sv * MPS_TO_MPH)
        values["path_yaw_deg"] = measurement(yaw, "deg", vm + "; ground direction, +Y positive",
                                             ang_sigma)
        values["club_path_deg"] = measurement(-sign * yaw, "deg",
                                              vm + "; positive = in-to-out", ang_sigma)
        values["attack_angle_deg"] = measurement(math.degrees(math.atan2(v[2], horiz)), "deg",
                                                 vm + "; elevation, negative = descending",
                                                 ang_sigma)

    # swing direction from the fitted plane
    pm = (f"total-least-squares plane through the tracking centre over ±{plane_window_ms:g} ms "
          "of impact; its horizontal line toward the target")
    t, x, _ = s.window(-plane_window_ms / 1000, plane_window_ms / 1000)
    why = []
    if len(t) < 7 or (t < 0).sum() < 2 or (t > 0).sum() < 2:
        why.append(f"needs >= 7 usable samples spanning impact within ±{plane_window_ms:g} ms "
                   f"(have {len(t)})")
        why += s.dropped_near(-plane_window_ms / 1000, plane_window_ms / 1000)
    else:
        _, n, _, _, rms = _fit_plane(x)
        incl = math.degrees(math.acos(min(1.0, abs(n[2]))))
        if rms > max_plane_rms_m:
            why.append(f"samples are {rms * 1000:.1f} mm RMS off a plane (> "
                       f"{max_plane_rms_m * 1000:.0f} mm)")
        elif incl < 15:
            why.append(f"plane only {incl:.0f}° from horizontal: its ground line is undefined")
    if why:
        values["swing_direction_deg"] = unavailable("deg", pm, *why)
        values["swing_plane_inclination_deg"] = unavailable("deg", pm, *why)
    else:
        d = np.cross(n, [0, 0, 1.0])
        d /= np.linalg.norm(d)
        if d[0] < 0:
            d = -d
        psi = math.degrees(math.atan2(d[1], d[0]))
        values["swing_direction_deg"] = measurement(-sign * psi, "deg",
                                                    pm + "; positive = right of target (RH)")
        values["swing_plane_inclination_deg"] = measurement(incl, "deg", pm)

    # observed low point
    lo, hi = low_window_ms
    lm = (f"lowest usable tracking-centre sample in [{lo:g}, {hi:g}] ms of impact, bracketed "
          "by samples on both sides; uncertainty includes half the sample spacing")
    t, x, sig = s.window(lo / 1000, hi / 1000)
    why = []
    if len(t) < 3:
        why.append(f"fewer than 3 usable samples in [{lo:g}, {hi:g}] ms")
    else:
        k = int(np.argmin(x[:, 2]))
        if k == 0 or k == len(t) - 1:
            why.append("the lowest sample is at the edge of the observed samples: the low "
                       "point isn't bracketed")
        elif not _gap_ok(t[k - 1:k + 2], s.dt):
            why.append("a gap next to the lowest sample")
    if why:
        why += s.dropped_near(lo / 1000, hi / 1000)
        p = {**unavailable("m", lm, *why), "value": None, "time_ms": None, "frame": None}
        for key, u in (("low_point_ahead_of_ball_m", "m"), ("low_point_height_m", "m"),
                       ("low_point_time_ms", "ms")):
            values[key] = unavailable(u, lm, *why)
    else:
        spacing = 0.5 * max(np.linalg.norm(x[k] - x[k - 1]), np.linalg.norm(x[k + 1] - x[k]))
        unc = math.hypot(float(sig[k]), spacing)
        p = {**measurement(None, "m", lm, unc), "value": [float(c) for c in x[k]],
             "time_ms": float(t[k] * 1000), "frame": None}
        values["low_point_ahead_of_ball_m"] = measurement(x[k, 0], "m", lm, unc)
        values["low_point_height_m"] = measurement(x[k, 2], "m", lm + " (tracking centre, "
                                                   "not the sole)", float(sig[k]))
        values["low_point_time_ms"] = measurement(t[k] * 1000, "ms", lm,
                                                  (s.dt or 0) * 1000)
    points["observed_low_point"] = p

    # fitted free-arc low point (optional, inferred)
    a0, a1 = arc_window_ms
    am = (f"circle fitted in its plane to usable pre-impact samples in [{a0:g}, {a1:g}] ms (the "
          "free arc before contact); its lowest point. A model extrapolation, distinct from "
          "the observed low point")
    t, x, _ = s.window(a0 / 1000, a1 / 1000)
    why = []
    low = None
    if len(t) < min_arc_points:
        why.append(f"needs >= {min_arc_points} usable pre-impact samples (have {len(t)})")
    else:
        low, r, rms, _ = _circle_low_point(x)
        if low is None:
            why.append("the fitted arc's plane is horizontal")
        elif rms > max_arc_rms_m:
            why.append(f"samples are {rms * 1000:.1f} mm RMS off a circle")
    if why:
        why += s.dropped_near(a0 / 1000, a1 / 1000)
        points["fitted_free_arc_low_point"] = {**unavailable("m", am, *why), "value": None,
                                               "time_ms": None, "frame": None}
        values["fitted_arc_low_point_ahead_of_ball_m"] = unavailable("m", am, *why)
    else:
        reason = [f"fitted circle radius {r:.2f} m, residual {rms * 1000:.1f} mm"]
        points["fitted_free_arc_low_point"] = {
            **measurement(None, "m", am, rms, "inferred", reason),
            "value": [float(c) for c in low], "time_ms": None, "frame": None}
        values["fitted_arc_low_point_ahead_of_ball_m"] = measurement(low[0], "m", am, rms,
                                                                     "inferred", reason)

    # series: per usable sample, displacement from the first and speed by central difference
    speed = [None] * len(s.t)
    for k in range(1, len(s.t) - 1):
        if _gap_ok(s.t[k - 1:k + 2], s.dt):
            speed[k] = float(np.linalg.norm(s.x[k + 1] - s.x[k - 1]) / (s.t[k + 1] - s.t[k - 1]))
    disp = ([float(np.linalg.norm(p - s.x[0])) for p in s.x] if len(s.x) else [])
    series = {"time_ms": [float(v * 1000) for v in s.t], "speed_m_s": speed,
              "displacement_m": disp}
    return {"impact_time_ms": impact_ms, "values": values, "points": points, "series": series}


def impact_time_ms(annotations, reference_stream):
    """The reference stream's P7 label time (its clock = the reconstruction's)."""
    p7 = annotations["streams"].get(reference_stream, {}).get("positions", {}).get("P7")
    return None if p7 is None else float(p7["timestamp_ms"])


def measurements_document(rec, manifest_sha256, reconstruction_id, handedness, impact_ms,
                          alignment, club, created_at=None):
    from utils.alignment_3d import ALGORITHM as ALIGN
    from utils.catalog import now_iso
    return {"schema_version": "1.0", "kind": "measurements_3d",
            "reconstruction_id": reconstruction_id, "shot_id": rec.shot_id,
            "manifest_sha256": manifest_sha256, "created_at": created_at or now_iso(),
            "algorithm": {"name": f"{ALIGN['name']}+{ALGORITHM['name']}",
                          "version": f"{ALIGN['version']}+{ALGORITHM['version']}",
                          "parameters": {}},
            "handedness": handedness, "alignment_p1": alignment,
            "club": {"impact_time_ms": impact_ms, "values": club["values"],
                     "points": club["points"], "series": club["series"]}}
