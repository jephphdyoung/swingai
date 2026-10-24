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

**Trajectory values** (world frame, metres, seconds of the swing's *real*
time -- the frames' `motion_time_ms`; impact = the reference stream's P7
time). A frame is usable only if it is a measured dynamic frame, the tracking
centre is *observed* (not inferred), its 1-sigma position uncertainty is <=
`max_sigma_m`, every point it comes from passes the timing gate (the motion
between the cameras' exposures, `utils/timing_3d.py`), and the club geometry
is consistent. With the reference stream's motion time unknown (retimed or
undocumented footage) nothing time-dependent is reported.

**Impact time.** P7 marks the frame containing or nearest impact, so the
true impact lies within one frame interval of the P7 time; that interval (in
real ms) is reported as `impact_time_uncertainty_ms`, and nothing finer is
claimed.

**Timing uncertainty** of a value (`timing_uncertainty`, a worst-case bound,
kept apart from the 1-sigma `uncertainty`): for speed, path and attack angle
it is the first-order change of the value over the impact-time uncertainty
(from the fit's acceleration) plus the effect of the samples' timing
position-error bounds on the fitted velocity (sum of |fit weight| x bound);
for the observed low point, that sample's timing bound (and the impact-time
uncertainty for its time). Swing direction and the fitted arc don't propagate
it (null); their samples each pass the timing gate.

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

ALGORITHM = {"name": "swingai-club-trajectory", "version": "1.1.0"}
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
    """(xyz, cov, basis, point names) of the clubhead tracking centre at frame i, or None."""
    c = _pt(rec, i, "clubhead_center")
    if c is not None:
        return c[0], c[1], "clubhead_center", ("clubhead_center",)
    h, t = _pt(rec, i, "clubhead_heel"), _pt(rec, i, "clubhead_toe")
    if h is not None and t is not None:
        return ((h[0] + t[0]) / 2, (h[1] + t[1]) / 4, "heel-toe midpoint",
                ("clubhead_heel", "clubhead_toe"))
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

    def __init__(self, rec, impact_motion_ms, max_sigma_m):
        """impact_motion_ms: impact in real ms (the frames' motion_time_ms clock)."""
        self.t, self.x, self.sig, self.err, self.dropped = [], [], [], [], []
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
                else:
                    why = next((p for p in (rec.problem(i, rec.point_index(n))
                                            for n in tc[3]) if p), None)
            t_ms = (f["time_ms"] if f["motion_time_ms"] is None
                    else f["motion_time_ms"]) - impact_motion_ms
            if why:
                self.dropped.append((t_ms, why))
                continue
            self.t.append(t_ms / 1000.0)
            self.x.append(tc[0])
            self.sig.append(s)
            self.err.append(max(float(rec.timing_error_m[i, rec.point_index(n)])
                                for n in tc[3]))
        order = np.argsort(self.t)
        self.t = np.asarray(self.t, float)[order]
        self.x = np.asarray(self.x, float).reshape(-1, 3)[order]
        self.sig = np.asarray(self.sig, float)[order]
        self.err = np.asarray(self.err, float)[order]
        d = np.diff(self.t)
        self.dt = float(np.median(d)) if len(d) else None

    def window(self, lo_s, hi_s, with_err=False):
        m = (self.t >= lo_s) & (self.t <= hi_s)
        return (self.t[m], self.x[m], self.sig[m]) + ((self.err[m],) if with_err else ())

    def dropped_near(self, lo_s, hi_s):
        return [why for t, why in self.dropped if lo_s * 1000 <= t <= hi_s * 1000]


def _gap_ok(t, dt):
    return dt is not None and len(t) > 1 and float(np.max(np.diff(t))) <= 1.6 * dt


def _velocity(samples, window_s):
    """(velocity, its 1-sigma, acceleration, velocity timing bound, reasons)."""
    t, x, sig, err = samples.window(-window_s, window_s, with_err=True)
    reasons = []
    if (t < 0).sum() < 2 or (t > 0).sum() < 2:
        reasons.append(f"needs two usable samples each side of impact within "
                       f"±{window_s * 1000:.0f} ms (have {(t < 0).sum()} before, "
                       f"{(t > 0).sum()} after)")
    elif not _gap_ok(t, samples.dt):
        reasons.append("a gap in the samples around impact")
    if reasons:
        return None, None, None, None, reasons + samples.dropped_near(-window_s, window_s)
    A = np.column_stack([np.ones_like(t), t, t ** 2])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    v = coef[1]
    s = float(np.max(sig))
    AtA_inv = np.linalg.inv(A.T @ A)
    cov_coef = s ** 2 * AtA_inv
    # Each sample may be displaced by up to err[k] in any direction by the
    # exposure timing; the fitted velocity is linear in the samples.
    w = (AtA_inv @ A.T)[1]
    v_timing = float(np.abs(w) @ err)
    return v, math.sqrt(cov_coef[1, 1]), 2 * coef[2], v_timing, []


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
    scale = rec.motion_scale
    ref = rec.streams[0]
    impact_unc = (None if scale is None or ref.frame_interval_ms is None
                  else ref.frame_interval_ms * scale)
    timing = {"impact_time_uncertainty_ms": impact_unc,
              "impact_time_basis": (None if impact_unc is None else
                                    "P7 marks the frame containing or nearest impact: true "
                                    "impact within one frame interval of its time"),
              "motion_scale": scale}
    if rec.world["status"] != "defined":
        gate = "the world frame is undefined: " + "; ".join(rec.world["reasons"])
    elif impact_ms is None:
        gate = "no impact (P7) time on the reference stream"
    elif scale is None:
        gate = (f"the reference stream's motion time is unknown, so its timestamps aren't the "
                f"swing's real time: " + "; ".join(ref.motion["reasons"]))
    elif impact_unc is None:
        gate = "the reference stream has no frame rate, so impact time can't be bounded"
    if gate:
        out = {k: unavailable(u, "", gate) for k, u in names.items()}
        pts = {k: {**unavailable("m", "", gate), "value": None, "time_ms": None, "frame": None}
               for k in ("observed_low_point", "fitted_free_arc_low_point")}
        return {"impact_time_ms": impact_ms, **timing, "values": out, "points": pts,
                "series": {"time_ms": [], "speed_m_s": [], "displacement_m": []}}
    s = _Samples(rec, impact_ms * scale, max_sigma_m)
    dt_imp = impact_unc / 1000.0

    # velocity-derived values
    v, sv, acc, v_timing, why = _velocity(s, fit_window_ms / 1000.0)
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
        # worst-case timing terms: impact anywhere within dt_imp of P7 (first order,
        # from the fitted acceleration), plus the samples' exposure-timing bounds
        speed_rate = abs(float(v @ acc)) / speed
        h = max(horiz, 1e-9)                    # a vertical velocity has no path direction
        yaw_rate = abs(v[0] * acc[1] - v[1] * acc[0]) / h ** 2
        horiz_rate = (v[0] * acc[0] + v[1] * acc[1]) / h
        elev_rate = abs(acc[2] * horiz - v[2] * horiz_rate) / (horiz ** 2 + v[2] ** 2)
        t_speed = speed_rate * dt_imp + v_timing
        t_ang = math.degrees(v_timing / speed)
        t_yaw = math.degrees(yaw_rate * dt_imp) + t_ang
        t_elev = math.degrees(elev_rate * dt_imp) + t_ang
        vm_t = (vm + f"; impact time ±{impact_unc:.2f} ms (P7 frame), so timing_uncertainty "
                "includes the change over that interval")
        values["clubhead_speed_m_s"] = measurement(speed, "m/s", vm_t, sv,
                                                   timing_uncertainty=t_speed)
        values["clubhead_speed_mph"] = measurement(speed * MPS_TO_MPH, "mph", vm_t,
                                                   sv * MPS_TO_MPH,
                                                   timing_uncertainty=t_speed * MPS_TO_MPH)
        values["path_yaw_deg"] = measurement(yaw, "deg",
                                             vm_t + "; ground direction, +Y positive",
                                             ang_sigma, timing_uncertainty=t_yaw)
        values["club_path_deg"] = measurement(-sign * yaw, "deg",
                                              vm_t + "; positive = in-to-out", ang_sigma,
                                              timing_uncertainty=t_yaw)
        values["attack_angle_deg"] = measurement(math.degrees(math.atan2(v[2], horiz)), "deg",
                                                 vm_t + "; elevation, negative = descending",
                                                 ang_sigma, timing_uncertainty=t_elev)

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
    t, x, sig, err = s.window(lo / 1000, hi / 1000, with_err=True)
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
        e = float(err[k])
        p = {**measurement(None, "m", lm, unc, timing_uncertainty=e),
             "value": [float(c) for c in x[k]], "time_ms": float(t[k] * 1000), "frame": None}
        values["low_point_ahead_of_ball_m"] = measurement(x[k, 0], "m", lm, unc,
                                                          timing_uncertainty=e)
        values["low_point_height_m"] = measurement(x[k, 2], "m", lm + " (tracking centre, "
                                                   "not the sole)", float(sig[k]),
                                                   timing_uncertainty=e)
        values["low_point_time_ms"] = measurement(t[k] * 1000, "ms", lm + "; relative to the "
                                                  "P7 time", (s.dt or 0) * 1000,
                                                  timing_uncertainty=impact_unc)
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
    return {"impact_time_ms": impact_ms, **timing, "values": values, "points": points,
            "series": series}


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
            "club": {"impact_time_ms": impact_ms,
                     **{k: club[k] for k in ("impact_time_uncertainty_ms", "impact_time_basis",
                                             "motion_scale")},
                     "values": club["values"], "points": club["points"],
                     "series": club["series"]}}
