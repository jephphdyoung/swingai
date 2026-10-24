"""Synthetic 3D truth for the geometry tests: a two-camera rig, a P1 body with
known alignment, scene references and a clubhead on a known circular arc.

World convention (utils/world_frame.py): +X toward the target, +Y left of
the target line, +Z up, origin at the ball's ground projection. A
right-handed golfer stands on +Y facing -Y.
"""

import math

import numpy as np

from utils.camera_model import Camera, look_at

W, H = 1440, 1080
DIST = (-0.12, 0.06, 0.0005, -0.0003, 0.0)
BALL = np.array([0.0, 0.0, 0.0214])


def camera(camera_id, center, target, f=1500.0, dist=DIST):
    R, t = look_at(center, target)
    return Camera(camera_id, W, H, f, f * 1.001, W / 2 + 3.5, H / 2 - 2.0, dist).with_pose(R, t)


def rig(handedness="right"):
    """DTL behind the hands looking down the target line; face-on facing the
    golfer. Mirrored across the target line for a left-hander."""
    m = 1 if handedness == "right" else -1
    dtl = camera("down_the_line", (-4.2, 0.95 * m, 1.25), (0.0, 0.45 * m, 0.8))
    fo = camera("face_on", (0.1, -3.6 * m, 1.15), (0.0, 0.55 * m, 0.8))
    return dtl, fo


def _line(center, half, yaw_deg):
    d = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    c = np.asarray(center, float)
    return c + half * d, c - half * d          # (lead, trail) for a right-handed golfer


# Yaw of each trail->lead line (degrees, world). For a right-handed golfer a
# positive yaw is open (pointing left of the target line).
TRUE_YAW = {"shoulder": 4.0, "pelvis": 8.0, "knee_center": 2.0, "stance": -3.0}
TRUE_FLARE = {"lead": 25.0, "trail": 10.0}     # toe turned out from square, degrees


def p1_body(handedness="right"):
    """{point name: world xyz} at address."""
    pts = {}
    for name, (c, half) in {"shoulder": ((0.05, 0.62, 1.45), 0.19),
                            "hip": ((0.03, 0.70, 0.95), 0.17),
                            "knee": ((0.03, 0.72, 0.50), 0.13)}.items():
        key = {"hip": "pelvis", "knee": "knee_center"}.get(name, name)
        lead, trail = _line(c, half, TRUE_YAW[key])
        pts[f"left_{name}"], pts[f"right_{name}"] = lead, trail
    lead_toe, trail_toe = _line((0.0, 0.55, 0.02), 0.25, TRUE_YAW["stance"])
    pts["left_big_toe"], pts["right_big_toe"] = lead_toe, trail_toe
    a, b = math.radians(TRUE_FLARE["lead"]), math.radians(TRUE_FLARE["trail"])
    lead_dir = np.array([math.sin(a), -math.cos(a), 0.0])       # square is -Y
    trail_dir = np.array([-math.sin(b), -math.cos(b), 0.0])
    pts["left_heel"] = lead_toe - 0.24 * lead_dir + [0, 0, 0.03]
    pts["right_heel"] = trail_toe - 0.24 * trail_dir + [0, 0, 0.03]
    pts["left_ankle"] = pts["left_heel"] + 0.06 * lead_dir + [0, 0, 0.06]
    pts["right_ankle"] = pts["right_heel"] + 0.06 * trail_dir + [0, 0, 0.06]
    pts["left_wrist"], pts["right_wrist"] = np.array([0.08, 0.38, 0.78]), np.array([0.06, 0.40, 0.80])
    pts["ball_center"] = BALL.copy()
    pts.update(scene())
    if handedness == "left":
        return mirror(pts)
    return pts


def scene():
    return {"floor_a": np.array([-0.8, -0.3, 0.0]), "floor_b": np.array([1.0, -0.45, 0.0]),
            "floor_c": np.array([0.3, 1.4, 0.0]),
            "target_line_back": np.array([-1.0, -0.25, 0.01]),
            "target_line_front": np.array([1.0, -0.25, 0.01])}


def mirror(pts):
    """The left-handed golfer: reflect y and swap anatomical sides."""
    out = {}
    for name, p in pts.items():
        q = np.array(p, float) * [1, -1, 1]
        if name.startswith("left_"):
            name = "right_" + name[5:]
        elif name.startswith("right_"):
            name = "left_" + name[6:]
        out[name] = q
    return out


# --- clubhead on a circular arc -------------------------------------------------

class Arc:
    """Clubhead tracking centre on a circle in a tilted plane, constant speed.

    The plane's horizontal line has yaw `plane_yaw_deg`; it rises toward the
    golfer (+Y) at `tilt_deg`. The arc's lowest point is `low_point`.
    """

    def __init__(self, radius=1.6, speed=40.0, tilt_deg=60.0, plane_yaw_deg=-3.0,
                 low_point=(0.04, 0.0, 0.012)):
        self.r, self.speed = radius, speed
        psi, tilt = math.radians(plane_yaw_deg), math.radians(tilt_deg)
        self.e1 = np.array([math.cos(psi), math.sin(psi), 0.0])
        h = np.array([-math.sin(psi), math.cos(psi), 0.0])
        self.u = math.cos(tilt) * h + math.sin(tilt) * np.array([0, 0, 1.0])
        self.low = np.asarray(low_point, float)
        self.center = self.low + self.r * self.u
        self.omega = speed / radius
        self.plane_yaw_deg = plane_yaw_deg
        # Impact: the tracking centre passes the ball (x = 0).
        self.t_impact = self._solve_x(0.0)

    def position(self, t):
        phi = self.omega * np.asarray(t, float)
        return (self.center - self.r * np.cos(phi)[..., None] * self.u
                + self.r * np.sin(phi)[..., None] * self.e1)

    def velocity(self, t):
        phi = self.omega * t
        return self.r * self.omega * (math.sin(phi) * self.u + math.cos(phi) * self.e1)

    def _solve_x(self, x):
        # Within one radian of the low point, where x increases monotonically.
        lo, hi = -1.0 / self.omega, 1.0 / self.omega
        for _ in range(100):
            mid = (lo + hi) / 2
            if self.position(mid)[0] < x:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def expected(self, t=None):
        """Velocity-derived truth at arc time t (default: the true impact)."""
        t = self.t_impact if t is None else t
        v = self.velocity(t)
        horiz = math.hypot(v[0], v[1])
        return {"speed": float(np.linalg.norm(v)),
                "path_yaw_deg": math.degrees(math.atan2(v[1], v[0])),
                "attack_deg": math.degrees(math.atan2(v[2], horiz)),
                "low_point": self.low.copy(),
                "low_point_time_ms": -self.t_impact * 1000.0}


def club_points(arc, t):
    """Club landmarks at time t: tracking centre, heel/toe either side, hosel
    and butt along a shaft toward the hands."""
    c = arc.position(t)
    radial = arc.center - c                        # toward the hands; turns with the club
    radial /= np.linalg.norm(radial)
    normal = np.cross(arc.e1, arc.u)               # fixed: the swing plane's normal
    hosel = c + 0.05 * normal + 0.03 * radial      # all offsets in the club's own frame
    return {"clubhead_center": c, "clubhead_heel": c + 0.045 * normal,
            "clubhead_toe": c - 0.045 * normal, "shaft_hosel": hosel,
            "shaft_butt": hosel + 0.9 * radial}


# --- a complete synthetic capture in the existing contracts ---------------------------

FPS = 240.0
CLUB_NAMES = ("clubhead_center", "clubhead_heel", "clubhead_toe", "shaft_hosel", "shaft_butt")


def _source(cat, source_id, golfer_id, view, n_frames, ts_status="verified",
            timing=None, fps=None):
    fps = fps or FPS
    import hashlib
    sha = hashlib.sha256(source_id.encode()).hexdigest()
    cat.sources[source_id] = {
        "source_id": source_id, "golfer_id": golfer_id, "sha256": sha, "size_bytes": 1000,
        "location": {"root": "user", "path": f"{source_id}.mp4"}, "also_found_at": [],
        "registered_at": "2026-11-01T10:00:00Z", "legacy_annotation_key": None,
        "provenance": {"source_type": "fox_capture", "url": None, "title": None,
                       "original_filename": f"{source_id}.mp4", "notes": None},
        "recording": {"date": "2026-11-01", "date_basis": "capture", "camera_view": view,
                      "notes": None},
        "video": {"frame_count": n_frames, "width": W, "height": H, "codec": "mp4v",
                  "nominal_fps": fps, "average_fps": fps, "variable_frame_rate": False,
                  "duration_ms": (n_frames - 1) * 1000 / fps, "start_offset_ms": 0.0,
                  "reported_frame_count": n_frames, "decoder": "synthetic",
                  "timestamps": {"status": ts_status, "estimated_frames": 0,
                                 "ffprobe_max_deviation_ms": 0.0, "detail": ""}},
        "timing": timing or {"motion_time": "real_time", "capture_fps": FPS,
                             "basis": "synthetic 240fps capture"}}
    return sha


def _digest(stream, f):
    import hashlib
    return hashlib.sha1(f"{stream}-{f}".encode()).hexdigest()


class Capture:
    """A two-camera capture of one right-handed swing.

    fo_shift_frames: the face-on camera started this many frames earlier.
    fo_phase_ms: extra sub-frame delay of face-on exposures (0 = genlocked).
    The true offset (t_ref = t_fo + offset_ms) is phase - shift * 1000/fps,
    on the files' timestamp clock (divided by the motion scale when slowed).
    same_swing_status: "verified", or "asserted" (as add_stream leaves it).
    Timing defaults describe a hardware-triggered pair (the synthetic exposures
    are exactly simultaneous): sync uncertainty 0.05 ms, exposure alignment
    bounded at 0.02 ms. exposure_alignment="unknown" leaves it unknown.
    timing: the sources' catalog timing record (default real time);
    playback_fps: the files' presented rate when it differs from the 240 fps
    capture (slowed footage).
    dynamic_body: body points also labelled in the club frames (the synthetic
    body is still, so they sit at their P1 positions).
    """

    def __init__(self, tmp_path, *, noise_px=0.3, fo_shift_frames=9, fo_phase_ms=0.0,
                 same_swing_status="verified", sync_status="verified", sync_offset_ms="true",
                 sync_uncertainty_ms=0.05, exposure_alignment="bounded",
                 exposure_uncertainty_ms=0.02,
                 calibration_status="verified", ts_status="verified", handedness="right",
                 world_to_rig=None, arc=None, seed=0, n_frames=100, impact_frame=70,
                 p1_frames=(5, 7), club_frames=range(40, 100), body=None, timing=None,
                 playback_fps=None, dynamic_body=()):
        from utils import catalog as cm
        from utils.calibration import calibration_document, load_calibration
        self.noise_px = noise_px
        self.rng = np.random.default_rng(seed)
        self.arc = arc or Arc()
        self.handedness = handedness
        self.body = body or p1_body(handedness)
        self.playback_fps = playback_fps or FPS
        self.motion_scale = self.playback_fps / FPS       # real ms per timestamp ms
        self.true_offset_ms = (fo_phase_ms - fo_shift_frames * 1000.0 / FPS) / self.motion_scale
        self.world_to_rig = (np.eye(4) if world_to_rig is None else np.asarray(world_to_rig))
        cams_world = rig(handedness)
        self.cams_world = cams_world
        cams_rig = [c.reframed(self.world_to_rig) for c in cams_world]
        # Arc time of reference (DTL) frame k.
        # The P7 frame is the last one before the ball is passed (by 0.3 frame).
        self.t0 = self.arc.t_impact - (impact_frame + 0.3) / FPS
        self.fo_shift, self.fo_phase_ms = fo_shift_frames, fo_phase_ms

        cat = cm.Catalog(root=str(tmp_path / "catalog"))
        cm.add_golfer(cat, "Synthetic Golfer", "user", handedness, golfer_id="golfer")
        _source(cat, "src-dtl", "golfer", "down_the_line", n_frames, ts_status, timing,
                self.playback_fps)
        _source(cat, "src-fo", "golfer", "face_on", n_frames, ts_status, timing,
                self.playback_fps)
        shot = cm.add_shot(cat, golfer_id="golfer", source_id="src-dtl",
                           camera_view="down_the_line", start_frame=0, end_frame=n_frames - 1,
                           swing_type="full", shot_id="shot-synthetic",
                           stream_id="down_the_line")
        cm.add_stream(cat, shot["shot_id"], source_id="src-fo", camera_view="face_on",
                      start_frame=0, end_frame=n_frames - 1, stream_id="face_on",
                      same_swing_method="synthetic co-triggered capture", asserted_by="test")
        if same_swing_status == "verified":
            cm.set_multi_view(cat, shot["shot_id"], "same_swing", "verified",
                              "synthetic: one simulated swing", "test")
        if sync_status != "unknown":
            offset = self.true_offset_ms if sync_offset_ms == "true" else sync_offset_ms
            cm.set_multi_view(cat, shot["shot_id"], "synchronization", sync_status,
                              "synthetic clocks", "test", offset_ms=offset,
                              uncertainty_ms=sync_uncertainty_ms,
                              exposure_alignment=exposure_alignment,
                              exposure_uncertainty_ms=(exposure_uncertainty_ms
                                                       if exposure_alignment == "bounded"
                                                       else None))
        if calibration_status != "unknown":
            cm.set_multi_view(cat, shot["shot_id"], "calibration", calibration_status,
                              "synthetic board", "test")
        self.cat, self.shot_id = cat, shot["shot_id"]

        doc = {"schema_version": "1.0", "kind": "shot_annotations", "revision": 1,
               "shot_id": self.shot_id, "p_convention": "p7-impact", "streams": {}}
        streams = (("down_the_line", "src-dtl", cams_world[0], 0.0, 0),
                   ("face_on", "src-fo", cams_world[1], fo_phase_ms, fo_shift_frames))
        for (sid, src_id, cam, phase, shift), p1 in zip(streams, p1_frames):
            frames = {}
            entry = {"source_id": src_id, "source_sha256": cat.sources[src_id]["sha256"],
                     "positions": {}, "frames": frames}
            frames[str(p1)] = self._frame(sid, p1, cam, self.body)
            entry["positions"]["P1"] = self._position(sid, p1)
            for k in club_frames:
                f = k + shift                      # same true instant, this camera's index
                if not 0 <= f < n_frames:
                    continue
                t = self.t0 + k / FPS + phase / 1000.0
                pts = {**club_points(self.arc, t), **{n: self.body[n] for n in dynamic_body}}
                frames[str(f)] = self._frame(sid, f, cam, pts)
            entry["positions"]["P7"] = self._position(sid, impact_frame + shift)
            doc["streams"][sid] = entry
        self.annotations = doc
        cal = calibration_document("synthetic-rig", 1, cams_rig, calibration_status
                                   if calibration_status != "unknown" else "estimated",
                                   "synthetic", "test", "2026-11-01T09:00:00Z",
                                   created_at="2026-11-01T09:00:00Z")
        self.calibration = load_calibration(cal)

    def _frame(self, sid, f, cam, points):
        out, unavailable = {}, {}
        for name, X in points.items():
            px, _ = cam.project(np.asarray(X)[None])
            p = px[0] + self.rng.normal(scale=self.noise_px, size=2) if self.noise_px else px[0]
            if not cam.in_image(p[None], margin=-1).all():
                unavailable[name] = "out_of_frame"      # as a person would record it
                continue
            out[name] = {"x": float(p[0]), "y": float(p[1]), "visible": True,
                         "origin": "manual", "model": None, "confidence": None,
                         "set_by": "synthetic", "set_at": "2026-11-01T10:00:00Z"}
        return {"timestamp_ms": round(f * 1000.0 / self.playback_fps, 3),
                "timestamp_basis": "container",
                "image_sha1": _digest(sid, f), "points": out, "unavailable": unavailable,
                "review": {"status": "reviewed", "by": "synthetic",
                           "at": "2026-11-01T10:00:00Z", "notes": None}}

    def _position(self, sid, f):
        return {"frame": f, "timestamp_ms": round(f * 1000.0 / self.playback_fps, 3),
                "timestamp_basis": "container", "image_sha1": _digest(sid, f),
                "origin": "manual", "set_by": "synthetic", "set_at": "2026-11-01T10:00:00Z"}

    def arc_time(self, ref_ms):
        """Arc time of a reference-clock time (ms)."""
        return self.t0 + ref_ms / 1000.0

    def club_truth(self, ref_frame):
        return club_points(self.arc, self.t0 + ref_frame / FPS)
