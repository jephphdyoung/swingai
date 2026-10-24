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


def rig():
    """DTL behind the hands looking down the target line; face-on facing the golfer."""
    dtl = camera("down_the_line", (-4.2, 0.95, 1.25), (0.0, 0.45, 0.8))
    fo = camera("face_on", (0.1, -3.6, 1.15), (0.0, 0.55, 0.8))
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
        lo, hi = -0.5, 0.5
        for _ in range(100):
            mid = (lo + hi) / 2
            if self.position(mid)[0] < x:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def expected(self):
        v = self.velocity(self.t_impact)
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
    v = arc.velocity(t)
    vdir = v / np.linalg.norm(v)
    toward_golfer = arc.center - c
    toward_golfer /= np.linalg.norm(toward_golfer)
    face_axis = np.cross(vdir, np.cross(toward_golfer, vdir))
    face_axis /= np.linalg.norm(face_axis)
    hosel = c + 0.05 * face_axis + [0, 0, 0.03]
    return {"clubhead_center": c, "clubhead_heel": c + 0.045 * face_axis,
            "clubhead_toe": c - 0.045 * face_axis, "shaft_hosel": hosel,
            "shaft_butt": hosel + 0.9 * toward_golfer}
