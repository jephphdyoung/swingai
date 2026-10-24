"""The shot's world frame: target line, ground plane and ball origin.

Convention (right-handed, metres):

- **+X** toward the target, along the target line;
- **+Y** left of the target line (when looking down it toward the target);
- **+Z** up, normal to the ground plane;
- **origin** at the ground projection of the ball centre at address.

So a right-handed golfer stands on the +Y side facing -Y, a left-handed golfer
on the -Y side facing +Y.

Camera calibration alone determines neither the ground nor the target line:
cameras calibrated against a board know where the board was, not where the
floor or the target is. Each of the three pieces therefore needs its own
explicit evidence, recorded with the frame:

- ground: >= 3 points on the ground (fitted plane, residual reported), or a
  calibration board an operator confirms was lying on the ground;
- target line: a direction from two points on a target-line marker (e.g. an
  alignment stick laid parallel to the target line), or an operator-confirmed
  board edge laid along it;
- origin: the ball centre at address (P1), triangulated, dropped to the ground.

`define_world_frame` refuses rather than guesses when any piece is missing or
degenerate.
"""

from dataclasses import dataclass, field

import numpy as np

from utils.camera_model import make_transform

CONVENTION = ("right-handed, metres: +X toward the target along the target line, +Y left of "
              "the target line, +Z up normal to the ground; origin at the ground projection "
              "of the ball centre at address")
MAX_GROUND_RMS_M = 0.01
MIN_TARGET_BASELINE_M = 0.3


class WorldFrameError(ValueError):
    pass


def fit_plane(points):
    """Total-least-squares plane through (N>=3,3) points: (unit normal, d, rms)
    with n . p + d = 0."""
    P = np.asarray(points, float)
    if len(P) < 3:
        raise WorldFrameError("a plane needs at least three points")
    c = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - c)
    if s[1] < 1e-6 * max(1.0, s[0]):
        raise WorldFrameError("ground points are collinear: they don't define a plane")
    n = vt[-1]
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((P @ n + d) ** 2)))
    return n, d, rms


@dataclass
class WorldFrame:
    rig_to_world: np.ndarray            # 4x4
    evidence: dict = field(default_factory=dict)

    def to_world(self, X):
        from utils.camera_model import apply_transform
        return apply_transform(self.rig_to_world, X)


def ground_from_points(points, method, max_rms_m=MAX_GROUND_RMS_M):
    n, d, rms = fit_plane(points)
    if rms > max_rms_m:
        raise WorldFrameError(f"ground points are not planar: rms {rms * 1000:.1f} mm > "
                              f"{max_rms_m * 1000:.0f} mm")
    return {"normal": n, "d": d, "rms_m": rms, "n_points": len(points), "method": method}


def define_world_frame(ground, target_direction, ball_center, up_hint, target_method,
                       origin_method):
    """Build rig->world from explicit evidence.

    ground: {"normal", "d", "method", ...} in rig coordinates.
    target_direction: rig vector pointing toward the target (any length).
    ball_center: rig coordinates of the ball centre at address.
    up_hint: a rig point known to be above the ground (e.g. a camera centre),
        used only to choose which way the plane normal points.
    """
    if ground is None:
        raise WorldFrameError("no ground-plane evidence: calibration alone doesn't say "
                              "where the floor is")
    if target_direction is None:
        raise WorldFrameError("no target-line evidence: calibration alone doesn't say "
                              "where the target is")
    if ball_center is None or not np.all(np.isfinite(ball_center)):
        raise WorldFrameError("no observed ball centre at address for the origin")
    n = np.asarray(ground["normal"], float)
    n = n / np.linalg.norm(n)
    d = float(ground["d"])
    if n @ np.asarray(up_hint, float) + d < 0:
        n, d = -n, -d
    v = np.asarray(target_direction, float)
    horiz = v - (v @ n) * n
    if np.linalg.norm(horiz) < 0.5 * np.linalg.norm(v):
        raise WorldFrameError("the target direction is closer to vertical than horizontal: "
                              "check the target-line marker points")
    x = horiz / np.linalg.norm(horiz)
    z = n
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    b = np.asarray(ball_center, float)
    origin = b - (n @ b + d) * n
    T = make_transform(R, -R @ origin)
    evidence = {"ground": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in ground.items()},
                "target_line": {"method": target_method,
                                "direction_rig": [float(c) for c in x]},
                "origin": {"method": origin_method,
                           "ball_center_rig": [float(c) for c in b],
                           "ball_height_m": float(n @ b + d)}}
    return WorldFrame(T, evidence)


def target_direction_from_points(back, front, min_baseline_m=MIN_TARGET_BASELINE_M):
    """Direction from a back point to a front point on a target-line marker."""
    v = np.asarray(front, float) - np.asarray(back, float)
    if not np.all(np.isfinite(v)) or np.linalg.norm(v) < min_baseline_m:
        raise WorldFrameError(f"target-line marker points are less than {min_baseline_m} m "
                              "apart (or missing): direction too uncertain")
    return v
