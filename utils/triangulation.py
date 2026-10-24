"""Multi-view triangulation with reprojection error, uncertainty and checks.

`triangulate` takes one pixel observation per calibrated camera of the same
physical point at the same instant and returns its 3D position:

1. linear (DLT) solution on undistorted normalized coordinates;
2. Gauss-Newton refinement minimizing the reprojection error in *pixels*
   through the full distortion model;
3. per-view reprojection error, the largest angle between the viewing rays
   (parallax), and whether the point lies in front of every camera;
4. first-order covariance: with pixel noise sigma, cov = sigma^2 (J^T J)^-1
   where J is the pixel Jacobian of all projections at the solution. Poorly
   conditioned geometry (small parallax) shows up as large covariance.

`check_point` turns those into the reasons a point is not a valid
measurement. Nothing is clamped or repaired: a point that fails is reported
missing with its reasons by the caller.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_SIGMA_PX = 1.0
MAX_REPROJECTION_PX = 3.0
MIN_PARALLAX_DEG = 5.0


@dataclass(frozen=True)
class TriangulatedPoint:
    xyz: np.ndarray            # (3,) in the cameras' reference frame
    cov: np.ndarray            # (3,3)
    reprojection_px: np.ndarray  # (V,) per-view error
    parallax_deg: float
    in_front: bool
    converged: bool

    @property
    def sigma(self):
        """1-sigma position uncertainty: sqrt of the largest covariance eigenvalue
        (inf when the covariance isn't finite, e.g. a point behind a camera)."""
        if not np.all(np.isfinite(self.cov)):
            return float("inf")
        return float(np.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0)))


def _dlt(cameras, normalized):
    rows = []
    for cam, (x, y) in zip(cameras, normalized):
        P = np.hstack([np.asarray(cam.R), np.asarray(cam.t).reshape(3, 1)])
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def _residuals(cameras, pixels, X):
    out = []
    for cam, p in zip(cameras, pixels):
        q, _ = cam.project(X[None])
        out.append(q[0] - p)
    return np.concatenate(out)


def _jacobian(cameras, X, h=1e-6):
    cols = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = h * max(1.0, abs(X[k]))
        plus = np.concatenate([c.project((X + d)[None])[0][0] for c in cameras])
        minus = np.concatenate([c.project((X - d)[None])[0][0] for c in cameras])
        cols.append((plus - minus) / (2 * d[k]))
    return np.column_stack(cols)


def triangulate(cameras, pixels, sigma_px=DEFAULT_SIGMA_PX, iterations=20):
    """cameras: [Camera] (>= 2); pixels: (V,2) observations, one per camera."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    if len(cameras) < 2 or len(cameras) != len(pixels):
        raise ValueError("need one pixel observation per camera, from at least two cameras")
    normalized = np.vstack([c.undistort(p[None]) for c, p in zip(cameras, pixels)])
    X = _dlt(cameras, normalized)
    nan = np.full(3, np.nan)
    if X is None or not np.all(np.isfinite(X)):
        return TriangulatedPoint(nan, np.full((3, 3), np.nan), np.full(len(cameras), np.nan),
                                 0.0, False, False)
    converged = False
    for _ in range(iterations):
        r = _residuals(cameras, pixels, X)
        J = _jacobian(cameras, X)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
            break
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        X = X + step
        if np.linalg.norm(step) < 1e-10 * max(1.0, np.linalg.norm(X)):
            converged = True
            break
    r = _residuals(cameras, pixels, X).reshape(-1, 2)
    errors = np.linalg.norm(r, axis=1)
    J = _jacobian(cameras, X)
    try:
        cov = sigma_px ** 2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    depths = [c.to_camera(X[None])[0, 2] for c in cameras]
    rays = [X - c.center for c in cameras]
    parallax = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            a, b = rays[i], rays[j]
            cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            parallax = max(parallax, float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return TriangulatedPoint(X, cov, errors, parallax, bool(min(depths) > 0), converged)


def check_point(p: TriangulatedPoint, max_reprojection_px=MAX_REPROJECTION_PX,
                min_parallax_deg=MIN_PARALLAX_DEG, max_sigma_m=None):
    """Reasons this triangulation is not a valid measurement ([] = valid)."""
    reasons = []
    if not np.all(np.isfinite(p.xyz)):
        return ["triangulation failed (rays don't meet)"]
    if not p.in_front:
        reasons.append("behind a camera")
    errors = np.asarray(p.reprojection_px, float)
    worst = float(errors.max()) if np.all(np.isfinite(errors)) else None
    if worst is None:
        reasons.append("doesn't reproject into every view")
    elif not worst <= max_reprojection_px:
        reasons.append(f"reprojection error {worst:.1f} px > {max_reprojection_px} px "
                       "(views disagree: mislabelled point, wrong frame pairing or "
                       "calibration)")
    if p.parallax_deg < min_parallax_deg:
        reasons.append(f"rays only {p.parallax_deg:.1f}° apart (< {min_parallax_deg}°): "
                       "depth is ill-conditioned")
    if max_sigma_m is not None and not p.sigma <= max_sigma_m:
        reasons.append(f"position uncertainty {p.sigma * 1000:.1f} mm > "
                       f"{max_sigma_m * 1000:.1f} mm")
    return reasons
