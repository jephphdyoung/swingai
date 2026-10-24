"""Calibrated pinhole cameras with lens distortion, and rigid transforms.

A `Camera` is the OpenCV model SwingAI calibrates with (`utils/calibration.py`):

- intrinsics fx, fy, cx, cy (pixels, no skew);
- Brown-Conrady distortion (k1, k2, p1, p2, k3) -- OpenCV's 5-coefficient
  model, so values from `cv2.calibrateCamera` are used unchanged;
- a pose: `R`, `t` map a point X in the camera's reference frame (the rig or
  the world frame) into camera coordinates, x_cam = R @ X + t, with the camera
  looking along its +z axis.

Pixel coordinates are the source frame's own pixels (top-left origin, x right,
y down), the same coordinates the shot labels are stored in, so a label maps
to a ray without any rescaling.

`camera_to_frame` is the inverse pose as a 4x4 matrix; it is what contracts
store ("camera-to-world" once the shot's world frame is defined).
"""

from dataclasses import dataclass, field, replace

import numpy as np

DISTORTION_MODEL = "opencv_brown_conrady_5"


# --- rigid transforms -------------------------------------------------------

def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def invert_transform(T):
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def apply_transform(T, X):
    """Apply a 4x4 rigid transform to (N,3) or (3,) points."""
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    return X @ T[:3, :3].T + T[:3, 3]


def is_rigid(T, tol=1e-6):
    """A proper rotation (det +1, orthonormal) and a homogeneous last row."""
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def look_at(center, target, up=(0, 0, 1)):
    """World->camera (R, t) for a camera at `center` looking at `target`, image
    y pointing down (OpenCV). For building synthetic rigs and tests."""
    center, target, up = (np.asarray(v, float) for v in (center, target, up))
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("view direction is parallel to up")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


# --- camera -----------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)     # k1, k2, p1, p2, k3
    R: np.ndarray = field(default_factory=lambda: np.eye(3), compare=False)
    t: np.ndarray = field(default_factory=lambda: np.zeros(3), compare=False)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def center(self):
        """Camera centre in the reference frame."""
        return -np.asarray(self.R).T @ np.asarray(self.t)

    @property
    def camera_to_frame(self):
        return invert_transform(make_transform(self.R, self.t))

    def with_pose(self, R, t):
        return replace(self, R=np.asarray(R, float), t=np.asarray(t, float).reshape(3))

    def with_camera_to_frame(self, T):
        W = invert_transform(T)
        return self.with_pose(W[:3, :3], W[:3, 3])

    def reframed(self, T_old_to_new):
        """The same physical camera expressed in another reference frame."""
        return self.with_camera_to_frame(np.asarray(T_old_to_new) @ self.camera_to_frame)

    # projection -------------------------------------------------------------

    def to_camera(self, X):
        return np.atleast_2d(np.asarray(X, float)) @ np.asarray(self.R).T + np.asarray(self.t)

    def distort(self, xn):
        """Normalized undistorted (N,2) -> normalized distorted (N,2)."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.column_stack([xd, yd])

    def project(self, X):
        """(N,3) points -> ((N,2) pixels, (N,) depth along the optical axis).
        Points behind the camera project to NaN."""
        Xc = self.to_camera(X)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, :2] / z[:, None]
        d = self.distort(xn)
        px = np.column_stack([self.fx * d[:, 0] + self.cx, self.fy * d[:, 1] + self.cy])
        px[z <= 0] = np.nan
        return px, z

    def undistort(self, pixels):
        """(N,2) pixels -> (N,2) normalized undistorted image coordinates
        (iterative inverse of `distort`, to 1e-12)."""
        import cv2
        p = np.asarray(pixels, float).reshape(-1, 1, 2)
        if not any(self.dist):
            xn = (p.reshape(-1, 2) - [self.cx, self.cy]) / [self.fx, self.fy]
            return xn
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)
        out = cv2.undistortPointsIter(p, self.K, np.asarray(self.dist, float), None, None, crit)
        return out.reshape(-1, 2)

    def rays(self, pixels):
        """Unit ray directions in the reference frame for (N,2) pixels."""
        xn = self.undistort(pixels)
        d_cam = np.column_stack([xn, np.ones(len(xn))])
        d = d_cam @ np.asarray(self.R)      # R^T applied to each row
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def in_image(self, px, margin=0.0):
        px = np.atleast_2d(px)
        return ((px[:, 0] >= -margin) & (px[:, 0] <= self.width + margin)
                & (px[:, 1] >= -margin) & (px[:, 1] <= self.height + margin))

    # serialization ----------------------------------------------------------

    def to_json(self, frame_key="camera_to_rig"):
        return {"camera_id": self.camera_id,
                "image_size": {"width": int(self.width), "height": int(self.height)},
                "model": DISTORTION_MODEL,
                "intrinsics": {"fx": float(self.fx), "fy": float(self.fy),
                               "cx": float(self.cx), "cy": float(self.cy)},
                "distortion": [float(v) for v in self.dist],
                frame_key: [[float(v) for v in row] for row in self.camera_to_frame]}

    @classmethod
    def from_json(cls, d, frame_key="camera_to_rig"):
        if d.get("model") != DISTORTION_MODEL:
            raise ValueError(f"camera {d.get('camera_id')}: unsupported model {d.get('model')!r}")
        i = d["intrinsics"]
        cam = cls(camera_id=d["camera_id"], width=d["image_size"]["width"],
                  height=d["image_size"]["height"], fx=i["fx"], fy=i["fy"], cx=i["cx"],
                  cy=i["cy"], dist=tuple(float(v) for v in d["distortion"]))
        T = np.asarray(d[frame_key], float)
        if not is_rigid(T):
            raise ValueError(f"camera {d['camera_id']}: {frame_key} is not a rigid transform")
        return cam.with_camera_to_frame(T)
