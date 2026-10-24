"""Camera calibration: intrinsics, two-camera extrinsics and the calibration
document (`schemas/geometry-3d.schema.json`, kind "camera_calibration").

Workflow (see docs/geometry-3d.md for the FOX procedure):

1. **Intrinsics per camera** from checkerboard images (`cv2.calibrateCamera`):
   focal lengths, principal point and distortion, with RMS reprojection error.
2. **Extrinsics** from frame pairs in which *both* cameras see the same board
   at the same moment. The board must be orientable (inner-corner counts
   summing to an odd number, e.g. 9x6) so both cameras number its corners
   from the same end (`detect_board(..., canonical=True)`). `cv2.stereoCalibrate` (intrinsics fixed) gives camera B
   relative to camera A over all pairs; a chosen reference pair fixes the
   **rig frame** = that board's frame, its z axis turned (if needed) to point
   toward the cameras. The board's in-plane axes depend on which corner the
   detector starts from, so they carry no meaning.
3. Optional scene evidence, which calibration alone never implies: if an
   operator confirms the reference board was **lying on the ground**, the
   ground plane is the board plane; if they confirm a board edge was laid
   **along the target line**, that gives the target direction.

Units are metres throughout (board square size in metres).
"""

import json

import numpy as np

from utils.camera_model import Camera, make_transform

SCHEMA_VERSION = "1.0"


class CalibrationError(ValueError):
    pass


def board_object_points(cols, rows, square_m):
    """Inner-corner grid of a cols x rows checkerboard in its own frame (z=0)."""
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return g


def board_is_orientable(cols, rows):
    """A board whose 180° rotation swaps square colours (cols + rows odd, e.g.
    9x6 inner corners). Only such a board tells which corner is "first", which
    two cameras must agree on for extrinsic calibration."""
    return (cols + rows) % 2 == 1


def _quad_brightness(gray, c):
    import cv2
    centre = c.mean(axis=0)
    r = max(1, int(0.15 * np.linalg.norm(c[0] - c[3])))
    x, y = int(round(centre[0])), int(round(centre[1]))
    patch = gray[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return float(cv2.mean(patch)[0]) if patch.size else 0.0


def detect_board(image_bgr, cols, rows, canonical=False):
    """Sub-pixel inner corners (N,2) or None.

    canonical=True (orientable boards only) puts the corners in one order for
    every view: the square between the first four corners is the dark one.
    The detector alone may start from either end of the board.
    """
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit).reshape(-1, 2)
    if canonical:
        if not board_is_orientable(cols, rows):
            raise CalibrationError(f"a {cols}x{rows} board looks the same rotated 180°, so "
                                   "two cameras can't agree which corner is first; use one "
                                   "whose inner-corner counts sum to an odd number (e.g. 9x6)")
        g = corners.reshape(rows, cols, 2)
        first = _quad_brightness(gray, np.array([g[0, 0], g[0, 1], g[1, 0], g[1, 1]]))
        last = _quad_brightness(gray, np.array([g[-1, -1], g[-1, -2], g[-2, -1], g[-2, -2]]))
        if first > last:
            corners = corners[::-1].copy()
    return corners


def calibrate_intrinsics(camera_id, image_size, image_points, object_points):
    """Camera (identity pose) and RMS reprojection error in pixels."""
    import cv2
    if len(image_points) < 3:
        raise CalibrationError(f"{camera_id}: at least 3 board views are needed, got "
                               f"{len(image_points)}")
    w, h = image_size
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [np.asarray(o, np.float32) for o in object_points],
        [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in image_points], (w, h), None, None)
    d = np.zeros(5)
    d[:min(5, dist.size)] = dist.ravel()[:5]
    cam = Camera(camera_id, w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], tuple(d))
    return cam, float(rms)


def solve_pose(camera, object_points, image_points):
    """Board->camera pose (R, t) and RMS reprojection error (px)."""
    import cv2
    ok, rvec, tvec = cv2.solvePnP(np.asarray(object_points, np.float64),
                                  np.asarray(image_points, np.float64), camera.K,
                                  np.asarray(camera.dist, float), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise CalibrationError(f"{camera.camera_id}: pose estimation failed")
    R, _ = cv2.Rodrigues(rvec)
    posed = camera.with_pose(R, tvec.ravel())
    px, _ = posed.project(object_points)
    rms = float(np.sqrt(np.mean(np.sum((px - image_points) ** 2, axis=1))))
    return R, tvec.ravel(), rms


def calibrate_extrinsics(cam_a, cam_b, object_points, pairs, reference_pair=0):
    """Poses of both cameras in the rig frame (the reference pair's board).

    pairs: [(image_points_a, image_points_b)] -- the same board pose seen by
    both cameras at the same moment.
    Returns (cam_a, cam_b, info) with each camera posed in the rig frame.
    """
    import cv2
    if len(pairs) < 1:
        raise CalibrationError("at least one shared board view is needed")
    obj = np.asarray(object_points, np.float32)
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    rms, *_, R_ab, T_ab, _, _ = cv2.stereoCalibrate(
        [obj] * len(pairs),
        [np.asarray(a, np.float32).reshape(-1, 1, 2) for a, _ in pairs],
        [np.asarray(b, np.float32).reshape(-1, 1, 2) for _, b in pairs],
        cam_a.K, np.asarray(cam_a.dist, float), cam_b.K, np.asarray(cam_b.dist, float),
        (cam_a.width, cam_a.height), criteria=crit, flags=flags)
    Ra, ta, rms_a = solve_pose(cam_a, obj, np.asarray(pairs[reference_pair][0], float))
    A = make_transform(Ra, ta)                 # rig -> camera A
    B = make_transform(R_ab, T_ab.ravel()) @ A  # rig -> camera B
    # A checkerboard's detected corner order can start from either end, so the
    # board normal may point away from the cameras. Turn the rig frame 180°
    # about its x axis if so (still right-handed): rig +z then always points
    # toward the cameras -- up, when the board lies on the ground. The in-plane
    # direction stays arbitrary; the world frame never relies on it.
    centers = [-(M[:3, :3].T @ M[:3, 3]) for M in (A, B)]
    flipped = bool(np.mean([c[2] for c in centers]) < 0)
    if flipped:
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        A, B = A @ F, B @ F
    cam_a = cam_a.with_pose(A[:3, :3], A[:3, 3])
    cam_b = cam_b.with_pose(B[:3, :3], B[:3, 3])
    _, _, rms_b = solve_pose(cam_b, obj, np.asarray(pairs[reference_pair][1], float))
    return cam_a, cam_b, {"stereo_rms_px": float(rms), "n_pairs": len(pairs),
                          "reference_pair": reference_pair,
                          "reference_pose_rms_px": [rms_a, rms_b],
                          "board_normal_flipped": flipped}


def calibration_document(calibration_id, version, cameras, status, method, by, at,
                         intrinsic_rms=None, extrinsic=None, ground=None,
                         target_direction=None, notes=None, created_at=None):
    """The JSON calibration record. `cameras` are posed in the rig frame."""
    from utils.catalog import now_iso
    intrinsic_rms = intrinsic_rms or {}
    cams = []
    for c in cameras:
        d = c.to_json("camera_to_rig")
        d["intrinsic_rms_px"] = intrinsic_rms.get(c.camera_id)
        cams.append(d)
    return {"schema_version": SCHEMA_VERSION, "kind": "camera_calibration",
            "calibration_id": calibration_id, "version": int(version),
            "created_at": created_at or now_iso(), "units": "m",
            "status": {"status": status, "method": method, "by": by, "at": at, "notes": None},
            "rig_frame": "board frame of the reference board pose, z toward the cameras; "
                         "in-plane axes arbitrary (detector corner order)",
            "cameras": cams,
            "extrinsic": extrinsic,
            "ground": ground, "target_direction": target_direction, "notes": notes}


def load_calibration(path_or_doc):
    """Validate a calibration document; returns (doc, {camera_id: Camera}, sha256)."""
    from utils.catalog import canonical_sha256
    from utils.geometry_contract import schema_problems
    doc = path_or_doc
    if isinstance(path_or_doc, str):
        with open(path_or_doc) as f:
            doc = json.load(f)
    problems = schema_problems(doc, "calibration: ")
    if problems:
        raise CalibrationError("; ".join(problems))
    cams = {}
    for d in doc["cameras"]:
        try:
            cams[d["camera_id"]] = Camera.from_json(d, "camera_to_rig")
        except ValueError as e:
            raise CalibrationError(str(e)) from None
    return doc, cams, canonical_sha256(doc)


def ground_from_board(R_board_to_rig=np.eye(3), t=np.zeros(3), method=None):
    """The reference board's plane as the ground, in rig coordinates. Only
    valid when an operator confirms the board was lying on the ground."""
    n = np.asarray(R_board_to_rig) @ np.array([0.0, 0.0, 1.0])
    p = np.asarray(t, float)
    return {"normal": [float(v) for v in n], "d": float(-n @ p),
            "method": method or "reference calibration board lying on the ground "
                                "(operator-confirmed)",
            "rms_m": None, "n_points": None}

