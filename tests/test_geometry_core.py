"""Camera model, calibration, triangulation and the world frame, against
synthetic truth with known camera positions."""

import cv2
import numpy as np
import pytest

from _geometry_fixtures import BALL, camera, p1_body, rig, scene
from utils import calibration as cal
from utils import world_frame as wf
from utils.camera_model import (Camera, apply_transform, invert_transform, is_rigid,
                                make_transform)
from utils.geometry_contract import schema_problems
from utils.triangulation import check_point, triangulate

pytest.importorskip("jsonschema")


def _random_rigid(seed):
    rng = np.random.default_rng(seed)
    R, _ = cv2.Rodrigues(rng.normal(size=3))
    return make_transform(R, rng.normal(scale=2.0, size=3))


# --- camera model --------------------------------------------------------------

def test_projection_matches_opencv_including_distortion():
    cam = rig()[0]
    X = np.array(list(p1_body().values()))
    px, depth = cam.project(X)
    rvec, _ = cv2.Rodrigues(cam.R)
    ref, _ = cv2.projectPoints(X, rvec, cam.t, cam.K, np.asarray(cam.dist))
    assert np.allclose(px, ref.reshape(-1, 2), atol=1e-6)
    assert (depth > 0).all() and cam.in_image(px).all()


def test_undistort_inverts_distortion():
    cam = rig()[1]
    px, _ = cam.project(np.array(list(p1_body().values())))
    xn = cam.undistort(px)
    back = cam.distort(xn) * [cam.fx, cam.fy] + [cam.cx, cam.cy]
    assert np.allclose(back, px, atol=1e-6)
    rays = cam.rays(px)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1)


def test_points_behind_the_camera_do_not_project():
    cam = rig()[0]
    px, depth = cam.project(cam.center[None] - 2 * (np.array([0.0, 0.45, 0.8]) - cam.center))
    assert depth[0] < 0 and np.isnan(px).all()


def test_camera_json_round_trip_and_rigid_checks():
    cam = rig()[0]
    back = Camera.from_json(cam.to_json())
    assert np.allclose(back.R, cam.R) and np.allclose(back.t, cam.t)
    assert back.dist == cam.dist and back.fx == cam.fx
    bad = cam.to_json()
    bad["camera_to_rig"][0][0] = 2.0
    with pytest.raises(ValueError, match="not a rigid transform"):
        Camera.from_json(bad)
    T = _random_rigid(1)
    assert is_rigid(T) and np.allclose(invert_transform(T) @ T, np.eye(4))
    mirror = np.diag([1.0, -1.0, 1.0, 1.0])
    assert not is_rigid(mirror)            # a reflection is not a proper rotation


def test_reframing_a_camera_keeps_its_projections():
    cam = rig()[1]
    T = _random_rigid(2)
    moved = cam.reframed(T)
    X = np.array(list(p1_body().values()))
    assert np.allclose(moved.project(apply_transform(T, X))[0], cam.project(X)[0], atol=1e-6)


# --- triangulation -------------------------------------------------------------

def test_noise_free_triangulation_is_exact():
    cams = rig()
    for name, X in p1_body().items():
        obs = [c.project(X[None])[0][0] for c in cams]
        p = triangulate(cams, obs)
        assert np.allclose(p.xyz, X, atol=1e-7), name
        assert p.reprojection_px.max() < 1e-6 and p.in_front and p.converged
        assert p.parallax_deg > 60
        assert check_point(p) == []


def test_covariance_predicts_the_scatter_under_pixel_noise():
    cams = rig()
    X = p1_body()["left_shoulder"]
    rng = np.random.default_rng(0)
    sigma = 0.7
    sols = []
    for _ in range(400):
        obs = [c.project(X[None])[0][0] + rng.normal(scale=sigma, size=2) for c in cams]
        p = triangulate(cams, obs, sigma_px=sigma)
        sols.append(p.xyz)
    emp = np.cov(np.array(sols).T)
    pred = p.cov
    ratio = np.sqrt(np.diag(emp) / np.diag(pred))
    assert np.all((ratio > 0.8) & (ratio < 1.25)), ratio
    assert np.linalg.norm(np.mean(sols, axis=0) - X) < 0.002


def test_corrupted_observations_are_rejected_not_absorbed():
    cams = rig()
    body = p1_body()
    X = body["left_knee"]
    good = [c.project(X[None])[0][0] for c in cams]
    shifted = [good[0], good[1] + [35.0, -20.0]]
    assert any("reprojection error" in r for r in check_point(triangulate(cams, shifted)))
    # Left/right swapped in one view: a different physical point.
    swapped = [good[0], cams[1].project(body["right_knee"][None])[0][0]]
    assert any("reprojection error" in r for r in check_point(triangulate(cams, swapped)))


def test_small_baseline_is_ill_conditioned():
    a = camera("a", (-4.0, 0.9, 1.2), (0, 0.5, 0.8))
    b = camera("b", (-4.0, 1.05, 1.2), (0, 0.5, 0.8))
    X = p1_body()["left_hip"]
    p = triangulate([a, b], [c.project(X[None])[0][0] for c in (a, b)])
    reasons = check_point(p, max_sigma_m=0.005)
    assert any("ill-conditioned" in r for r in reasons)
    assert any("uncertainty" in r for r in reasons)


def test_a_point_behind_the_cameras_is_rejected():
    # Pixels of the point mirrored through each camera centre: the viewing lines
    # meet behind both cameras, where nothing projects (NaN covariance).
    cams = rig()
    X = np.array([-6.0, -6.0, 3.0])
    assert all(c.to_camera(X[None])[0, 2] < 0 for c in cams)
    p = triangulate(cams, [c.project((2 * c.center - X)[None])[0][0] for c in cams])
    assert not p.in_front and p.sigma == np.inf
    reasons = check_point(p, max_sigma_m=0.02)
    assert "behind a camera" in reasons and any("uncertainty" in r for r in reasons)


def test_triangulation_needs_two_views():
    with pytest.raises(ValueError):
        triangulate([rig()[0]], [[10.0, 10.0]])


# --- world frame -----------------------------------------------------------------

def _world_from_rig(T_world_to_rig, cams_rig):
    s = {k: apply_transform(T_world_to_rig, v) for k, v in scene().items()}
    ground = wf.ground_from_points([s["floor_a"], s["floor_b"], s["floor_c"]], "floor points")
    target = wf.target_direction_from_points(s["target_line_back"], s["target_line_front"])
    ball = apply_transform(T_world_to_rig, BALL)
    return wf.define_world_frame(ground, target, ball, cams_rig[0].center, "marker", "ball")


def test_world_frame_axes_follow_the_convention_from_explicit_evidence():
    T = _random_rigid(3)                          # world -> rig (an arbitrary rig frame)
    cams_rig = [c.reframed(T) for c in rig()]
    frame = _world_from_rig(T, cams_rig)
    M = frame.rig_to_world
    assert is_rigid(M) and np.isclose(np.linalg.det(M[:3, :3]), 1.0)
    assert np.allclose(M @ T, np.eye(4), atol=1e-9)        # recovers the true world frame
    origin = frame.to_world(apply_transform(T, np.array([0, 0, 0.0])))
    assert np.allclose(origin, 0, atol=1e-9)
    # +X toward the target, +Y to its left, +Z up.
    x = frame.to_world(apply_transform(T, np.array([1.0, 0, 0])))
    y = frame.to_world(apply_transform(T, np.array([0, 1.0, 0])))
    assert np.allclose(x, [1, 0, 0]) and np.allclose(y, [0, 1, 0])
    assert np.allclose(np.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
    assert frame.evidence["origin"]["ball_height_m"] == pytest.approx(BALL[2])
    for c in cams_rig:
        assert frame.to_world(c.center)[2] > 0            # cameras above the ground


def test_world_frame_refuses_missing_or_degenerate_evidence():
    s = scene()
    ground = wf.ground_from_points([s["floor_a"], s["floor_b"], s["floor_c"]], "floor")
    up = np.array([0, 0, 2.0])
    with pytest.raises(wf.WorldFrameError, match="ground"):
        wf.define_world_frame(None, [1, 0, 0], BALL, up, "m", "b")
    with pytest.raises(wf.WorldFrameError, match="target"):
        wf.define_world_frame(ground, None, BALL, up, "m", "b")
    with pytest.raises(wf.WorldFrameError, match="ball"):
        wf.define_world_frame(ground, [1, 0, 0], None, up, "m", "b")
    with pytest.raises(wf.WorldFrameError, match="vertical"):
        wf.define_world_frame(ground, [0.1, 0, 1.0], BALL, up, "m", "b")
    with pytest.raises(wf.WorldFrameError, match="collinear"):
        wf.ground_from_points([[0, 0, 0], [1, 0, 0], [2, 0, 0]], "floor")
    with pytest.raises(wf.WorldFrameError, match="not planar"):
        wf.ground_from_points([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.1]], "floor")
    with pytest.raises(wf.WorldFrameError, match="apart"):
        wf.target_direction_from_points([0, 0, 0], [0.1, 0, 0])


# --- calibration -------------------------------------------------------------------

def _board_views(cam, obj, n, rng):
    """Board views spread over the whole image, as a real calibration session
    should be (a centred-only set leaves the principal point unconstrained)."""
    views = []
    while len(views) < n:
        R, _ = cv2.Rodrigues(rng.normal(scale=0.4, size=3))
        depth = rng.uniform(1.2, 2.8)
        t = np.array([rng.uniform(-0.9, 0.6) * depth / 2, rng.uniform(-0.6, 0.4) * depth / 2,
                      depth])
        px, z = cam.with_pose(R, t).project(obj)
        if (z > 0).all() and cam.in_image(px, margin=-10).all():
            views.append(px)
    return views


def test_intrinsic_calibration_recovers_focal_length_and_distortion():
    truth = rig()[0]
    obj = cal.board_object_points(9, 6, 0.08)
    rng = np.random.default_rng(4)
    views = _board_views(truth, obj, 25, rng)
    views = [v + rng.normal(scale=0.15, size=v.shape) for v in views]
    cam, rms = cal.calibrate_intrinsics("down_the_line", (truth.width, truth.height), views,
                                        [obj] * len(views))
    assert rms < 0.3
    assert cam.fx == pytest.approx(truth.fx, rel=0.01)
    assert cam.cx == pytest.approx(truth.cx, abs=5)
    assert cam.dist[0] == pytest.approx(truth.dist[0], abs=0.02)
    with pytest.raises(cal.CalibrationError, match="at least 3"):
        cal.calibrate_intrinsics("x", (100, 100), views[:2], [obj] * 2)


def test_extrinsic_calibration_recovers_both_camera_positions_in_the_board_frame():
    a_true, b_true = rig()
    obj = cal.board_object_points(9, 6, 0.06)
    # The board lies on the ground in front of the golfer; rig frame = board frame.
    board_to_world = make_transform(np.eye(3), [-0.3, -0.6, 0.0])
    world_to_board = invert_transform(board_to_world)
    a_b, b_b = a_true.reframed(world_to_board), b_true.reframed(world_to_board)
    rng = np.random.default_rng(5)
    pairs = []
    for k in range(6):
        if k == 0:
            M = np.eye(4)                      # the reference pose: on the ground
        else:
            R, _ = cv2.Rodrigues(rng.normal(scale=0.25, size=3))
            M = make_transform(R, rng.normal(scale=0.2, size=3) + [0, 0, 0.4])
        X = apply_transform(M, obj)
        pairs.append(tuple(c.project(X)[0] + rng.normal(scale=0.1, size=(len(obj), 2))
                           for c in (a_b, b_b)))
    a0 = a_b.with_pose(np.eye(3), np.zeros(3))
    b0 = b_b.with_pose(np.eye(3), np.zeros(3))
    a, b, info = cal.calibrate_extrinsics(a0, b0, obj, pairs, reference_pair=0)
    assert np.linalg.norm(a.center - a_b.center) < 0.01
    assert np.linalg.norm(b.center - b_b.center) < 0.01
    assert info["stereo_rms_px"] < 0.5
    doc = cal.calibration_document(
        "fox-2026-11-01", 1, [a, b], "estimated", "synthetic", "tester", None,
        intrinsic_rms={"down_the_line": 0.2}, extrinsic={"method": "stereoCalibrate", **info},
        ground=cal.ground_from_board())
    assert schema_problems(doc) == []
    loaded, cams, sha = cal.load_calibration(doc)
    assert set(cams) == {"down_the_line", "face_on"} and len(sha) == 64
    assert np.allclose(cams["face_on"].center, b.center)


def test_detect_board_finds_corners_in_a_rendered_image():
    cols, rows, sq = 7, 5, 40
    img = np.full((rows * sq + 3 * sq, cols * sq + 3 * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = int(1.5 * sq) + r * sq, int(1.5 * sq) + c * sq
                img[y:y + sq, x:x + sq] = 0
    corners = cal.detect_board(img, cols, rows)
    assert corners is not None and corners.shape == (cols * rows, 2)
    assert cal.detect_board(np.full((200, 200), 255, np.uint8), cols, rows) is None



def test_canonical_corner_order_needs_an_orientable_board():
    def render(cols, rows, flip):
        sq = 40
        img = np.full((rows * sq + 3 * sq, cols * sq + 3 * sq), 255, np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y, x = int(1.5 * sq) + r * sq, int(1.5 * sq) + c * sq
                    img[y:y + sq, x:x + sq] = 0
        return img[::-1, ::-1].copy() if flip else img
    a = cal.detect_board(render(7, 4, False), 7, 4, canonical=True)
    b = cal.detect_board(render(7, 4, True), 7, 4, canonical=True)
    h, w = render(7, 4, False).shape
    # The same physical corner comes first whichever way up the board is seen.
    assert np.allclose(b[0], [w - 1, h - 1] - a[0], atol=1.0)
    with pytest.raises(cal.CalibrationError, match="looks the same rotated"):
        cal.detect_board(render(7, 5, False), 7, 5, canonical=True)


def test_geometry_contract_shares_the_capture_contracts_definitions():
    import json
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
    geo = json.load(open(os.path.join(root, "geometry-3d.schema.json")))["$defs"]
    cap = json.load(open(os.path.join(root, "capture-manifest.schema.json")))["$defs"]
    for name in ("identifier", "rfc3339DateTime", "relativePath", "cameraView"):
        assert geo[name] == cap[name], name
