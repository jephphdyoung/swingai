"""scripts/geometry_3d.py on rendered checkerboard images and a saved synthetic shot."""

import importlib.util
import json
import os

import cv2
import numpy as np
import pytest

from _geometry_fixtures import Capture, _digest
from utils.camera_model import Camera, apply_transform, look_at, make_transform

pytest.importorskip("jsonschema")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS, ROWS, SQ = 7, 4, 0.06          # orientable: 7 + 4 is odd


def _cli():
    spec = importlib.util.spec_from_file_location(
        "geometry_cli", os.path.join(REPO, "scripts", "geometry_3d.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(cam, board_to_world):
    """Ray-cast every pixel onto the board plane: white margin, black squares."""
    h, w = cam.height, cam.width
    px = np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5), -1).reshape(-1, 2)
    rays = cam.rays(px)
    B = np.asarray(board_to_world)
    n, o = B[:3, 2], B[:3, 3]
    s = ((o - cam.center) @ n) / (rays @ n)
    hit = cam.center + s[:, None] * rays
    local = (hit - o) @ B[:3, :3]                      # board coordinates
    u, v = local[:, 0] / SQ + 1, local[:, 1] / SQ + 1  # squares start one square before
    inside = (s > 0) & (u >= 0) & (u < COLS + 1) & (v >= 0) & (v < ROWS + 1)
    black = inside & ((np.floor(u) + np.floor(v)) % 2 == 0)
    img = np.full(len(px), 255, np.uint8)
    img[black] = 0
    img = img.reshape(h, w)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def _cams():
    dist = (-0.08, 0.02, 0.0, 0.0, 0.0)
    a = Camera("down_the_line", 640, 480, 620.0, 620.0, 322.0, 238.0, dist)
    b = Camera("face_on", 640, 480, 600.0, 600.0, 318.0, 242.0, dist)
    a = a.with_pose(*look_at((-1.6, 0.1, 1.0), (0.2, 0.15, 0.0)))
    b = b.with_pose(*look_at((0.25, -1.5, 1.0), (0.2, 0.15, 0.0)))
    return a, b


def _poses(rng, n):
    out = [make_transform(np.eye(3), [0.0, 0.0, 0.0])]          # flat on the ground
    while len(out) < n:
        R, _ = cv2.Rodrigues(rng.normal(scale=0.3, size=3))
        out.append(make_transform(R, rng.normal(scale=0.08, size=3) + [0, 0, 0.25]))
    return out


def test_calibrate_check_and_verify_from_rendered_boards(tmp_path):
    cli = _cli()
    a_true, b_true = _cams()
    rng = np.random.default_rng(0)
    # Intrinsics: board views filling each camera's image.
    for cam in (a_true, b_true):
        d = tmp_path / cam.camera_id
        d.mkdir()
        k = 0
        while k < 14:
            R, _ = cv2.Rodrigues(rng.normal(scale=0.35, size=3))
            depth = rng.uniform(0.6, 1.1)
            local = make_transform(R, [rng.uniform(-0.25, 0.05), rng.uniform(-0.2, 0.0), depth])
            B = cam.camera_to_frame @ local                     # board pose in the world
            corners = apply_transform(B, cv2_board())
            px, z = cam.project(corners)
            if not ((z > 0).all() and cam.in_image(px, margin=-15).all()):
                continue
            cv2.imwrite(str(d / f"{k:02d}.png"), _render(cam, B))
            k += 1
        assert cli.main(["calibrate-intrinsics", "--camera", cam.camera_id, "--images",
                         str(d / "*.png"), "--board", f"{COLS}x{ROWS}", "--square-mm",
                         str(SQ * 1000), "--out", str(tmp_path / f"{cam.camera_id}.json")]) == 0
        got = json.load(open(tmp_path / f"{cam.camera_id}.json"))
        assert got["intrinsics"]["fx"] == pytest.approx(cam.fx, rel=0.02)
    # Extrinsics: the same board in both views; the first pose lies on the ground.
    da, db = tmp_path / "pair_a", tmp_path / "pair_b"
    da.mkdir(), db.mkdir()
    for k, B in enumerate(_poses(rng, 10)):
        cv2.imwrite(str(da / f"{k:02d}.png"), _render(a_true, B))
        cv2.imwrite(str(db / f"{k:02d}.png"), _render(b_true, B))
    cal_path = tmp_path / "fox.json"
    assert cli.main(["calibrate-extrinsics", "--intrinsics", str(tmp_path / "down_the_line.json"),
                     str(tmp_path / "face_on.json"), "--images-a", str(da / "*.png"),
                     "--images-b", str(db / "*.png"), "--board", f"{COLS}x{ROWS}",
                     "--square-mm", str(SQ * 1000), "--reference-pair", "0",
                     "--board-on-ground", "--calibration-id", "fox-test", "--by", "tester",
                     "--out", str(cal_path)]) == 0
    doc = json.load(open(cal_path))
    assert doc["status"]["status"] == "estimated" and doc["ground"] is not None
    # The rig frame is the ground board's, up to its in-plane orientation (corner
    # order), so compare what doesn't depend on that.
    from utils.calibration import load_calibration
    _, cams, _ = load_calibration(str(cal_path))
    ca, cb = cams["down_the_line"], cams["face_on"]
    assert ca.center[2] == pytest.approx(a_true.center[2], abs=0.02)      # height above board
    assert cb.center[2] == pytest.approx(b_true.center[2], abs=0.02)
    assert np.linalg.norm(ca.center - cb.center) == pytest.approx(
        np.linalg.norm(a_true.center - b_true.center), abs=0.02)
    rel, rel_true = cb.R @ ca.R.T, b_true.R @ a_true.R.T
    assert np.degrees(np.arccos(np.clip((np.trace(rel.T @ rel_true) - 1) / 2, -1, 1))) < 0.5
    # Known-distance check: a 0.5 m stick.
    ends = np.array([[0.0, 0.3, 0.0], [0.5, 0.3, 0.0]])
    pa = [f"{x:.2f},{y:.2f}" for x, y in a_true.project(ends)[0]]
    pb = [f"{x:.2f},{y:.2f}" for x, y in b_true.project(ends)[0]]
    out = tmp_path / "fox-v2.json"
    assert cli.main(["check-distance", "--calibration", str(cal_path), "--a", *pa, "--b", *pb,
                     "--known-m", "0.5", "--tolerance-mm", "5", "--by", "tester",
                     "--out", str(out)]) == 0
    v2 = json.load(open(out))
    assert v2["status"]["status"] == "verified" and v2["version"] == 2
    assert cli.main(["check-distance", "--calibration", str(cal_path), "--a", *pa, "--b", *pb,
                     "--known-m", "0.55", "--tolerance-mm", "5"]) == 1


def cv2_board():
    from utils.calibration import board_object_points
    return board_object_points(COLS, ROWS, SQ).astype(float)


def _fake_index(sid, n=100):
    from utils.frame_access import FrameIndex
    return FrameIndex(path="x", file_sha256="0" * 64, size_bytes=1, frame_count=n,
                      width=1440, height=1080,
                      timestamps_ms=tuple(f * 1000 / 240 for f in range(n)),
                      digests=tuple(_digest(sid, f) for f in range(n)), nominal_fps=240.0,
                      reported_frame_count=n, codec="x", decoder="x")


def test_reconstruct_writes_artifacts_and_measurements(tmp_path, monkeypatch):
    from utils import catalog as cm
    from utils import reconstruction_io as rio
    monkeypatch.setenv("SWINGAI_DATA_DIR", str(tmp_path))
    cap = Capture(tmp_path)
    cm.save_catalog(cap.cat)
    cm.atomic_write_json(str(tmp_path / "annotations" / f"{cap.shot_id}.json"), cap.annotations)
    cal_path = tmp_path / "cal.json"
    json.dump(cap.calibration[0], open(cal_path, "w"))
    cli = _cli()
    monkeypatch.setattr(cli, "_frame_indexes",
                        lambda cat, shot: {s["stream_id"]: _fake_index(s["stream_id"])
                                           for s in shot["streams"]})
    out = tmp_path / "out"
    assert cli.main(["reconstruct", "--shot", cap.shot_id, "--calibration", str(cal_path),
                     "--out", str(out)]) == 0
    (rid,) = os.listdir(out / cap.shot_id)
    d = out / cap.shot_id / rid
    m, _ = rio.load(str(d))
    meas = json.load(open(d / "measurements.json"))
    assert meas["manifest_sha256"] == rio.manifest_sha256(m)
    assert meas["club"]["values"]["clubhead_speed_m_s"]["status"] == "measured"
    assert meas["alignment_p1"]["shoulder_open_deg"]["status"] == "measured"

    # A single-view shot is refused.
    cat = cm.load_catalog()
    cm.add_shot(cat, golfer_id="golfer", source_id="src-fo", camera_view="face_on",
                start_frame=0, end_frame=99, swing_type="full", shot_id="shot-single")
    cm.save_catalog(cat)
    doc = {**cap.annotations, "shot_id": "shot-single", "revision": 0}
    cm.atomic_write_json(str(tmp_path / "annotations" / "shot-single.json"), doc)
    assert cli.main(["reconstruct", "--shot", "shot-single", "--calibration", str(cal_path),
                     "--out", str(out)]) == 2
