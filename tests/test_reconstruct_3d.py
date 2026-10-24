"""Reconstruction from the catalog + shot labels, statuses, world frame and
artifacts, on synthetic captures with known truth."""

import copy
import json
import os

import numpy as np
import pytest

from _geometry_fixtures import FPS, Capture
from utils import catalog as cm
from utils import reconstruct_3d as r3
from utils import reconstruction_io as rio
from utils import shot_annotations as sa
from utils.camera_model import make_transform

pytest.importorskip("jsonschema")


def _rot(seed):
    import cv2
    rng = np.random.default_rng(seed)
    R, _ = cv2.Rodrigues(rng.normal(size=3))
    return make_transform(R, rng.normal(scale=1.5, size=3))


def _run(cap, **req):
    return r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, cap.calibration,
                               requirements=r3.Requirements(**req) if req else None)


def _point(rec, frame, name):
    j = rec.point_index(name)
    return rec.xyz[frame, j], rec.state[frame, j]


def test_labels_are_valid_shot_annotations(tmp_path):
    cap = Capture(tmp_path)
    assert sa.check(cap.annotations, cap.cat) == []


def test_measured_reconstruction_in_the_world_frame(tmp_path):
    cap = Capture(tmp_path, world_to_rig=_rot(7))
    rec = _run(cap)
    assert rec.world["status"] == "defined", rec.world["reasons"]
    s = rec.frame_with("static_address")
    assert rec.frames[s]["status"] == "measured"
    for name, X in cap.body.items():
        got, state = _point(rec, s, name)
        assert state == r3.STATE_OBSERVED, name
        assert np.linalg.norm(got - X) < 0.004, (name, got, X)
    # Dynamic club frames: measured, and the clubhead where the arc says.
    dyn = [i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic"]
    assert len(dyn) >= 50
    assert all(rec.frames[i]["status"] == "measured" for i in dyn)
    errs = []
    for i in dyn:
        k = rec.frames[i]["views"][0]["source_frame"]
        got, state = _point(rec, i, "clubhead_center")
        if rec.n_views[i, rec.point_index("clubhead_center")] < 2:
            continue                    # out of one camera's frame (labelled unavailable)
        assert state == r3.STATE_OBSERVED
        errs.append(np.linalg.norm(got - cap.club_truth(k)["clubhead_center"]))
    assert len(errs) >= 35 and max(errs) < 0.006
    # Uncertainty is reported and of the right order for 0.3 px noise.
    j = rec.point_index("clubhead_center")
    i = next(i for i in dyn if rec.state[i, j] == r3.STATE_OBSERVED)
    sig = np.sqrt(np.trace(rec.cov[i, j]))
    assert 1e-4 < sig < 0.01


def test_each_camera_keeps_its_own_source_frames_and_timestamps(tmp_path):
    cap = Capture(tmp_path, fo_shift_frames=9)
    rec = _run(cap)
    i = next(i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic")
    dtl, fo = rec.frames[i]["views"]
    assert (dtl["stream_id"], fo["stream_id"]) == ("down_the_line", "face_on")
    assert fo["source_frame"] == dtl["source_frame"] + 9
    assert fo["timestamp_ms"] == pytest.approx(dtl["timestamp_ms"] + 9 * 1000 / FPS, abs=1e-3)
    assert rec.frames[i]["time_ms"] == dtl["timestamp_ms"]          # reference clock
    s = rec.frames[rec.frame_with("static_address")]
    assert [v["source_frame"] for v in s["views"]] == [5, 7]        # each stream's own P1
    assert all(len(v["image_sha1"]) == 40 for v in s["views"])


def test_independent_or_unassociated_views_are_refused_as_stereo(tmp_path):
    cap = Capture(tmp_path)
    shot = cm.add_shot(cap.cat, golfer_id="golfer", source_id="src-fo", camera_view="face_on",
                       start_frame=0, end_frame=99, swing_type="full", shot_id="shot-single")
    with pytest.raises(r3.StereoInputError, match="one camera view"):
        r3.reconstruct_shot(cap.cat, shot["shot_id"], {**cap.annotations, "shot_id":
                                                      shot["shot_id"]}, cap.calibration)
    with pytest.raises(r3.StereoInputError, match="different shot"):
        r3.reconstruct_shot(cap.cat, cap.shot_id, {**cap.annotations, "shot_id": "other"},
                            cap.calibration)
    doc, cams, sha = cap.calibration
    with pytest.raises(r3.StereoInputError, match="no camera for stream face_on"):
        r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations,
                            (doc, {"down_the_line": cams["down_the_line"]}, sha))
    from dataclasses import replace
    small = {k: replace(c, width=720, height=540) for k, c in cams.items()}
    with pytest.raises(r3.StereoInputError, match="calibrated at 720x540"):
        r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, (doc, small, sha))


def test_unknown_synchronization_pairs_only_the_static_address(tmp_path):
    cap = Capture(tmp_path, sync_status="unknown", calibration_status="verified")
    rec = _run(cap)
    assert [f["kind"] for f in rec.frames] == ["static_address"]
    assert rec.frames[0]["status"] == "measured"
    assert "synchronization unknown" in rec.association["notes"][0]


def test_estimated_or_loose_synchronization_leaves_moving_frames_unverified(tmp_path):
    rec = _run(Capture(tmp_path, sync_status="estimated"))
    dyn = [f for f in rec.frames if f["kind"] == "dynamic"]
    assert dyn and all(f["status"] == "unverified" for f in dyn)
    assert "synchronization is estimated" in dyn[0]["reasons"][0]
    rec = _run(Capture(tmp_path, sync_uncertainty_ms=3.0))
    dyn = [f for f in rec.frames if f["kind"] == "dynamic"]
    assert all(any("uncertainty 3.0 ms" in r for r in f["reasons"]) for f in dyn)


def test_a_wrong_offset_or_unsynchronized_exposures_never_pair(tmp_path):
    # Offset recorded as 0 although face-on started 9 frames early: pairing by
    # time joins different instants, which the views then disagree about.
    rec = _run(Capture(tmp_path, sync_offset_ms=0.0))
    j = rec.point_index("clubhead_center")
    dyn = [i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic"]
    assert all(rec.state[i, j] == r3.STATE_MISSING for i in dyn)
    assert any("reprojection error" in m["reason"] for m in rec.missing)
    # Free-running cameras 1.3 ms apart: no exposure pair is close enough.
    rec = _run(Capture(tmp_path, fo_phase_ms=1.3))
    assert all(f["kind"] == "static_address" for f in rec.frames)
    # ...unless the tolerance is widened, when the motion error shows up.
    rec = _run(Capture(tmp_path, fo_phase_ms=1.3), max_pair_gap_ms=2.0)
    dyn = [i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic"]
    assert dyn and all(rec.frames[i]["pair_gap_ms"] == pytest.approx(1.3, abs=0.01)
                       for i in dyn)


def test_corrupted_hidden_and_single_view_labels(tmp_path):
    cap = Capture(tmp_path)
    p1 = cap.annotations["streams"]["face_on"]["frames"]["7"]["points"]
    p1["left_knee"]["x"] += 40                                  # mislabelled in one view
    p1["right_hip"]["visible"] = False                          # placed as hidden
    del p1["left_wrist"]                                        # only DTL has it
    rec = _run(cap)
    s = rec.frame_with("static_address")
    assert _point(rec, s, "left_knee")[1] == r3.STATE_MISSING
    assert _point(rec, s, "right_hip")[1] == r3.STATE_INFERRED
    assert _point(rec, s, "left_wrist")[1] == r3.STATE_MISSING
    reasons = {m["point"]: m["reason"] for m in rec.missing if m["frame"] == s}
    assert "reprojection error" in reasons["left_knee"]
    assert reasons["left_wrist"] == "seen in one view only"
    assert np.isnan(_point(rec, s, "left_knee")[0]).all()


def test_estimated_timestamps_and_unverified_calibration_are_not_measured(tmp_path):
    cap = Capture(tmp_path)
    f = cap.annotations["streams"]["face_on"]["frames"]["80"]
    f["timestamp_basis"] = "estimated"
    rec = _run(cap)
    bad = next(fr for fr in rec.frames if any(v["source_frame"] == 80 and
                                              v["stream_id"] == "face_on" for v in fr["views"]))
    assert bad["status"] == "unverified" and "estimated timestamp" in bad["reasons"][0]
    rec = _run(Capture(tmp_path, ts_status="unverified"))
    dyn = [f for f in rec.frames if f["kind"] == "dynamic"]
    assert all(any("source timestamps are unverified" in r for r in f["reasons"]) for f in dyn)
    rec = _run(Capture(tmp_path, calibration_status="estimated"))
    assert all(f["status"] == "unverified" for f in rec.frames)
    assert "not verified" in rec.frames[0]["reasons"][0]


def test_world_frame_stays_undefined_without_scene_evidence(tmp_path):
    cap = Capture(tmp_path, world_to_rig=_rot(8))
    pts = cap.annotations["streams"]["down_the_line"]["frames"]["5"]["points"]
    del pts["floor_c"], pts["target_line_front"]
    rec = _run(cap)
    assert rec.world["status"] == "undefined"
    assert any("floor points" in r for r in rec.world["reasons"])
    assert any("target-line" in r for r in rec.world["reasons"])
    # Coordinates stay in the rig frame: the truth mapped by world->rig.
    s = rec.frame_with("static_address")
    X = cap.body["left_shoulder"]
    rig_truth = cap.world_to_rig[:3, :3] @ X + cap.world_to_rig[:3, 3]
    assert np.linalg.norm(_point(rec, s, "left_shoulder")[0] - rig_truth) < 0.004


def test_world_frame_from_an_operator_confirmed_ground_board(tmp_path):
    cap = Capture(tmp_path)
    pts = cap.annotations["streams"]["down_the_line"]["frames"]["5"]["points"]
    for n in ("floor_a", "floor_b", "floor_c"):
        del pts[n]
    doc, cams, _ = cap.calibration
    doc = copy.deepcopy(doc)
    doc["ground"] = {"normal": [0.0, 0.0, 1.0], "d": 0.0, "method": "board on the ground",
                     "rms_m": None, "n_points": None}
    from utils.calibration import load_calibration
    rec = r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, load_calibration(doc))
    assert rec.world["status"] == "defined"
    assert rec.world["evidence"]["ground"]["method"] == "board on the ground"


def test_labels_on_the_wrong_image_are_refused(tmp_path):
    from utils.frame_access import FrameIndex
    cap = Capture(tmp_path)

    def index(sid):
        n = 100
        return FrameIndex(path="x", file_sha256="0" * 64, size_bytes=1, frame_count=n,
                          width=1440, height=1080,
                          timestamps_ms=tuple(f * 1000 / FPS for f in range(n)),
                          digests=tuple(_d(sid, f) for f in range(n)), nominal_fps=FPS,
                          reported_frame_count=n, codec="x", decoder="x")
    from _geometry_fixtures import _digest as _d
    good = {s: index(s) for s in ("down_the_line", "face_on")}
    r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, cap.calibration, indexes=good)
    cap.annotations["streams"]["face_on"]["frames"]["7"]["image_sha1"] = _d("face_on", 6)
    with pytest.raises(r3.StereoInputError, match="placed on a different image"):
        r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, cap.calibration,
                            indexes=good)


# --- artifacts ---------------------------------------------------------------------

def test_artifacts_are_versioned_linked_and_reproducible(tmp_path):
    cap = Capture(tmp_path)
    rec = _run(cap)
    d = rio.write(rec, str(tmp_path / "out"))
    m, arrays = rio.load(d)
    assert m["shot_id"] == "shot-synthetic" and m["reconstruction_id"].startswith("rec-")
    assert m["inputs"]["calibration"]["calibration_id"] == "synthetic-rig"
    assert {s["sha256"] for s in m["inputs"]["sources"]} == {
        cap.cat.sources["src-dtl"]["sha256"], cap.cat.sources["src-fo"]["sha256"]}
    assert m["algorithm"]["version"] == r3.ALGORITHM["version"]
    assert m["arrays"]["coordinates"] == "world"
    assert all(c["camera_to_world"] is not None for c in m["cameras"])
    assert np.allclose(arrays["xyz"], rec.xyz, equal_nan=True)
    assert arrays["state"].dtype == np.int8
    # Same inputs -> same id and identical array content.
    again = rio.write(_run(cap), str(tmp_path / "out2"))
    m2, _ = rio.load(again)
    assert m2["reconstruction_id"] == m["reconstruction_id"]
    assert m2["arrays"]["layout"] == m["arrays"]["layout"]
    # Different parameters -> a different reconstruction.
    other = rio.write(_run(cap, max_reprojection_px=2.0), str(tmp_path / "out3"))
    assert rio.load(other)[0]["reconstruction_id"] != m["reconstruction_id"]


def test_tampered_artifacts_are_refused(tmp_path):
    d = rio.write(_run(Capture(tmp_path)), str(tmp_path / "out"))
    path = os.path.join(d, "trajectory.npz")
    data = bytearray(open(path, "rb").read())
    data[-40] ^= 0xFF
    open(path, "wb").write(bytes(data))
    with pytest.raises(rio.ArtifactError):
        rio.load(d)
    d = rio.write(_run(Capture(tmp_path)), str(tmp_path / "out2"))
    m = json.load(open(os.path.join(d, "manifest.json")))
    m["arrays"]["layout"]["xyz"]["sha256"] = "0" * 64
    json.dump(m, open(os.path.join(d, "manifest.json"), "w"))
    with pytest.raises(rio.ArtifactError, match="array xyz"):
        rio.load(d)
