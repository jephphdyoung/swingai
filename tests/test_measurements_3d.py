"""P1 alignment and clubhead trajectory measurements against known 3D truth."""

import math

import numpy as np
import pytest

from _geometry_fixtures import FPS, TRUE_FLARE, TRUE_YAW, Arc, Capture
from utils import club_3d
from utils import reconstruct_3d as r3
from utils.alignment_3d import p1_alignment
from utils.geometry_contract import schema_problems

pytest.importorskip("jsonschema")


def _rec(cap, **req):
    return r3.reconstruct_shot(cap.cat, cap.shot_id, cap.annotations, cap.calibration,
                               requirements=r3.Requirements(**req) if req else None)


def _club(cap, rec=None, **kw):
    rec = rec or _rec(cap)
    impact = club_3d.impact_time_ms(cap.annotations, "down_the_line")
    return club_3d.club_metrics(rec, impact, cap.handedness, **kw), rec


# --- P1 alignment ----------------------------------------------------------------

@pytest.mark.parametrize("handedness", ["right", "left"])
def test_p1_alignment_recovers_the_known_lines_and_feet(tmp_path, handedness):
    cap = Capture(tmp_path, handedness=handedness)
    a = p1_alignment(_rec(cap), handedness)
    sign = 1 if handedness == "right" else -1
    for line, truth in TRUE_YAW.items():
        m = a[f"{line}_open_deg"]
        assert m["status"] == "measured", (line, m["reasons"])
        assert m["value"] == pytest.approx(truth, abs=1.0), line
        assert a[f"{line}_yaw_deg"]["value"] == pytest.approx(sign * truth, abs=1.0)
        assert 0 < m["uncertainty"] < 2.0
    assert a["lead_foot_flare_deg"]["value"] == pytest.approx(TRUE_FLARE["lead"], abs=1.5)
    assert a["trail_foot_flare_deg"]["value"] == pytest.approx(TRUE_FLARE["trail"], abs=1.5)
    assert a["shoulder_minus_pelvis_open_deg"]["value"] == pytest.approx(
        TRUE_YAW["shoulder"] - TRUE_YAW["pelvis"], abs=1.5)
    assert a["pelvis_minus_stance_open_deg"]["value"] == pytest.approx(
        TRUE_YAW["pelvis"] - TRUE_YAW["stance"], abs=1.5)
    assert "not kneecap direction" in a["knee_center_open_deg"]["method"]


def test_unobservable_and_hidden_points(tmp_path):
    cap = Capture(tmp_path)
    fo = cap.annotations["streams"]["face_on"]["frames"]["7"]["points"]
    del fo["left_knee"]                       # one view only
    fo["right_hip"]["visible"] = False        # placed as hidden
    a = p1_alignment(_rec(cap), "right")
    k = a["knee_center_open_deg"]
    assert k["status"] == "unavailable" and k["value"] is None
    assert "left_knee: seen in one view only" in k["reasons"]
    p = a["pelvis_open_deg"]
    assert p["status"] == "inferred" and p["value"] is not None
    assert "right_hip placed as hidden" in p["reasons"][0]
    assert a["shoulder_minus_pelvis_open_deg"]["status"] == "inferred"
    assert a["shoulder_open_deg"]["status"] == "measured"


def test_alignment_needs_a_world_frame_and_a_measured_address(tmp_path):
    cap = Capture(tmp_path)
    del cap.annotations["streams"]["face_on"]["frames"]["7"]["points"]["target_line_front"]
    a = p1_alignment(_rec(cap), "right")
    assert all(m["status"] == "unavailable" for m in a.values())
    assert "target-line" in a["shoulder_open_deg"]["reasons"][0]
    a = p1_alignment(_rec(Capture(tmp_path, calibration_status="estimated")), "right")
    assert all(m["status"] == "unavailable" for m in a.values())
    assert "address frame is not measured" in a["stance_open_deg"]["reasons"][0]


# --- club trajectory ---------------------------------------------------------------

def test_club_trajectory_matches_the_known_arc(tmp_path):
    cap = Capture(tmp_path)
    out, rec = _club(cap)
    v = out["values"]
    # Impact is the P7 frame's time (frame-quantized: here 1.25 ms after the
    # ball is passed), so velocity-derived truth is taken at that time.
    exp = cap.arc.expected(cap.arc_time(out["impact_time_ms"]))
    exp["low_point"] = cap.arc.low
    for k in ("clubhead_speed_m_s", "club_path_deg", "attack_angle_deg",
              "swing_direction_deg", "low_point_ahead_of_ball_m"):
        assert v[k]["status"] == "measured", (k, v[k]["reasons"])
    assert v["clubhead_speed_m_s"]["value"] == pytest.approx(exp["speed"], rel=0.02)
    assert v["clubhead_speed_mph"]["value"] == pytest.approx(exp["speed"] * 2.23694, rel=0.02)
    assert v["path_yaw_deg"]["value"] == pytest.approx(exp["path_yaw_deg"], abs=0.75)
    assert v["club_path_deg"]["value"] == pytest.approx(-exp["path_yaw_deg"], abs=0.75)
    assert v["attack_angle_deg"]["value"] == pytest.approx(exp["attack_deg"], abs=0.75)
    assert exp["attack_deg"] < 0                       # descending into the ball
    # The plane's horizontal line yaw is -3°: right of target for a right-hander.
    assert v["swing_direction_deg"]["value"] == pytest.approx(-cap.arc.plane_yaw_deg, abs=0.75)
    assert v["swing_plane_inclination_deg"]["value"] == pytest.approx(60, abs=1.5)
    # Observed low point: the lowest sample, within its stated sampling uncertainty.
    lp = out["points"]["observed_low_point"]
    assert lp["status"] == "measured"
    assert abs(lp["value"][0] - exp["low_point"][0]) <= lp["uncertainty"]
    assert v["low_point_ahead_of_ball_m"]["value"] > 0       # after the ball
    # Fitted free-arc low point: close to the true circle's, and labelled inferred.
    fit = out["points"]["fitted_free_arc_low_point"]
    assert fit["status"] == "inferred"
    assert np.linalg.norm(np.array(fit["value"]) - exp["low_point"]) < 0.01
    assert "distinct from the observed low point" in fit["method"]
    # Series: per sample, with speeds near the true constant speed.
    speeds = [s for s in out["series"]["speed_m_s"] if s is not None]
    assert speeds and np.median(speeds) == pytest.approx(exp["speed"], rel=0.03)


def test_left_handed_signs(tmp_path):
    # Mirror the swing: a left-hander's arc yawed +3° is right-of-target for them.
    cap = Capture(tmp_path, handedness="left", arc=Arc(plane_yaw_deg=3.0, tilt_deg=-60.0))
    out, _ = _club(cap)
    exp = cap.arc.expected(cap.arc_time(out["impact_time_ms"]))
    assert out["values"]["club_path_deg"]["value"] == pytest.approx(exp["path_yaw_deg"],
                                                                    abs=0.75)
    assert out["values"]["swing_direction_deg"]["value"] == pytest.approx(3.0, abs=0.75)


def test_clubhead_is_never_taken_from_the_shaft(tmp_path):
    cap = Capture(tmp_path)
    for st in cap.annotations["streams"].values():
        for fl in st["frames"].values():
            fl["points"].pop("clubhead_center", None)
    out, rec = _club(cap)
    # Heel and toe are observed, so their midpoint is the tracking centre.
    assert out["values"]["clubhead_speed_m_s"]["status"] == "measured"
    i = next(i for i, f in enumerate(rec.frames)
             if f["kind"] == "dynamic" and club_3d.tracking_center(rec, i) is not None)
    assert club_3d.tracking_center(rec, i)[2] == "heel-toe midpoint"
    for st in cap.annotations["streams"].values():
        for fl in st["frames"].values():
            for n in ("clubhead_heel", "clubhead_toe"):
                fl["points"].pop(n, None)
    out, rec = _club(cap)
    v = out["values"]
    assert all(v[k]["status"] == "unavailable" for k in
               ("clubhead_speed_m_s", "attack_angle_deg", "low_point_ahead_of_ball_m"))
    assert "tracking centre not observed in two views" in " ".join(
        v["clubhead_speed_m_s"]["reasons"])
    # ...although the shaft itself is fully observed.
    assert club_3d.club_structure(rec, i)["shaft_length_m"] == pytest.approx(0.9, abs=0.01)


def _frame_at(cap, offset_frames, stream="face_on"):
    k = 70 + offset_frames + (cap.fo_shift if stream == "face_on" else 0)
    return cap.annotations["streams"][stream]["frames"][str(k)]


def test_corrupted_samples_at_impact_make_speed_unavailable_not_wrong(tmp_path):
    cap = Capture(tmp_path)
    # Shifted across the epipolar line (here: vertically), which two views do
    # detect. A shift *along* it only moves the point in depth -- see the
    # rigidity check below for that case.
    for off in (1, 2):                                   # both samples after impact
        pts = _frame_at(cap, off)["points"]
        pts["clubhead_center"]["y"] += 30.0
        del pts["clubhead_heel"]                         # and no heel-toe fallback
    out, rec = _club(cap)
    s = out["values"]["clubhead_speed_m_s"]
    assert s["status"] == "unavailable" and s["value"] is None
    assert any("reprojection" in m["reason"] for m in rec.missing)
    assert "two usable samples each side" in s["reasons"][0]


def test_inconsistent_club_geometry_is_excluded(tmp_path):
    cap = Capture(tmp_path)
    # Move the butt 15 cm along the shaft in *both* views of one frame: it
    # triangulates cleanly but the shaft is now the wrong length.
    k = 66
    truth = cap.club_truth(k)
    moved = truth["shaft_butt"] + 0.15 * (truth["shaft_butt"] - truth["shaft_hosel"]) / 0.9
    for cam, stream in zip(cap.cams_world, ("down_the_line", "face_on")):
        px, _ = cam.project(moved[None])
        f = _frame_at(cap, k - 70, stream)["points"]["shaft_butt"]
        f["x"], f["y"] = float(px[0, 0]), float(px[0, 1])
    rec = _rec(cap)
    dyn = [i for i, f in enumerate(rec.frames) if f["kind"] == "dynamic"]
    bad = club_3d.rigidity(rec, dyn)
    (frame,) = bad
    assert rec.frames[frame]["views"][0]["source_frame"] == k
    assert "shaft_butt-shaft_hosel" in bad[frame][0]


def test_unverified_moving_frames_publish_no_trajectory(tmp_path):
    out, _ = _club(Capture(tmp_path, sync_status="estimated"))
    assert all(m["status"] == "unavailable" for m in out["values"].values())
    assert any("synchronization is estimated" in r
               for r in out["values"]["clubhead_speed_m_s"]["reasons"])
    cap = Capture(tmp_path)
    for off in (-1, 1):
        _frame_at(cap, off, "down_the_line")["timestamp_basis"] = "estimated"
    out, _ = _club(cap)
    assert out["values"]["clubhead_speed_m_s"]["status"] == "unavailable"
    assert any("estimated timestamp" in r for r in out["values"]["clubhead_speed_m_s"]["reasons"])


def test_low_point_must_be_bracketed(tmp_path):
    cap = Capture(tmp_path, club_frames=range(40, 70))     # observations stop before it
    out, _ = _club(cap)
    lp = out["values"]["low_point_ahead_of_ball_m"]
    assert lp["status"] == "unavailable" and "isn't bracketed" in lp["reasons"][0]
    # The fitted arc can still be reported -- as an inference.
    assert out["points"]["fitted_free_arc_low_point"]["status"] == "inferred"


def test_no_impact_time_or_world_frame(tmp_path):
    cap = Capture(tmp_path)
    rec = _rec(cap)
    out = club_3d.club_metrics(rec, None, "right")
    assert all(m["status"] == "unavailable" for m in out["values"].values())
    del cap.annotations["streams"]["down_the_line"]["frames"]["5"]["points"]["ball_center"]
    out, _ = _club(cap)
    assert "world frame is undefined" in out["values"]["club_path_deg"]["reasons"][0]


def test_measurements_document_is_valid(tmp_path):
    from utils import reconstruction_io as rio
    cap = Capture(tmp_path)
    rec = _rec(cap)
    d = rio.write(rec, str(tmp_path / "out"))
    m, _ = rio.load(d)
    club, _ = _club(cap, rec)
    doc = club_3d.measurements_document(rec, rio.manifest_sha256(m), m["reconstruction_id"],
                                        "right", club["impact_time_ms"],
                                        p1_alignment(rec, "right"), club)
    assert schema_problems(doc) == []
    assert doc["club"]["values"]["low_point_ahead_of_ball_m"]["status"] == "measured"
    assert math.isfinite(doc["club"]["values"]["clubhead_speed_m_s"]["value"])
    assert FPS == 240.0
