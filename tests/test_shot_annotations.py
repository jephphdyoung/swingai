"""Per-shot labels on exact source frames; edits as reversible transforms;
setup-measurement readiness."""

import json
import os

import pytest

from _catalog_fixtures import dataset, golfers, register  # noqa: F401  (fixture)
from _video_fixtures import seek_trap_video
from utils import catalog as cm
from utils import frame_access as fa
from utils import label_points as lp
from utils import setup_measurements as sm
from utils import shot_annotations as sa
from utils import shot_edits as se
from utils.joint_labels import decode_frames, frame_digest

pytest.importorskip("jsonschema")


def _setup(d, key="gw_long", view="face_on", start=0, end=None):
    cat = golfers(cm.load_catalog())
    src_id = register(cat, d, key, "grant-waite", view)
    src = cat.source(src_id)
    end = src["video"]["frame_count"] - 1 if end is None else end
    shot = cm.add_shot(cat, golfer_id="grant-waite", source_id=src_id, camera_view=view,
                       start_frame=start, end_frame=end, swing_type="full")
    cm.save_catalog(cat)
    index = fa.get_frame_index(d.videos[key])
    return cat, shot, src, index, shot["streams"][0]["stream_id"]


def test_p_positions_survive_save_and_reload_and_can_change_or_be_removed(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    doc = sa.load(shot["shot_id"])
    for p, f in (("P1", 10), ("P4", 40), ("P7", 70), ("P10", 110)):
        sa.set_position(doc, shot, src, index, sid, p, f, by="tester")
    sa.save(doc, cat)

    doc = sa.load(shot["shot_id"])
    assert sa.positions(doc, sid) == {"P1": 10, "P4": 40, "P7": 70, "P10": 110}
    lab = doc["streams"][sid]["positions"]["P7"]
    assert lab["image_sha1"] == index.digests[70]
    assert lab["timestamp_ms"] == pytest.approx(index.timestamps_ms[70], abs=1e-3)
    assert doc["p_convention"] == "p7-impact"

    sa.set_position(doc, shot, src, index, sid, "P4", 42, by="tester")
    sa.clear_position(doc, sid, "P10")
    sa.save(doc, cat)
    doc = sa.load(shot["shot_id"])
    assert sa.positions(doc, sid) == {"P1": 10, "P4": 42, "P7": 70}
    assert doc["revision"] == 2
    with pytest.raises(cm.CatalogError):
        sa.set_position(doc, shot, src, index, sid, "P11", 1)
    with pytest.raises(cm.CatalogError):
        sa.set_position(doc, shot, src, index, sid, "P2", index.frame_count)


def test_swing_window_is_separate_from_p1_and_p10(dataset):
    cat, shot, src, index, sid = _setup(dataset, start=20, end=90)
    doc = sa.load(shot["shot_id"])
    sa.set_position(doc, shot, src, index, sid, "P1", 25)
    sa.set_position(doc, shot, src, index, sid, "P10", 95)       # past the window: allowed
    sa.set_position(doc, shot, src, index, sid, "P4", 22)        # before P1: flagged
    sa.save(doc, cat)
    warnings = sa.window_warnings(doc, cat.shot(shot["shot_id"]))
    assert any("P10 (frame 95) is outside the swing window 20–90" in w for w in warnings)
    assert any("P4 (frame 22) is earlier than P1 (frame 25)" in w for w in warnings)
    assert cat.shot(shot["shot_id"])["streams"][0]["window"] == {"start_frame": 20,
                                                                 "end_frame": 90}


def _seeked_digest(path, f):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    return frame_digest(img) if ok else None


def test_labels_land_on_the_exact_frame_where_seeking_would_be_off_by_one(dataset):
    path = seek_trap_video(dataset.ref, n=700, name="trap.mp4")
    dataset.videos["trap"] = path
    cat, shot, src, index, sid = _setup(dataset, key="trap")
    f = next(f for f in range(400, index.frame_count, 5)
             if _seeked_digest(path, f) != index.digests[f])
    doc = sa.load(shot["shot_id"])
    sa.set_position(doc, shot, src, index, sid, "P7", f)
    sa.set_point(doc, shot, src, index, sid, f, "ball_center", 40.0, 30.0)
    sa.save(doc, cat)
    want = frame_digest(decode_frames(path, [f])[f])
    doc = sa.load(shot["shot_id"])
    assert doc["streams"][sid]["positions"]["P7"]["image_sha1"] == want
    assert doc["streams"][sid]["frames"][str(f)]["image_sha1"] == want
    assert fa.read_frame(path, f).digest == want
    assert sa.identity_problems(doc, sid, index) == []


def test_points_origins_unavailable_and_review(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    doc = sa.load(shot["shot_id"])
    f = 15
    sa.set_point(doc, shot, src, index, sid, f, "left_wrist", 100, 80, by="tester")
    sa.set_point(doc, shot, src, index, sid, f, "right_knee", 150, 200, visible=False,
                 origin="model_proposed", model="rtmpose-m", confidence=0.71)
    sa.set_point(doc, shot, src, index, sid, f, "shaft_hosel", 160, 220, origin="model_proposed",
                 model="rtmpose-m")
    sa.mark_unavailable(doc, shot, src, index, sid, f, "clubhead_toe", "motion_blur")
    with pytest.raises(cm.CatalogError, match="reviewer"):
        sa.set_frame_review(doc, sid, f, "reviewed", by="")
    sa.set_frame_review(doc, sid, f, "reviewed", by="tester")
    sa.save(doc, cat)

    doc = sa.load(shot["shot_id"])
    fl = sa.frame_labels(doc, sid, f)
    assert fl["review"]["status"] == "reviewed"
    assert fl["points"]["right_knee"]["origin"] == "model_proposed"   # accepted as placed
    assert fl["points"]["right_knee"]["visible"] is False
    assert fl["unavailable"] == {"clubhead_toe": "motion_blur"}

    # Moving a proposal makes it a correction and reopens review.
    sa.set_point(doc, shot, src, index, sid, f, "right_knee", 152, 199, by="tester")
    fl = sa.frame_labels(doc, sid, f)
    assert fl["points"]["right_knee"]["origin"] == "corrected"
    assert fl["points"]["right_knee"]["model"] == "rtmpose-m"
    assert fl["review"]["status"] == "unreviewed"
    # Marking a placed point unavailable replaces it; placing clears unavailable.
    sa.mark_unavailable(doc, shot, src, index, sid, f, "left_wrist", "occluded")
    assert "left_wrist" not in fl["points"]
    sa.set_point(doc, shot, src, index, sid, f, "clubhead_toe", 170, 230)
    assert "clubhead_toe" not in fl["unavailable"]
    sa.remove_point(doc, sid, f, "shaft_hosel")
    sa.save(doc, cat)

    w, h = src["video"]["width"], src["video"]["height"]
    for bad in ((w + 1, 5), (-1, 5)):
        with pytest.raises(cm.CatalogError, match="outside"):
            sa.set_point(doc, shot, src, index, sid, f, "nose", *bad)
    with pytest.raises(cm.CatalogError, match="not a known point"):
        sa.set_point(doc, shot, src, index, sid, f, "third_eye", 1, 1)
    with pytest.raises(cm.CatalogError):
        sa.mark_unavailable(doc, shot, src, index, sid, f, "nose", "felt like it")


def test_every_labelling_group_is_available_and_the_benchmark_catalogue_is_untouched():
    assert set(sa.GROUPS) == {"body", "feet", "hands_club", "ball"}
    for needed in ("left_wrist", "right_wrist", "left_ankle", "left_heel", "ball_center",
                   "shaft_butt", "shaft_hosel", "clubhead_heel", "clubhead_toe"):
        assert needed in sa.POINT
    assert "ball_center" not in lp.POINT_NAMES
    assert lp.BENCHMARK_POINTS == tuple(n for n in lp.POINT_NAMES
                                        if lp.POINT[n]["group"] == "body")


def test_changed_image_identity_is_reported_not_repaired(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    doc = sa.load(shot["shot_id"])
    sa.set_position(doc, shot, src, index, sid, "P1", 5)
    sa.set_point(doc, shot, src, index, sid, 5, "nose", 10, 10)
    doc["streams"][sid]["positions"]["P1"]["image_sha1"] = index.digests[4]
    problems = sa.identity_problems(doc, sid, index)
    assert problems == ["P1 at frame 5: placed on a different image than this frame"]
    assert doc["streams"][sid]["positions"]["P1"]["frame"] == 5
    other = fa.get_frame_index(dataset.videos["gw_dtl"])
    with pytest.raises(cm.CatalogError, match="different bytes"):
        sa.set_position(doc, shot, src, other, sid, "P2", 6)


def test_labels_are_checked_against_the_catalog(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    doc = sa.load(shot["shot_id"])
    sa.set_point(doc, shot, src, index, sid, 3, "nose", 10, 10)
    doc["streams"]["ghost"] = json.loads(json.dumps(doc["streams"][sid]))
    with pytest.raises(cm.CatalogError, match="no such stream"):
        sa.save(doc, cat)
    assert not os.path.exists(sa.annotations_path(shot["shot_id"]))


def test_concurrent_label_saves_conflict(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    a, b = sa.load(shot["shot_id"]), sa.load(shot["shot_id"])
    sa.set_position(a, shot, src, index, sid, "P1", 1)
    sa.set_position(b, shot, src, index, sid, "P1", 2)
    sa.save(a, cat)
    with pytest.raises(cm.ConflictError):
        sa.save(b, cat)
    assert sa.positions(sa.load(shot["shot_id"]), sid) == {"P1": 1}


def test_a_shot_with_labels_cannot_be_removed(dataset):
    cat, shot, src, index, sid = _setup(dataset)
    doc = sa.load(shot["shot_id"])
    sa.set_position(doc, shot, src, index, sid, "P1", 1)
    sa.save(doc, cat)
    with pytest.raises(cm.CatalogError, match="saved labels"):
        cm.remove_shot(cat, shot["shot_id"])


# --- edits ------------------------------------------------------------------

def test_edit_transforms_round_trip():
    stream = {"window": {"start_frame": 30, "end_frame": 90},
              "crop": {"x": 40, "y": 12, "width": 100, "height": 80}}
    for k in (0, 1, 60):
        assert se.source_to_edit_frame(stream, se.edit_to_source_frame(stream, k)) == k
    assert se.source_to_edit_frame(stream, 29) is None
    with pytest.raises(se.EditRangeError):
        se.edit_to_source_frame(stream, 61)
    for x, y in ((40, 12), (139.5, 91.25)):
        assert se.edit_to_source_point(stream, *se.source_to_edit_point(stream, x, y)) == (x, y)
    assert se.point_in_crop(stream, 40, 12) and not se.point_in_crop(stream, 140, 12)
    import numpy as np
    img = np.arange(200 * 300 * 3, dtype=np.uint32).reshape(200, 300, 3)
    crop = se.crop_image(img, stream)
    assert crop.shape == (80, 100, 3)
    assert (crop[0, 0] == img[12, 40]).all()            # same pixels, no resize
    assert se.crop_image(img, {**stream, "crop": None}) is img


# --- setup measurements -----------------------------------------------------

def test_setup_readiness_reports_what_p1_labels_support(dataset):
    cat, shot, src, index, sid = _setup(dataset, view="face_on")
    doc = sa.load(shot["shot_id"])
    rows = {r["id"]: r for r in sm.setup_readiness(doc, shot, sid)}
    assert rows["ball_position"]["status"] == "no_p1"
    assert rows["spine_inclination"]["status"] == "not_this_view"
    sa.set_position(doc, shot, src, index, sid, "P1", 8)
    for name, x in (("ball_center", 120), ("left_ankle", 150), ("right_ankle", 90),
                    ("shaft_butt", 125), ("shaft_hosel", 118)):
        sa.set_point(doc, shot, src, index, sid, 8, name, x, 200)
    rows = {r["id"]: r for r in sm.setup_readiness(doc, shot, sid)}
    assert rows["ball_position"]["status"] == "observable_2d"
    assert rows["shaft_orientation"]["status"] == "observable_2d"
    assert rows["stance_width"]["status"] == "observable_2d"
    assert rows["hand_position"]["status"] == "missing_points"
    assert "left_wrist" in rows["hand_position"]["missing"]
    assert "real units" in rows["ball_position"]["calibrated"]
    spec = sm.club_specification(shot)
    assert spec["length_known"] is False and spec["length_in"] is None
