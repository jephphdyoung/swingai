"""The swing dataset catalog: golfers, immutable sources, shots, collections."""

import json
import os
import shutil

import pytest

from _catalog_fixtures import dataset, golfers, register  # noqa: F401  (fixture)
from utils import annotations as ann
from utils import catalog as cm
from utils.frame_access import file_sha256

jsonschema = pytest.importorskip("jsonschema")


def _fresh(d):
    return golfers(cm.load_catalog())


def _shot(cat, golfer, source, view, start=0, end=20, **kw):
    return cm.add_shot(cat, golfer_id=golfer, source_id=source, camera_view=view,
                       start_frame=start, end_frame=end, swing_type=kw.pop("swing_type", "full"),
                       **kw)["shot_id"]


def test_two_golfers_keep_independent_shot_libraries(dataset):
    cat = _fresh(dataset)
    a_dtl = register(cat, dataset, "alice_dtl", "alice", "down_the_line")
    b_fo = register(cat, dataset, "bob_fo", "bob", "face_on")
    a1 = _shot(cat, "alice", a_dtl, "down_the_line")
    a2 = _shot(cat, "alice", a_dtl, "down_the_line", 30, 50)
    b1 = _shot(cat, "bob", b_fo, "face_on")
    cm.save_catalog(cat)

    cat = cm.load_catalog()
    assert {s["shot_id"] for s in cm.shots_for_golfer(cat, "alice")} == {a1, a2}
    assert {s["shot_id"] for s in cm.shots_for_golfer(cat, "bob")} == {b1}
    assert cat.shot(b1)["handedness"] == "left"        # from Bob's golfer record
    # One golfer's recording can't become another's shot...
    with pytest.raises(cm.CatalogError, match="alice's, not bob's"):
        _shot(cat, "bob", a_dtl, "down_the_line")
    # ...and users' shots never go into a (shared, reference) collection.
    with pytest.raises(cm.CatalogError, match="is a user"):
        cm.create_collection(cat, "alice-v1", "alice", "Alice")


def test_grant_waite_collection_from_independent_dtl_and_face_on(dataset):
    cat = _fresh(dataset)
    dtl = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line", source_type="youtube")
    fo = register(cat, dataset, "gw_fo", "grant-waite", "face_on", source_type="youtube")
    s_dtl = _shot(cat, "grant-waite", dtl, "down_the_line",
                  club={"category": "iron", "name": "6-iron"}, recording_date="2010-10-27")
    s_fo = _shot(cat, "grant-waite", fo, "face_on", recording_date="2009")
    # Different swings are never one shot: adding the face-on recording as a
    # second view of the DTL swing needs an explicit same-swing assertion.
    with pytest.raises(cm.CatalogError, match="same swing"):
        cm.add_stream(cat, s_dtl, source_id=fo, camera_view="face_on", start_frame=0,
                      end_frame=20, same_swing_method="", asserted_by="")
    cm.create_collection(cat, "grant-waite-full-swing-v1", "grant-waite",
                         "Grant Waite full swings v1")
    for s in (s_dtl, s_fo):
        cm.add_to_collection(cat, "grant-waite-full-swing-v1", s)
    with pytest.raises(cm.CatalogError, match="three_quarter swing"):
        cm.add_to_collection(cat, "grant-waite-full-swing-v1",
                             _shot(cat, "grant-waite", dtl, "down_the_line", 21, 39,
                                   swing_type="three_quarter"))
    cm.save_catalog(cat)

    cat = cm.load_catalog()
    coll = cat.collection("grant-waite-full-swing-v1")
    assert coll["shot_ids"] == [s_dtl, s_fo]
    for s in (s_dtl, s_fo):
        shot = cat.shot(s)
        assert len(shot["streams"]) == 1
        assert shot["multi_view"]["same_swing"]["status"] == "not_applicable"
        gate = cm.eligibility(cat, s)["stereo_3d"]
        assert not gate["eligible"] and "not a stereo measurement" in gate["reasons"][0]
    assert cat.shot(s_dtl)["club"]["name"] == "6-iron"
    assert cat.shot(s_fo)["recording_date"] == "2009"

    cm.freeze_collection(cat, "grant-waite-full-swing-v1", by="tester")
    cm.save_catalog(cat)
    cat = cm.load_catalog()
    with pytest.raises(cm.CatalogError, match="frozen"):
        cm.remove_from_collection(cat, "grant-waite-full-swing-v1", s_fo)
    with pytest.raises(cm.CatalogError, match="frozen"):
        cm.update_shot(cat, s_fo, swing_type="half")
    # Editing the record behind the API's back is caught at save.
    cat.collections["grant-waite-full-swing-v1"]["shot_ids"].pop()
    with pytest.raises(cm.CatalogError, match="frozen and can't be changed"):
        cm.save_catalog(cat)


def test_several_swings_from_one_long_source(dataset):
    cat = _fresh(dataset)
    src = register(cat, dataset, "gw_long", "grant-waite", "unknown")
    a = _shot(cat, "grant-waite", src, "face_on", 5, 35)
    b = _shot(cat, "grant-waite", src, "down_the_line", 60, 95)
    cm.save_catalog(cat)
    cat = cm.load_catalog()
    shots = cm.shots_for_source(cat, src)
    assert [s["shot_id"] for s in shots] == [a, b]
    assert [s["streams"][0]["window"] for s in shots] == [
        {"start_frame": 5, "end_frame": 35}, {"start_frame": 60, "end_frame": 95}]
    assert len(cat.sources) == 1


def test_window_and_crop_edits_never_touch_the_source(dataset):
    path = dataset.videos["gw_long"]
    before = (file_sha256(path), os.stat(path).st_mtime_ns)
    cat = _fresh(dataset)
    src = register(cat, dataset, "gw_long", "grant-waite", "face_on")
    shot = _shot(cat, "grant-waite", src, "face_on", 0, 119)
    sid = cat.shot(shot)["streams"][0]["stream_id"]
    cm.set_window(cat, shot, sid, 12, 80)
    cm.set_crop(cat, shot, sid, {"x": 40, "y": 10, "width": 200, "height": 220})
    cm.save_catalog(cat)
    cat = cm.load_catalog()
    st = cat.shot(shot)["streams"][0]
    assert st["window"] == {"start_frame": 12, "end_frame": 80}
    assert st["crop"] == {"x": 40, "y": 10, "width": 200, "height": 220}
    cm.set_window(cat, shot, sid, 20, 70)            # reopened and modified
    cm.set_crop(cat, shot, sid, None)
    cm.save_catalog(cat)
    assert cm.load_catalog().shot(shot)["streams"][0]["crop"] is None

    for bad in ((50, 40), (0, 120), (-1, 5)):
        with pytest.raises(cm.CatalogError):
            cm.set_window(cat, shot, sid, *bad)
    for crop in ({"x": 300, "y": 0, "width": 40, "height": 10},
                 {"x": 0, "y": 0, "width": 0, "height": 10}):
        with pytest.raises(cm.CatalogError):
            cm.set_crop(cat, shot, sid, crop)
    assert cm.load_catalog().shot(shot)["streams"][0]["window"]["start_frame"] == 20
    assert (file_sha256(path), os.stat(path).st_mtime_ns) == before


def test_same_swing_views_are_explicit_and_statuses_stay_unknown_until_set(dataset):
    cat = _fresh(dataset)
    dtl = register(cat, dataset, "alice_dtl", "alice", "down_the_line")
    fo = register(cat, dataset, "alice_fo", "alice", "face_on")
    shot = _shot(cat, "alice", dtl, "down_the_line", 0, 59)
    cm.add_stream(cat, shot, source_id=fo, camera_view="face_on", start_frame=0, end_frame=59,
                  same_swing_method="co-triggered Swing Catalyst pair", asserted_by="alice")
    mv = cat.shot(shot)["multi_view"]
    assert mv["same_swing"]["status"] == "asserted"
    assert mv["synchronization"]["status"] == "unknown"
    assert mv["calibration"]["status"] == "unknown"
    gate = cm.eligibility(cat, shot)["stereo_3d"]
    assert not gate["eligible"]
    assert any("synchronization is unknown" in r for r in gate["reasons"])

    with pytest.raises(cm.CatalogError, match="only asserted"):
        cm.set_multi_view(cat, shot, "synchronization", "verified", "clap", "alice")
    with pytest.raises(cm.CatalogError, match="no method"):
        cm.set_multi_view(cat, shot, "same_swing", "verified", None, "alice")
    cm.set_multi_view(cat, shot, "same_swing", "verified", "same impact sound", "alice")
    cm.set_multi_view(cat, shot, "synchronization", "verified", "audio clap", "alice",
                      offset_ms=2.1, uncertainty_ms=4.2)
    cm.set_multi_view(cat, shot, "calibration", "verified", "checkerboard", "alice")
    assert cm.eligibility(cat, shot)["stereo_3d"]["eligible"]
    cm.save_catalog(cat)
    assert cm.load_catalog().shot(shot)["multi_view"]["synchronization"]["offset_ms"] == 2.1

    # A single-stream shot can't claim a same-swing partner.
    single = _shot(cat, "alice", fo, "face_on", 0, 59)
    with pytest.raises(cm.CatalogError, match="not_applicable"):
        cm.set_multi_view(cat, single, "same_swing", "asserted", "x", "alice")


def test_motion_timing_needs_known_real_time(dataset):
    cat = _fresh(dataset)
    yt = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line", source_type="youtube")
    cap = register(cat, dataset, "alice_dtl", "alice", "down_the_line",
                   source_type="swing_catalyst", motion_time="real_time", capture_fps=240,
                   timing_basis="Swing Catalyst 240fps capture")
    slow = register(cat, dataset, "gw_fo", "grant-waite", "face_on", motion_time="slowed")
    t_yt = cm.eligibility(cat, _shot(cat, "grant-waite", yt, "down_the_line"))["motion_timing"]
    assert not t_yt["eligible"] and "playback fps is not real swing time" in t_yt["reasons"][0]
    assert cm.eligibility(cat, _shot(cat, "alice", cap, "down_the_line"))["motion_timing"][
        "eligible"]
    s_slow = _shot(cat, "grant-waite", slow, "face_on")
    assert not cm.eligibility(cat, s_slow)["motion_timing"]["eligible"]
    cm.update_source(cat, slow, timing={"capture_fps": 480, "basis": "broadcast 480fps"})
    assert cm.eligibility(cat, s_slow)["motion_timing"]["eligible"]
    # Playback rate is recorded separately from capture timing.
    src = cat.source(yt)
    assert src["video"]["nominal_fps"] == pytest.approx(30.0)
    assert src["timing"] == {"motion_time": "unknown", "capture_fps": None, "basis": None}


def test_unassessed_quality_and_review_are_not_eligible(dataset):
    cat = _fresh(dataset)
    src = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line")
    shot = _shot(cat, "grant-waite", src, "down_the_line")
    gates = cm.eligibility(cat, shot)
    assert not any(gates[k]["eligible"] for k in ("body_landmarks", "club_landmarks",
                                                  "resolution", "human_review"))
    sid = cat.shot(shot)["streams"][0]["stream_id"]
    cm.set_stream_quality(cat, shot, sid, body_visibility="good", club_visibility="poor",
                          resolution_usable="yes")
    with pytest.raises(cm.CatalogError, match="needs a reviewer"):
        cm.set_review(cat, shot, "reviewed")
    cm.set_review(cat, shot, "reviewed", by="tester")
    gates = cm.eligibility(cat, shot)
    assert gates["body_landmarks"]["eligible"] and gates["resolution"]["eligible"]
    assert gates["human_review"]["eligible"]
    assert not gates["club_landmarks"]["eligible"]


def test_missing_moved_and_changed_sources(dataset):
    cat = _fresh(dataset)
    dtl = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line")
    fo = register(cat, dataset, "gw_fo", "grant-waite", "face_on")
    cm.save_catalog(cat)
    assert all(s.state == "ok" for s in cm.verify_sources(cat))

    os.makedirs(dataset.ref / "moved")
    shutil.move(dataset.videos["gw_dtl"], dataset.ref / "moved" / "renamed.avi")
    st = cm.resolve_source(cat, dtl)
    assert st.state == "relocated" and st.path.endswith("renamed.avi") and st.usable
    assert cm.resolve_source(cat, dtl, search=False).state == "missing"

    with open(dataset.videos["gw_fo"], "r+b") as f:     # same name, different bytes
        f.seek(100)
        f.write(b"\x00" * 16)
    cm._sha_memo.clear()
    st = cm.resolve_source(cat, fo)
    assert st.state == "changed" and not st.usable and st.path is None

    with pytest.raises(cm.CatalogError, match="no video file"):
        cm.register_source(cat, str(dataset.ref / "nope.mp4"), golfer_id="grant-waite")
    junk = dataset.ref / "junk.mp4"
    junk.write_bytes(b"definitely not a video")
    with pytest.raises(cm.CatalogError):
        cm.register_source(cat, str(junk), golfer_id="grant-waite")


def test_same_bytes_register_once_with_other_copies_recorded(dataset):
    cat = _fresh(dataset)
    src = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line")
    copy = dataset.tmp / "downloads" / "Grant Waite DTL.avi"
    os.makedirs(copy.parent)
    shutil.copy(dataset.videos["gw_dtl"], copy)
    rec, created = cm.register_source(cat, str(copy), golfer_id="grant-waite")
    assert not created and rec["source_id"] == src
    assert rec["also_found_at"] == [{"root": "absolute", "path": str(copy)}]
    assert len(cat.sources) == 1
    cm.save_catalog(cat)


def test_locations_are_relative_to_the_video_libraries(dataset):
    cat = _fresh(dataset)
    rec, _ = cm.register_source(cat, dataset.videos["alice_dtl"], golfer_id="alice")
    assert rec["location"] == {"root": "user", "path": "alice/dtl.avi"}
    rec, _ = cm.register_source(cat, dataset.videos["gw_dtl"], golfer_id="grant-waite")
    assert rec["location"] == {"root": "reference", "path": "GW_DTL.avi"}
    assert cm.resolve_location(rec["location"]) == dataset.videos["gw_dtl"]


def test_legacy_annotations_are_pointed_to_never_converted(dataset):
    path = dataset.videos["gw_fo"]
    ann.save_annotations(path, {"P1": 3, "P6": 20}, 30.0, view_angle="face_on")
    with open(ann.ANNOTATIONS_PATH) as f:
        legacy = json.load(f)
    key = next(iter(legacy))
    del legacy[key]["_p_convention"]          # a pre-convention entry (may call impact P6)
    with open(ann.ANNOTATIONS_PATH, "w") as f:
        json.dump(legacy, f)
    raw = open(ann.ANNOTATIONS_PATH, "rb").read()

    cat = _fresh(dataset)
    rec, _ = cm.register_source(cat, path, golfer_id="grant-waite", camera_view="face_on")
    _shot(cat, "grant-waite", rec["source_id"], "face_on")
    cm.save_catalog(cat)
    assert rec["legacy_annotation_key"] == key
    assert open(ann.ANNOTATIONS_PATH, "rb").read() == raw


def test_concurrent_saves_conflict_and_invalid_saves_write_nothing(dataset):
    cat = _fresh(dataset)
    cm.save_catalog(cat)
    one, two = cm.load_catalog(), cm.load_catalog()
    cm.add_golfer(one, "Carol", "user")
    cm.add_golfer(two, "Dan", "user")
    cm.save_catalog(one)
    with pytest.raises(cm.ConflictError):
        cm.save_catalog(two)
    assert "dan" not in cm.load_catalog().golfers

    before = open(one.path("golfers"), "rb").read()
    one.golfers["carol"]["role"] = "coach"             # not a role
    with pytest.raises(cm.CatalogError, match="role"):
        cm.save_catalog(one)
    assert open(one.path("golfers"), "rb").read() == before
    assert not [f for f in os.listdir(one.root) if f.startswith(".tmp")]


def test_a_corrupt_catalog_file_is_refused_with_its_problems(dataset):
    cat = _fresh(dataset)
    cm.save_catalog(cat)
    path = cat.path("golfers")
    doc = json.load(open(path))
    doc["golfers"][0]["golfer_id"] = "con"              # a reserved device name
    doc["golfers"].append(dict(doc["golfers"][1]))       # duplicate id
    json.dump(doc, open(path, "w"))
    with pytest.raises(cm.CatalogError) as e:
        cm.load_catalog()
    assert any("golfer_id" in p for p in e.value.problems)


def test_removing_a_shot_is_refused_while_it_is_used(dataset):
    cat = _fresh(dataset)
    src = register(cat, dataset, "gw_dtl", "grant-waite", "down_the_line")
    shot = _shot(cat, "grant-waite", src, "down_the_line")
    cm.create_collection(cat, "gw-v1", "grant-waite", "GW")
    cm.add_to_collection(cat, "gw-v1", shot)
    with pytest.raises(cm.CatalogError, match="in collection"):
        cm.remove_shot(cat, shot)
    cm.remove_from_collection(cat, "gw-v1", shot)
    cm.remove_shot(cat, shot)
    assert shot not in cat.shots and src in cat.sources


# --- the contract's shared definitions --------------------------------------

def _schemas():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
    load = lambda n: json.load(open(os.path.join(root, n)))  # noqa: E731
    return load("swing-dataset.schema.json"), load("capture-manifest.schema.json")


def test_identifier_and_camera_view_are_the_capture_contracts():
    ds, cap = _schemas()
    for name in ("identifier", "cameraView", "rfc3339DateTime", "relativePath"):
        assert ds["$defs"][name] == cap["$defs"][name], name


@pytest.mark.parametrize("value", ["grant-waite", "shot-3f9a1c2b7e", "2026-07-31T14-22-05.412Z",
                                   "fox-dtl", "a b", "CON", "nul.txt", "x.", "x ", "a/b",
                                   "a:b", ".", "..", "", "com1", "lpt9.log", "tab\tname"])
def test_python_identifier_rule_agrees_with_the_schema(value):
    ds, _ = _schemas()
    v = jsonschema.Draft202012Validator(ds["$defs"]["identifier"])
    assert cm.is_identifier(value) == v.is_valid(value), value


def test_the_schema_examples_form_a_consistent_catalog(tmp_path):
    from utils import shot_annotations as sa
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "schemas", "examples", "swing-dataset")
    for kind in cm.KINDS:
        shutil.copy(os.path.join(src, f"{kind}.example.json"), tmp_path / f"{kind}.json")
    cat = cm.load_catalog(str(tmp_path))
    doc = json.load(open(os.path.join(src, "shot-annotations.example.json")))
    assert sa.check(doc, cat) == []
    # The user's co-triggered pair is one shot; Grant Waite's views are two.
    assert len(cat.shot("shot-jy-2026-07-27-135006")["streams"]) == 2
    coll = cat.collection("grant-waite-full-swing-v1")
    assert all(len(cat.shot(s)["streams"]) == 1 for s in coll["shot_ids"])
