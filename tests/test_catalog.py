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
    # The working shot stays editable; the frozen collection keeps its snapshot.
    cm.update_shot(cat, s_fo, swing_type="half")
    cm.save_catalog(cat)
    cat = cm.load_catalog()
    frozen = {e["shot"]["shot_id"]: e["shot"] for e in
              cm.load_snapshot(cat, "grant-waite-full-swing-v1")["shots"]}
    assert frozen[s_fo]["swing_type"] == "full"
    assert cm.frozen_drift(cat, "grant-waite-full-swing-v1") == {s_fo: ["swing_type changed"]}
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


def test_exposure_alignment_is_explicit_and_never_assumed(dataset):
    cat = _fresh(dataset)
    dtl = register(cat, dataset, "alice_dtl", "alice", "down_the_line")
    fo = register(cat, dataset, "alice_fo", "alice", "face_on")
    shot = _shot(cat, "alice", dtl, "down_the_line", 0, 59)
    cm.add_stream(cat, shot, source_id=fo, camera_view="face_on", start_frame=0, end_frame=59,
                  same_swing_method="co-triggered pair", asserted_by="alice")
    cm.set_multi_view(cat, shot, "same_swing", "verified", "same impact sound", "alice")
    # Equal frame rates and a measured offset say nothing about the exposures.
    sync = cm.set_multi_view(cat, shot, "synchronization", "verified", "audio clap", "alice",
                             offset_ms=2.1, uncertainty_ms=0.4)["multi_view"]["synchronization"]
    assert sync["exposure_alignment"] == "unknown" and sync["exposure_uncertainty_ms"] is None
    for bad, match in (({"exposure_alignment": "bounded"}, "no exposure_uncertainty_ms"),
                       ({"exposure_alignment": "included", "exposure_uncertainty_ms": 0.1},
                        "only for a bounded"),
                       ({"exposure_alignment": "bounded", "exposure_uncertainty_ms": -1},
                        "must be >= 0"),
                       ({"exposure_alignment": "sometimes"}, "must be one of")):
        with pytest.raises(cm.CatalogError, match=match):
            cm.set_multi_view(cat, shot, "synchronization", "verified", "audio clap", "alice",
                              offset_ms=2.1, uncertainty_ms=0.4, **bad)
    with pytest.raises(cm.CatalogError, match="needs a measured synchronization"):
        cm.set_multi_view(cat, shot, "synchronization", "unknown",
                          exposure_alignment="included")
    cm.set_multi_view(cat, shot, "synchronization", "verified", "LED flash", "alice",
                      offset_ms=2.1, uncertainty_ms=0.05, exposure_alignment="bounded",
                      exposure_uncertainty_ms=0.02)
    cm.save_catalog(cat)
    sync = cm.load_catalog().shot(shot)["multi_view"]["synchronization"]
    assert (sync["exposure_alignment"], sync["exposure_uncertainty_ms"]) == ("bounded", 0.02)


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
    assert cm.motion_clock(cat.source(slow))["scale"] is None
    cm.update_source(cat, slow, timing={"capture_fps": 480, "basis": "broadcast 480fps"})
    assert cm.eligibility(cat, s_slow)["motion_timing"]["eligible"]
    # Real time per timestamp ms: presented at 30 fps, captured at 480 fps.
    clock = cm.motion_clock(cat.source(slow))
    assert clock["scale"] == pytest.approx(30.0 / 480)
    assert "one presented frame per captured frame" in clock["basis"]
    assert cm.motion_clock(cat.source(cap))["scale"] == 1.0
    assert cm.motion_clock(cat.source(yt))["scale"] is None
    # A variable-rate file can't be rescaled by a single capture rate.
    vfr = {**cat.source(slow), "video": {**cat.source(slow)["video"],
                                         "variable_frame_rate": True}}
    assert cm.motion_clock(vfr)["scale"] is None
    assert "variable frame rate" in cm.motion_clock(vfr)["reasons"][0]
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


# --- frozen collections are reproducible ------------------------------------

def _labelled_collection(d):
    from utils import frame_access as fa
    from utils import shot_annotations as sa
    cat = _fresh(d)
    dtl = register(cat, d, "gw_dtl", "grant-waite", "down_the_line")
    fo = register(cat, d, "gw_fo", "grant-waite", "face_on")
    s1 = _shot(cat, "grant-waite", dtl, "down_the_line", 2, 30)
    s2 = _shot(cat, "grant-waite", fo, "face_on", 0, 39)
    cm.create_collection(cat, "gw-v1", "grant-waite", "GW v1")
    cm.add_to_collection(cat, "gw-v1", s1)
    cm.add_to_collection(cat, "gw-v1", s2)
    cm.save_catalog(cat)
    shot, src = cat.shot(s1), cat.source(dtl)
    index = fa.get_frame_index(d.videos["gw_dtl"])
    sid = shot["streams"][0]["stream_id"]
    doc = sa.load(s1)
    sa.set_position(doc, shot, src, index, sid, "P1", 5, by="tester")
    sa.set_point(doc, shot, src, index, sid, 5, "ball_center", 100, 150, by="tester")
    sa.save(doc, cat)
    return cat, s1, s2, sid, shot, src, index


def test_a_frozen_collection_reproduces_its_inputs_after_the_shots_are_edited(dataset):
    from utils import shot_annotations as sa
    cat, s1, s2, sid, shot, src, index = _labelled_collection(dataset)
    cm.freeze_collection(cat, "gw-v1", by="tester")
    cm.save_catalog(cat)
    cat = cm.load_catalog()
    ref = cat.collection("gw-v1")["snapshot"]
    assert os.path.exists(os.path.join(cat.root, ref["path"]))
    before = json.loads(json.dumps(cm.load_snapshot(cat, "gw-v1")))
    entry = before["shots"][0]
    assert entry["shot"]["streams"][0]["window"] == {"start_frame": 2, "end_frame": 30}
    assert entry["sources"][0]["sha256"] == src["sha256"]
    assert entry["annotations"]["streams"][sid]["positions"]["P1"]["frame"] == 5
    assert entry["annotations_sha256"] == cm.canonical_sha256(entry["annotations"])
    assert before["shots"][1]["annotations"] is None
    assert cm.frozen_drift(cat, "gw-v1") == {}

    # Normal editing of the working shots carries on...
    cm.set_window(cat, s1, sid, 4, 25)
    cm.set_crop(cat, s1, sid, {"x": 10, "y": 10, "width": 100, "height": 100})
    cm.update_shot(cat, s1, club={"name": "7-iron"})
    cm.set_stream_quality(cat, s2, cat.shot(s2)["streams"][0]["stream_id"],
                          body_visibility="good")
    cm.save_catalog(cat)
    doc = sa.load(s1)
    sa.set_position(doc, cat.shot(s1), src, index, sid, "P1", 6, by="tester")
    sa.clear_position(doc, sid, "P1")
    sa.set_position(doc, cat.shot(s1), src, index, sid, "P4", 20, by="tester")
    sa.save(doc, cat)

    # ...and the frozen collection still represents exactly what was frozen.
    cat = cm.load_catalog()
    assert cm.load_snapshot(cat, "gw-v1") == before
    assert cat.collection("gw-v1")["snapshot"] == ref
    assert cat.shot(s1)["streams"][0]["window"] == {"start_frame": 4, "end_frame": 25}
    drift = cm.frozen_drift(cat, "gw-v1")
    assert set(drift[s1]) >= {f"{sid} window changed", f"{sid} crop changed", "club changed",
                              "labels changed"}
    assert drift[s2] == [f"{cat.shot(s2)['streams'][0]['stream_id']} quality changed"]
    # Freezing again needs a new collection; the old one can't be re-frozen.
    with pytest.raises(cm.CatalogError, match="frozen"):
        cm.freeze_collection(cat, "gw-v1", by="tester")


def test_an_altered_or_missing_snapshot_is_refused(dataset):
    cat, *_ = _labelled_collection(dataset)
    cm.freeze_collection(cat, "gw-v1", by="tester")
    cm.save_catalog(cat)
    path = os.path.join(cat.root, cat.collection("gw-v1")["snapshot"]["path"])
    doc = json.load(open(path))
    doc["shots"][0]["shot"]["streams"][0]["window"]["end_frame"] = 31
    json.dump(doc, open(path, "w"), indent=2)
    with pytest.raises(cm.CatalogError, match="does not match its recorded sha256"):
        cm.load_snapshot(cm.load_catalog(), "gw-v1")
    os.remove(path)
    with pytest.raises(cm.CatalogError, match="unreadable"):
        cm.load_snapshot(cm.load_catalog(), "gw-v1")
    # Re-pointing the collection at another snapshot is a change to a frozen record.
    cat = cm.load_catalog()
    cat.collections["gw-v1"]["snapshot"]["sha256"] = "0" * 64
    with pytest.raises(cm.CatalogError, match="frozen and can't be changed"):
        cm.save_catalog(cat)


def test_freezing_is_refused_when_inputs_are_not_reproducible(dataset):
    cat, s1, *_ = _labelled_collection(dataset)
    os.rename(dataset.videos["gw_fo"], dataset.videos["gw_fo"] + ".away")
    try:
        with pytest.raises(cm.CatalogError, match="is missing"):
            cm.freeze_collection(cat, "gw-v1", by="tester")
    finally:
        os.rename(dataset.videos["gw_fo"] + ".away", dataset.videos["gw_fo"])
    assert cat.collection("gw-v1")["status"] == "draft"
    with pytest.raises(cm.CatalogError, match="name of who froze"):
        cm.freeze_collection(cat, "gw-v1", by=" ")
    # A failed save leaves no frozen record behind.
    cm.freeze_collection(cat, "gw-v1", by="tester")
    cat.golfers["alice"]["role"] = "coach"
    with pytest.raises(cm.CatalogError):
        cm.save_catalog(cat)
    assert cm.load_catalog().collection("gw-v1")["status"] == "draft"


# --- timing reliability -------------------------------------------------------

def test_motion_timing_needs_verified_timestamps_and_a_basis(dataset, monkeypatch):
    from utils import frame_access as fa
    cat = _fresh(dataset)
    ok = register(cat, dataset, "alice_dtl", "alice", "down_the_line", motion_time="real_time",
                  capture_fps=60, timing_basis="phone slow-mo off, 60fps")
    assert cat.source(ok)["video"]["timestamps"]["status"] == "verified"
    assert cm.eligibility(cat, _shot(cat, "alice", ok, "down_the_line"))["motion_timing"][
        "eligible"]

    no_basis = register(cat, dataset, "alice_fo", "alice", "face_on", motion_time="real_time")
    gate = cm.eligibility(cat, _shot(cat, "alice", no_basis, "face_on"))["motion_timing"]
    assert not gate["eligible"] and "without a basis" in gate["reasons"][0]

    # Estimated frame times never qualify, whatever is claimed about the source.
    fa._memo.clear()
    real_pos = fa._pos_msec
    calls = {"n": 0}

    def flaky(cap):
        calls["n"] += 1
        return 0.0 if calls["n"] == 10 else real_pos(cap)
    monkeypatch.setattr(fa, "_pos_msec", flaky)
    monkeypatch.setattr(fa, "get_frame_index",
                        lambda p, progress=None, **k: fa.build_frame_index(p))
    est = register(cat, dataset, "bob_fo", "bob", "face_on", motion_time="real_time",
                   capture_fps=60, timing_basis="claimed")
    ts = cat.source(est)["video"]["timestamps"]
    assert ts["status"] == "estimated" and ts["estimated_frames"] == 1
    gate = cm.eligibility(cat, _shot(cat, "bob", est, "face_on"))["motion_timing"]
    assert not gate["eligible"] and "estimated" in gate["reasons"][0]
    cm.save_catalog(cat)


# --- freezing checks each label's image against its exact source frame ---------

def _tamper_digest(shot_id, mutate):
    """Rewrite a saved label's image_sha1 behind the API, as a stale or
    hand-edited file would have it."""
    from utils import shot_annotations as sa
    path = sa.annotations_path(shot_id)
    doc = json.load(open(path))
    mutate(doc)
    json.dump(doc, open(path, "w"), indent=2)


def _assert_still_draft(cat_root):
    cat = cm.load_catalog()
    c = cat.collection("gw-v1")
    assert c["status"] == "draft" and c["snapshot"] is None and c["frozen_at"] is None
    snaps = os.path.join(cat_root, "snapshots")
    assert not os.path.isdir(snaps) or not os.listdir(snaps)


def test_freezing_accepts_labels_on_their_exact_source_frames(dataset):
    cat, s1, s2, sid, shot, src, index = _labelled_collection(dataset)
    doc = json.load(open(os.path.join(str(dataset.data), "annotations", f"{s1}.json")))
    assert doc["streams"][sid]["positions"]["P1"]["image_sha1"] == index.digests[5]
    assert doc["streams"][sid]["frames"]["5"]["image_sha1"] == index.digests[5]
    cm.freeze_collection(cat, "gw-v1", by="tester")
    cm.save_catalog(cat)
    assert cm.load_catalog().collection("gw-v1")["status"] == "frozen"


@pytest.mark.parametrize("which", ["p_position", "landmark_frame"])
def test_freezing_refuses_a_label_whose_image_is_not_its_source_frame(dataset, which):
    cat, s1, s2, sid, shot, src, index = _labelled_collection(dataset)
    wrong = index.digests[4]                      # the neighbouring frame's image

    def mutate(doc):
        if which == "p_position":
            doc["streams"][sid]["positions"]["P1"]["image_sha1"] = wrong
        else:
            doc["streams"][sid]["frames"]["5"]["image_sha1"] = wrong
    _tamper_digest(s1, mutate)
    labels_before = open(os.path.join(str(dataset.data), "annotations", f"{s1}.json")).read()

    cat = cm.load_catalog()
    with pytest.raises(cm.CatalogError) as e:
        cm.freeze_collection(cat, "gw-v1", by="tester")
    expected = ("P1 at frame 5: placed on a different image" if which == "p_position"
                else "labels at frame 5: placed on a different image")
    assert any(expected in p and s1 in p for p in e.value.problems), e.value.problems

    # A failed freeze changes nothing in memory, so a later save publishes nothing.
    assert cat.collection("gw-v1")["status"] == "draft"
    assert not cat._pending_snapshots
    cm.save_catalog(cat)
    _assert_still_draft(cat.root)
    # The label file is reported, never repaired or re-pointed.
    assert open(os.path.join(str(dataset.data), "annotations",
                             f"{s1}.json")).read() == labels_before
