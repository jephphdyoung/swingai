"""The Reference Editor tab, driven headlessly through Streamlit's AppTest.

The frame canvas (a custom component) can't be clicked in AppTest; the same
placement goes through the numeric form, which calls the same code.
"""

import os

import pytest

pytest.importorskip("jsonschema")
from streamlit.testing.v1 import AppTest  # noqa: E402

from _catalog_fixtures import dataset, golfers, register  # noqa: E402,F401  (fixture)
from utils import catalog as cm  # noqa: E402
from utils import shot_annotations as sa  # noqa: E402
from utils.frame_access import file_sha256, get_frame_index  # noqa: E402
from utils.joint_labels import decode_frames, frame_digest  # noqa: E402

SCRIPT = """
from reference_editor_tab import render_reference_editor_tab
render_reference_editor_tab()
"""
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def editor(dataset, monkeypatch):
    cat = golfers(cm.load_catalog())
    src = register(cat, dataset, "gw_long", "grant-waite", "face_on", source_type="youtube")
    cm.save_catalog(cat)
    monkeypatch.chdir(REPO)
    dataset.src = src
    dataset.before = {k: file_sha256(p) for k, p in dataset.videos.items()}

    def run(**state):
        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["re_operator"] = "tester"
        at.session_state["re_golfer"] = "grant-waite"
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    dataset.run = run
    yield dataset
    assert {k: file_sha256(p) for k, p in dataset.videos.items()} == dataset.before


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def test_mark_two_swings_in_one_video_then_edit_window_crop_and_labels(editor):
    at = editor.run()
    assert at.selectbox(key="re_shot").value == "__new__"

    # Swing 1: frames 10-50, from the navigator.
    at.session_state["re_frame"] = 10
    _ok(at.run())
    at.button(key="re_new_start_here").click().run()
    at.session_state["re_frame"] = 50
    _ok(at.run())
    at.button(key="re_new_end_here").click().run()
    at.selectbox(key="re_new_club_cat").set_value("iron")
    at.text_input(key="re_new_club").set_value("6-iron")
    _ok(at.button(key="re_new_submit").click().run())
    cat = cm.load_catalog()
    (first,) = cat.shots.values()
    assert first["streams"][0]["window"] == {"start_frame": 10, "end_frame": 50}
    assert first["club"]["name"] == "6-iron"

    # Swing 2 from the same video.
    at.selectbox(key="re_shot").set_value("__new__").run()
    at.number_input(key="re_new_start").set_value(70)
    at.number_input(key="re_new_end").set_value(110)
    _ok(at.button(key="re_new_submit").click().run())
    cat = cm.load_catalog()
    assert len(cm.shots_for_source(cat, editor.src)) == 2
    second = next(s for s in cat.shots.values() if s["shot_id"] != first["shot_id"])

    # Reopen swing 1: move its start, crop it, mark P-positions and a point.
    at.selectbox(key="re_shot").set_value(first["shot_id"]).run()
    at.session_state["re_frame"] = 14
    _ok(at.run())
    _ok(at.button(key="re_win_start").click().run())
    for k, v in (("x", 20), ("y", 10), ("width", 200), ("height", 180)):
        at.number_input(key=f"re_crop_{k}").set_value(v)
    _ok(at.button(key="re_crop_set").click().run())
    at.session_state["re_frame"] = 37
    _ok(at.run())
    _ok(at.button(key="re_set_P7").click().run())
    _ok(at.button(key="re_set_P4").click().run())
    at.session_state["re_frame"] = 20
    _ok(at.run())
    _ok(at.button(key="re_set_P1").click().run())
    at.selectbox(key="re_group").set_value("ball").run()
    at.number_input(key="re_px").set_value(123.0)
    at.number_input(key="re_py").set_value(201.0)
    _ok(at.button(key="re_place").click().run())
    _ok(at.button(key="re_clear_P4").click().run())
    _ok(at.button(key="re_review_reviewed").click().run())

    cat = cm.load_catalog()
    st_ = cat.shot(first["shot_id"])["streams"][0]
    assert st_["window"] == {"start_frame": 14, "end_frame": 50}
    assert st_["crop"] == {"x": 20, "y": 10, "width": 200, "height": 180}
    assert cat.shot(second["shot_id"])["streams"][0]["window"] == {"start_frame": 70,
                                                                   "end_frame": 110}
    doc = sa.load(first["shot_id"])
    sid = st_["stream_id"]
    assert sa.positions(doc, sid) == {"P1": 20, "P7": 37}
    decoded = decode_frames(editor.videos["gw_long"], [20, 37])
    assert doc["streams"][sid]["positions"]["P7"]["image_sha1"] == frame_digest(decoded[37])
    assert doc["streams"][sid]["positions"]["P7"]["set_by"] == "tester"
    fl = sa.frame_labels(doc, sid, 20)
    assert fl["points"]["ball_center"]["x"] == 123.0
    assert fl["points"]["ball_center"]["origin"] == "manual"
    assert fl["review"] == {**fl["review"], "status": "reviewed", "by": "tester"}

    # Reload from scratch: everything is still there.
    at = editor.run(re_source=editor.src, re_shot=first["shot_id"], re_frame=37)
    _ok(at)
    assert any("P7" in c.value for c in at.caption)


def test_register_a_video_through_the_form(editor):
    at = editor.run()
    at.selectbox(key="re_reg_pick").set_value(editor.videos["gw_dtl"])
    at.selectbox(key="re_reg_view").set_value("down_the_line")
    at.text_input(key="re_reg_date").set_value("2011-05")
    _ok(at.button(key="re_reg_submit").click().run())
    cat = cm.load_catalog()
    rec = next(s for s in cat.sources.values() if s["location"]["path"] == "GW_DTL.avi")
    assert rec["recording"] == {**rec["recording"], "date": "2011-05",
                                "camera_view": "down_the_line"}
    assert rec["timing"]["motion_time"] == "unknown"
    assert at.selectbox(key="re_source").value == rec["source_id"]


def test_a_missing_source_is_reported_and_nothing_is_changed(editor):
    cat = cm.load_catalog()
    shot = cm.add_shot(cat, golfer_id="grant-waite", source_id=editor.src,
                       camera_view="face_on", start_frame=0, end_frame=30, swing_type="full")
    cm.save_catalog(cat)
    path = editor.videos["gw_long"]
    os.rename(path, path + ".away")
    try:
        at = editor.run(re_source=editor.src, re_shot=shot["shot_id"])
        assert any("unavailable (missing)" in e.value for e in at.error)
        assert cm.load_catalog().shot(shot["shot_id"]) == shot
    finally:
        os.rename(path + ".away", path)


def test_invalid_edits_show_problems_instead_of_saving(editor):
    at = editor.run()
    at.number_input(key="re_new_start").set_value(60)
    at.number_input(key="re_new_end").set_value(20)
    at.button(key="re_new_submit").click().run()
    assert any("not within" in e.value for e in at.error)
    assert cm.load_catalog().shots == {}
