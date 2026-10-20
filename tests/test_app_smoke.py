"""Headless smoke test of app.py: all three tabs render on a deterministic fixture.

Poses are pre-seeded into the cache under the real MediaPipe backend identity,
so the Dual-View tab finds them without running inference.
"""

import os

import pytest

pytest.importorskip("mediapipe")
from streamlit.testing.v1 import AppTest  # noqa: E402

from _dual_view_fixtures import FakeBackend, write_video  # noqa: E402
from utils import annotations as ann  # noqa: E402
from utils import pose_data as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTL = "Test Golfer - 2026-01-01 120000 Fox Down the line 61fc.avi"
FO = "Test Golfer - 2026-01-01 120000 Fox Face on right aaf0.avi"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    user, ref, out = tmp_path / "user", tmp_path / "reference", tmp_path / "output"
    for d in (user, ref, out):
        d.mkdir()
    monkeypatch.setenv("SWINGAI_USER_DIR", str(user))
    monkeypatch.setenv("SWINGAI_REFERENCE_DIR", str(ref))
    monkeypatch.setenv("SWINGAI_OUTPUT_DIR", str(out))
    monkeypatch.setattr(ann, "ANNOTATIONS_PATH", str(tmp_path / "annotations.json"))

    videos = {
        ("user", "dtl"): write_video(user / DTL, 60, 60.0),
        ("user", "face_on"): write_video(user / FO, 60, 60.0),
        ("reference", "dtl"): write_video(ref / "Ref_DTL.avi", 30, 30.0, (240, 180)),
        ("reference", "face_on"): write_video(ref / "Ref_faceon.avi", 30, 30.0),
    }
    for (role, view), path in videos.items():
        # Your DTL starts without P4, so a test can add it through the UI.
        marks = {"P1": 2} if (role, view) == ("user", "dtl") else {"P1": 2, "P4": 20}
        ann.save_annotations(path, marks, 60.0 if role == "user" else 30.0, view_angle=view)
        seq = pd.extract_pose_sequence(path, FakeBackend(view=view, n_frames=60))
        pd.save_pose_sequence(seq, pd.cached_pose_path(path))
    monkeypatch.chdir(REPO)
    return videos


def test_all_tabs_render_and_dual_view_shows_four_panels(app_env):
    at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=60)
    at.session_state["dv_position"] = "P1"
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    assert [t.label for t in at.tabs] == ["Analyze", "Annotate", "Dual-View Comparison"]
    # Analyze and Annotate still build their widgets.
    assert at.selectbox(key="analyze_user") is not None
    assert at.selectbox(key="annot_video") is not None
    # Dual view: pair found, poses served from cache, two measurement tables.
    assert "Test Golfer" in at.selectbox(key="dv_pair").format_func(0)
    assert not any("Pose not yet extracted" in w.value for w in at.warning)
    assert len(at.dataframe) == 2
    dtl = at.dataframe[0].value
    assert list(dtl.columns[:4]) == ["Measurement", "Your value", "Reference value",
                                      "Difference (you − ref)"]
    assert dtl["Your value"].notna().any()


def test_switching_position_and_skeleton_toggle(app_env):
    at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=60)
    at.run()
    at.radio(key="dv_position").set_value("P4").run()
    at.toggle(key="dv_skeleton").set_value(False).run()
    at.button(key="dv_next").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="dv_position").value == "P5"


def test_saving_an_annotation_refreshes_counts_and_tables(app_env):
    at = AppTest.from_file(os.path.join(REPO, "app.py"), default_timeout=60)
    at.session_state["dv_position"] = "P4"
    at.run()
    assert "P4 (3/4)" in at.radio(key="dv_position").options
    assert at.dataframe[0].value["Your value"].isna().all()     # your DTL has no P4

    prefix = "dv_annot_user_dtl"
    at.slider(key=f"{prefix}_slider").set_value(20).run()
    at.button(key=f"{prefix}_mark_P4").click().run()
    at.button(key=f"{prefix}_save").click().run()
    assert not at.exception, [e.value for e in at.exception]

    # Same run that handled the click: counts, panels and tables already reflect it.
    assert any("Saved 2 positions" in s.value for s in at.success)
    assert "P4 (4/4)" in at.radio(key="dv_position").options
    assert at.dataframe[0].value["Your value"].notna().any()
    assert ann.load_annotations(app_env[("user", "dtl")])["P4"] == 20
