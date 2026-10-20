"""Tests for P-position handling and the offline dual-view comparison pipeline."""

import json

import numpy as np
import pytest

from _dual_view_fixtures import FakeBackend, write_video
from utils import annotations as ann
from utils import dual_view as dv
from utils.pose_data import get_pose_sequence


@pytest.fixture
def annotations_file(tmp_path, monkeypatch):
    path = tmp_path / "annotations.json"
    monkeypatch.setattr(ann, "ANNOTATIONS_PATH", str(path))
    return path


def slot(role="user", view="dtl", fps=60.0, n=100, frames=None, timestamps=None, pose=None):
    rec = ann.AnnotationRecord(video_path=f"/v/{role}_{view}.mp4",
                               frames=dict(frames or {}), timestamps_ms=dict(timestamps or {}),
                               view_angle=view, convention=ann.P_CONVENTION)
    return dv.Slot(role=role, view=view, path=rec.video_path,
                   info=dv.VideoInfo(fps=fps, frame_count=n, width=320, height=240),
                   record=rec, pose=pose)


class TestCanonicalDefinitions:
    def test_p6_is_shaft_parallel_and_p7_is_impact(self):
        assert ann.P_DEFINITIONS["P7"] == "Impact"
        assert "Shaft parallel" in ann.P_DEFINITIONS["P6"]
        assert ann.P_POSITIONS == [f"P{i}" for i in range(1, 11)]


class TestAnnotationStore:
    def test_save_records_convention_and_handedness(self, annotations_file):
        ann.save_annotations("/v/a.mp4", {"P1": 10, "P7": 100}, 250.0,
                             view_angle="dtl", handedness="left")
        rec = ann.load_annotation_record("/v/a.mp4")
        assert rec.frames == {"P1": 10, "P7": 100}
        assert rec.timestamps_ms["P7"] == pytest.approx(400.0)
        assert rec.view_angle == "dtl"
        assert rec.handedness == "left"
        assert rec.convention == ann.P_CONVENTION
        assert ann.load_annotations("/v/a.mp4") == {"P1": 10, "P7": 100}

    def test_resave_keeps_handedness(self, annotations_file):
        ann.save_annotations("/v/a.mp4", {"P1": 1}, 30.0, handedness="left")
        ann.save_annotations("/v/a.mp4", {"P1": 2}, 30.0)
        assert ann.load_annotation_record("/v/a.mp4").handedness == "left"

    def test_legacy_entry_warns_until_confirmed(self, annotations_file):
        legacy = {"/v/old.mp4": {"P1": {"frame": 5, "timestamp_ms": 166.7},
                                 "P6": {"frame": 50, "timestamp_ms": 1666.7},
                                 "P7": {"frame": 55, "timestamp_ms": 1833.3},
                                 "_view_angle": "face_on"}}
        annotations_file.write_text(json.dumps(legacy))
        rec = ann.load_annotation_record("/v/old.mp4")
        assert rec.convention is None
        assert any("P7 = impact" in w for w in ann.convention_warnings(rec))
        ann.confirm_annotations("/v/old.mp4")
        rec = ann.load_annotation_record("/v/old.mp4")
        assert rec.frames == {"P1": 5, "P6": 50, "P7": 55}
        assert ann.convention_warnings(rec) == []
        assert ann.is_confirmed(rec)

    def _legacy(self, annotations_file):
        annotations_file.write_text(json.dumps({"/v/old.mp4": {
            "P1": {"frame": 5, "timestamp_ms": 166.7},
            "P6": {"frame": 50, "timestamp_ms": 1666.7},
            "P7": {"frame": 55, "timestamp_ms": 1833.3},
            "_view_angle": "face_on"}}))

    def test_saving_an_unrelated_position_does_not_verify_legacy_p6_p7(self, annotations_file):
        self._legacy(annotations_file)
        # What the annotator does after fixing P2: re-save every loaded mark.
        marks = ann.load_annotations("/v/old.mp4") | {"P2": 20}
        ann.save_annotations("/v/old.mp4", marks, 30.0, view_angle="face_on")
        rec = ann.load_annotation_record("/v/old.mp4")
        assert rec.frames["P2"] == 20
        assert rec.convention is None
        assert not ann.is_confirmed(rec)
        assert any("P7 = impact" in w for w in ann.convention_warnings(rec))

    def test_resaving_p6_p7_themselves_still_needs_confirmation(self, annotations_file):
        self._legacy(annotations_file)
        ann.save_annotations("/v/old.mp4", {"P1": 5, "P6": 48, "P7": 56}, 30.0)
        assert not ann.is_confirmed(ann.load_annotation_record("/v/old.mp4"))

    def test_new_entry_is_recorded_under_the_convention(self, annotations_file):
        ann.save_annotations("/v/new.mp4", {"P1": 1}, 30.0)
        assert ann.is_confirmed(ann.load_annotation_record("/v/new.mp4"))

    def test_confirmed_entry_stays_confirmed_on_resave(self, annotations_file):
        ann.save_annotations("/v/new.mp4", {"P1": 1}, 30.0)
        ann.save_annotations("/v/new.mp4", {"P1": 2, "P7": 40}, 30.0)
        assert ann.is_confirmed(ann.load_annotation_record("/v/new.mp4"))

    def test_detector_output_can_declare_its_convention(self, annotations_file):
        self._legacy(annotations_file)
        ann.save_annotations("/v/old.mp4", {"P1": 4, "P7": 54}, 30.0,
                             convention=ann.P_CONVENTION)
        assert ann.load_annotation_record("/v/old.mp4").convention == ann.P_CONVENTION

    def test_needs_review_survives_saves_until_confirmed(self, annotations_file):
        ann.save_annotations("/v/p.mp4", {"P1": 1, "P4": 9}, 30.0)
        ann.mark_needs_review("/v/p.mp4", "proposed by Claude")
        ann.save_annotations("/v/p.mp4", {"P1": 1, "P4": 10}, 30.0)
        rec = ann.load_annotation_record("/v/p.mp4")
        assert rec.needs_review == "proposed by Claude"
        assert not ann.is_confirmed(rec)
        assert any("proposed by Claude" in w for w in ann.convention_warnings(rec))
        ann.confirm_annotations("/v/p.mp4")
        assert ann.is_confirmed(ann.load_annotation_record("/v/p.mp4"))

    def test_p6_without_p7_is_flagged(self):
        rec = ann.AnnotationRecord("/v", frames={"P1": 1, "P6": 50},
                                   convention=ann.P_CONVENTION)
        assert any("P6 may have been used for impact" in w
                   for w in ann.convention_warnings(rec))

    def test_out_of_order_positions_are_flagged(self):
        rec = ann.AnnotationRecord("/v", frames={"P6": 60, "P7": 55},
                                   convention=ann.P_CONVENTION)
        assert any("earlier than P6" in w for w in ann.convention_warnings(rec))

    def test_clean_record_has_no_warnings(self):
        rec = ann.AnnotationRecord("/v", frames={"P1": 1, "P4": 40, "P6": 52, "P7": 55},
                                   convention=ann.P_CONVENTION)
        assert ann.convention_warnings(rec) == []


class TestResolvePosition:
    def test_uses_this_videos_saved_frame(self):
        pf = dv.resolve_position(slot(frames={"P4": 40}), "P4")
        assert (pf.frame, pf.source) == (40, "annotation")
        assert pf.timestamp_ms == pytest.approx(40 * 1000 / 60)

    def test_missing_position_is_not_fabricated(self):
        pf = dv.resolve_position(slot(frames={"P1": 3, "P7": 80}), "P4")
        assert pf.frame is None
        assert "not annotated" in pf.note

    def test_no_annotations_at_all(self):
        s = slot()
        s.record = None
        assert dv.resolve_position(s, "P1").frame is None

    def test_timestamp_only_annotation_uses_this_videos_timeline(self):
        pf = dv.resolve_position(slot(fps=240.0, n=1200, timestamps={"P4": 1000.0}), "P4")
        assert pf.frame == 240
        assert pf.source == "annotation timestamp"

    def test_out_of_range_frame_falls_back_to_timestamp(self):
        pf = dv.resolve_position(slot(fps=30.0, n=60, frames={"P4": 500},
                                      timestamps={"P4": 1000.0}), "P4")
        assert pf.frame == 30
        assert "outside this video" in pf.note

    def test_timestamp_after_the_end_is_unavailable_not_last_frame(self):
        # 60 frames at 30fps end at 1966.7 ms
        pf = dv.resolve_position(slot(fps=30.0, n=60, timestamps={"P10": 5000.0}), "P10")
        assert pf.frame is None
        assert "outside this video" in pf.note

    def test_timestamp_before_the_start_is_unavailable_not_first_frame(self):
        pf = dv.resolve_position(slot(fps=30.0, n=60, timestamps={"P1": -200.0}), "P1")
        assert pf.frame is None

    def test_out_of_range_frame_and_timestamp_is_unavailable(self):
        pf = dv.resolve_position(slot(fps=30.0, n=60, frames={"P4": 500},
                                      timestamps={"P4": 16666.7}), "P4")
        assert pf.frame is None
        assert "saved frame 500" in pf.note and "outside this video" in pf.note

    def test_timestamp_within_half_a_frame_of_the_end_is_the_last_frame(self):
        last = 59 * 1000 / 30
        pf = dv.resolve_position(slot(fps=30.0, n=60, timestamps={"P10": last + 10}), "P10")
        assert pf.frame == 59

    def test_partner_instant_beyond_this_video_is_unavailable(self):
        dtl = slot(view="dtl", fps=30.0, n=300, frames={"P10": 250})   # 8333 ms
        fo = slot(view="face_on", fps=30.0, n=120, frames={})            # ends 3967 ms
        pf = dv.resolve_position(fo, "P10", partner=dtl)
        assert pf.frame is None
        assert "outside this video" in pf.note

    def test_frame_at_timestamp(self):
        from utils.pose_data import frame_at_timestamp
        ts = [0.0, 10.0, 20.0, 30.0]
        assert frame_at_timestamp(ts, 14) == 1
        assert frame_at_timestamp(ts, 34.9) == 3
        assert frame_at_timestamp(ts, 35.1) is None
        assert frame_at_timestamp(ts, -5.1) is None
        assert frame_at_timestamp([], 0) is None

    def test_different_frame_rates_select_each_videos_own_frame(self):
        user = slot(fps=249.3, n=1244, frames={"P4": 652})
        ref = slot(role="reference", fps=29.97, n=540, frames={"P4": 278})
        assert dv.resolve_position(user, "P4").frame == 652
        assert dv.resolve_position(ref, "P4").frame == 278

    def test_partner_view_maps_by_timestamp_not_frame_number(self):
        dtl = slot(view="dtl", fps=30.0, n=60, frames={"P4": 30})          # 1000 ms
        fo = slot(view="face_on", fps=60.0, n=120, frames={})
        pf = dv.resolve_position(fo, "P4", partner=dtl)
        assert pf.frame == 60
        assert pf.source == "partner view"
        assert dv.resolve_position(fo, "P4").frame is None

    def test_own_annotation_wins_over_partner(self):
        dtl = slot(view="dtl", frames={"P4": 30})
        fo = slot(view="face_on", frames={"P4": 33})
        assert dv.resolve_position(fo, "P4", partner=dtl).frame == 33


class TestBuildComparison:
    def test_partner_view_is_never_used_for_the_reference(self):
        slots = {
            ("user", "dtl"): slot("user", "dtl", frames={"P4": 30}),
            ("user", "face_on"): slot("user", "face_on"),
            ("reference", "dtl"): slot("reference", "dtl", frames={"P4": 30}),
            ("reference", "face_on"): slot("reference", "face_on"),
        }
        c = dv.build_comparison(slots, "P4", use_partner_view=True)
        assert c.slots[("user", "face_on")].position.frame == 30
        assert c.slots[("reference", "face_on")].position.frame is None

    def test_without_pose_everything_is_unavailable(self):
        slots = {(r, v): slot(r, v, frames={"P1": 1, "P4": 30})
                 for r in ("user", "reference") for v in ("dtl", "face_on")}
        c = dv.build_comparison(slots, "P4")
        for rows in c.rows.values():
            assert rows and all(r.difference is None for r in rows)

    def test_view_mismatch_warning(self):
        s = slot("user", "dtl", frames={"P1": 1})
        s.record.view_angle = "face_on"
        res = dv.measure_slot(s, "P1")
        assert any("face_on" in w for w in res.warnings)


@pytest.fixture
def library(tmp_path, annotations_file):
    """Deterministic four-video fixture with mismatched rates, sizes and lengths."""
    user = tmp_path / "user" / "session"
    ref = tmp_path / "reference"
    user.mkdir(parents=True)
    ref.mkdir()
    specs = {
        ("user", "dtl"): (user / "Test Golfer - 2026-01-01 120000 Fox Down the line 61fc.avi",
                          90, 60.0, (320, 240)),
        ("user", "face_on"): (user / "Test Golfer - 2026-01-01 120000 Fox Face on right aaf0.avi",
                              90, 60.0, (320, 240)),
        ("reference", "dtl"): (ref / "Ref_DTL.avi", 50, 30.0, (240, 180)),
        ("reference", "face_on"): (ref / "Ref_faceon.avi", 40, 25.0, (320, 180)),
    }
    paths = {k: write_video(p, n, fps, size) for k, (p, n, fps, size) in specs.items()}
    ann.save_annotations(paths[("user", "dtl")], {"P1": 5, "P4": 45}, 60.0, view_angle="dtl")
    ann.save_annotations(paths[("user", "face_on")], {"P1": 5}, 60.0, view_angle="face_on")
    ann.save_annotations(paths[("reference", "dtl")], {"P1": 2, "P4": 20}, 30.0,
                         view_angle="dtl")
    ann.save_annotations(paths[("reference", "face_on")], {"P1": 3, "P4": 25}, 25.0,
                         view_angle="face_on")
    backends = {k: FakeBackend(view=k[1], n_frames=specs[k][1]) for k in specs}
    return tmp_path, paths, specs, backends


def load_slots(paths, backends, cache_dir):
    slots = {}
    for key, path in paths.items():
        pose = get_pose_sequence(path, backends[key], cache_dir=cache_dir)
        slots[key] = dv.Slot(role=key[0], view=key[1], path=path, info=dv.probe_video(path),
                             record=ann.load_annotation_record(path), pose=pose)
    return slots


class TestOfflinePipelineSmoke:
    def test_four_panel_comparison_end_to_end(self, library):
        root, paths, specs, backends = library
        cache = str(root / "cache")
        slots = load_slots(paths, backends, cache)

        c = dv.build_comparison(slots, "P4", use_partner_view=True)

        frames = {k: r.position.frame for k, r in c.slots.items()}
        assert frames == {("user", "dtl"): 45, ("user", "face_on"): 45,
                          ("reference", "dtl"): 20, ("reference", "face_on"): 25}
        assert c.slots[("user", "face_on")].position.source == "partner view"

        for key, res in c.slots.items():
            img = dv.render_panel(res, highlight_key="lead_elbow_flexion")
            w, h = specs[key][3]
            assert img.shape == (h, w, 3), key            # aspect ratio preserved
            plain = dv.render_panel(res, show_skeleton=False)
            assert not np.array_equal(img, plain)          # skeleton was drawn

        assert set(c.rows) == {"dtl", "face_on"}
        for view, rows in c.rows.items():
            assert all(r.view == view for r in rows)
            arm = {r.key: r for r in rows}["lead_arm_inclination"]
            assert arm.difference == pytest.approx(arm.user - arm.reference)
            hips = {r.key: r for r in rows}["hips_dy"]
            assert hips.user == pytest.approx(0.0, abs=1e-9)   # fixture hips don't move

    def test_switching_position_does_not_rerun_inference(self, library):
        root, paths, specs, backends = library
        cache = str(root / "cache")
        load_slots(paths, backends, cache)
        calls = {k: b.calls for k, b in backends.items()}
        slots = load_slots(paths, backends, cache)
        for p in ("P1", "P4", "P7"):
            dv.build_comparison(slots, p)
        assert {k: b.calls for k, b in backends.items()} == calls

    def test_missing_position_shows_no_frame_and_no_values(self, library):
        root, paths, specs, backends = library
        slots = load_slots(paths, backends, str(root / "cache"))
        c = dv.build_comparison(slots, "P7")
        for res in c.slots.values():
            assert res.position.frame is None
            assert dv.render_panel(res) is None
            assert all(m.value is None for m in res.measurements)

    def test_undetected_frame_gives_unavailable_measurements(self, library):
        root, paths, specs, backends = library
        key = ("reference", "dtl")
        backends[key] = FakeBackend(view="dtl", n_frames=50, missing={20},
                                    identity="fake-missing")
        slots = load_slots(paths, backends, str(root / "cache"))
        res = dv.build_comparison(slots, "P4").slots[key]
        assert res.position.frame == 20
        assert all(m.value is None and m.reason == "no pose detected in this frame"
                   for m in res.measurements)

    def test_table_rows_keep_unavailable_as_none(self, library):
        root, paths, specs, backends = library
        slots = load_slots(paths, backends, str(root / "cache"))
        rows = dv.build_comparison(slots, "P4").rows["dtl"]
        table = dv.rows_as_table(rows)
        shoulder = next(t for t in table if t["Measurement"].startswith("Shoulder tilt"))
        assert shoulder["Your value"] is None              # DTL shoulders overlap
        assert "overlap" in shoulder["Note"]
