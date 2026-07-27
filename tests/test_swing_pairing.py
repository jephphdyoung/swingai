"""Tests for parsing and pairing Swing Catalyst capture filenames."""

import os
import pytest

from utils import swing_pairing as sp

DTL = "Jeff Young - 2026-07-27 135006 Fox Down the line 61fc.mp4"
FACE = "Jeff Young - 2026-07-27 135006 Fox Face on right aaf0.mp4"
DTL_LATER = "Jeff Young - 2026-07-27 135043 Fox Down the line 61fc.mp4"
FACE_LATER = "Jeff Young - 2026-07-27 135043 Fox Face on right aaf0.mp4"


class TestParseCapture:
    def test_parses_every_field(self):
        c = sp.parse_capture(DTL)
        assert c.golfer == "Jeff Young"
        assert c.date == "2026-07-27"
        assert c.time == "135006"
        assert c.camera == "Fox Down the line"
        assert c.camera_id == "61fc"
        assert c.view == "dtl"

    def test_recognises_the_face_on_camera(self):
        c = sp.parse_capture(FACE)
        assert c.view == "face_on"
        assert c.camera_id == "aaf0"

    def test_works_on_a_full_path(self):
        c = sp.parse_capture(os.path.join("/a/b", "2026-07-27 135033", DTL))
        assert c is not None and c.time == "135006"

    def test_camera_id_is_lowercased(self):
        c = sp.parse_capture("Jeff Young - 2026-07-27 135006 Fox Down the line 61FC.mp4")
        assert c.camera_id == "61fc"

    def test_golfer_names_containing_hyphens(self):
        c = sp.parse_capture("Anne-Marie Smith-Jones - 2026-07-27 135006 Fox Face on 1234.mp4")
        assert c is not None
        assert c.golfer == "Anne-Marie Smith-Jones"

    def test_returns_none_for_unrelated_names(self):
        assert sp.parse_capture("comparison_output.mp4") is None
        assert sp.parse_capture("GW_faceon.mp4") is None

    def test_unknown_camera_label_leaves_view_unset(self):
        c = sp.parse_capture("Jeff Young - 2026-07-27 135006 Overhead Rig 9ab1.mp4")
        assert c is not None
        assert c.view is None

    def test_swing_key_matches_across_views(self):
        assert sp.parse_capture(DTL).swing_key == sp.parse_capture(FACE).swing_key

    def test_swing_key_differs_across_swings(self):
        assert sp.parse_capture(DTL).swing_key != sp.parse_capture(DTL_LATER).swing_key


class TestViewFromLabel:
    @pytest.mark.parametrize("label", ["Fox Down the line", "DTL cam", "downtheline"])
    def test_down_the_line_variants(self, label):
        assert sp.view_from_label(label) == "dtl"

    @pytest.mark.parametrize("label", ["Fox Face on right", "faceon", "Front camera"])
    def test_face_on_variants(self, label):
        assert sp.view_from_label(label) == "face_on"

    def test_unknown_label(self):
        assert sp.view_from_label("Overhead") is None


class TestPairCaptures:
    def test_pairs_the_two_views_of_one_swing(self):
        captures = [sp.parse_capture(DTL), sp.parse_capture(FACE)]
        pairs, unpaired = sp.pair_captures(captures)
        assert len(pairs) == 1 and not unpaired
        assert pairs[0].dtl.view == "dtl"
        assert pairs[0].face_on.view == "face_on"

    def test_keeps_swings_separate(self):
        captures = [sp.parse_capture(n) for n in (DTL, FACE, DTL_LATER, FACE_LATER)]
        pairs, unpaired = sp.pair_captures(captures)
        assert len(pairs) == 2 and not unpaired
        assert pairs[0].captured_at != pairs[1].captured_at

    def test_pairs_are_ordered_by_capture_time(self):
        captures = [sp.parse_capture(n) for n in (DTL_LATER, FACE_LATER, DTL, FACE)]
        pairs, _ = sp.pair_captures(captures)
        assert pairs[0].captured_at < pairs[1].captured_at

    def test_a_lone_view_is_reported_not_dropped(self):
        pairs, unpaired = sp.pair_captures([sp.parse_capture(DTL)])
        assert pairs == []
        assert len(unpaired) == 1

    def test_unknown_camera_is_reported_not_dropped(self):
        captures = [
            sp.parse_capture(DTL),
            sp.parse_capture(FACE),
            sp.parse_capture("Jeff Young - 2026-07-27 140000 Overhead Rig 9ab1.mp4"),
        ]
        pairs, unpaired = sp.pair_captures(captures)
        assert len(pairs) == 1
        assert len(unpaired) == 1

    def test_different_golfers_do_not_pair(self):
        captures = [
            sp.parse_capture(DTL),
            sp.parse_capture("Someone Else - 2026-07-27 135006 Fox Face on right aaf0.mp4"),
        ]
        pairs, unpaired = sp.pair_captures(captures)
        assert pairs == []
        assert len(unpaired) == 2


class TestDirectoryHelpers:
    @pytest.fixture
    def session(self, tmp_path):
        for name in (DTL, FACE, DTL_LATER, FACE_LATER):
            (tmp_path / name).write_bytes(b"")
        (tmp_path / "notes.txt").write_text("ignore me")
        return tmp_path

    def test_lists_only_parseable_videos(self, session):
        captures = sp.list_captures(str(session))
        assert len(captures) == 4

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert sp.list_captures(str(tmp_path / "nope")) == []

    def test_finds_pairs_in_a_session(self, session):
        pairs, unpaired = sp.find_pairs(str(session))
        assert len(pairs) == 2 and not unpaired

    def test_recurses_into_session_subdirectories(self, tmp_path):
        sub = tmp_path / "2026-07-27 135033"
        sub.mkdir()
        (sub / DTL).write_bytes(b"")
        (sub / FACE).write_bytes(b"")
        pairs, _ = sp.find_pairs(str(tmp_path))
        assert len(pairs) == 1

    def test_find_partner_from_either_view(self, session):
        from_dtl = sp.find_partner(str(session / DTL))
        from_face = sp.find_partner(str(session / FACE))
        assert from_dtl.face_on.path.endswith(FACE)
        assert from_face.dtl.path.endswith(DTL)
        # Same swing either way round.
        assert from_dtl.captured_at == from_face.captured_at

    def test_find_partner_returns_none_when_alone(self, tmp_path):
        (tmp_path / DTL).write_bytes(b"")
        assert sp.find_partner(str(tmp_path / DTL)) is None

    def test_find_partner_returns_none_for_unparseable_name(self, tmp_path):
        (tmp_path / "random.mp4").write_bytes(b"")
        assert sp.find_partner(str(tmp_path / "random.mp4")) is None
