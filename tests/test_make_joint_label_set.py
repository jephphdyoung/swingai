"""scripts/make_joint_label_set.py: keeping the set, sequences, review candidates.

Synthetic videos only (tmp_path); nothing is written to data/joint_labels/.
"""

import importlib.util
import json
import os

import pytest

from _dual_view_fixtures import write_video
from utils import joint_labels as jl
from utils.annotations import AnnotationRecord

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "make_joint_label_set", os.path.join(REPO, "scripts", "make_joint_label_set.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "v.avi", 30, 30.0)


def frames_of(video, wanted):
    decoded = jl.decode_frames(video, wanted)
    return [mk.item_for(video, "dtl", f, decoded[f], "test", 0) for f in wanted]


class TestKeptFrames:
    def test_existing_frames_are_kept_with_their_images(self, video):
        frames = frames_of(video, [2, 5])
        kept, images = mk.kept_frames(frames)
        assert [(f["frame"], f["image_sha1"], f["set"]) for f in kept] == \
            [(2, frames[0]["image_sha1"], "benchmark"), (5, frames[1]["image_sha1"], "benchmark")]
        assert jl.frame_digest(images[(video, 5)]) == frames[1]["image_sha1"]

    def test_a_changed_image_stops_the_rebuild(self, video):
        frames = frames_of(video, [2, 5])
        frames[1]["image_sha1"] = frames[0]["image_sha1"]
        with pytest.raises(SystemExit, match="frame 5 decodes to a different image"):
            mk.kept_frames(frames)

    def test_a_frame_past_the_end_stops_the_rebuild(self, video):
        frames = frames_of(video, [2])
        frames.append({**frames[0], "frame": 99})
        with pytest.raises(SystemExit, match="frame 99 no longer decodes"):
            mk.kept_frames(frames)


class TestSequences:
    def test_sequence_frames_are_consecutive_and_outside_the_benchmark(self, video):
        items, images = mk.sequence_frames(video, "dtl", "p5-p8", 10, 16, step=2,
                                           purpose="P5-P8", existing=[(video, 12)])
        assert [i["frame"] for i in items] == [10, 14, 16]       # 12 is already in the set
        for i in items:
            assert i["set"] == "sequence"
            assert i["sequence"] == {"name": "p5-p8", "start": 10, "end": 16, "step": 2,
                                     "purpose": "P5-P8"}
            assert jl.frame_digest(images[(video, i["frame"])]) == i["image_sha1"]
        sequential = jl.decode_frames(video, [14])
        assert items[1]["image_sha1"] == jl.frame_digest(sequential[14])

    def test_bad_ranges_are_refused(self, video):
        with pytest.raises(SystemExit, match="bad sequence range"):
            mk.sequence_frames(video, "dtl", "x", 9, 3)
        with pytest.raises(SystemExit, match="ends before frame 30"):
            mk.sequence_frames(video, "dtl", "x", 28, 31)

    def test_adding_a_sequence_leaves_benchmark_evaluation_alone(self, video):
        bench = frames_of(video, [0, 1])
        seq, _ = mk.sequence_frames(video, "dtl", "heel-lift", 5, 6)
        m = jl.load_manifest_dict({"joints": [], "frames": bench + seq})
        assert [k for k, v in m["frames"].items() if v["set"] == "benchmark"] == \
            [(video, 0), (video, 1)]


class TestPPosition:
    def rec(self, **frames):
        r = AnnotationRecord(video_path="v", view_angle="dtl", convention="p7-impact")
        r.frames.update(frames)
        return r

    @pytest.mark.parametrize("frame, label", [
        (100, "P1"), (50, "before P1"), (150, "P1–P4"), (260, "P4–P7"), (999, "after P10"),
        (300, "P7")])
    def test_labels(self, frame, label):
        rec = self.rec(P1=100, P4=200, P7=300, P10=400)
        assert mk.p_position(frame, rec)["label"] == label

    def test_unconfirmed_annotations_say_so(self):
        rec = self.rec(P1=100)
        assert "not confirmed" not in mk.p_position(100, rec)["note"]
        rec.needs_review = "proposed by Claude"
        assert "not confirmed" in mk.p_position(100, rec)["note"]
        assert "approximate" in mk.p_position(100, rec)["note"]

    def test_unknown_without_annotations(self):
        assert mk.p_position(10, None) is None
        assert mk.p_position(10, self.rec()) is None


class TestPriorReview:
    def test_an_export_without_digests_is_checked_by_seeking(self, tmp_path, video):
        bench = frames_of(video, [3, 7])
        manifest = jl.load_manifest_dict({"joints": [], "frames": bench})
        point = {"nose": {"x": 10.0, "y": 20.0, "visible": True}}
        prior = {"schema": 1, "verified_by": "Earlier Labeller",
                 "videos": {video: {"frames": {"3": point, "7": point}}}}
        path = tmp_path / "old.json"
        path.write_text(json.dumps(prior))
        out = mk.prior_review(str(path), manifest)
        assert out["image_check"].startswith("re-decoding by seeking")
        # MJPG AVI seeks exactly, so both frames show the identical image here.
        assert sorted(f["frame"] for f in out["frames"]) == [3, 7]
        assert all(f["points"] == point for f in out["frames"])
        assert json.loads(path.read_text()) == prior          # the earlier file is untouched

    def test_an_export_with_digests_uses_them(self, tmp_path, video):
        bench = frames_of(video, [3])
        manifest = jl.load_manifest_dict({"joints": [], "frames": bench})
        prior = {"schema": 2, "verified_by": "x",
                 "videos": {video: {"frames": {"3": {}}, "image_sha1": {"3": "0" * 40}}}}
        path = tmp_path / "new.json"
        path.write_text(json.dumps(prior))
        out = mk.prior_review(str(path), manifest)
        assert out["image_check"] == "its own image digests"
        assert out["frames"] == [] and "different image" in out["refused"][0]["why"]


def test_labeler_template_has_one_data_slot_and_exports_the_digests():
    with open(os.path.join(REPO, "scripts", "joint_labeler.html")) as f:
        html = f.read()
    assert html.count("/*__DATA__*/null") == 1
    for s in ("image_sha1", "frame_status", "groups_complete", "cannot_place",
              "points_unchanged", "confirmed_by", "schema: DATA.schema"):
        assert s in html
    # progress is keyed by image digest, so new frames never reset it
    assert 'state.labels[f.image_sha1]' in html
