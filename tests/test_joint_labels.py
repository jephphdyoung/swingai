"""Human-label evaluation: validation, scoring per view, and the CLI's refusals.

All labels here are synthetic test fixtures built in tmp_path. Nothing is
written to data/joint_labels/, and no test supplies real human labels.
"""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

from test_pose_metrics import body, seq_from
from utils import joint_labels as jl
from utils.pose_metrics import COMMON_JOINTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTL, FO = "videos/x/dtl.mp4", "videos/x/fo.mp4"


def sha(video, frame):
    return hashlib.sha1(f"{video}#{frame}".encode()).hexdigest()


def manifest_dict():
    frames = []
    for video, view in ((DTL, "dtl"), (FO, "face_on")):
        for f in (0, 1, 2):
            frames.append({"id": f"{len(frames):02d}", "video": video, "view": view, "frame": f,
                           "reason": "test", "p1_frame": 0, "width": 640, "height": 480,
                           "display_scale": 1.0, "image_sha1": sha(video, f)})
    return {"joints": list(COMMON_JOINTS), "frames": frames}


def labelled(dx=0.0, occluded=()):
    return {j: {"x": x + dx, "y": y, "visible": j not in occluded}
            for j, (x, y) in body().items()}


def labels_dict(**over):
    d = {"schema": 1, "verified_by": "Test Labeller", "videos": {
        DTL: {"verified_by": "Test Labeller", "p1_frame": 0, "view": "dtl",
              "frames": {"0": labelled(), "1": labelled(occluded=("left_wrist",)),
                         "2": labelled()},
              "image_sha1": {str(f): sha(DTL, f) for f in range(3)}},
        FO: {"verified_by": "Test Labeller", "p1_frame": 0, "view": "face_on",
             "frames": {"0": labelled(), "1": labelled(), "2": labelled()},
             "image_sha1": {str(f): sha(FO, f) for f in range(3)}},
    }}
    d.update(over)
    return d


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest_dict()))
    return jl.load_manifest(str(p))


class TestValidation:
    def test_valid_labels_load_with_int_frames(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        assert set(labels) == {DTL, FO}
        assert set(labels[DTL]["frames"]) == {0, 1, 2}
        assert labels[DTL]["frames"][1]["left_wrist"]["visible"] is False

    def test_missing_file_is_refused(self, tmp_path):
        with pytest.raises(jl.LabelError, match="no labels at"):
            jl.load_labels(str(tmp_path / "labels.json"), str(tmp_path / "m.json"))

    @pytest.mark.parametrize("mutate, message", [
        (lambda d: d.update(verified_by=""), "no verified_by"),
        (lambda d: d.update(schema=2), "schema is 2"),
        (lambda d: d.update(videos={}), "no labelled videos"),
        (lambda d: d["videos"][DTL].update(verified_by=" "), "dtl.mp4: no verified_by"),
        (lambda d: d["videos"][DTL].update(view="face_on"), "manifest says 'dtl'"),
        (lambda d: d["videos"][DTL].update(p1_frame=1), "p1_frame 1"),
        (lambda d: d["videos"].update({"videos/x/other.mp4": d["videos"][DTL]}),
         "not in the labelling manifest"),
        (lambda d: d["videos"][DTL]["frames"].update({"7": labelled()}),
         "frame 7: not a frame of the labelling set"),
        (lambda d: d["videos"][DTL]["frames"]["0"].update(left_thumb={"x": 1, "y": 1,
                                                                      "visible": True}),
         "left_thumb: not a labelled joint"),
        (lambda d: d["videos"][DTL]["frames"]["0"]["nose"].update(x=700),
         "outside the 640x480 frame"),
        (lambda d: d["videos"][DTL]["frames"]["0"]["nose"].update(y=float("nan")),
         "not finite numbers"),
        (lambda d: d["videos"][DTL]["frames"]["0"]["nose"].update(x=True),
         "not finite numbers"),
        (lambda d: d["videos"][DTL]["frames"]["0"]["nose"].update(visible="yes"),
         "visible must be true or false"),
        # labelled on another image: an export from the old seek-based labeller
        (lambda d: d["videos"][DTL]["image_sha1"].update({"2": "0" * 40}),
         "frame 2: labelled on a different image"),
        (lambda d: d["videos"][DTL].pop("image_sha1"), "frame 0: labelled on a different image"),
    ])
    def test_problems_are_refused_not_repaired(self, manifest, mutate, message):
        d = labels_dict()
        mutate(d)
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest)
        assert any(message in p for p in e.value.problems), e.value.problems

    def test_manifest_without_digests_is_refused(self):
        m = manifest_dict()
        for f in m["frames"]:
            del f["image_sha1"]
        with pytest.raises(jl.LabelError, match="the manifest has no image digest"):
            jl.validate(labels_dict(), jl.load_manifest_dict(m))

    def test_every_problem_is_listed(self, manifest):
        d = labels_dict(verified_by="")
        d["videos"][DTL]["frames"]["0"]["nose"]["x"] = -5
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest)
        assert len(e.value.problems) == 2

    def test_coverage_and_unscorable_video(self, manifest):
        d = labels_dict()
        del d["videos"][FO]["frames"]["0"]["left_hip"]          # no P1 torso
        del d["videos"][DTL]["frames"]["2"]
        labels = jl.validate(d, manifest)
        cov = jl.coverage(labels, manifest)
        assert cov["frames_expected"] == 6 and cov["frames_labelled"] == 5
        assert list(cov["unscorable"]) == [FO]
        assert cov["videos_unlabelled"] == []


def sequences(rt_dx=0.0, rt_missing_frame=None, mp_low_conf_joint=None):
    """Two fake models over both videos: 'mediapipe' exact, 'rtmpose' shifted."""
    mp = seq_from([body() for _ in range(3)])
    if mp_low_conf_joint:
        from utils.landmarks import LANDMARK_INDEX
        mp.confidence[:, LANDMARK_INDEX[mp_low_conf_joint]] = 0.1
    rtm = seq_from([body(dx=rt_dx) for _ in range(3)], min_conf=0.3,
                   detected=[f != rt_missing_frame for f in range(3)])
    if rt_missing_frame is not None:
        rtm.xy[rt_missing_frame] = np.nan
    return {"mediapipe": {DTL: mp, FO: mp}, "rtmpose": {DTL: rtm, FO: rtm}}


class TestScoring:
    def test_errors_are_normalized_by_the_labelled_p1_torso(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        rep = jl.evaluate(labels, sequences(rt_dx=7.5))       # torso is 100 px
        for view in ("dtl", "face_on"):
            mp, rtm = (rep["by_view"][view][b]["all"] for b in ("mediapipe", "rtmpose"))
            assert mp["mean"] == pytest.approx(0) and mp["pck@0.05"] == 1.0
            assert rtm["mean"] == pytest.approx(0.075)
            assert rtm["pck@0.05"] == 0.0 and rtm["pck@0.1"] == 1.0

    def test_views_are_scored_separately(self, manifest):
        d = labels_dict()
        for f in d["videos"][FO]["frames"].values():           # face-on labels moved
            for lab in f.values():
                lab["x"] += 15
        rep = jl.evaluate(jl.validate(d, manifest), sequences())
        assert rep["by_view"]["dtl"]["mediapipe"]["all"]["pck@0.1"] == 1.0
        assert rep["by_view"]["face_on"]["mediapipe"]["all"]["pck@0.1"] == 0.0
        assert set(rep["by_video"]) == {DTL, FO}

    def test_frame_misses_and_joint_misses(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        rep = jl.evaluate(labels, sequences(rt_missing_frame=2, mp_low_conf_joint="left_knee"))
        n = len(COMMON_JOINTS) * 3
        rtm = rep["by_view"]["dtl"]["rtmpose"]["all"]
        assert (rtm["frames"], rtm["frames_missed"]) == (3, 1)
        assert rtm["joints_missed"] == len(COMMON_JOINTS) and rtm["n"] == n
        assert rtm["pck@0.2"] == pytest.approx(2 / 3)           # a miss is a PCK failure
        mp = rep["by_view"]["dtl"]["mediapipe"]
        assert mp["all"]["frames_missed"] == 0 and mp["all"]["joints_missed"] == 3
        assert mp["per_joint"]["left_knee"]["pck@0.2"] == 0.0

    def test_occluded_labels_report_how_often_the_model_answered(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        rep = jl.evaluate(labels, sequences(mp_low_conf_joint="left_wrist"))
        dtl = rep["by_view"]["dtl"]
        assert dtl["rtmpose"]["occluded"]["n"] == 1
        assert dtl["rtmpose"]["occluded"]["answered"] == 1.0
        assert dtl["mediapipe"]["occluded"]["answered"] == 0.0
        assert "answered" not in rep["by_view"]["face_on"]["rtmpose"]["occluded"]

    def test_paired_bootstrap(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        rep = jl.evaluate(labels, sequences(rt_dx=12))
        c = rep["comparison"]["dtl"]["rtmpose - mediapipe"]["pck@0.1"]
        assert c["difference"] == pytest.approx(-1.0) and c["frames"] == 3
        assert c["ci95"] == pytest.approx([-1.0, -1.0])
        same = rep["comparison"]["dtl"]["rtmpose - mediapipe"]["pck@0.2"]
        assert same["difference"] == 0 and same["ci95"] == [0, 0]

    def test_bootstrap_refuses_different_label_sets(self):
        a = [{"video": "v", "frame": 0, "error": 0.0}]
        b = [{"video": "v", "frame": 1, "error": 0.0}]
        with pytest.raises(ValueError, match="different labelled frames"):
            jl.paired_bootstrap(a, b)

    def test_missing_sequence_is_an_error_not_a_skip(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        seqs = sequences()
        del seqs["rtmpose"][FO]
        with pytest.raises(KeyError, match="rtmpose has no pose sequence"):
            jl.evaluate(labels, seqs)


def _cli():
    spec = importlib.util.spec_from_file_location(
        "evaluate_joint_labels", os.path.join(REPO, "scripts", "evaluate_joint_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCli:
    def test_no_labels_stops_and_says_how_to_make_them(self, tmp_path, capsys):
        m = tmp_path / "manifest.json"
        m.write_text(json.dumps(manifest_dict()))
        rc = _cli().main(["--labels", str(tmp_path / "labels.json"), "--manifest", str(m)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "no labels at" in err and "labeler.html" in err

    def test_invalid_labels_are_refused(self, tmp_path, capsys):
        m, lab = tmp_path / "manifest.json", tmp_path / "labels.json"
        m.write_text(json.dumps(manifest_dict()))
        lab.write_text(json.dumps(labels_dict(verified_by="")))
        rc = _cli().main(["--labels", str(lab), "--manifest", str(m)])
        assert rc == 2 and "no verified_by" in capsys.readouterr().err

    def test_render_covers_both_views_and_the_comparison(self, manifest):
        labels = jl.validate(labels_dict(), manifest)
        result = {"labels_file": "labels.json", "verified_by": "Test Labeller",
                  "coverage": jl.coverage(labels, manifest),
                  "report": jl.evaluate(labels, sequences(rt_dx=7.5))}
        md = _cli().render(result)
        assert "### Down the line" in md and "### Face-on" in md
        assert "RTMPose - MediaPipe PCK" in md and "left wrist" in md

    def test_frames_are_decoded_sequentially_and_digests_checked(self, tmp_path):
        from _dual_view_fixtures import write_video
        video = write_video(tmp_path / "v.avi", 12, 30.0)
        decoded = jl.decode_frames(video, [3, 10, 30])
        assert set(decoded) == {3, 10}                        # 30 is past the end
        assert jl.frame_digest(decoded[3]) != jl.frame_digest(decoded[10])
        items = [{"video": video, "frame": f, "image_sha1": jl.frame_digest(decoded[f])}
                 for f in (3, 10)]
        manifest = jl.load_manifest_dict({"joints": [], "frames": items})
        labels = {video: {"frames": {3: {}, 10: {}}}}
        assert _cli().frame_alignment_problems(labels, manifest) == []
        items[1]["image_sha1"] = jl.frame_digest(decoded[3])  # frame 10 labelled on frame 3
        manifest = jl.load_manifest_dict({"joints": [], "frames": items})
        problems = _cli().frame_alignment_problems(labels, manifest)
        assert len(problems) == 1 and "frame 10: decodes to a different image" in problems[0]


def test_committed_manifest_is_valid():
    m = jl.load_manifest(os.path.join(REPO, "data/joint_labels/manifest.json"))
    assert len(m["frames"]) == 20
    views = {}
    for (video, frame), item in m["frames"].items():
        views.setdefault(item["view"], set()).add(video)
        # every video's P1 is in the set, so its errors can be normalized
        assert (video, item["p1_frame"]) in m["frames"]
        assert len(item["image_sha1"]) == 40
    assert {v: len(s) for v, s in views.items()} == {"dtl": 2, "face_on": 2}
