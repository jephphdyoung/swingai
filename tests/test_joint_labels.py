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
from utils import label_points as lp
from utils.pose_metrics import COMMON_JOINTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTL, FO = "videos/x/dtl.mp4", "videos/x/fo.mp4"


def sha(video, frame):
    return hashlib.sha1(f"{video}#{frame}".encode()).hexdigest()


def manifest_dict(schema2=False):
    """Benchmark frames 0-2 of each video; schema 2 adds the catalogue and a
    non-benchmark sequence (DTL frames 3-4)."""
    frames = []
    for video, view in ((DTL, "dtl"), (FO, "face_on")):
        for f in (0, 1, 2):
            frames.append({"id": f"{len(frames):02d}", "video": video, "view": view, "frame": f,
                           "reason": "test", "p1_frame": 0, "width": 640, "height": 480,
                           "display_scale": 1.0, "image_sha1": sha(video, f)})
    m = {"joints": list(COMMON_JOINTS), "frames": frames}
    if schema2:
        for f in (3, 4):
            frames.append({"id": f"{len(frames):02d}", "video": DTL, "view": "dtl", "frame": f,
                           "set": "sequence", "sequence": {"name": "p5-p8"},
                           "reason": "sequence p5-p8", "p1_frame": 0, "width": 640,
                           "height": 480, "display_scale": 1.0, "image_sha1": sha(DTL, f)})
        m.update(schema=2, catalogue=lp.catalogue())
    return m


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
        (lambda d: d.update(schema=4), "schema is 4"),
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


def extended(d):
    """Schema-2 export of labels_dict(): feet/hands points on the benchmark
    frames, a labelled sequence frame, and frame_status."""
    d["schema"] = 2
    v = d["videos"][DTL]
    v["frames"]["0"].update(left_heel={"x": 300, "y": 470, "visible": True},
                            left_shoe_heel_contact={"x": 302, "y": 476, "visible": True},
                            floor_a={"x": 100, "y": 478, "visible": True},
                            left_thumb={"x": 310, "y": 250, "visible": False},
                            shaft_butt={"x": 320, "y": 240, "visible": True})
    v["frames"]["3"] = labelled(dx=40)
    v["image_sha1"]["3"] = sha(DTL, 3)
    v["frame_status"] = {"0": {"groups_complete": ["body"], "cannot_place": ["right_heel"]},
                         "3": {"groups_complete": ["body"], "cannot_place": []}}
    return d


@pytest.fixture
def manifest2(tmp_path):
    p = tmp_path / "manifest2.json"
    p.write_text(json.dumps(manifest_dict(schema2=True)))
    return jl.load_manifest(str(p))


class TestSchema2:
    def test_feet_hands_points_and_sequence_frames_are_accepted(self, manifest2):
        labels = jl.validate(extended(labels_dict()), manifest2)
        v = labels[DTL]
        assert {"left_heel", "left_shoe_heel_contact", "floor_a", "left_thumb",
                "shaft_butt"} <= set(v["frames"][0])
        assert v["sets"] == {0: "benchmark", 1: "benchmark", 2: "benchmark", 3: "sequence"}
        assert v["frame_status"][0] == {"groups_complete": ["body"], "cannot_place": ["right_heel"]}

    def test_benchmark_results_are_unchanged_by_extra_points_and_frames(self, manifest,
                                                                         manifest2):
        seqs = sequences(rt_dx=7.5, mp_low_conf_joint="left_knee")
        plain = jl.evaluate(jl.validate(labels_dict(), manifest), seqs)
        rich = jl.evaluate(jl.validate(extended(labels_dict()), manifest2), seqs)
        assert rich == plain
        cov = jl.coverage(jl.validate(extended(labels_dict()), manifest2), manifest2)
        assert cov["frames_expected"] == 6 and cov["frames_labelled"] == 6
        assert cov["joints_labelled"] == 6 * len(COMMON_JOINTS)
        assert cov["frames_outside_benchmark"] == 1

    def test_benchmark_labels_drops_sequence_frames(self, manifest2):
        labels = jl.validate(extended(labels_dict()), manifest2)
        assert set(jl.benchmark_labels(labels)[DTL]["frames"]) == {0, 1, 2}

    @pytest.mark.parametrize("mutate, message", [
        # MediaPipe's foot_index is not a label point, and never stands in for a toe.
        (lambda d: d["videos"][DTL]["frames"]["0"].update(
            left_foot_index={"x": 1, "y": 1, "visible": True}),
         "left_foot_index: not a labelled joint"),
        (lambda d: d["videos"][DTL]["frame_status"]["0"].update(groups_complete=["arms"]),
         "groups_complete must be a list"),
        (lambda d: d["videos"][DTL]["frame_status"]["0"].update(cannot_place=["left_heel"]),
         "both placed and marked as impossible"),
        (lambda d: d["videos"][DTL]["frame_status"]["0"].update(cannot_place=["toe"]),
         "cannot_place must list known points"),
        (lambda d: d["videos"][DTL]["frame_status"].update({"9": {}}),
         "frame_status for a frame outside the labelling set"),
        # carried-over points that nobody confirmed are refused
        (lambda d: d["videos"][DTL]["frame_status"]["0"].update(
            review={"source": "old.json", "points_unchanged": ["nose"]}),
         "were not confirmed on this image"),
        (lambda d: d["videos"][DTL]["frame_status"]["0"].update(
            review={"source": "old.json", "confirmed_by": " ", "confirmed_at": "t"}),
         "were not confirmed on this image"),
    ])
    def test_problems_are_refused(self, manifest2, mutate, message):
        d = extended(labels_dict())
        mutate(d)
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest2)
        assert any(message in p for p in e.value.problems), e.value.problems

    def test_schema_1_stays_body_and_benchmark_only(self, manifest2):
        d = labels_dict()
        d["videos"][DTL]["frames"]["0"]["left_heel"] = {"x": 1, "y": 1, "visible": True}
        d["videos"][DTL]["frames"]["3"] = labelled()
        d["videos"][DTL]["image_sha1"]["3"] = sha(DTL, 3)
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest2)
        text = " | ".join(e.value.problems)
        assert "left_heel: not a labelled joint" in text
        assert "schema-1 export can't label frames outside the benchmark" in text

    def test_confirmed_review_is_kept_and_counted(self, manifest2):
        d = extended(labels_dict())
        d["videos"][DTL]["frame_status"]["1"] = {
            "groups_complete": ["body"], "cannot_place": [],
            "review": {"source": "old.json", "confirmed_by": "Test Labeller",
                       "confirmed_at": "2026-10-21T00:00:00Z", "points_unchanged": ["nose"]}}
        labels = jl.validate(d, manifest2)
        assert labels[DTL]["frame_status"][1]["review"]["points_unchanged"] == ["nose"]
        assert jl.coverage(labels, manifest2)["frames_confirmed_from_earlier_export"] == 1


def manifest3():
    """Schema-3 manifest: DTL frames 0-1 and face-on 0-1 blind, frame 2 of each assisted."""
    m = manifest_dict(schema2=True)
    m["schema"] = 3
    for f in m["frames"]:
        f["blind"] = f.get("set", "benchmark") == "benchmark" and f["frame"] in (0, 1)
    return jl.load_manifest_dict(m)


def assisted_export(**frame2):
    """Schema-3 export: blind frames 0-1 placed by hand; frame 2 accepted from models."""
    d = labels_dict()
    d["schema"] = 3
    for video in (DTL, FO):
        st = {}
        for f in ("0", "1"):
            st[f] = {"mode": "blind", "groups_complete": ["body"], "cannot_place": [],
                     "origins": {j: {"origin": "placed"} for j in d["videos"][video]["frames"][f]}}
        origins = {j: {"origin": "model_accepted", "model": "rtmpose"}
                   for j in d["videos"][video]["frames"]["2"]}
        origins["nose"] = {"origin": "model_corrected", "model": ["mediapipe", "rtmpose"]}
        origins["left_ear"] = {"origin": "placed"}
        st["2"] = {"mode": "assisted", "groups_complete": ["body"], "cannot_place": [],
                   "origins": origins,
                   "assist": {"confirmed_by": "Test Labeller", "confirmed_at": "t",
                              "primary": "rtmpose", "models": {}}, **frame2}
        d["videos"][video]["frame_status"] = st
    return d


class TestAssisted:
    def test_modes_and_origins_are_kept(self):
        labels = jl.validate(assisted_export(), manifest3())
        v = labels[DTL]
        assert v["modes"] == {0: "blind", 1: "blind", 2: "assisted"}
        o = v["frame_status"][2]["origins"]
        assert o["left_wrist"] == {"origin": "model_accepted", "model": "rtmpose"}
        assert o["nose"]["model"] == ["mediapipe", "rtmpose"]
        assert v["frame_status"][2]["assist"]["confirmed_by"] == "Test Labeller"

    def test_earlier_schemas_had_no_model_output_so_every_frame_is_blind(self, manifest2):
        labels = jl.validate(extended(labels_dict()), manifest2)
        assert set(labels[DTL]["modes"].values()) == {"blind"}

    @pytest.mark.parametrize("mutate, message", [
        # model output on a blind frame
        (lambda st: st["0"]["origins"].update(nose={"origin": "model_accepted",
                                                    "model": "mediapipe"}),
         "a model proposal on a blind frame"),
        # a frame's mode must be the manifest's
        (lambda st: st["0"].update(mode="assisted"), "the manifest makes this frame 'blind'"),
        (lambda st: st["2"].update(mode="blind"), "the manifest makes this frame 'assisted'"),
        (lambda st: st["2"].update(mode="maybe"), "mode must be one of"),
        (lambda st: st.pop("1"), "frame 1: no frame_status mode"),
        # accepting a suggestion is only valid with the frame signed off
        (lambda st: st["2"].pop("assist"), "without the frame's assist.confirmed_by"),
        (lambda st: st["2"]["assist"].update(confirmed_by=""), "without the frame's"),
        # a model can only be credited with points it really produces
        (lambda st: st["2"]["origins"].update(left_wrist={"origin": "model_accepted",
                                                          "model": "openpose"}),
         "'openpose' does not produce this point"),
        (lambda st: st["2"]["origins"].update(nose={"origin": "model_accepted",
                                                    "model": ["mediapipe", "rtmpose"]}),
         "does not produce this point"),
        (lambda st: st["2"]["origins"].update(nose={"origin": "guessed"}),
         "origin must be one of"),
        (lambda st: st["2"]["origins"].update(left_heel={"origin": "placed"}),
         "origin given for a point that isn't placed"),
        (lambda st: st["2"]["origins"].update(nose={"origin": "earlier_kept"}),
         "earlier_kept but not in review.points_unchanged"),
        (lambda st: st["2"]["origins"].update(nose={"origin": "floor_carried",
                                                    "from_frame": 0}),
         "floor_carried needs a floor point"),
    ])
    def test_problems_are_refused(self, mutate, message):
        d = assisted_export()
        mutate(d["videos"][DTL]["frame_status"])
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest3())
        assert any(message in p for p in e.value.problems), e.value.problems

    def test_model_points_a_model_does_not_have_are_refused(self):
        d = assisted_export()
        f2 = d["videos"][DTL]["frames"]["2"]
        f2["left_big_toe"] = {"x": 300, "y": 470, "visible": True}
        f2["left_thumb"] = {"x": 310, "y": 250, "visible": True}
        st = d["videos"][DTL]["frame_status"]["2"]["origins"]
        st["left_big_toe"] = {"origin": "model_accepted", "model": "mediapipe"}   # foot_index
        st["left_thumb"] = {"origin": "model_accepted", "model": "rtmpose"}       # no fingers
        with pytest.raises(jl.LabelError) as e:
            jl.validate(d, manifest3())
        text = " | ".join(e.value.problems)
        assert "left_big_toe: 'mediapipe' does not produce" in text
        assert "left_thumb: 'rtmpose' does not produce" in text

    def test_floor_carried_within_the_video(self):
        d = assisted_export()
        for f in ("0", "2"):
            d["videos"][DTL]["frames"][f]["floor_a"] = {"x": 100, "y": 470, "visible": True}
        d["videos"][DTL]["frame_status"]["2"]["origins"]["floor_a"] = {
            "origin": "floor_carried", "from_frame": 0}
        labels = jl.validate(d, manifest3())
        assert labels[DTL]["frame_status"][2]["origins"]["floor_a"]["from_frame"] == 0

    def test_independent_and_assisted_are_scored_separately(self):
        labels = jl.validate(assisted_export(), manifest3())
        # MediaPipe exact; RTMPose off by 7.5 px (0.075 torso) on frame 2 only.
        mp = seq_from([body() for _ in range(3)])
        rt = seq_from([body(), body(), body(dx=7.5)], min_conf=0.3)
        seqs = {"mediapipe": {DTL: mp, FO: mp}, "rtmpose": {DTL: rt, FO: rt}}
        split = jl.evaluate_split(labels, seqs)
        ind, ast = split["independent"]["by_view"]["dtl"], split["assisted"]["by_view"]["dtl"]
        assert ind["rtmpose"]["all"]["frames"] == 2 and ast["rtmpose"]["all"]["frames"] == 1
        assert ind["rtmpose"]["all"]["pck@0.05"] == 1.0
        assert ast["rtmpose"]["all"]["pck@0.05"] == 0.0
        assert split["independent"]["comparison"]["dtl"]["rtmpose - mediapipe"]["pck@0.1"][
            "frames"] == 2
        # Unsplit evaluate still scores every benchmark frame, as before.
        assert jl.evaluate(labels, seqs)["by_view"]["dtl"]["rtmpose"]["all"]["frames"] == 3

    def test_coverage_counts_blind_assisted_and_model_points(self):
        labels = jl.validate(assisted_export(), manifest3())
        cov = jl.coverage(labels, manifest3())
        assert cov["frames_blind_expected"] == 4 and cov["frames_blind_labelled"] == 4
        assert cov["frames_assisted_labelled"] == 2
        n = len(COMMON_JOINTS) - 2                      # nose corrected, left_ear placed
        assert cov["assisted_points"] == {"model_accepted": {"rtmpose": 2 * n},
                                          "model_corrected": {"mediapipe+rtmpose": 2}}


class TestReviewPrior:
    """Offering an earlier export again: only on identical images, never certified."""

    def prior(self):
        return {"schema": 1, "verified_by": "Earlier Labeller", "exported_at": "t",
                "videos": {DTL: {"frames": {"0": labelled(), "1": labelled(), "2": labelled()}},
                           "videos/x/gone.mp4": {"frames": {"0": labelled()}}}}

    def test_only_frames_labelled_on_the_identical_image_are_offered(self, manifest2):
        shown = {(DTL, 0): sha(DTL, 0), (DTL, 1): sha(DTL, 0), (DTL, 2): "f" * 40}
        previous = {(DTL, 1): sha(DTL, 0)}
        out = jl.review_prior(self.prior(), manifest2, shown, previous, source="old.json")
        assert [(f["video"], f["frame"]) for f in out["frames"]] == [(DTL, 0)]
        assert out["frames"][0]["points"] == labelled()           # copied, not altered
        assert out["frames"][0]["image_sha1"] == sha(DTL, 0)
        why = {(r["video"], r["frame"]): r["why"] for r in out["refused"]}
        assert "image of frame 0" in why[(DTL, 1)] and "relabel" in why[(DTL, 1)]
        assert "different image" in why[(DTL, 2)]
        assert "not in the current labelling set" in why[("videos/x/gone.mp4", 0)]
        assert out["source"] == "old.json" and out["verified_by"] == "Earlier Labeller"
        # Nothing in the result marks a point as confirmed or verified.
        assert "confirmed" not in json.dumps(out["frames"])

    def test_unknown_image_or_bad_points_are_refused(self, manifest2):
        p = self.prior()
        p["videos"][DTL]["frames"]["2"]["nose"]["x"] = 9999
        shown = {(DTL, 0): sha(DTL, 0), (DTL, 2): sha(DTL, 2)}     # frame 1: no record
        out = jl.review_prior(p, manifest2, shown)
        why = {(r["video"], r["frame"]): r["why"] for r in out["refused"]}
        assert "no record of which image" in why[(DTL, 1)]
        assert "invalid points ['nose']" in why[(DTL, 2)]
        assert [f["frame"] for f in out["frames"]] == [0]

    def test_the_committed_earlier_export_is_left_unchanged(self):
        path = os.path.join(REPO, "data/joint_labels/labels-old-labeller-2026-10-21.json")
        with open(path, "rb") as f:
            assert hashlib.sha1(f.read()).hexdigest() == \
                "aa53e8a7733518c003152703f20806627f9dc929"


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
                  "report": jl.evaluate_split(labels, sequences(rt_dx=7.5))}
        md = _cli().render(result)
        assert "### Down the line" in md and "### Face-on" in md
        assert "RTMPose - MediaPipe PCK" in md and "left wrist" in md
        # schema-1 labels came from a labeller with no model output: all independent
        assert md.index("Independent (blind) frames") < md.index("### Down the line")
        assert "Assisted frames: reported separately" in md
        assert md.split("## Assisted frames")[1].count("No labelled frames.") == 1

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
    with open(os.path.join(REPO, "data/joint_labels/manifest.json")) as f:
        raw = json.load(f)
    assert raw["schema"] == jl.SCHEMA and raw["joints"] == list(COMMON_JOINTS)
    assert raw["catalogue"] == json.loads(json.dumps(lp.catalogue()))
    m = jl.load_manifest_dict(raw)
    bench = {k: v for k, v in m["frames"].items() if v["set"] == "benchmark"}
    assert len(bench) == 20                     # the original benchmark set, unchanged
    views = {}
    for (video, frame), item in bench.items():
        views.setdefault(item["view"], set()).add(video)
        # every video's P1 is in the set, so its errors can be normalized
        assert (video, item["p1_frame"]) in m["frames"]
        assert len(item["image_sha1"]) == 40
    assert {v: len(s) for v, s in views.items()} == {"dtl": 2, "face_on": 2}


def test_benchmark_frames_and_images_are_pinned():
    """The 20 frames MediaPipe and RTMPose are compared on, and the exact
    images (SHA-1), are those of the 2026-10-21 set. Adding points, groups or
    sequences must never change them."""
    with open(os.path.join(REPO, "data/joint_labels/manifest.json")) as f:
        m = json.load(f)
    key = sorted((f["video"], f["frame"], f["image_sha1"]) for f in m["frames"]
                 if f.get("set", "benchmark") == "benchmark")
    assert hashlib.sha1(json.dumps(key).encode()).hexdigest() == \
        "6d8150a103dbb5f11698ba6e56e3b42e6121f416"
