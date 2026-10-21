"""The labelling catalogue: groups, kinds, and what each pose model really has."""

import pytest

from utils import label_points as lp
from utils.pose_metrics import COMMON_JOINTS
from utils.pose_rtmpose import HALPE26


def test_body_group_is_exactly_the_benchmark_joints_in_order():
    assert lp.BENCHMARK_POINTS == COMMON_JOINTS
    assert lp.GROUP_POINTS["body"] == COMMON_JOINTS


def test_names_are_unique_and_every_point_is_complete():
    assert len(set(lp.POINT_NAMES)) == len(lp.POINT_NAMES)
    for p in lp.POINTS:
        assert p["group"] in lp.GROUPS and p["kind"] in lp.KINDS
        assert p["help"].strip()
        assert set(p["models"]) == set(lp.MODELS)
        for m in p["models"].values():
            assert m["match"] in (None, "same", "approximate")
            assert m["extracted"] in (True, False)
            if m["point"] is None:                    # no point: nothing to match or extract
                assert m["match"] is None and m["extracted"] is False


def test_requested_points_exist():
    for side in ("left", "right"):
        for part in ("heel", "big_toe", "little_toe", "shoe_heel_contact", "shoe_toe_contact",
                     "thumb", "index", "pinky", "hand_back", "forearm_marker"):
            assert f"{side}_{part}" in lp.POINT
    for name in ("floor_a", "floor_b", "shaft_butt", "shaft_hosel", "clubhead_heel",
                 "clubhead_toe", "clubface_center"):
        assert name in lp.POINT


@pytest.mark.parametrize("kind", ["contact", "reference", "marker", "equipment"])
def test_manual_points_have_no_model_equivalent(kind):
    names = [p["name"] for p in lp.POINTS if p["kind"] == kind]
    assert names
    for n in names:
        assert all(m["point"] is None for m in lp.POINT[n]["models"].values()), n


def test_contact_points_are_distinct_from_anatomical_foot_points():
    for side in ("left", "right"):
        assert lp.POINT[f"{side}_heel"]["kind"] == "anatomical"
        assert lp.POINT[f"{side}_shoe_heel_contact"]["kind"] == "contact"
        assert lp.POINT[f"{side}_big_toe"]["kind"] == "anatomical"
        assert lp.POINT[f"{side}_shoe_toe_contact"]["kind"] == "contact"


def test_mediapipe_foot_index_is_not_the_big_toe():
    assert "left_foot_index" not in lp.POINT                 # not a label point at all
    mp = lp.POINT["left_big_toe"]["models"]["mediapipe"]
    assert mp["point"] is None and "not the big toe" in mp["note"]
    rt = lp.POINT["left_big_toe"]["models"]["rtmpose"]
    assert rt["point"].startswith("left_big_toe") and rt["match"] == "same"


def test_rtmpose_has_no_finger_points_and_mediapipes_are_approximate():
    for side in ("left", "right"):
        for finger in ("thumb", "index", "pinky"):
            m = lp.POINT[f"{side}_{finger}"]["models"]
            assert m["rtmpose"]["point"] is None
            assert m["mediapipe"]["match"] == "approximate" and m["mediapipe"]["extracted"]
    assert not any("thumb" in k or "index" in k or "pinky" in k for k in HALPE26)


def test_rtmpose_points_are_real_halpe26_keypoints():
    for p in lp.POINTS:
        rt = p["models"]["rtmpose"]["point"]
        if rt:
            name, idx = rt.split(" (")
            assert HALPE26[int(idx.rstrip(")"))] == name, p["name"]


def test_points_not_kept_by_the_pose_layer_say_so():
    from utils.landmarks import LANDMARK_NAMES
    for p in lp.POINTS:
        for model, m in p["models"].items():
            if m["point"] and not m["extracted"]:
                assert "drops it" in m["note"], (p["name"], model)
            if p["group"] == "body":
                assert m["extracted"] and p["name"] in LANDMARK_NAMES


def test_catalogue_is_json_ready_and_grouped():
    import json
    cat = json.loads(json.dumps(lp.catalogue()))
    assert [g["id"] for g in cat["groups"]] == list(lp.GROUPS)
    assert sum(len(g["points"]) for g in cat["groups"]) == len(cat["points"])
