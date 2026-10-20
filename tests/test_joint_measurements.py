"""Tests for the 2D joint measurement engine."""

import math

import numpy as np
import pytest

from utils import joint_measurements as jm
from utils.landmarks import sided


def face_on_rh():
    """Right-handed golfer facing the camera: their left (lead) is image-right.

    Torso is exactly 100px (shoulder centre (100,100) to hip centre (100,200)).
    Lead arm hangs straight; trail elbow is bent 90°; trail knee 45° flexed.
    """
    s45 = 100 * math.sin(math.radians(45))
    return {
        "nose": (100, 45),
        "left_ear": (110, 40), "right_ear": (90, 40),
        "left_shoulder": (130, 100), "right_shoulder": (70, 100),
        "left_elbow": (130, 160), "left_wrist": (130, 220),
        "right_elbow": (70, 160), "right_wrist": (100, 160),
        "left_hip": (120, 200), "right_hip": (80, 200),
        "left_knee": (120, 300), "left_ankle": (120, 400),
        "right_knee": (80, 300), "right_ankle": (80 + s45, 300 + s45),
    }


def dtl_rh():
    """Down-the-line: hands hang toward the ball, which is image-right."""
    return {
        "left_ear": (95, 40), "right_ear": (85, 40),
        "left_shoulder": (130, 100), "right_shoulder": (120, 102),
        "left_elbow": (150, 160), "left_wrist": (160, 220),
        "right_elbow": (145, 160), "right_wrist": (158, 222),
        "left_hip": (80, 190), "right_hip": (78, 192),
        "left_knee": (110, 290), "left_ankle": (100, 390),
        "right_knee": (108, 292), "right_ankle": (98, 392),
    }


def by_key(measurements):
    return {m.key: m for m in measurements}


def run(points, view="face_on", hand="right", p1=None):
    ref, why = jm.body_reference(p1 if p1 is not None else points, view, hand)
    return by_key(jm.measure(points, view, hand, ref, why))


def transform(points, scale=1.0, shift=(0.0, 0.0)):
    return {k: (v[0] * scale + shift[0], v[1] * scale + shift[1]) for k, v in points.items()}


def mirror_to_left_handed(points, width=200):
    """Mirror image of a right-hander is a left-hander: flip x, swap sides."""
    out = {}
    for name, (x, y) in points.items():
        if name.startswith("left_"):
            name = "right_" + name[5:]
        elif name.startswith("right_"):
            name = "left_" + name[6:]
        out[name] = (width - x, y)
    return out


class TestKnownAngles:
    def test_included_angle(self):
        assert jm.included_angle((0, -1), (0, 0), (1, 0)) == pytest.approx(90)
        assert jm.included_angle((0, -1), (0, 0), (0, 1)) == pytest.approx(180)
        assert jm.included_angle((1, 0), (0, 0), (1, 1)) == pytest.approx(45)

    def test_elevation_is_up_positive_in_image_coordinates(self):
        assert jm.elevation((0, 0), (10, -10)) == pytest.approx(45)
        assert jm.elevation((0, 0), (-10, -10)) == pytest.approx(45)
        assert jm.elevation((0, 0), (0, 10)) == pytest.approx(-90)

    def test_flexion_values(self):
        m = run(face_on_rh())
        assert m["lead_elbow_flexion"].value == pytest.approx(0)
        assert m["trail_elbow_flexion"].value == pytest.approx(90)
        assert m["lead_knee_flexion"].value == pytest.approx(0)
        assert m["trail_knee_flexion"].value == pytest.approx(45)

    def test_upright_spine_and_level_shoulders(self):
        m = run(face_on_rh())
        assert m["spine_inclination"].value == pytest.approx(0)
        assert m["shoulder_tilt"].value == pytest.approx(0)
        assert m["lead_arm_inclination"].value == pytest.approx(-90)

    def test_spine_lean_toward_target_is_positive(self):
        p1 = face_on_rh()
        pts = dict(p1)
        lean = 100 * math.tan(math.radians(20))
        pts["left_shoulder"] = (130 + lean, 100)
        pts["right_shoulder"] = (70 + lean, 100)
        assert run(pts, p1=p1)["spine_inclination"].value == pytest.approx(20)

    def test_lead_shoulder_higher_is_positive_tilt(self):
        pts = face_on_rh()
        pts["left_shoulder"] = (130, 100 - 60 * math.tan(math.radians(10)))
        assert run(pts)["shoulder_tilt"].value == pytest.approx(10)

    def test_dtl_forward_bend_is_positive(self):
        m = run(dtl_rh(), view="dtl")
        assert m["spine_inclination"].value > 20

    def test_angles_carry_their_view_and_2d_unit(self):
        for m in jm.measure(face_on_rh(), "face_on", "right", None):
            assert m.view == "face_on"
        m = run(face_on_rh())
        assert m["lead_elbow_flexion"].unit == jm.ANGLE_UNIT
        assert "2D" in jm.ANGLE_UNIT


class TestDisplacement:
    def test_zero_at_p1(self):
        m = run(face_on_rh())
        for key, v in m.items():
            if key.endswith(("_dx", "_dy")):
                assert v.value == pytest.approx(0), key

    def test_units_are_p1_torso_lengths(self):
        p1 = face_on_rh()
        pts = transform(p1, shift=(25, -10))   # whole body 0.25 toward target, 0.1 up
        m = run(pts, p1=p1)
        assert m["hips_dx"].value == pytest.approx(0.25)
        assert m["hips_dy"].value == pytest.approx(0.10)
        assert m["head_dx"].value == pytest.approx(0.25)
        assert m["hands_dy"].value == pytest.approx(0.10)

    def test_scale_is_fixed_at_p1_not_recomputed_per_frame(self):
        p1 = face_on_rh()
        later = transform(p1, scale=2.0)       # torso is 200px in this frame
        m = run(later, p1=p1)
        # hip centre (100,200) -> (200,400): 100px right, 200px down, over P1's 100px
        assert m["hips_dx"].value == pytest.approx(1.0)
        assert m["hips_dy"].value == pytest.approx(-2.0)

    @pytest.mark.parametrize("scale,shift", [(1.0, (300, 50)), (3.5, (0, 0)), (0.4, (-80, 12))])
    def test_translation_and_scale_invariance(self, scale, shift):
        p1 = face_on_rh()
        later = transform(p1, shift=(20, -15))
        later["left_wrist"] = (150, 140)
        base = run(later, p1=p1)
        moved = run(transform(later, scale, shift), p1=transform(p1, scale, shift))
        for key, m in base.items():
            assert moved[key].value == pytest.approx(m.value, abs=1e-9), key

    def test_reference_filmed_from_the_other_side_has_same_signs(self):
        p1 = face_on_rh()
        later = transform(p1, shift=(25, 0))
        mirrored = lambda pts: {k: (400 - x, y) for k, (x, y) in pts.items()}  # noqa: E731
        a = run(later, p1=p1)
        b = run(mirrored(later), p1=mirrored(p1))
        assert a["hips_dx"].value == pytest.approx(0.25)
        assert b["hips_dx"].value == pytest.approx(0.25)


class TestHandedness:
    def test_sided(self):
        assert sided("lead", "elbow", "right") == "left_elbow"
        assert sided("lead", "elbow", "left") == "right_elbow"
        assert sided("trail", "knee", "left") == "left_knee"

    @pytest.mark.parametrize("view,builder", [("face_on", face_on_rh), ("dtl", dtl_rh)])
    def test_left_hander_mirror_measures_the_same(self, view, builder):
        p1 = builder()
        later = transform(p1, shift=(12, -4))
        later["left_wrist"] = (later["left_wrist"][0] + 30, later["left_wrist"][1] - 70)
        rh = run(later, view, "right", p1)
        lh = run(mirror_to_left_handed(later), view, "left", mirror_to_left_handed(p1))
        for key, m in rh.items():
            if m.value is None:
                assert lh[key].value is None, key
            else:
                assert lh[key].value == pytest.approx(m.value, abs=1e-9), key

    def test_wrong_handedness_swaps_lead_and_trail(self):
        right = run(face_on_rh(), hand="right")
        wrong = run(face_on_rh(), hand="left")
        assert right["trail_elbow_flexion"].value == pytest.approx(90)
        assert wrong["lead_elbow_flexion"].value == pytest.approx(90)


class TestUnavailable:
    def test_missing_landmark_only_affects_its_measurements(self):
        pts = face_on_rh()
        del pts["left_elbow"]
        m = run(pts, p1=face_on_rh())
        assert m["lead_elbow_flexion"].value is None
        assert m["lead_elbow_flexion"].reason == "landmark not detected"
        assert m["trail_elbow_flexion"].value == pytest.approx(90)

    def test_nan_landmark_is_missing(self):
        pts = face_on_rh()
        pts["right_knee"] = (np.nan, np.nan)
        assert run(pts, p1=face_on_rh())["trail_knee_flexion"].value is None

    def test_zero_length_segment(self):
        pts = face_on_rh()
        pts["left_wrist"] = pts["left_elbow"]
        m = run(pts, p1=face_on_rh())
        assert m["lead_elbow_flexion"].value is None
        assert m["lead_elbow_flexion"].reason == "zero-length segment"

    def test_zero_length_lead_arm(self):
        pts = face_on_rh()
        pts["left_wrist"] = pts["left_shoulder"]
        m = run(pts, p1=face_on_rh())
        assert m["lead_arm_inclination"].value is None
        assert m["lead_arm_inclination"].reason == "zero-length segment"

    def test_missing_p1_torso_disables_displacement_not_angles(self):
        p1 = face_on_rh()
        del p1["left_hip"]
        ref, why = jm.body_reference(p1, "face_on", "right")
        assert ref is None and "torso" in why
        m = by_key(jm.measure(face_on_rh(), "face_on", "right", ref, why))
        assert m["hips_dx"].value is None and m["hips_dx"].reason == why
        assert m["spine_inclination"].value is None
        assert m["trail_elbow_flexion"].value == pytest.approx(90)

    def test_zero_torso_is_not_a_scale(self):
        p1 = face_on_rh()
        p1["left_hip"], p1["right_hip"] = p1["left_shoulder"], p1["right_shoulder"]
        ref, why = jm.body_reference(p1, "face_on", "right")
        assert ref is None

    def test_overlapping_shoulders_have_no_tilt(self):
        pts = face_on_rh()
        pts["left_shoulder"] = (101, 98)
        pts["right_shoulder"] = (99, 102)
        m = run(pts, p1=face_on_rh())
        assert m["shoulder_tilt"].value is None
        assert "overlap" in m["shoulder_tilt"].reason

    def test_unknown_orientation_blocks_signed_horizontal_only(self):
        p1 = face_on_rh()
        p1["left_hip"], p1["right_hip"] = (101, 200), (99, 200)
        m = run(face_on_rh(), p1=p1)
        assert m["hips_dx"].value is None
        assert m["spine_inclination"].value is None
        assert m["hips_dy"].value is not None

    def test_dtl_orientation_does_not_need_the_occluded_lead_arm(self):
        p1 = dtl_rh()
        del p1["left_wrist"], p1["left_elbow"]
        ref, why = jm.body_reference(p1, "dtl", "right")
        assert ref.horizontal_sign == 1.0

    def test_dtl_without_forward_bend_has_no_orientation(self):
        p1 = dtl_rh()
        p1["left_shoulder"], p1["right_shoulder"] = (80, 100), (78, 102)
        m = run(dtl_rh(), view="dtl", p1=p1)
        assert m["hips_dx"].value is None and "forward bend" in m["hips_dx"].reason
        assert m["hips_dy"].value is not None

    def test_landmark_missing_at_p1_only(self):
        p1 = face_on_rh()
        del p1["left_ear"]
        m = run(face_on_rh(), p1=p1)
        assert m["head_dx"].value is None
        assert m["head_dx"].reason == "not detected at P1"


class TestCompare:
    def test_difference_is_user_minus_reference(self):
        user = jm.measure(face_on_rh(), "face_on", "right", None)
        ref_pts = face_on_rh()
        ref_pts["right_wrist"] = (70, 220)          # trail arm straight
        ref = jm.measure(ref_pts, "face_on", "right", None)
        rows = {r.key: r for r in jm.compare(user, ref)}
        assert rows["trail_elbow_flexion"].difference == pytest.approx(90)
        assert rows["spine_inclination"].difference is None

    def test_unavailable_on_either_side_gives_no_difference(self):
        user_pts = face_on_rh()
        del user_pts["left_knee"]
        user = jm.measure(user_pts, "face_on", "right", None)
        ref = jm.measure(face_on_rh(), "face_on", "right", None)
        row = {r.key: r for r in jm.compare(user, ref)}["lead_knee_flexion"]
        assert row.user is None and row.reference == pytest.approx(0)
        assert row.difference is None
        assert "yours" in row.note

    def test_refuses_cross_view_comparison(self):
        user = jm.measure(face_on_rh(), "face_on", "right", None)
        ref = jm.measure(dtl_rh(), "dtl", "right", None)
        with pytest.raises(ValueError):
            jm.compare(user, ref)
