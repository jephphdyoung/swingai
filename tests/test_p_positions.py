"""Tests for P-position signal extraction and crossing detection."""

import math
import pytest

from utils import p_positions as pp


def make_frame(lead_wrist_dy=0.2, scale=0.3):
    """A minimal landmark frame.

    Shoulders at y=0.4, hips at y=0.4+scale so body scale is `scale`.
    The left (lead, for a right-hander) wrist sits `lead_wrist_dy` below the
    shoulder, in raw normalized units.
    """
    frame = [(0.5, 0.3, 0.0)] * 21
    frame[pp.L_SHOULDER] = (0.45, 0.40, 0.0)
    frame[pp.R_SHOULDER] = (0.55, 0.40, 0.0)
    frame[pp.L_HIP] = (0.47, 0.40 + scale, 0.0)
    frame[pp.R_HIP] = (0.53, 0.40 + scale, 0.0)
    frame[pp.L_WRIST] = (0.50, 0.40 + lead_wrist_dy, 0.0)
    frame[pp.R_WRIST] = (0.50, 0.40 + lead_wrist_dy, 0.0)
    return frame


class TestValidFrameMask:
    def test_flags_carried_forward_frames(self):
        a, b = make_frame(0.2), make_frame(0.1)
        # pose_extractor repeats the previous frame's list object on failure.
        landmarks = [a, b, b, b, a]
        valid = pp._valid_frame_mask(landmarks)
        assert valid == [True, True, False, False, True]

    def test_first_frame_always_valid(self):
        assert pp._valid_frame_mask([make_frame()]) == [True]

    def test_all_distinct_frames_are_valid(self):
        landmarks = [make_frame(0.2), make_frame(0.1), make_frame(0.0)]
        assert pp._valid_frame_mask(landmarks) == [True, True, True]


class TestBodyScale:
    def test_measures_shoulder_to_hip_distance(self):
        assert pp._body_scale(make_frame(scale=0.3)) == pytest.approx(0.3, abs=1e-6)

    def test_never_returns_zero(self):
        degenerate = [(0.5, 0.5, 0.0)] * 21
        assert pp._body_scale(degenerate) > 0


class TestLeadArmElevation:
    def test_zero_when_arm_is_horizontal(self):
        # Wrist level with the shoulder means the arm is parallel to the ground.
        frames = [make_frame(lead_wrist_dy=0.0)]
        assert pp._lead_arm_elevation(frames, "right")[0] == pytest.approx(0.0)

    def test_positive_when_wrist_below_shoulder(self):
        frames = [make_frame(lead_wrist_dy=0.15)]
        assert pp._lead_arm_elevation(frames, "right")[0] > 0

    def test_negative_when_wrist_above_shoulder(self):
        frames = [make_frame(lead_wrist_dy=-0.15)]
        assert pp._lead_arm_elevation(frames, "right")[0] < 0

    def test_normalized_by_body_scale(self):
        # Same pose at two camera distances must give the same elevation.
        near = pp._lead_arm_elevation([make_frame(0.30, scale=0.6)], "right")[0]
        far = pp._lead_arm_elevation([make_frame(0.15, scale=0.3)], "right")[0]
        assert near == pytest.approx(far, abs=1e-9)

    def test_handedness_selects_the_other_side(self):
        frame = make_frame()
        frame[pp.R_WRIST] = (0.50, 0.20, 0.0)   # trail wrist high
        right = pp._lead_arm_elevation([frame], "right")[0]
        left = pp._lead_arm_elevation([frame], "left")[0]
        assert right != left


class TestHandSpeed:
    def test_zero_for_a_still_golfer(self):
        landmarks = [make_frame(0.2), make_frame(0.2), make_frame(0.2)]
        valid = [True, True, True]
        speed = pp._hand_speed(landmarks, [0.0, 10.0, 20.0], valid)
        assert all(s == pytest.approx(0.0) for s in speed)

    def test_scales_with_displacement(self):
        slow = [make_frame(0.20), make_frame(0.21)]
        fast = [make_frame(0.20), make_frame(0.30)]
        ts, valid = [0.0, 10.0], [True, True]
        assert (pp._hand_speed(fast, ts, valid)[1]
                > pp._hand_speed(slow, ts, valid)[1])

    def test_uses_elapsed_time_not_frame_count(self):
        landmarks = [make_frame(0.20), make_frame(0.30)]
        quick = pp._hand_speed(landmarks, [0.0, 5.0], [True, True])[1]
        slow = pp._hand_speed(landmarks, [0.0, 50.0], [True, True])[1]
        assert quick == pytest.approx(slow * 10, rel=1e-6)

    def test_carried_forward_frames_do_not_read_as_zero_speed(self):
        # A repeated frame is missing data, not a stationary golfer. Reporting
        # zero there would look like address or finish.
        landmarks = [make_frame(0.20), make_frame(0.30), make_frame(0.30)]
        valid = [True, True, False]
        speed = pp._hand_speed(landmarks, [0.0, 10.0, 20.0], valid)
        assert speed[2] == pytest.approx(speed[1])
        assert speed[2] > 0


class TestCrossings:
    def test_interpolates_between_frames(self):
        # Crossing sits exactly halfway between the two samples.
        ts, frame, slope = pp._interpolate_crossing([1.0, -1.0], [0.0, 10.0], 1)
        assert ts == pytest.approx(5.0)
        assert slope == pytest.approx(200.0)

    def test_interpolation_is_proportional(self):
        # 0.75 of the way from +3 to -1.
        ts, _, _ = pp._interpolate_crossing([3.0, -1.0], [0.0, 100.0], 1)
        assert ts == pytest.approx(75.0)

    def test_nearest_frame_follows_the_fraction(self):
        _, near, _ = pp._interpolate_crossing([0.1, -0.9], [0.0, 10.0], 1)
        assert near == 0                     # crossing is closer to frame 0
        _, far, _ = pp._interpolate_crossing([0.9, -0.1], [0.0, 10.0], 1)
        assert far == 1

    def test_finds_descending_and_ascending_crossings(self):
        values = [1.0, -1.0, 1.0]
        ts = [0.0, 10.0, 20.0]
        found = pp._find_crossings(values, [True] * 3, ts, 0, 2)
        assert len(found) == 2

    def test_ignores_crossings_touching_invalid_frames(self):
        values = [1.0, -1.0]
        found = pp._find_crossings(values, [True, False], [0.0, 10.0], 0, 1)
        assert found == []

    def test_respects_the_search_window(self):
        values = [1.0, -1.0, 1.0, -1.0]
        ts = [0.0, 10.0, 20.0, 30.0]
        found = pp._find_crossings(values, [True] * 4, ts, 2, 3)
        assert len(found) == 1

    def test_no_crossing_when_sign_never_changes(self):
        found = pp._find_crossings([1.0, 2.0, 3.0], [True] * 3, [0.0, 1.0, 2.0], 0, 2)
        assert found == []


class TestPickCrossing:
    def test_rejects_crossings_that_are_too_shallow(self):
        shallow = [(10.0, 1, pp.MIN_CROSSING_SLOPE / 2)]
        assert pp._pick_crossing(shallow) is None

    def test_keeps_a_steep_crossing(self):
        steep = [(10.0, 1, pp.MIN_CROSSING_SLOPE * 4)]
        assert pp._pick_crossing(steep) is not None

    def test_first_and_last_selection(self):
        crossings = [
            (10.0, 1, 1.0),
            (20.0, 2, 1.0),
        ]
        assert pp._pick_crossing(crossings, "first")[0] == 10.0
        assert pp._pick_crossing(crossings, "last")[0] == 20.0

    def test_empty_input(self):
        assert pp._pick_crossing([]) is None


class TestConfidence:
    def test_bounded_to_unit_range(self):
        assert pp._confidence_from_slope(0.0) == 0.0
        assert pp._confidence_from_slope(1e6) == 1.0

    def test_increases_with_slope(self):
        assert pp._confidence_from_slope(0.5) < pp._confidence_from_slope(1.5)


class TestKinematicDetection:
    def _signals(self, n=60):
        landmarks = [make_frame(0.2) for _ in range(n)]
        ts = [i * 10.0 for i in range(n)]
        return pp.ViewSignals(
            path="x.mp4", view="face_on", fps=100.0, frame_count=n,
            timestamps_ms=ts, landmarks=landmarks, valid=[True] * n,
            lead_arm_elevation=[0.2] * n,
            hand_speed=[0.0] * n,
        )

    class Window:
        """Stand-in for swing_detector.SwingWindow.

        `impact_frame` is deliberately wrong here: the detector derives it from
        motion-burst boundaries, and detect_kinematic must locate impact from
        the hand-speed profile instead of trusting it.
        """
        start_frame, takeaway_frame, impact_frame, end_frame = 0, 5, 50, 55
        confidence = 1.0

    def _swing_profile(self, n=60, top=20, impact=40):
        """A hand-speed profile shaped like a real swing."""
        speed = []
        for i in range(n):
            if i < 5 or i > 55:
                speed.append(0.05)              # address / finish
            elif i < top:
                speed.append(4.0)               # backswing
            elif i == top:
                speed.append(0.1)               # transition
            elif i < impact:
                speed.append(8.0)               # downswing
            elif i == impact:
                speed.append(20.0)              # impact
            else:
                speed.append(6.0)               # follow-through
        return speed

    def test_returns_nothing_without_a_swing_window(self):
        signals = self._signals()
        signals.swing = None
        assert pp.detect_kinematic(signals) == {}

    def test_locates_impact_at_peak_speed_not_the_window_estimate(self):
        signals = self._signals()
        signals.swing = self.Window()
        signals.hand_speed = self._swing_profile()

        result = pp.detect_kinematic(signals)
        # The window claims frame 50; the speed profile peaks at 40.
        assert result["P7"].frame == 40

    def test_locates_the_top_at_the_hand_speed_minimum(self):
        signals = self._signals()
        signals.swing = self.Window()
        signals.hand_speed = self._swing_profile()

        result = pp.detect_kinematic(signals)
        assert result["P4"].frame == 20
        assert result["P1"].frame == 0
        assert result["P10"].frame == 55

    def test_top_precedes_impact(self):
        signals = self._signals()
        signals.swing = self.Window()
        signals.hand_speed = self._swing_profile()
        result = pp.detect_kinematic(signals)
        assert result["P4"].frame < result["P7"].frame

    def test_shallow_transition_lowers_top_confidence(self):
        signals = self._signals()
        signals.swing = self.Window()

        signals.hand_speed = self._swing_profile()
        signals.hand_speed[20] = 3.9        # barely a dip
        shallow = pp.detect_kinematic(signals)["P4"].confidence

        signals.hand_speed = self._swing_profile()
        signals.hand_speed[20] = 0.05       # clear reversal
        deep = pp.detect_kinematic(signals)["P4"].confidence
        assert deep > shallow

    def test_flat_speed_profile_yields_low_impact_confidence(self):
        # No pronounced peak means we should not claim to have found impact.
        signals = self._signals()
        signals.swing = self.Window()
        signals.hand_speed = [5.0] * 60
        assert pp.detect_kinematic(signals)["P7"].confidence < 0.1

    def test_pronounced_peak_yields_high_impact_confidence(self):
        signals = self._signals()
        signals.swing = self.Window()
        signals.hand_speed = self._swing_profile()
        assert pp.detect_kinematic(signals)["P7"].confidence > 0.5

    def test_address_confidence_reflects_stillness(self):
        signals = self._signals()
        signals.swing = self.Window()

        signals.hand_speed = self._swing_profile()
        still = pp.detect_kinematic(signals)["P1"].confidence

        signals.hand_speed = self._swing_profile()
        signals.hand_speed[0] = 15.0        # already moving at the window start
        moving = pp.detect_kinematic(signals)["P1"].confidence
        assert still > moving


class TestFramesWithin:
    def test_converts_milliseconds_to_frames(self):
        ts = [i * 10.0 for i in range(100)]      # 100fps
        assert pp._frames_within(ts, 40.0) == 4

    def test_scales_with_frame_rate(self):
        fast = [i * (1000 / 240) for i in range(240)]
        assert pp._frames_within(fast, 40.0) == pytest.approx(10, abs=1)

    def test_degenerate_series(self):
        assert pp._frames_within([], 40.0) == 0
        assert pp._frames_within([0.0], 40.0) == 0
