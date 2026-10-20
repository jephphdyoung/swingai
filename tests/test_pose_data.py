"""Tests for the named-landmark pose layer and its cache."""

import os

import numpy as np
import pytest

from _dual_view_fixtures import FakeBackend, write_video
from utils import pose_data as pd
from utils.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, MEDIAPIPE_INDEX
from utils.pose_extractor import SELECTED_LANDMARK_INDICES


class TestLandmarkNames:
    def test_names_are_unique_and_indexed_in_order(self):
        assert len(set(LANDMARK_NAMES)) == len(LANDMARK_NAMES)
        assert [LANDMARK_INDEX[n] for n in LANDMARK_NAMES] == list(range(len(LANDMARK_NAMES)))

    def test_covers_the_legacy_selected_landmarks(self):
        assert set(SELECTED_LANDMARK_INDICES) <= set(MEDIAPIPE_INDEX.values())

    def test_mediapipe_mapping_reads_named_landmarks(self):
        class LM:
            def __init__(self, i):
                self.x, self.y, self.z, self.visibility = i / 100, i / 50, 0.0, 0.8
        obs = pd.mediapipe_observation([LM(i) for i in range(33)])
        j = LANDMARK_INDEX["left_wrist"]
        assert obs.xy_norm[j] == pytest.approx((0.15, 0.30))
        assert obs.confidence[j] == pytest.approx(0.8)


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "clip.avi", 12, 30, size=(320, 240))


class TestExtraction:
    def test_missing_detection_is_not_carried_forward(self, video):
        seq = pd.extract_pose_sequence(video, FakeBackend(n_frames=12, missing={4, 5}))
        assert seq.frame_count == 12
        assert not seq.detected[4] and not seq.detected[5]
        assert np.isnan(seq.xy[4]).all()
        assert seq.point(4, "left_wrist") is None
        assert seq.points(5) == {}
        assert seq.point(6, "left_wrist") is not None

    def test_coordinates_are_pixels_for_non_square_frames(self, video):
        seq = pd.extract_pose_sequence(video, FakeBackend(n_frames=12))
        assert (seq.width, seq.height) == (320, 240)
        assert seq.point(0, "nose") == pytest.approx((0.50 * 320, 0.20 * 240))

    def test_timestamps_are_monotonic_and_match_fps(self, video):
        seq = pd.extract_pose_sequence(video, FakeBackend(n_frames=12))
        assert np.all(np.diff(seq.timestamps_ms) > 0)
        assert seq.timestamps_ms[3] == pytest.approx(100.0, abs=1.0)

    def test_low_confidence_landmark_is_not_valid(self, video):
        seq = pd.extract_pose_sequence(video, FakeBackend(n_frames=12))
        seq.confidence[2, LANDMARK_INDEX["nose"]] = 0.1
        assert seq.point(2, "nose") is None
        assert seq.point(2, "nose", min_confidence=0.05) is not None

    def test_nearest_frame_by_timestamp(self):
        assert pd.nearest_frame([0, 10, 20, 30], 14) == 1
        assert pd.nearest_frame([0, 10, 20, 30], 16) == 2
        assert pd.nearest_frame([0, 10, 20, 30], 999) == 3


class TestCache:
    def test_round_trip(self, video, tmp_path):
        seq = pd.extract_pose_sequence(video, FakeBackend(n_frames=12, missing={3}))
        path = tmp_path / "c" / "x.npz"
        pd.save_pose_sequence(seq, str(path))
        back = pd.load_pose_sequence(str(path), video)
        np.testing.assert_array_equal(back.detected, seq.detected)
        np.testing.assert_allclose(back.xy, seq.xy, equal_nan=True)
        assert back.backend == "fake-backend-1"

    def test_second_call_does_not_rerun_inference(self, video, tmp_path):
        backend = FakeBackend(n_frames=12)
        pd.get_pose_sequence(video, backend, cache_dir=str(tmp_path))
        assert backend.calls == 12
        pd.get_pose_sequence(video, backend, cache_dir=str(tmp_path))
        assert backend.calls == 12

    def test_cache_miss_without_extraction_returns_none(self, video, tmp_path):
        assert pd.get_pose_sequence(video, FakeBackend(), cache_dir=str(tmp_path),
                                    extract_if_missing=False) is None

    def test_key_changes_with_backend_and_with_the_file(self, video):
        k1 = pd.cache_key(video, "a")
        assert pd.cache_key(video, "b") != k1
        st = os.stat(video)
        os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert pd.cache_key(video, "a") != k1

    def test_key_ignores_the_directory(self, tmp_path):
        a = write_video(tmp_path / "a.avi", 3, 30)
        os.makedirs(tmp_path / "elsewhere")
        b = tmp_path / "elsewhere" / "a.avi"
        os.link(a, b)
        assert pd.cache_key(a, "x") == pd.cache_key(str(b), "x")
