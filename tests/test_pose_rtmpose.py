"""RTMPose backend: mapping, coordinate transforms, confidence, CPU-only behaviour.

Everything here runs without onnxruntime or a GPU. Model inference is replaced
by fake sessions so the full detect() path -- letterbox, golfer box, crop,
SimCC decode, inverse transform -- is checked numerically.
"""

import importlib.util

import cv2
import numpy as np
import pytest

from _dual_view_fixtures import FakeBackend, write_video
from utils import pose_data as pd
from utils import pose_rtmpose as rt
from utils.landmarks import LANDMARK_INDEX, LANDMARK_NAMES

HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


class TestHalpe26Mapping:
    def test_every_swingai_landmark_maps_to_a_real_keypoint_or_none(self):
        assert set(rt.HALPE26_FOR) == set(LANDMARK_NAMES)
        for name, src in rt.HALPE26_FOR.items():
            assert src is None or src in rt.HALPE26, name

    def test_body_joints_map_one_to_one(self):
        for name in ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                     "left_hip", "right_hip", "left_knee", "right_knee",
                     "left_ankle", "right_ankle"):
            assert rt.HALPE26_FOR[name] == name

    def test_no_side_swaps(self):
        for name, src in rt.HALPE26_FOR.items():
            if src and name.startswith(("left_", "right_")):
                assert src.split("_")[0] == name.split("_")[0], name

    def test_hand_points_are_unsupported(self):
        for side in ("left", "right"):
            for part in ("pinky", "index", "thumb"):
                assert rt.HALPE26_FOR[f"{side}_{part}"] is None

    def test_foot_index_is_big_toe(self):
        assert rt.HALPE26_FOR["left_foot_index"] == "left_big_toe"

    def test_observation_marks_unsupported_and_missing_as_nan(self):
        kp = np.arange(52, dtype=float).reshape(26, 2) * 10
        kp[rt.HALPE26.index("left_knee")] = np.nan
        scores = np.linspace(0.1, 0.9, 26)
        obs = rt.halpe26_observation(kp, scores, 1000, 500)
        j = LANDMARK_INDEX["left_wrist"]
        k = rt.HALPE26.index("left_wrist")
        assert obs.xy_norm[j] == pytest.approx(kp[k] / (1000, 500))
        assert obs.confidence[j] == pytest.approx(scores[k])
        for name in ("left_thumb", "right_index", "left_knee"):
            assert np.isnan(obs.xy_norm[LANDMARK_INDEX[name]]).all()
            assert np.isnan(obs.confidence[LANDMARK_INDEX[name]])
        assert np.isnan(obs.z).all()      # RTMPose 2D has no depth


class TestGeometry:
    def test_fix_aspect_grows_the_short_side(self):
        assert rt.fix_aspect(np.array([300.0, 100.0]), 0.75) == pytest.approx([300, 400])
        assert rt.fix_aspect(np.array([30.0, 400.0]), 0.75) == pytest.approx([300, 400])

    def test_bbox_center_scale_pads(self):
        c, s = rt.bbox_center_scale([100, 200, 300, 600])
        assert c == pytest.approx([200, 400])
        assert s == pytest.approx([250, 500])

    @pytest.mark.parametrize("size", [(1440, 1080), (480, 360), (1080, 1920), (1280, 720)])
    def test_crop_and_inverse_round_trip_through_cv2(self, size):
        w, h = size
        box = [w * 0.25, h * 0.2, w * 0.6, h * 0.95]
        out_wh = rt.CONFIG["pose_input_wh"]
        center, scale = rt.bbox_center_scale(box)
        scale = rt.fix_aspect(scale, out_wh[0] / out_wh[1])
        # A dot a few crop pixels across, wherever the crop scale puts it.
        frame = np.zeros((h, w, 3), np.uint8)
        target = np.array([w * 0.43, h * 0.61])
        radius = int(np.ceil(3 * scale[0] / out_wh[0]))
        cv2.circle(frame, tuple(int(v) for v in target), radius, (255, 255, 255), -1)
        crop = cv2.warpAffine(frame, rt.crop_matrix(center, scale, out_wh), out_wh,
                              flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(crop[..., 0] > 60)
        found = np.array([[xs.mean(), ys.mean()]])
        back = rt.crop_to_frame(found, center, scale, out_wh)[0]
        px_per_crop_px = scale[0] / out_wh[0]
        assert np.abs(back - target).max() <= max(1.5, px_per_crop_px)

    def test_inverse_is_exact_inverse_of_matrix(self):
        center, scale, out_wh = np.array([700.0, 500.0]), np.array([600.0, 800.0]), (192, 256)
        m = rt.crop_matrix(center, scale, out_wh)
        pts = np.array([[650.0, 420.0], [400.0, 100.0], [700.0, 500.0]])
        crop_pts = pts @ m[:, :2].T + m[:, 2]
        assert rt.crop_to_frame(crop_pts, center, scale, out_wh) == pytest.approx(pts)
        assert crop_pts[2] == pytest.approx([96, 128])     # box centre -> crop centre

    def test_letterbox(self):
        img = np.full((1080, 1440, 3), 7, np.uint8)
        out, ratio = rt.letterbox(img)
        assert out.shape == (640, 640, 3)
        assert ratio == pytest.approx(640 / 1440)
        assert out[0, 0, 0] == 7 and out[-1, -1, 0] == 114.0   # padded bottom-right

    def test_decode_simcc(self):
        sx, sy = np.zeros((26, 384)), np.zeros((26, 512))
        sx[:, 100], sy[:, 300] = 0.8, 0.6
        sx[3, :] = 0.0                                   # no peak at all
        locs, scores = rt.decode_simcc(sx, sy)
        assert locs[0] == pytest.approx([50.0, 150.0])   # bins / split ratio 2
        assert scores[0] == pytest.approx(0.6)           # min of the two peaks
        assert np.isnan(locs[3]).all() and scores[3] == 0.0


class TestGolferSelection:
    def test_threshold_and_largest(self):
        boxes = [[0, 0, 10, 10, 0.9], [0, 0, 100, 200, 0.8], [0, 0, 500, 500, 0.2]]
        assert rt.select_golfer(boxes)[2] == 100

    def test_none_when_nobody_confident(self):
        assert rt.select_golfer([[0, 0, 10, 10, 0.1]]) is None
        assert rt.select_golfer(np.zeros((0, 5))) is None

    def test_follows_previous_golfer_over_a_bigger_bystander(self):
        prev = np.array([100, 100, 200, 300, 0.9])
        boxes = [[105, 102, 205, 305, 0.9], [300, 0, 700, 600, 0.95]]
        assert rt.select_golfer(boxes, prev)[0] == 105


class FakeSession:
    def __init__(self, fn):
        self.fn = fn

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def run(self, _outputs, feeds):
        return self.fn(feeds["input"])


def fake_backend(frame_wh, box_px, peak_crop_xy, score=0.9):
    """Detector reports `box_px` (frame pixels); pose peaks at `peak_crop_xy`."""
    w, h = frame_wh
    ratio = min(640 / w, 640 / h)

    def det(inp):
        assert inp.shape == (1, 3, 640, 640)
        b = np.array(box_px, float) * ratio
        return [np.array([[[*b, 0.95]]]), np.array([[0]])]

    def pose(inp):
        assert inp.shape == (1, 3, 256, 192)
        sx, sy = np.zeros((1, 26, 384)), np.zeros((1, 26, 512))
        sx[0, :, int(peak_crop_xy[0] * 2)] = score
        sy[0, :, int(peak_crop_xy[1] * 2)] = score
        return [sx, sy]

    b = rt.RTMPoseBackend()
    b._det, b._pose = FakeSession(det), FakeSession(pose)
    return b


class TestDetectPipeline:
    @pytest.mark.parametrize("size", [(1440, 1080), (480, 360), (720, 1280)])
    def test_crop_centre_maps_back_to_box_centre(self, size):
        w, h = size
        box = [w * 0.3, h * 0.1, w * 0.55, h * 0.9]
        obs = fake_backend(size, box, (96, 128)).detect(np.zeros((h, w, 3), np.uint8), 0)
        nose = obs.xy_norm[LANDMARK_INDEX["nose"]] * (w, h)
        assert nose == pytest.approx([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], abs=0.5)

    def test_off_centre_peak(self):
        w, h = 1440, 1080
        box = [500, 100, 800, 1000]
        obs = fake_backend((w, h), box, (48, 64)).detect(np.zeros((h, w, 3), np.uint8), 0)
        center, scale = rt.bbox_center_scale(box)
        scale = rt.fix_aspect(scale, 192 / 256)
        expect = center + (np.array([48, 64]) / [192, 256] - 0.5) * scale
        assert obs.xy_norm[LANDMARK_INDEX["nose"]] * (w, h) == pytest.approx(expect, abs=0.5)

    def test_no_person_means_no_observation(self):
        b = fake_backend((640, 480), [0, 0, 10, 10], (96, 128))
        b._det = FakeSession(lambda inp: [np.zeros((1, 0, 5)), np.zeros((1, 0))])
        assert b.detect(np.zeros((480, 640, 3), np.uint8), 0) is None

    def test_non_person_classes_ignored(self):
        b = fake_backend((640, 480), [0, 0, 300, 400], (96, 128))
        b._det = FakeSession(lambda inp: [np.array([[[0, 0, 300, 400, 0.99]]]), np.array([[5]])])
        assert b.detect(np.zeros((480, 640, 3), np.uint8), 0) is None


class TestConfidence:
    def test_backend_specific_defaults(self):
        assert rt.RTMPoseBackend.min_confidence == rt.RTMPOSE_MIN_CONFIDENCE == 0.3
        assert pd.MediaPipeBackend.min_confidence == 0.5

    def test_sequence_uses_its_backends_threshold(self, tmp_path):
        video = write_video(tmp_path / "v.avi", 4, 30)

        class Scored(FakeBackend):
            min_confidence = 0.3

            def detect(self, rgb, ts):
                obs = super().detect(rgb, ts)
                obs.confidence[:] = 0.4
                return obs

        rtm_like = pd.extract_pose_sequence(video, Scored(n_frames=4))
        mp_like = pd.extract_pose_sequence(video, FakeBackend(n_frames=4))
        mp_like.confidence[:] = 0.4
        assert rtm_like.default_min_confidence == 0.3
        assert rtm_like.point(1, "nose") is not None        # 0.4 >= 0.3
        assert mp_like.point(1, "nose") is None              # 0.4 <  0.5
        assert rtm_like.point(1, "nose", min_confidence=0.5) is None

    def test_threshold_survives_the_cache(self, tmp_path):
        video = write_video(tmp_path / "v.avi", 3, 30)
        backend = FakeBackend(n_frames=3)
        backend.min_confidence = 0.3
        seq = pd.get_pose_sequence(video, backend, cache_dir=str(tmp_path / "c"))
        again = pd.get_pose_sequence(video, backend, cache_dir=str(tmp_path / "c"))
        assert seq.default_min_confidence == again.default_min_confidence == 0.3


class TestBackendSelectionAndCache:
    def test_make_backend(self):
        assert isinstance(pd.make_backend("mediapipe"), pd.MediaPipeBackend)
        assert isinstance(pd.make_backend("rtmpose"), rt.RTMPoseBackend)
        with pytest.raises(ValueError):
            pd.make_backend("openpose")

    def test_backends_never_share_a_cache_entry(self, tmp_path):
        video = write_video(tmp_path / "v.avi", 5, 30)
        cache = str(tmp_path / "c")
        a = FakeBackend(n_frames=5, identity="mediapipe-x")
        b = FakeBackend(n_frames=5, identity="rtmpose-y", missing={2})
        sa = pd.get_pose_sequence(video, a, cache_dir=cache)
        sb = pd.get_pose_sequence(video, b, cache_dir=cache)
        assert b.calls == 5                        # b ran, did not reuse a's result
        assert sa.backend == "mediapipe-x" and sb.backend == "rtmpose-y"
        assert sa.detected[2] and not sb.detected[2]
        pd.get_pose_sequence(video, a, cache_dir=cache)
        pd.get_pose_sequence(video, b, cache_dir=cache)
        assert (a.calls, b.calls) == (5, 5)        # switching back: cache only

    def test_any_config_change_changes_the_key(self, monkeypatch, tmp_path):
        if not HAS_ORT:
            pytest.skip("identity needs onnxruntime")
        b = rt.RTMPoseBackend()
        before = b.identity()
        monkeypatch.setitem(rt.CONFIG, "bbox_padding", 1.3)
        assert b.identity() != before


@pytest.mark.skipif(HAS_ORT, reason="checks the CPU-only image, which has no onnxruntime")
class TestCpuOnlyImage:
    def test_rtmpose_reports_unavailable_instead_of_crashing(self):
        with pytest.raises(rt.RTMPoseUnavailable, match="onnxruntime-gpu is not installed"):
            rt.RTMPoseBackend().identity()
        with pytest.raises(rt.RTMPoseUnavailable):
            rt.RTMPoseBackend().open()

    def test_mediapipe_still_works_without_gpu_packages(self):
        assert pd.MediaPipeBackend().identity().startswith("mediapipe-")


def test_identity_includes_the_runtime_environment(monkeypatch):
    before = pd.MediaPipeBackend().identity()
    assert before.endswith("-env" + pd.runtime_fingerprint())
    monkeypatch.setattr(pd, "runtime_fingerprint", lambda: "deadbeef")
    assert pd.MediaPipeBackend().identity() != before


def test_missing_model_files_are_reported(tmp_path):
    b = rt.RTMPoseBackend(model_dir=str(tmp_path))
    with pytest.raises(rt.RTMPoseUnavailable, match="fetch_rtmpose_models"):
        b._paths()
