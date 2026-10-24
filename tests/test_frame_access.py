"""Exact frame access: frame f is the f-th sequentially decoded frame, always.

Each ffmpeg fixture is first checked to really break OpenCV's seek, so these
tests can't pass by accident on a video where seeking happens to work.
"""

import os
import random

import cv2
import pytest

from _video_fixtures import (ffprobe_timestamps_ms, offset_video, repeated_frames_video,
                             seek_trap_video, vfr_video)
from utils import frame_access as fa
from utils.joint_labels import decode_frames, frame_digest


def _naive_seek_digest(path, f):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    return frame_digest(img) if ok else None


def _seek_misses(path, index, step=7):
    return [f for f in range(0, index.frame_count, step)
            if _naive_seek_digest(path, f) != index.digests[f]]


@pytest.fixture(autouse=True)
def _fresh_memo(tmp_path, monkeypatch):
    monkeypatch.setenv("SWINGAI_OUTPUT_DIR", str(tmp_path / "output"))
    fa._memo.clear()
    fa._readers.clear()
    yield
    fa._memo.clear()
    fa._readers.clear()


@pytest.mark.parametrize("make", [seek_trap_video, offset_video, vfr_video],
                         ids=["high-fps-seek-trap", "nonzero-start-offset", "variable-frame-rate"])
def test_reader_returns_the_sequentially_decoded_frame_where_seeking_does_not(tmp_path, make):
    path = make(tmp_path)
    index = fa.build_frame_index(path)
    assert _seek_misses(path, index), "fixture no longer reproduces the OpenCV seek bug"

    expected = decode_frames(path, range(index.frame_count))
    rng = random.Random(0)
    order = list(range(index.frame_count))
    rng.shuffle(order)
    with fa.FrameReader(index) as reader:
        for f in order[:120] + [index.frame_count - 1, 0]:
            frame = reader.read(f)
            assert frame.index == f
            assert frame_digest(frame.image) == frame_digest(expected[f])
            assert frame.timestamp_ms == index.timestamps_ms[f]


def test_stepping_forward_and_backward_one_frame_at_a_time(tmp_path):
    path = seek_trap_video(tmp_path)
    index = fa.build_frame_index(path)
    start = next(f for f in _seek_misses(path, index))     # a frame seeking gets wrong
    with fa.FrameReader(index) as reader:
        for f in list(range(start, start + 20)) + list(range(start + 19, start - 20, -1)):
            assert reader.read(f).digest == index.digests[f]


def test_timestamps_are_presentation_times_relative_to_frame_zero(tmp_path):
    path = offset_video(tmp_path, offset_s=0.25)
    index = fa.build_frame_index(path)
    probed = ffprobe_timestamps_ms(path)
    assert probed[0] == pytest.approx(250.0, abs=0.5)
    assert index.timestamps_ms[0] == 0.0
    assert index.start_offset_ms == pytest.approx(250.0, abs=0.5)
    assert index.presentation_ms(10) == pytest.approx(probed[10], abs=0.5)
    assert index.ffprobe_max_deviation_ms < 0.5


def test_variable_frame_rate_is_detected_and_timestamps_follow_the_container(tmp_path):
    path = vfr_video(tmp_path)
    index = fa.build_frame_index(path)
    assert index.is_variable_frame_rate
    probed = ffprobe_timestamps_ms(path)
    assert len(probed) == index.frame_count
    for f in (0, 30, 59, 60, 61, 120, index.frame_count - 1):
        assert index.timestamps_ms[f] == pytest.approx(probed[f] - probed[0], abs=0.5)
    # 120 fps then 240 fps: no single nominal rate describes it.
    d = [b - a for a, b in zip(index.timestamps_ms, index.timestamps_ms[1:])]
    assert d[10] == pytest.approx(1000 / 120, abs=0.1)
    assert d[-10] == pytest.approx(1000 / 240, abs=0.1)
    assert index.nearest_frame(index.timestamps_ms[61] + 0.1) == 61


def test_constant_rate_video_is_not_variable(tmp_path):
    index = fa.build_frame_index(seek_trap_video(tmp_path))
    assert not index.is_variable_frame_rate
    assert index.average_fps == pytest.approx(240.0, rel=0.01)


class _EarlySeekCapture:
    """Wraps a VideoCapture whose frame seek lands `early` frames early."""

    def __init__(self, cap, early):
        self._cap, self._early = cap, early

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            value = max(0, value - self._early)
        return self._cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _reader_with_early_seek(index, early):
    reader = fa.FrameReader(index)
    original = reader._open

    def _open():
        original()
        reader._cap = _EarlySeekCapture(reader._cap, early)
    reader._open = _open
    return reader


def test_repeated_images_are_disambiguated_not_misidentified(tmp_path):
    path = repeated_frames_video(tmp_path, n=120, repeat=3)
    index = fa.build_frame_index(path)
    assert len(set(index.digests)) == 40      # every image appears three times
    reader = _reader_with_early_seek(index, early=2)
    rng = random.Random(1)
    for f in rng.sample(range(index.frame_count), 60):
        frame = reader.read(f)
        assert frame.index == f
        # The image alone can't prove which of three copies it is; the position can.
        assert frame.digest == index.digests[f]
    reader.close()


def test_a_wrong_image_is_refused_never_returned(tmp_path):
    a = seek_trap_video(tmp_path, n=300, name="a.mp4")
    index = fa.build_frame_index(a)
    b = seek_trap_video(tmp_path, n=300, name="b.mp4", seed=5)
    other = fa.build_frame_index(b)
    assert other.frame_count == index.frame_count and other.digests != index.digests
    reader = fa.FrameReader(index, path=b)
    with pytest.raises(fa.FrameAccessError, match="not the indexed one"):
        reader.read(10)


def test_out_of_range_missing_and_truncated_files_raise(tmp_path):
    path = seek_trap_video(tmp_path, n=200)
    index = fa.build_frame_index(path)
    with fa.FrameReader(index) as reader:
        for bad in (-1, index.frame_count, 1.5, "3"):
            with pytest.raises(fa.FrameAccessError):
                reader.read(bad)
    with pytest.raises(fa.FrameAccessError, match="no video file"):
        fa.get_frame_index(str(tmp_path / "gone.mp4"))
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(fa.FrameAccessError):
        fa.build_frame_index(str(junk))


def test_index_is_cached_by_content_and_survives_a_rename(tmp_path):
    path = seek_trap_video(tmp_path, n=200)
    cache = tmp_path / "cache"
    first = fa.get_frame_index(path, cache_dir=str(cache))
    assert len(os.listdir(cache)) == 1
    moved = tmp_path / "renamed.mp4"
    os.rename(path, moved)
    fa._memo.clear()
    second = fa.get_frame_index(str(moved), cache_dir=str(cache))
    assert second.digests == first.digests and second.path == str(moved)
    assert len(os.listdir(cache)) == 1


def test_shared_read_frame_matches_sequential_decode(tmp_path):
    path = seek_trap_video(tmp_path)
    index = fa.get_frame_index(path)
    want = [f for f in _seek_misses(path, index)][:5]
    decoded = decode_frames(path, want)
    for f in want:
        assert frame_digest(fa.read_frame(path, f).image) == frame_digest(decoded[f])


# --- timestamp reliability ----------------------------------------------------

def test_container_times_checked_by_ffprobe_are_verified(tmp_path):
    index = fa.build_frame_index(offset_video(tmp_path))
    assert index.timestamp_status == "verified" and index.timestamps_verified
    assert index.estimated_frames == () and index.ffprobe_max_deviation_ms < 0.5
    assert index.summary()["timestamps"] == {"status": "verified", "estimated_frames": 0,
                                             "ffprobe_max_deviation_ms":
                                                 index.ffprobe_max_deviation_ms,
                                             "detail": ""}


@pytest.mark.parametrize("probed,status,detail", [
    (None, "unverified", "ffprobe unavailable"),
    ("short", "unverified", "can't be matched frame by frame"),
    ("na", "unverified", "no timestamp for 2 frame(s)"),
    ("shifted", "inconsistent", "differ by up to"),
], ids=["no-ffprobe", "count-mismatch", "ffprobe-NA", "disagreeing"])
def test_unchecked_or_disagreeing_times_are_not_verified(tmp_path, monkeypatch, probed,
                                                         status, detail):
    path = offset_video(tmp_path)
    real = ffprobe_timestamps_ms(path)
    fake = {None: None, "short": real[:-3], "na": real[:5] + [None, None] + real[7:],
            "shifted": real[:50] + [t + 2.0 for t in real[50:]]}[probed]
    monkeypatch.setattr(fa, "_ffprobe_timestamps_ms", lambda p: fake)
    index = fa.build_frame_index(path)
    assert index.timestamp_status == status and not index.timestamps_verified
    assert detail in index.timestamp_detail
    assert index.estimated_frames == ()


def test_missing_or_non_increasing_decoder_times_are_estimated_and_listed(tmp_path,
                                                                        monkeypatch):
    path = seek_trap_video(tmp_path, n=120)
    real_pos = fa._pos_msec
    calls = {"n": 0}

    def flaky(cap):
        i = calls["n"]
        calls["n"] += 1
        t = real_pos(cap)
        if i in (5, 6):
            return 0.0                  # decoder gave no time
        if i == 40:
            return 1.0                  # goes backwards
        return t
    monkeypatch.setattr(fa, "_pos_msec", flaky)
    index = fa.build_frame_index(path)
    assert index.estimated_frames == (5, 6, 40)
    assert index.timestamp_status == "estimated"
    assert "3 frame time(s) estimated" in index.timestamp_detail
    assert index.timestamp_is_estimated(5) and not index.timestamp_is_estimated(7)
    # Estimates still keep time monotonic, as pose_data does.
    assert all(b > a for a, b in zip(index.timestamps_ms, index.timestamps_ms[1:]))
    # The flags survive the disk cache.
    restored = fa.FrameIndex.from_json(path, index.to_json())
    assert restored.estimated_frames == (5, 6, 40)
    assert restored.timestamp_status == "estimated"


def test_index_times_match_what_pose_extraction_stores():
    from utils.pose_data import _frame_timestamp

    class Cap:
        def __init__(self, t):
            self.t = t

        def get(self, prop):
            return self.t
    previous = [0.0, 4.0]
    for raw, idx in ((8.0, 2), (0.0, 2), (3.0, 2), (0.0, 0), (5.0, 0)):
        t, _ = fa._container_time(raw, idx, 250.0, previous if idx else [])
        assert t == _frame_timestamp(Cap(raw), idx, 250.0, previous if idx else [])
