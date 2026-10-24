"""Exact source-frame access: frame `f` is the f-th frame a sequential decode
produces, and nothing else.

OpenCV's random seek can't be trusted for that. On the 240fps Swing Catalyst
captures (nominal 249.1 fps, average 249.3, first frame at 4.014 ms) a seek to
frame f lands on f-1 from about frame 700 on, and the container's frame count
says 1245 where 1244 frames decode. Pose extraction and the joint-label
benchmark decode sequentially, so a seeked image and a pose/label "frame f"
can disagree by one.

A `FrameIndex` is built once per file by decoding every frame in order. It
records each frame's presentation time (as OpenCV reports it during the
sequential decode -- the same clock `pose_data` stores, relative to the first
frame) and the SHA-1 of its pixels (`joint_labels.frame_digest`, the digest the
labelling benchmark already pins). `FrameReader` then serves any frame:

- forward steps decode sequentially from the current position;
- a jump seeks a few frames *before* the target, identifies where the seek
  actually landed by looking its digest up in the index, and decodes forward;
- when the landing frame can't be identified uniquely (a repeated image, a
  seek past the target) it decodes from the start, which is exact by
  construction.

The returned frame's digest is always checked against the index. A mismatch
raises `FrameAccessError`; an adjacent frame is never returned in its place.

Indexes are cached on disk keyed by the file's SHA-256 and the decoder
identity (OpenCV and runtime fingerprint), since a different decoder build may
produce different pixels.

Timestamp reliability is explicit. A frame's time is either the container's
presentation time as the decoder reports it, or -- when the decoder gives
nothing usable (0 after frame 0, or not increasing) -- an *estimate* from the
nominal frame rate, listed in `estimated_frames`. The file as a whole gets a
`timestamp_status`:

- ``verified``: every frame has a container time and ffprobe's per-frame
  presentation times agree with them within `TIMESTAMP_TOLERANCE_MS`;
- ``unverified``: every frame has a container time, but there was no
  independent check (no ffprobe, a frame-count mismatch, or ffprobe frames
  without a time);
- ``inconsistent``: ffprobe disagrees beyond the tolerance;
- ``estimated``: at least one frame's time is an estimate.

Only ``verified`` times are fit for motion timing. Even then they are
*playback* times: whether playback is the swing's real time is a separate,
recorded fact about the source (`utils/catalog.py`, `timing.motion_time`).
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

INDEX_SCHEMA = 2
# Read forward instead of seeking when the target is at most this far ahead.
FORWARD_WINDOW = 48
# Seek this many frames before the target: the seek may land early or late.
SEEK_BACKOFF = 4
# Frame intervals varying by more than this fraction mean variable frame rate.
VFR_TOLERANCE = 0.02
# Largest decoder-vs-ffprobe disagreement still counted as the same time.
TIMESTAMP_TOLERANCE_MS = 0.5
TIMESTAMP_STATUSES = ("verified", "unverified", "inconsistent", "estimated")


class FrameAccessError(RuntimeError):
    """The exact requested frame can't be produced (missing, unreadable or
    changed file, frame out of range, or a decoded image that isn't the
    indexed one)."""


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def decoder_identity():
    """Decoded pixels depend on the decoder build, so indexes are per decoder."""
    import cv2
    from utils.pose_data import runtime_fingerprint
    return f"opencv-{cv2.__version__}-{runtime_fingerprint()}"


@dataclass(frozen=True)
class FrameIndex:
    path: str
    file_sha256: str
    size_bytes: int
    frame_count: int                  # frames a sequential decode produces
    width: int
    height: int
    timestamps_ms: tuple              # per frame, relative to the first frame
    digests: tuple                    # per frame, joint_labels.frame_digest
    nominal_fps: float                # the container's declared rate (CAP_PROP_FPS)
    reported_frame_count: int         # the container's count; may disagree
    codec: str
    decoder: str
    # Presentation time of frame 0 in the container (ffprobe), so that
    # start_offset_ms + timestamps_ms[f] is the stream's own PTS. None when
    # ffprobe is unavailable or disagrees with the decode about frame count.
    start_offset_ms: Optional[float] = None
    # Largest |OpenCV - ffprobe| relative timestamp difference, as a check.
    ffprobe_max_deviation_ms: Optional[float] = None
    # Frames whose time is a nominal-rate estimate, not a container time.
    estimated_frames: tuple = ()
    timestamp_status: str = "unverified"
    timestamp_detail: str = ""
    _by_digest: dict = field(default=None, compare=False, repr=False)

    @property
    def duration_ms(self):
        if self.frame_count < 2:
            return 0.0
        return self.timestamps_ms[-1] - self.timestamps_ms[0]

    @property
    def average_fps(self):
        """Frames per second of presentation time; None for a single frame."""
        if self.frame_count < 2 or self.duration_ms <= 0:
            return None
        return (self.frame_count - 1) * 1000.0 / self.duration_ms

    @property
    def is_variable_frame_rate(self):
        if self.frame_count < 3:
            return False
        d = np.diff(np.asarray(self.timestamps_ms, dtype=float))
        mean = d.mean()
        return bool(mean > 0 and (d.max() - d.min()) / mean > VFR_TOLERANCE)

    def frames_with_digest(self, digest):
        by = self._by_digest
        if by is None:
            by = {}
            for i, d in enumerate(self.digests):
                by.setdefault(d, []).append(i)
            object.__setattr__(self, "_by_digest", by)
        return by.get(digest, [])

    @property
    def timestamps_verified(self):
        return self.timestamp_status == "verified"

    def timestamp_is_estimated(self, f):
        return f in self._estimated_set()

    def _estimated_set(self):
        cached = self.__dict__.get("_estimated_cache")
        if cached is None:
            cached = frozenset(self.estimated_frames)
            object.__setattr__(self, "_estimated_cache", cached)
        return cached

    def presentation_ms(self, f):
        """The container's own timestamp of frame f, when known."""
        if self.start_offset_ms is None:
            return None
        return self.start_offset_ms + self.timestamps_ms[f]

    def nearest_frame(self, timestamp_ms):
        ts = np.asarray(self.timestamps_ms, dtype=float)
        i = int(np.searchsorted(ts, timestamp_ms))
        if i <= 0:
            return 0
        if i >= len(ts):
            return len(ts) - 1
        return i if ts[i] - timestamp_ms < timestamp_ms - ts[i - 1] else i - 1

    def summary(self):
        """JSON-ready video facts, as recorded in the catalog."""
        return {"frame_count": self.frame_count, "width": self.width, "height": self.height,
                "codec": self.codec, "nominal_fps": round(self.nominal_fps, 6),
                "average_fps": None if self.average_fps is None else round(self.average_fps, 6),
                "variable_frame_rate": self.is_variable_frame_rate,
                "duration_ms": round(self.duration_ms, 3),
                "start_offset_ms": self.start_offset_ms,
                "reported_frame_count": self.reported_frame_count,
                "timestamps": {"status": self.timestamp_status,
                               "estimated_frames": len(self.estimated_frames),
                               "ffprobe_max_deviation_ms": self.ffprobe_max_deviation_ms,
                               "detail": self.timestamp_detail}}

    def to_json(self):
        return {"schema": INDEX_SCHEMA, "file_sha256": self.file_sha256,
                "size_bytes": self.size_bytes, "frame_count": self.frame_count,
                "width": self.width, "height": self.height,
                "timestamps_ms": list(self.timestamps_ms), "digests": list(self.digests),
                "nominal_fps": self.nominal_fps,
                "reported_frame_count": self.reported_frame_count, "codec": self.codec,
                "decoder": self.decoder, "start_offset_ms": self.start_offset_ms,
                "ffprobe_max_deviation_ms": self.ffprobe_max_deviation_ms,
                "estimated_frames": list(self.estimated_frames),
                "timestamp_status": self.timestamp_status,
                "timestamp_detail": self.timestamp_detail}

    @classmethod
    def from_json(cls, path, d):
        return cls(path=path, file_sha256=d["file_sha256"], size_bytes=d["size_bytes"],
                   frame_count=d["frame_count"], width=d["width"], height=d["height"],
                   timestamps_ms=tuple(d["timestamps_ms"]), digests=tuple(d["digests"]),
                   nominal_fps=d["nominal_fps"],
                   reported_frame_count=d["reported_frame_count"], codec=d["codec"],
                   decoder=d["decoder"], start_offset_ms=d.get("start_offset_ms"),
                   ffprobe_max_deviation_ms=d.get("ffprobe_max_deviation_ms"),
                   estimated_frames=tuple(d["estimated_frames"]),
                   timestamp_status=d["timestamp_status"],
                   timestamp_detail=d["timestamp_detail"])


def _fourcc(cap):
    import cv2
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
    return text if text.isprintable() and text else "unknown"


def _ffprobe_timestamps_ms(path):
    """Per-frame best-effort presentation times (None for a frame ffprobe gives
    no time), or None when ffprobe is unavailable or fails."""
    import shutil
    import subprocess
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=600, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    out_ms = []
    for line in out.splitlines():
        v = line.strip().rstrip(",")
        if not v:
            continue
        try:
            out_ms.append(float(v) * 1000.0)
        except ValueError:          # "N/A"
            out_ms.append(None)
    return out_ms


def _pos_msec(cap):
    import cv2
    return cap.get(cv2.CAP_PROP_POS_MSEC)


def _container_time(raw, idx, fps, previous):
    """(time, estimated) for the frame just decoded.

    The same values pose_data._frame_timestamp stores (so index and pose
    frames share one clock), plus whether the value is an estimate.
    """
    if idx == 0:
        return (raw if raw and raw > 0 else 0.0), False
    if not raw or raw <= previous[-1]:
        return max(idx * 1000.0 / fps, previous[-1] + 1e-3), True
    return raw, False


def timestamp_check(timestamps, estimated, probed):
    """(status, start_offset_ms, max_deviation_ms, detail) -- see the module doc."""
    n = len(timestamps)
    start = deviation = None
    detail = ""
    if probed is None:
        status, detail = "unverified", "ffprobe unavailable: no independent timestamp check"
    elif len(probed) != n:
        status, detail = "unverified", (f"ffprobe lists {len(probed)} frames, the decode "
                                        f"{n}: timestamps can't be matched frame by frame")
    elif any(p is None for p in probed):
        missing = sum(p is None for p in probed)
        status, detail = "unverified", f"ffprobe has no timestamp for {missing} frame(s)"
    else:
        start = round(probed[0], 3)
        deviation = round(max(abs((p - probed[0]) - t) for p, t in zip(probed, timestamps)), 3)
        if deviation > TIMESTAMP_TOLERANCE_MS:
            status, detail = "inconsistent", (f"decoder and ffprobe times differ by up to "
                                              f"{deviation:.3f} ms")
        else:
            status = "verified"
    if estimated:
        status = "estimated"
        detail = (f"{len(estimated)} frame time(s) estimated from the nominal frame rate "
                  f"(first: frame {estimated[0]})" + (f"; {detail}" if detail else ""))
    return status, start, deviation, detail


def build_frame_index(path, progress: Optional[Callable[[int, int], None]] = None,
                      sha256=None):
    """Decode every frame in order and record its timestamp and digest."""
    import cv2
    from utils.joint_labels import frame_digest

    if not os.path.isfile(path):
        raise FrameAccessError(f"no video file at {path}")
    size = os.path.getsize(path)
    sha = sha256 or file_sha256(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FrameAccessError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc(cap)
        timestamps, estimated, digests, shape = [], [], [], None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = len(timestamps)
            t, guessed = _container_time(_pos_msec(cap), idx, fps, timestamps)
            timestamps.append(t)
            if guessed:
                estimated.append(idx)
            digests.append(frame_digest(frame))
            shape = frame.shape
            if progress is not None:
                progress(len(digests), reported)
    finally:
        cap.release()
    if not digests:
        raise FrameAccessError(f"no frames could be decoded from {path}")
    status, start, deviation, detail = timestamp_check(timestamps, estimated,
                                                       _ffprobe_timestamps_ms(path))
    return FrameIndex(path=path, file_sha256=sha, size_bytes=size, frame_count=len(digests),
                      width=int(shape[1]), height=int(shape[0]),
                      timestamps_ms=tuple(float(t) for t in timestamps),
                      digests=tuple(digests), nominal_fps=float(fps),
                      reported_frame_count=reported, codec=codec, decoder=decoder_identity(),
                      start_offset_ms=start, ffprobe_max_deviation_ms=deviation,
                      estimated_frames=tuple(estimated), timestamp_status=status,
                      timestamp_detail=detail)


_memo = {}
_memo_lock = threading.Lock()


def get_frame_index(path, cache_dir=None, progress=None):
    """The index for `path`, from memory, the disk cache, or a fresh decode.

    The disk cache is keyed by content (SHA-256) and decoder, so a renamed or
    moved file reuses its index and a changed file never does.
    """
    try:
        st = os.stat(path)
    except OSError:
        raise FrameAccessError(f"no video file at {path}") from None
    memo_key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    with _memo_lock:
        if memo_key in _memo:
            return _memo[memo_key]
    if cache_dir is None:
        from utils.paths import get_frame_index_dir
        cache_dir = get_frame_index_dir()
    sha = file_sha256(path)
    decoder = decoder_identity()
    cache_file = os.path.join(
        cache_dir, f"{sha[:32]}-{hashlib.sha1(decoder.encode()).hexdigest()[:8]}.json")
    index = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                d = json.load(f)
            if (d.get("schema") == INDEX_SCHEMA and d.get("file_sha256") == sha
                    and d.get("decoder") == decoder):
                index = FrameIndex.from_json(path, d)
        except (OSError, ValueError, KeyError):
            index = None
    if index is None:
        index = build_frame_index(path, progress=progress, sha256=sha)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{cache_file}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(index.to_json(), f)
            os.replace(tmp, cache_file)
        except OSError:
            pass   # a read-only cache only costs a re-decode next time
    with _memo_lock:
        _memo[memo_key] = index
    return index


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray       # BGR, the decoded source frame, never resized
    digest: str


class FrameReader:
    """Serves exact frames of one indexed video. Not thread-safe; see
    `read_frame` for a shared, locked reader."""

    def __init__(self, index: FrameIndex, path=None):
        self.index = index
        self.path = path or index.path
        self._cap = None
        self._last = None       # (frame index, BGR) of the most recent decode

    def close(self):
        if self._cap is not None:
            self._cap.release()
        self._cap, self._last = None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self):
        import cv2
        self.close()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FrameAccessError(f"cannot open video: {self.path}")
        self._cap = cap

    def _decode_next(self):
        ok, img = self._cap.read()
        nxt = -1 if self._last is None else self._last[0]
        if not ok:
            raise FrameAccessError(
                f"{os.path.basename(self.path)}: decoding stopped after frame {nxt}, but the "
                f"index has {self.index.frame_count} frames (file changed or unreadable)")
        self._last = (nxt + 1, img)

    def _restart(self):
        self._open()
        self._last = None

    def _reposition(self, f):
        import cv2
        from utils.joint_labels import frame_digest
        start = max(0, f - SEEK_BACKOFF)
        if start == 0:
            self._restart()
            return
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, img = self._cap.read()
        if not ok:
            self._restart()
            return
        # Where did the seek land? A repeated image (static address, frame
        # doubling in slowed footage) matches several frames: read on until
        # the run of digests fits only one of them.
        digests, n = self.index.digests, self.index.frame_count
        candidates = [c for c in self.index.frames_with_digest(frame_digest(img)) if c <= f]
        k = 0
        while len(candidates) > 1 and k < 2 * FORWARD_WINDOW:
            ok, img = self._cap.read()
            if not ok:
                break
            k += 1
            d = frame_digest(img)
            candidates = [c for c in candidates if c + k < n and digests[c + k] == d]
        if len(candidates) == 1 and candidates[0] + k <= f:
            self._last = (candidates[0] + k, img)
        else:
            self._restart()

    def read(self, f) -> Frame:
        from utils.joint_labels import frame_digest
        if not isinstance(f, (int, np.integer)) or not 0 <= f < self.index.frame_count:
            raise FrameAccessError(
                f"frame {f!r} is outside 0..{self.index.frame_count - 1}")
        f = int(f)
        if self._last is not None and self._last[0] == f:
            img = self._last[1]
        else:
            ahead = None if self._last is None else f - self._last[0]
            if self._cap is None or ahead is None or not 0 < ahead <= FORWARD_WINDOW:
                self._reposition(f)
            while self._last is None or self._last[0] < f:
                self._decode_next()
            img = self._last[1]
        digest = frame_digest(img)
        if digest != self.index.digests[f]:
            self.close()
            raise FrameAccessError(
                f"{os.path.basename(self.path)} frame {f}: the decoded image is not the "
                "indexed one (file changed, or a different decoder); refusing to show an "
                "unverified frame")
        return Frame(index=f, timestamp_ms=self.index.timestamps_ms[f], image=img,
                     digest=digest)


_readers = {}
_readers_lock = threading.Lock()
MAX_OPEN_READERS = 8


def read_frame(path, frame, cache_dir=None) -> Frame:
    """Exact frame `frame` of `path` from a shared reader per file."""
    index = get_frame_index(path, cache_dir=cache_dir)
    key = (os.path.abspath(path), index.file_sha256)
    with _readers_lock:
        entry = _readers.get(key)
        if entry is None:
            if len(_readers) >= MAX_OPEN_READERS:
                old_key = next(iter(_readers))
                old_reader, old_lock = _readers.pop(old_key)
                with old_lock:
                    old_reader.close()
            entry = (FrameReader(index, path), threading.Lock())
            _readers[key] = entry
    reader, lock = entry
    with lock:
        return reader.read(frame)
