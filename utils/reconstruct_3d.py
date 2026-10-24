"""3D reconstruction of one shot from its labelled camera views.

Inputs are the existing contracts, nothing parallel:

- the catalog shot (`utils/catalog.py`): its streams (one per camera, the
  stream id is the capture contract's camera id), the documented same-swing
  association, synchronization and calibration statuses;
- the shot's labels (`utils/shot_annotations.py`): 2D points on exact source
  frames, each with its source-frame index, timestamp, timestamp basis and
  image digest;
- a camera calibration (`utils/calibration.py`) with one camera per stream id.

**Stereo input rule.** A shot is triangulated only when it has two or more
streams whose same-swing association is documented (asserted with a method,
or verified). Independent recordings -- e.g. Grant Waite DTL and face-on
clips of different swings, which the catalog keeps as separate single-stream
shots -- are refused (`StereoInputError`): they can feed a later statistical
3D fit, never a stereo measurement. An asserted association is enough to
reconstruct experimentally; nothing is measured until it is verified.

**Frame association.** Each camera keeps its own source-frame ids and
timestamps. Times are put on the reference stream's clock (the shot's first
stream): t_ref = t_stream + synchronization.offset_ms.

- *static_address*: every stream's own P1 frame. The golfer is still at
  address, so these pair without relying on synchronization.
- *dynamic*: only with a known synchronization offset (estimated or
  verified); each other labelled reference frame pairs with the other
  stream's labelled frame nearest the synchronized time, only if the gap is
  <= `max_pair_gap_ms` (default 0.5 ms: effectively simultaneous capture). A
  moving point is somewhere else in a frame taken later, so a wider gap is
  not paired. Free-running unsynchronized cameras therefore produce no
  measured dynamic frames.

**Statuses.** Per point: observed (every contributing label visible), inferred
(a label placed as hidden, or an unreviewed model proposal) or missing (one
view only, or rejected by `triangulation.check_point`, with the reason). Per
frame: "measured" only when the calibration is verified (document and
catalog) and the same-swing association is verified with its provenance
(method, by, at) -- views of different swings make even a still address pose
meaningless in 3D; for dynamic frames also the synchronization is verified
with an offset and uncertainty, every view's timestamp is a verified
container time, every stream's motion time is known (`catalog.motion_clock`)
and the frame's timing-error bound is known; otherwise "unverified" with the
reasons.

**Timing** (`utils/timing_3d.py`). Each dynamic frame records its timing-error
bound in real ms (pair gap, synchronization uncertainty, exposure alignment)
and each triangulated point the position error that bound allows at the
point's a-priori speed bound, with a per-point timing status (within /
exceeds `max_timing_position_error_m` / unknown). This is orthogonal to the
observed/inferred/missing state and to the covariance: a fast clubhead can
fail it in a frame whose slow shoulders pass. `Reconstruction.problem()`
combines the gates for consumers. Static address frames are not timed (the
golfer is still).

**World frame.** Points are triangulated in the calibration's rig frame, then
moved to the shot's world frame (`utils/world_frame.py`) when the ground,
target line and ball are all evidenced -- from the calibration document
(operator-confirmed board) or from labelled scene points at address
(floor_a/b/c, target_line_back/front, ball_center). Otherwise the world frame
is "undefined" and stays so in the output.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils import timing_3d as tm
from utils import world_frame as wf
from utils.camera_model import apply_transform
from utils.catalog import canonical_sha256
from utils.triangulation import check_point, triangulate

ALGORITHM = {"name": "swingai-label-triangulation", "version": "1.1.0"}
STATE_MISSING, STATE_OBSERVED, STATE_INFERRED = 0, 1, 2
STATE_NAMES = {STATE_MISSING: "missing", STATE_OBSERVED: "observed",
               STATE_INFERRED: "inferred"}
SYNC_CONVENTION = "t_reference = t_stream + offset_ms for every non-reference stream"


class StereoInputError(ValueError):
    """The shot can't be a stereo input (independent views, no calibration...)."""


@dataclass(frozen=True)
class Requirements:
    sigma_px: float = 1.0
    max_reprojection_px: float = 3.0
    min_parallax_deg: float = 5.0
    max_point_sigma_m: float = 0.02
    max_pair_gap_ms: float = 0.5
    max_timing_position_error_m: float = 0.010
    speed_bounds_m_s: tuple = tuple(sorted(tm.SPEED_BOUNDS_M_S.items()))

    @property
    def speed_bounds(self):
        return dict(self.speed_bounds_m_s)

    def as_dict(self):
        return {**self.__dict__, "speed_bounds_m_s": self.speed_bounds}


@dataclass
class StreamInput:
    stream_id: str
    source_id: str
    source_sha256: str
    camera_view: str
    timestamp_status: str
    camera: object                      # Camera in the rig frame
    labels: dict                        # the stream's shot-annotation entry
    offset_ms: Optional[float] = 0.0    # t_ref = t_stream + offset_ms; None: unknown
    motion: dict = field(default_factory=lambda: {"scale": None, "reasons": ["not given"]})
    frame_interval_ms: Optional[float] = None   # timestamp ms per frame (1000 / nominal fps)

    @property
    def motion_scale(self):
        """Real ms per timestamp ms (catalog.motion_clock), None when unknown."""
        return self.motion.get("scale")


@dataclass
class Reconstruction:
    shot_id: str
    streams: list
    point_names: list
    frames: list                        # dicts as in the manifest
    xyz: np.ndarray                     # (F,P,3)
    cov: np.ndarray                     # (F,P,3,3)
    state: np.ndarray                   # (F,P) int8
    n_views: np.ndarray                 # (F,P) int8
    reprojection_px: np.ndarray         # (F,P,V)
    parallax_deg: np.ndarray            # (F,P)
    world: dict
    association: dict
    requirements: Requirements
    missing: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    timing_error_m: np.ndarray = None   # (F,P) worst-case displacement between exposures
    timing_status: np.ndarray = None    # (F,P) int8, timing_3d.TIMING_*

    @property
    def time_ms(self):
        return np.array([f["time_ms"] for f in self.frames], float)

    @property
    def motion_scale(self):
        """Real ms per reference-clock ms; None when the motion time is unknown."""
        return self.streams[0].motion_scale

    @property
    def motion_time_ms(self):
        """(F,) real swing time on the reference clock's origin; NaN when unknown."""
        return np.array([np.nan if f["motion_time_ms"] is None else f["motion_time_ms"]
                         for f in self.frames], float)

    @property
    def timing_bound_ms(self):
        """(F,) each dynamic frame's timing-error bound (real ms); NaN static/unknown."""
        return np.array([np.nan if not f["timing"] or f["timing"]["bound_ms"] is None
                         else f["timing"]["bound_ms"] for f in self.frames], float)

    @property
    def speed_bound_m_s(self):
        return np.array([tm.speed_bound(n, self.requirements.speed_bounds)
                         for n in self.point_names], float)

    def problem(self, i, j, allow_inferred=False):
        """Why point j at frame i isn't a trustworthy measured input (None if it is):
        frame not measured, point not observed, or (dynamic) timing gate failed."""
        fr = self.frames[i]
        if fr["status"] != "measured":
            return "frame not measured: " + "; ".join(fr["reasons"])
        ok = (STATE_OBSERVED, STATE_INFERRED) if allow_inferred else (STATE_OBSERVED,)
        if self.state[i, j] not in ok:
            return f"{self.point_names[j]} is {STATE_NAMES[int(self.state[i, j])]}"
        if fr["kind"] == "dynamic":
            return tm.describe(self.point_names[j], fr["timing"], self.timing_error_m[i, j],
                               self.timing_status[i, j],
                               self.requirements.max_timing_position_error_m,
                               self.requirements.speed_bounds)
        return None

    def point_index(self, name):
        return self.point_names.index(name)

    def frame_with(self, kind):
        return next((i for i, f in enumerate(self.frames) if f["kind"] == kind), None)


# --- inputs -----------------------------------------------------------------------

def stereo_inputs(cat, shot_id, annotations, calibration, indexes=None):
    """Check that the shot can be triangulated and gather its streams.

    calibration: (doc, {camera_id: Camera}, sha256) from load_calibration.
    indexes: optional {stream_id: FrameIndex}; when given, every label's image
    digest is checked against its source frame first.
    """
    from utils import shot_annotations as sa
    shot = cat.shot(shot_id)
    if len(shot["streams"]) < 2:
        raise StereoInputError(
            f"shot {shot_id} has one camera view. Independent recordings (e.g. separate "
            "DTL and face-on swings) are separate shots and can't be a stereo pair; they "
            "can only feed a later statistical 3D fit")
    same = shot["multi_view"]["same_swing"]
    if same["status"] not in ("asserted", "verified") or not same.get("method"):
        raise StereoInputError(f"shot {shot_id}: the streams' same-swing association is "
                               f"{same['status']} without a documented method")
    if annotations.get("shot_id") != shot_id:
        raise StereoInputError("labels are for a different shot")
    from utils.catalog import motion_clock
    doc, cams, _ = calibration
    sync = shot["multi_view"]["synchronization"]
    offset = sync.get("offset_ms") if sync["status"] in ("estimated", "verified") else None
    streams = []
    for i, st in enumerate(shot["streams"]):
        cam = cams.get(st["stream_id"])
        if cam is None:
            raise StereoInputError(f"calibration {doc['calibration_id']} has no camera for "
                                   f"stream {st['stream_id']}")
        src = cat.source(st["source_id"])
        if (cam.width, cam.height) != (src["video"]["width"], src["video"]["height"]):
            raise StereoInputError(
                f"stream {st['stream_id']}: calibrated at {cam.width}x{cam.height}, source "
                f"is {src['video']['width']}x{src['video']['height']}")
        labels = annotations["streams"].get(st["stream_id"])
        if labels is None:
            raise StereoInputError(f"no labels for stream {st['stream_id']}")
        if labels["source_sha256"] != src["sha256"]:
            raise StereoInputError(f"stream {st['stream_id']}: labels were placed on "
                                   "different bytes than the registered source")
        if indexes is not None:
            problems = sa.identity_problems(annotations, st["stream_id"], indexes[st["stream_id"]])
            if problems:
                raise StereoInputError(f"stream {st['stream_id']}: " + "; ".join(problems))
        fps = src["video"].get("nominal_fps")
        streams.append(StreamInput(
            st["stream_id"], st["source_id"], src["sha256"], st["camera_view"],
            src["video"]["timestamps"]["status"], cam, labels,
            0.0 if i == 0 else (None if offset is None else float(offset)),
            motion_clock(src, st["stream_id"]), 1000.0 / fps if fps else None))
    return shot, streams


# --- association --------------------------------------------------------------------

def _view(stream, f):
    fl = stream.labels["frames"][str(f)]
    return {"stream_id": stream.stream_id, "source_id": stream.source_id, "source_frame": f,
            "timestamp_ms": fl["timestamp_ms"], "timestamp_basis": fl["timestamp_basis"],
            "image_sha1": fl["image_sha1"]}


def _associate(streams, req):
    ref, others = streams[0], streams[1:]
    frames = []
    p1 = [s.labels["positions"].get("P1") for s in streams]
    static_ref = None
    if all(p1) and all(str(p["frame"]) in s.labels["frames"] for p, s in zip(p1, streams)):
        static_ref = p1[0]["frame"]
        frames.append({"kind": "static_address", "pair_gap_ms": None,
                       "views": [_view(s, p["frame"]) for s, p in zip(streams, p1)]})
    for fkey in sorted(ref.labels["frames"], key=int):
        f = int(fkey)
        if f == static_ref:
            continue
        t_ref = ref.labels["frames"][fkey]["timestamp_ms"]
        views, gap = [_view(ref, f)], 0.0
        for s in others:
            if s.offset_ms is None:
                continue                # no synchronization: never pair moving frames
            target = t_ref - s.offset_ms
            best = min(s.labels["frames"].items(),
                       key=lambda kv: abs(kv[1]["timestamp_ms"] - target), default=None)
            if best is None:
                continue
            g = abs(best[1]["timestamp_ms"] - target)
            if g <= req.max_pair_gap_ms:
                views.append(_view(s, int(best[0])))
                gap = max(gap, g)
        if len(views) >= 2:
            frames.append({"kind": "dynamic", "pair_gap_ms": round(gap, 4), "views": views})
    scale = ref.motion_scale
    for fr in frames:
        v0 = fr["views"][0]
        fr["time_ms"] = v0["timestamp_ms"]
        fr["motion_time_ms"] = None if scale is None else round(fr["time_ms"] * scale, 6)
    frames.sort(key=lambda fr: fr["time_ms"])
    for i, fr in enumerate(frames):
        fr["index"] = i
    return frames


def _frame_status(fr, streams, shot, calibration_doc, req):
    reasons = []
    by_id = {s.stream_id: s for s in streams}
    if calibration_doc["status"]["status"] != "verified":
        reasons.append(f"calibration {calibration_doc['calibration_id']} is "
                       f"{calibration_doc['status']['status']}, not verified")
    if shot["multi_view"]["calibration"]["status"] != "verified":
        reasons.append(f"the shot's calibration status is "
                       f"{shot['multi_view']['calibration']['status']}")
    same = shot["multi_view"]["same_swing"]
    if same["status"] != "verified":
        reasons.append(f"same swing is {same['status']}, not verified")
    else:
        gaps = [k for k in ("method", "by", "at") if not str(same.get(k) or "").strip()]
        if gaps:
            reasons.append(f"same swing is verified without provenance ({', '.join(gaps)})")
    if fr["kind"] == "dynamic":
        sync = shot["multi_view"]["synchronization"]
        if sync["status"] != "verified":
            reasons.append(f"synchronization is {sync['status']}, not verified")
        elif sync.get("offset_ms") is None or sync.get("uncertainty_ms") is None:
            reasons.append("synchronization has no offset / uncertainty")
        for v in fr["views"]:
            s = by_id[v["stream_id"]]
            if v["timestamp_basis"] != "container":
                reasons.append(f"{s.stream_id} frame {v['source_frame']}: estimated timestamp")
            if s.timestamp_status != "verified":
                reasons.append(f"{s.stream_id}: source timestamps are {s.timestamp_status}")
            reasons += s.motion["reasons"]
        reasons += [f"timing-error bound unknown: {u}" for u in fr["timing"]["unknown"]
                    if u != tm.MOTION_UNKNOWN]
    return ("measured" if not reasons else "unverified"), reasons


# --- reconstruction -------------------------------------------------------------------

def _inferred(label, frame_labels):
    if not label["visible"]:
        return True
    reviewed = (frame_labels.get("review") or {}).get("status") == "reviewed"
    return label["origin"] == "model_proposed" and not reviewed


def reconstruct(shot, streams, calibration, annotations, requirements=None):
    req = requirements or Requirements()
    cal_doc, _, cal_sha = calibration
    frames = _associate(streams, req)
    by_id = {s.stream_id: s for s in streams}
    names = sorted({n for fr in frames for v in fr["views"]
                    for n in by_id[v["stream_id"]].labels["frames"][str(v["source_frame"])]
                    ["points"]})
    F, P, V = len(frames), len(names), len(streams)
    xyz = np.full((F, P, 3), np.nan)
    cov = np.full((F, P, 3, 3), np.nan)
    state = np.zeros((F, P), np.int8)
    n_views = np.zeros((F, P), np.int8)
    reproj = np.full((F, P, V), np.nan)
    parallax = np.full((F, P), np.nan)
    missing = []
    col = {s.stream_id: i for i, s in enumerate(streams)}
    sync = shot["multi_view"]["synchronization"]
    for fr in frames:
        fr["timing"] = (tm.frame_timing(fr["pair_gap_ms"], sync,
                                        [by_id[v["stream_id"]].motion_scale
                                         for v in fr["views"]])
                        if fr["kind"] == "dynamic" else None)
    for i, fr in enumerate(frames):
        fr["status"], fr["reasons"] = _frame_status(fr, streams, shot, cal_doc, req)
        for j, name in enumerate(names):
            cams, pix, inferred, used = [], [], False, []
            for v in fr["views"]:
                s = by_id[v["stream_id"]]
                fl = s.labels["frames"][str(v["source_frame"])]
                lab = fl["points"].get(name)
                if lab is None:
                    continue
                cams.append(s.camera)
                pix.append((lab["x"], lab["y"]))
                used.append(col[s.stream_id])
                inferred |= _inferred(lab, fl)
            n_views[i, j] = len(cams)
            if len(cams) < 2:
                if cams:
                    missing.append({"frame": i, "point": name, "reason": "seen in one view only"})
                continue
            p = triangulate(cams, pix, sigma_px=req.sigma_px)
            reproj[i, j, used] = p.reprojection_px
            parallax[i, j] = p.parallax_deg
            reasons = check_point(p, req.max_reprojection_px, req.min_parallax_deg,
                                  req.max_point_sigma_m)
            if reasons:
                missing.append({"frame": i, "point": name, "reason": "; ".join(reasons)})
                continue
            xyz[i, j], cov[i, j] = p.xyz, p.cov
            state[i, j] = STATE_INFERRED if inferred else STATE_OBSERVED
    timing_err = np.full((F, P), np.nan)
    timing_status = np.zeros((F, P), np.int8)
    for i, fr in enumerate(frames):
        if fr["kind"] == "dynamic":
            timing_err[i], timing_status[i] = tm.point_timing(
                fr["timing"]["bound_ms"], names, state[i] > 0,
                req.max_timing_position_error_m, req.speed_bounds)
    association = {
        "reference_stream": streams[0].stream_id,
        "same_swing": {k: shot["multi_view"]["same_swing"].get(k)
                       for k in ("status", "method", "by", "at")},
        "synchronization": {"status": shot["multi_view"]["synchronization"]["status"],
                            "method": shot["multi_view"]["synchronization"].get("method"),
                            "offset_ms": shot["multi_view"]["synchronization"].get("offset_ms"),
                            "uncertainty_ms":
                                shot["multi_view"]["synchronization"].get("uncertainty_ms"),
                            "exposure_alignment": sync.get("exposure_alignment", "unknown"),
                            "exposure_uncertainty_ms": sync.get("exposure_uncertainty_ms"),
                            "convention": SYNC_CONVENTION},
        "motion_time": [{"stream_id": s.stream_id,
                         **{k: s.motion.get(k) for k in ("motion_time", "capture_fps",
                                                         "scale", "basis", "reasons")}}
                        for s in streams],
        "timing_convention": tm.BOUND_FORMULA,
        "catalog_calibration": {k: shot["multi_view"]["calibration"].get(k)
                                for k in ("status", "method")}}
    association["notes"] = [f"{s.stream_id}: synchronization unknown, so no moving frames were "
                            "paired with it" for s in streams[1:] if s.offset_ms is None]
    rec = Reconstruction(shot["shot_id"], streams, names, frames, xyz, cov, state, n_views,
                         reproj, parallax, {}, association, req, missing,
                         timing_error_m=timing_err, timing_status=timing_status)
    rec.world = _world_frame(rec, cal_doc)
    if rec.world["status"] == "defined":
        T = np.asarray(rec.world["rig_to_world"])
        R = T[:3, :3]
        ok = np.isfinite(rec.xyz).all(axis=2)
        rec.xyz[ok] = apply_transform(T, rec.xyz[ok])
        rec.cov[ok] = R @ rec.cov[ok] @ R.T
    rec.inputs = {
        "shot_sha256": canonical_sha256(shot),
        "annotations": {"revision": annotations.get("revision", 0),
                        "sha256": canonical_sha256({k: v for k, v in annotations.items()
                                                    if not k.startswith("_")})},
        "sources": [{"stream_id": s.stream_id, "source_id": s.source_id,
                     "sha256": s.source_sha256, "camera_view": s.camera_view,
                     "timestamp_status": s.timestamp_status} for s in streams],
        "calibration": {"calibration_id": cal_doc["calibration_id"],
                        "version": cal_doc["version"], "sha256": cal_sha,
                        "status": cal_doc["status"]["status"]}}
    return rec


def _static_point(rec, name):
    i = rec.frame_with("static_address")
    if i is None or name not in rec.point_names:
        return None
    j = rec.point_index(name)
    return rec.xyz[i, j] if rec.state[i, j] == STATE_OBSERVED else None


def _world_frame(rec, cal_doc):
    reasons, ground, target = [], None, None
    tmethod = None
    if cal_doc.get("ground"):
        ground = dict(cal_doc["ground"])
    else:
        pts = [p for p in (_static_point(rec, n) for n in ("floor_a", "floor_b", "floor_c"))
               if p is not None]
        if len(pts) >= 3:
            try:
                ground = wf.ground_from_points(pts, "floor_a/b/c labelled at address, "
                                                    "triangulated")
            except wf.WorldFrameError as e:
                reasons.append(str(e))
        else:
            reasons.append("no ground evidence: the calibration has none and fewer than three "
                           "floor points (floor_a/b/c) were observed at address")
    if cal_doc.get("target_direction"):
        target = cal_doc["target_direction"]["direction_rig"]
        tmethod = cal_doc["target_direction"]["method"]
    else:
        back, front = (_static_point(rec, n) for n in ("target_line_back", "target_line_front"))
        if back is not None and front is not None:
            try:
                target = wf.target_direction_from_points(back, front)
                tmethod = "target_line_back -> target_line_front labelled at address"
            except wf.WorldFrameError as e:
                reasons.append(str(e))
        else:
            reasons.append("no target-line evidence: the calibration has none and the "
                           "target-line marker points weren't observed at address")
    ball = _static_point(rec, "ball_center")
    if ball is None:
        reasons.append("ball_center not observed in both views at address (P1)")
    undefined = {"status": "undefined", "convention": wf.CONVENTION, "units": "m",
                 "rig_to_world": None, "evidence": None, "reasons": reasons}
    if reasons:
        return undefined
    up = np.mean([s.camera.center for s in rec.streams], axis=0)
    try:
        frame = wf.define_world_frame(ground, target, ball, up, tmethod,
                                      "ball_center at address (P1), triangulated, projected "
                                      "to the ground")
    except wf.WorldFrameError as e:
        return {**undefined, "reasons": [str(e)]}
    return {"status": "defined", "convention": wf.CONVENTION, "units": "m",
            "rig_to_world": frame.rig_to_world.tolist(), "evidence": frame.evidence,
            "reasons": []}


def reconstruct_shot(cat, shot_id, annotations, calibration, indexes=None, requirements=None):
    shot, streams = stereo_inputs(cat, shot_id, annotations, calibration, indexes)
    return reconstruct(shot, streams, calibration, annotations, requirements)
