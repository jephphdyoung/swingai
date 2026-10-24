"""Reconstruction artifacts: a versioned JSON manifest plus compact arrays.

    <output>/reconstructions/<shot_id>/<reconstruction_id>/
        manifest.json     kind "reconstruction_3d" (schemas/geometry-3d.schema.json)
        trajectory.npz    time_ms, xyz, cov, state, n_views, reprojection_px, parallax_deg,
                          motion_time_ms, timing_bound_ms, speed_bound_m_s,
                          timing_error_m, timing_status

The manifest links the arrays to everything that produced them: shot id and
the shot record's hash, each source's checksum, the label document's
revision and hash, the calibration id/version/hash, the algorithm name and
version and its parameters. The reconstruction id is derived from those, so
re-running on the same inputs gives the same id and the same arrays. Each
array's own SHA-256 (of its C-order bytes) is recorded as well as the file's,
so content can be checked whatever the zip container's metadata.

Timing arrays (`utils/timing_3d.py`): motion_time_ms (F; real ms, NaN when
the motion time is unknown), timing_bound_ms (F; NaN for static or unknown),
speed_bound_m_s (P), timing_error_m (F,P; worst-case displacement between
exposures, NaN where not applicable -- never mixed into `cov`) and
timing_status (F,P; 0 not applicable, 1 within, 2 exceeds, 3 unknown).
"""

import hashlib
import io
import json
import os

import numpy as np

from utils.catalog import atomic_write_json, canonical_sha256, now_iso
from utils.geometry_contract import SCHEMA_VERSION, schema_problems
from utils.reconstruct_3d import ALGORITHM, STATE_NAMES
from utils.timing_3d import TIMING_NAMES

ARRAYS = ("time_ms", "xyz", "cov", "state", "n_views", "reprojection_px", "parallax_deg",
          "motion_time_ms", "timing_bound_ms", "speed_bound_m_s", "timing_error_m",
          "timing_status")


class ArtifactError(ValueError):
    pass


def _array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def reconstruction_id(rec):
    seed = {"inputs": rec.inputs, "algorithm": ALGORITHM,
            "parameters": rec.requirements.as_dict()}
    return "rec-" + canonical_sha256(seed)[:16]


def arrays_of(rec):
    return {"time_ms": rec.time_ms.astype(np.float64), "xyz": rec.xyz.astype(np.float64),
            "cov": rec.cov.astype(np.float64), "state": rec.state.astype(np.int8),
            "n_views": rec.n_views.astype(np.int8),
            "reprojection_px": rec.reprojection_px.astype(np.float64),
            "parallax_deg": rec.parallax_deg.astype(np.float64),
            "motion_time_ms": rec.motion_time_ms.astype(np.float64),
            "timing_bound_ms": rec.timing_bound_ms.astype(np.float64),
            "speed_bound_m_s": rec.speed_bound_m_s.astype(np.float64),
            "timing_error_m": rec.timing_error_m.astype(np.float64),
            "timing_status": rec.timing_status.astype(np.int8)}


def manifest(rec, rid=None, created_at=None):
    arrays = arrays_of(rec)
    world = rec.world
    T = np.asarray(world["rig_to_world"]) if world["status"] == "defined" else None
    cams = []
    for s in rec.streams:
        c2r = s.camera.camera_to_frame
        cams.append({"camera_id": s.stream_id, "camera_to_rig": c2r.tolist(),
                     "camera_to_world": None if T is None else (T @ c2r).tolist()})
    counts = {name: int((rec.state == code).sum()) for code, name in STATE_NAMES.items()}
    # "missing" = labelled in at least one view but no valid 3D point;
    # points nobody labelled in a frame are counted apart.
    counts["not_labelled"] = int(((rec.state == 0) & (rec.n_views == 0)).sum())
    counts["missing"] -= counts["not_labelled"]
    reproj = rec.reprojection_px[rec.state > 0]
    timing = {name: int((rec.timing_status == code).sum())
              for code, name in TIMING_NAMES.items() if code}
    bounds = rec.timing_bound_ms[np.isfinite(rec.timing_bound_ms)]
    return {"schema_version": SCHEMA_VERSION, "kind": "reconstruction_3d",
            "reconstruction_id": rid or reconstruction_id(rec), "shot_id": rec.shot_id,
            "created_at": created_at or now_iso(),
            "algorithm": {**ALGORITHM, "parameters": rec.requirements.as_dict()},
            "inputs": rec.inputs, "association": rec.association, "world_frame": world,
            "cameras": cams, "points": list(rec.point_names), "frames": rec.frames,
            "arrays": {"path": "trajectory.npz", "sha256": "0" * 64,
                       "coordinates": "world" if T is not None else "rig",
                       "layout": {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                                      "sha256": _array_sha(v)} for k, v in arrays.items()}},
            "missing": rec.missing,
            "summary": {"frames": len(rec.frames),
                        "frames_measured": sum(f["status"] == "measured" for f in rec.frames),
                        "points": counts,
                        "dynamic_point_timing": timing,
                        "max_timing_bound_ms": float(bounds.max()) if bounds.size else None,
                        "max_reprojection_px": (float(np.nanmax(reproj))
                                                if reproj.size and np.isfinite(reproj).any()
                                                else None)}}


def write(rec, out_root=None):
    """Write manifest.json + trajectory.npz; returns the directory."""
    if out_root is None:
        from utils.paths import get_output_dir
        out_root = os.path.join(get_output_dir(), "reconstructions")
    m = manifest(rec)
    d = os.path.join(out_root, rec.shot_id, m["reconstruction_id"])
    os.makedirs(d, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays_of(rec))
    data = buf.getvalue()
    m["arrays"]["sha256"] = hashlib.sha256(data).hexdigest()
    problems = schema_problems(m, "manifest: ")
    if problems:
        raise ArtifactError("; ".join(problems))
    tmp = os.path.join(d, ".trajectory.npz.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, os.path.join(d, "trajectory.npz"))
    atomic_write_json(os.path.join(d, "manifest.json"), m)
    return d


def load(directory):
    """(manifest, {name: array}) after checking the schema and every hash."""
    with open(os.path.join(directory, "manifest.json")) as f:
        m = json.load(f)
    problems = schema_problems(m, "manifest: ")
    if problems:
        raise ArtifactError("; ".join(problems))
    path = os.path.join(directory, m["arrays"]["path"])
    with open(path, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != m["arrays"]["sha256"]:
        raise ArtifactError(f"{path} does not match the manifest's sha256")
    with np.load(io.BytesIO(data)) as z:
        arrays = {k: z[k] for k in z.files}
    for name, spec in m["arrays"]["layout"].items():
        a = arrays.get(name)
        if a is None or list(a.shape) != spec["shape"] or _array_sha(a) != spec["sha256"]:
            raise ArtifactError(f"array {name} does not match the manifest")
    return m, arrays


def manifest_sha256(m):
    return canonical_sha256(m)
