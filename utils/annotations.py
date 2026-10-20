import json
import os
from dataclasses import dataclass, field
from typing import Optional

from utils.paths import repo_relative

ANNOTATIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annotations.json")

P_POSITIONS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]

# Canonical P-system. P6 is shaft-parallel in the downswing and P7 is impact.
# Path A (pose_features / video_sync) still calls its velocity peak "P6" --
# that naming is legacy and is not what these labels mean.
P_DEFINITIONS = {
    "P1": "Address",
    "P2": "Takeaway: shaft parallel to ground",
    "P3": "Lead arm parallel to ground (backswing)",
    "P4": "Top of backswing",
    "P5": "Lead arm parallel to ground (downswing)",
    "P6": "Shaft parallel to ground (downswing)",
    "P7": "Impact",
    "P8": "Shaft parallel to ground (follow-through)",
    "P9": "Trail arm parallel to ground (follow-through)",
    "P10": "Finish",
}

# Stored as "_p_convention" on every entry saved under the definitions above.
# Entries without it predate the convention and may have marked impact as P6.
P_CONVENTION = "p7-impact"


def _normalize_path(video_path: str) -> str:
    """Normalize video path to a consistent relative key.

    Keys are relative to the repo root so a video keeps the same key whether
    it is addressed from the host or from /app inside the container.
    """
    return os.path.normpath(repo_relative(video_path))


def _load_all() -> dict:
    if not os.path.exists(ANNOTATIONS_PATH):
        return {}
    with open(ANNOTATIONS_PATH) as f:
        return json.load(f)


def _save_all(data: dict) -> None:
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_annotations(video_path: str) -> Optional[dict[str, int]]:
    """Load saved P-position annotations for a video.

    Returns dict mapping position names (P1-P10) to frame numbers, or None.
    """
    data = _load_all()
    key = _normalize_path(video_path)
    entry = data.get(key)
    if entry is None:
        return None
    return {name: pos["frame"] for name, pos in entry.items()
            if isinstance(pos, dict) and "frame" in pos}


def load_view_angle(video_path: str) -> Optional[str]:
    """Load saved view angle for a video ('face_on' or 'dtl')."""
    data = _load_all()
    key = _normalize_path(video_path)
    entry = data.get(key)
    if entry is None:
        return None
    return entry.get("_view_angle")


def save_annotations(video_path: str, positions: dict[str, int], fps: float,
                     view_angle: Optional[str] = None,
                     handedness: Optional[str] = None) -> None:
    """Save P-position annotations for a video, under the canonical definitions."""
    data = _load_all()
    key = _normalize_path(video_path)
    previous = data.get(key) or {}
    entry = {
        name: {
            "frame": frame,
            "timestamp_ms": round(frame * 1000.0 / fps, 1),
        }
        for name, frame in sorted(positions.items())
    }
    if view_angle:
        entry["_view_angle"] = view_angle
    handedness = handedness or previous.get("_handedness")
    if handedness:
        entry["_handedness"] = handedness
    entry["_p_convention"] = P_CONVENTION
    data[key] = entry
    _save_all(data)


def confirm_convention(video_path: str) -> None:
    """Mark an existing entry as following P_DEFINITIONS, frames unchanged."""
    data = _load_all()
    key = _normalize_path(video_path)
    if key in data:
        data[key]["_p_convention"] = P_CONVENTION
        _save_all(data)


def delete_annotations(video_path: str) -> None:
    """Remove saved annotations for a video."""
    data = _load_all()
    key = _normalize_path(video_path)
    if key in data:
        del data[key]
        _save_all(data)


def has_annotations(video_path: str) -> bool:
    """Check if a video has saved annotations."""
    data = _load_all()
    return _normalize_path(video_path) in data


@dataclass
class AnnotationRecord:
    """Everything saved for one video."""
    video_path: str
    frames: dict = field(default_factory=dict)          # {"P4": 652}
    timestamps_ms: dict = field(default_factory=dict)   # {"P4": 2615.3}
    view_angle: Optional[str] = None
    handedness: Optional[str] = None
    convention: Optional[str] = None


def load_annotation_record(video_path: str) -> Optional[AnnotationRecord]:
    entry = _load_all().get(_normalize_path(video_path))
    if entry is None:
        return None
    record = AnnotationRecord(
        video_path=video_path,
        view_angle=entry.get("_view_angle"),
        handedness=entry.get("_handedness"),
        convention=entry.get("_p_convention"),
    )
    for name, pos in entry.items():
        if not isinstance(pos, dict):
            continue
        if "frame" in pos:
            record.frames[name] = int(pos["frame"])
        if pos.get("timestamp_ms") is not None:
            record.timestamps_ms[name] = float(pos["timestamp_ms"])
    return record


def convention_warnings(record: Optional[AnnotationRecord]) -> list[str]:
    """Reasons to doubt that a record's P6/P7 mean shaft-parallel/impact.

    Annotations can't be checked against the video here -- these are the
    structural signs of a legacy "P6 = impact" labelling.
    """
    if record is None:
        return []
    warnings = []
    named = {p: f for p, f in record.frames.items() if p in P_DEFINITIONS}
    if record.convention != P_CONVENTION:
        warnings.append(
            "Saved before the P7 = impact convention was recorded. Older tooling "
            "called impact P6 -- check that P6 is shaft-parallel and P7 is impact, "
            "then confirm or correct.")
    if "P6" in named and "P7" not in named:
        warnings.append("P6 is marked but P7 is not: P6 may have been used for impact.")
    ordered = [(p, named[p]) for p in P_POSITIONS if p in named]
    for (pa, fa), (pb, fb) in zip(ordered, ordered[1:]):
        if fb < fa:
            warnings.append(f"{pb} (frame {fb}) is earlier than {pa} (frame {fa}).")
    return warnings
