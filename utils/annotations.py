import json
import os
from typing import Optional

from utils.paths import repo_relative

ANNOTATIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annotations.json")


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

    Returns dict mapping position names (P1-P9) to frame numbers, or None.
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
                     view_angle: Optional[str] = None) -> None:
    """Save P-position annotations for a video."""
    data = _load_all()
    key = _normalize_path(video_path)
    entry = {
        name: {
            "frame": frame,
            "timestamp_ms": round(frame * 1000.0 / fps, 1),
        }
        for name, frame in sorted(positions.items())
    }
    if view_angle:
        entry["_view_angle"] = view_angle
    data[key] = entry
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
