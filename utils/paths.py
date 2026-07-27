"""Central resolution of the video directories.

Layout (defaults, relative to the repo root):

    videos/reference/   reference swings shipped with the repo
    videos/user/        the golfer's own swings
    videos/output/      rendered comparison and overlay videos

Each can be repointed with an environment variable, which is how you aim the
tool at a capture library that lives outside the repo:

    SWINGAI_REFERENCE_DIR
    SWINGAI_USER_DIR
    SWINGAI_OUTPUT_DIR

run.sh reads the same variables and bind-mounts whatever they resolve to, so
the container and local dev agree on where footage lives.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEOS_ROOT = os.path.join(REPO_ROOT, "videos")

DEFAULT_REFERENCE_DIR = os.path.join(VIDEOS_ROOT, "reference")
DEFAULT_USER_DIR = os.path.join(VIDEOS_ROOT, "user")
DEFAULT_OUTPUT_DIR = os.path.join(VIDEOS_ROOT, "output")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _resolve(env_var, default, create=False):
    path = os.environ.get(env_var) or default
    path = os.path.abspath(os.path.expanduser(path))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_reference_dir():
    """Directory holding reference swings to compare against."""
    return _resolve("SWINGAI_REFERENCE_DIR", DEFAULT_REFERENCE_DIR)


def get_user_dir():
    """Directory holding the golfer's own swings."""
    return _resolve("SWINGAI_USER_DIR", DEFAULT_USER_DIR)


def get_output_dir():
    """Directory for rendered videos. Created if absent."""
    return _resolve("SWINGAI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR, create=True)


def list_videos(directory):
    """Video files under a directory, as paths relative to it, sorted.

    Recurses, because capture libraries nest clips in per-session folders —
    e.g. pointing SWINGAI_USER_DIR at a Swing Catalyst export gives
    "2026-07-27 135033/Jeff Young - ... .mp4". Returns [] if absent.
    """
    if not os.path.isdir(directory):
        return []

    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in files:
            if f.startswith(".") or not f.lower().endswith(VIDEO_EXTENSIONS):
                continue
            found.append(os.path.relpath(os.path.join(root, f), directory))
    return sorted(found)


def repo_relative(path):
    """Path relative to the repo root when it lives inside it, else absolute.

    Annotations are keyed by this so a video keeps the same key whether it is
    addressed from the host or from /app inside the container.
    """
    abs_path = os.path.abspath(path)
    if os.path.commonpath([abs_path, REPO_ROOT]) == REPO_ROOT:
        return os.path.relpath(abs_path, REPO_ROOT)
    return abs_path
