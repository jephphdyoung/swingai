"""A stream's edits -- swing window and crop -- as explicit, reversible
transforms between source and edited coordinates.

Edits are metadata on the shot stream (`utils/catalog.py`); the source file is
never rewritten. Everything stored (P-positions, landmark labels) is in
**source** frames and pixels, so changing a window or crop later never moves
a label. These functions are the only conversion between the two spaces:

- frames: edited frame k is source frame window.start_frame + k;
- pixels: edited (x, y) is source (x - crop.x, y - crop.y). No scaling and no
  rotation: a crop is a sub-rectangle of the original pixels.
"""

from typing import Optional


class EditRangeError(ValueError):
    pass


def window(stream):
    w = stream["window"]
    return w["start_frame"], w["end_frame"]


def window_length(stream):
    start, end = window(stream)
    return end - start + 1


def edit_to_source_frame(stream, k):
    start, end = window(stream)
    if not 0 <= k <= end - start:
        raise EditRangeError(f"edited frame {k} is outside the {end - start + 1}-frame window")
    return start + k


def source_to_edit_frame(stream, f) -> Optional[int]:
    """Edited index of source frame f, or None when f is outside the window."""
    start, end = window(stream)
    return f - start if start <= f <= end else None


def in_window(stream, f):
    start, end = window(stream)
    return start <= f <= end


def _crop(stream):
    c = stream.get("crop")
    return (0, 0) if c is None else (c["x"], c["y"])


def source_to_edit_point(stream, x, y):
    ox, oy = _crop(stream)
    return x - ox, y - oy


def edit_to_source_point(stream, x, y):
    ox, oy = _crop(stream)
    return x + ox, y + oy


def point_in_crop(stream, x, y):
    c = stream.get("crop")
    if c is None:
        return True
    return c["x"] <= x < c["x"] + c["width"] and c["y"] <= y < c["y"] + c["height"]


def crop_image(image, stream):
    """The cropped view of a source frame (a slice: same pixels, no resize)."""
    c = stream.get("crop")
    if c is None:
        return image
    return image[c["y"]:c["y"] + c["height"], c["x"]:c["x"] + c["width"]]


def describe(stream, fps=None):
    start, end = window(stream)
    n = end - start + 1
    text = f"frames {start}–{end} ({n} frames"
    if fps:
        text += f", {n / fps:.2f}s at {fps:.1f} fps playback"
    text += ")"
    c = stream.get("crop")
    text += (", full frame" if c is None else
             f", crop {c['width']}×{c['height']} at ({c['x']}, {c['y']})")
    return text
