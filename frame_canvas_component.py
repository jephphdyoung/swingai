"""Streamlit wrapper for the frame canvas: click to place a point, drag to crop.

Static bidirectional component (no Node build), like overlay_component.py:
serves components/frame_canvas/ and returns the last click or crop drag, in
source pixels, with a nonce so the caller can act on each event once.
"""

import base64
import os

import cv2
import streamlit.components.v1 as components

_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components",
                          "frame_canvas")

_frame_canvas = components.declare_component("frame_canvas", path=_BUILD_DIR)


def frame_canvas(image_bgr, *, frame, points=(), crop=None, view=None, mode="point",
                 active=None, show_names=False, key=None):
    """Show a decoded source frame with labels; return the latest event.

    Args:
        image_bgr: the full source frame, never resized. Sent as a quality-95
            JPEG for display only: labels are tied to the decoded frame's
            digest, not to this preview.
        frame: source frame index, echoed back with each event.
        points: [{"name", "x", "y", "visible", "origin"}] in source pixels.
        crop: {"x", "y", "width", "height"} to outline, or None.
        view: source rectangle to display (zoom), or None for the whole frame.
        mode: "point" (click places `active`) or "crop" (drag a rectangle).

    Returns:
        None, or {"kind": "point", "name", "x", "y", "frame", "nonce"} /
        {"kind": "crop", "x", "y", "width", "height", "frame", "nonce"}.
    """
    h, w = image_bgr.shape[:2]
    ok, jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("could not encode the frame")
    image = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    return _frame_canvas(image=image, width=w, height=h, frame=int(frame), points=list(points),
                         crop=crop, view=view, mode=mode, active=active,
                         show_names=show_names, key=key, default=None)
