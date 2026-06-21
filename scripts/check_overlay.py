"""Headless smoke test for the ghost-overlay pipeline (no Streamlit/browser)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import analyze_swing, render_overlay
from utils.overlay import auto_register, composite_overlay, DEFAULT_TRANSFORM
import cv2
import numpy as np

USER = "my_videos/TW_face.mp4"
REF = "sample_videos/GW_faceon.mp4"


def main():
    print("== analyze_swing ==")
    result = analyze_swing(USER, REF)

    assert result.still_pairs, "no still_pairs produced"
    print(f"still_pairs: {len(result.still_pairs)} P-positions -> "
          f"{[p['p'] for p in result.still_pairs]}")
    p0 = result.still_pairs[0]
    assert p0["user"].startswith("data:image/jpeg;base64,"), "bad user thumb"
    assert p0["ref"].startswith("data:image/jpeg;base64,"), "bad ref thumb"
    print(f"auto_transform: {result.auto_transform}")
    for k in ("left", "top", "width", "mirror", "alpha"):
        assert k in result.auto_transform, f"missing transform key {k}"

    print("== composite_overlay (single frame) ==")
    u = np.zeros((400, 300, 3), np.uint8)
    r = np.full((200, 150, 3), 128, np.uint8)
    out, m = composite_overlay(u, r, dict(DEFAULT_TRANSFORM))
    assert out.shape == u.shape, "overlay changed output dims"
    assert m.shape == (2, 3), "bad affine"
    print(f"  out shape {out.shape}, affine ok, mean {out.mean():.1f}")

    print("== render_overlay (full video) ==")
    sk, no_sk, p_ts = render_overlay(result, result.auto_transform)
    for path in (sk, no_sk):
        assert os.path.exists(path), f"missing {path}"
        size = os.path.getsize(path)
        assert size > 10000, f"{path} too small ({size} bytes)"
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  {path}: {size//1024}KB, {frames} frames, {w}x{h}")
    print(f"  overlay p_timestamps: {p_ts}")

    print("\nALL OVERLAY CHECKS PASSED")


if __name__ == "__main__":
    main()
