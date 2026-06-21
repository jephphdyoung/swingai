"""Ghost-overlay compositing and registration.

The overlay places the reference (pro) swing as a semi-transparent layer on top
of the user's swing. A single affine transform positions/sizes the pro layer.

Transform representation (resolution-independent, shared with the JS editor):
    {
        "left":   fraction of the user/base frame width  -> x of pro layer top-left
        "top":    fraction of the user/base frame height -> y of pro layer top-left
        "width":  fraction of the user/base frame width  -> displayed width of pro layer
        "mirror": bool                                   -> flip pro horizontally
        "alpha":  0..1                                   -> pro opacity in the blend
    }
Height is implied: the pro keeps its own aspect ratio (uniform scale).
"""

import cv2
import numpy as np

# Landmark indices in the extracted array (see pose_extractor.SELECTED_LANDMARK_INDICES):
# 1,2 = shoulders, 13,14 = hips.
_L_SHOULDER, _R_SHOULDER = 1, 2
_L_HIP, _R_HIP = 13, 14

DEFAULT_TRANSFORM = {"left": 0.0, "top": 0.0, "width": 1.0, "mirror": False, "alpha": 0.5}


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _hip_and_torso(landmarks, w, h):
    """Return (hip_center_px, torso_length_px) for one frame's landmarks."""
    sh = _midpoint(landmarks[_L_SHOULDER], landmarks[_R_SHOULDER])
    hp = _midpoint(landmarks[_L_HIP], landmarks[_R_HIP])
    sh_px = (sh[0] * w, sh[1] * h)
    hp_px = (hp[0] * w, hp[1] * h)
    torso = float(np.hypot(sh_px[0] - hp_px[0], sh_px[1] - hp_px[1]))
    return hp_px, torso


def auto_register(user_landmarks, ref_landmarks, user_dims, ref_dims):
    """Compute an initial transform that lines the pro up on the user.

    Matches body size (torso length) and position (hip center). Computed once,
    typically from the P1/address frame, then applied to the whole swing.

    Args:
        user_landmarks: user landmark list for one frame (normalized x,y,z).
        ref_landmarks:  reference landmark list for one frame.
        user_dims: (width, height) of the user/output frame in pixels.
        ref_dims:  (width, height) of the reference frame in pixels.

    Returns:
        transform dict (see module docstring).
    """
    w_u, h_u = user_dims
    w_r, h_r = ref_dims

    (uhx, uhy), utl = _hip_and_torso(user_landmarks, w_u, h_u)
    (rhx, rhy), rtl = _hip_and_torso(ref_landmarks, w_r, h_r)

    # Uniform scale mapping reference pixels -> output pixels.
    s = (utl / rtl) if rtl > 1e-6 else 1.0

    return {
        "left": (uhx - s * rhx) / w_u,
        "top": (uhy - s * rhy) / h_u,
        "width": (s * w_r) / w_u,
        "mirror": False,
        "alpha": 0.5,
    }


def _affine_matrix(transform, user_frame, ref_frame):
    """Build the 2x3 affine mapping reference-frame px -> output(user) px."""
    h_u, w_u = user_frame.shape[:2]
    h_r, w_r = ref_frame.shape[:2]

    s = (transform["width"] * w_u) / w_r if w_r else 1.0
    tx = transform["left"] * w_u
    ty = transform["top"] * h_u

    if transform.get("mirror"):
        # Flip about the layer's own vertical axis, then translate.
        m = np.float32([[-s, 0, tx + s * w_r], [0, s, ty]])
    else:
        m = np.float32([[s, 0, tx], [0, s, ty]])
    return m


def composite_overlay(user_frame, ref_frame, transform):
    """Blend the reference frame onto the user frame per ``transform``.

    Returns (output_frame, affine_matrix). The affine matrix maps reference-frame
    pixel coords to output coords (reuse it to draw the reference skeleton).
    """
    h_u, w_u = user_frame.shape[:2]
    m = _affine_matrix(transform, user_frame, ref_frame)

    warped = cv2.warpAffine(ref_frame, m, (w_u, h_u))
    mask = cv2.warpAffine(
        np.full(ref_frame.shape[:2], 255, np.uint8), m, (w_u, h_u)
    )

    alpha = float(transform.get("alpha", 0.5))
    out = user_frame.copy()
    idx = mask > 0
    out[idx] = (
        (1.0 - alpha) * user_frame[idx] + alpha * warped[idx]
    ).astype(np.uint8)
    return out, m


def transform_points(landmarks, ref_w, ref_h, affine_matrix):
    """Map a reference frame's normalized landmarks into output pixel coords."""
    pts = []
    a = affine_matrix
    for x, y, *_ in landmarks:
        px = x * ref_w
        py = y * ref_h
        ox = a[0, 0] * px + a[0, 1] * py + a[0, 2]
        oy = a[1, 0] * px + a[1, 1] * py + a[1, 2]
        pts.append((int(ox), int(oy)))
    return pts
