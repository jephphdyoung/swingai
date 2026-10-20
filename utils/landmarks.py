"""Named body landmarks, independent of any pose backend's index layout.

Downstream code (measurements, drawing) addresses joints by name -- never by a
backend's raw index -- so a second backend (RTMPose, PR #4) only has to map its
own keypoints onto these names. A backend that has no keypoint for a name
reports it as unobserved.

Left/right are the *golfer's anatomical* sides, which is MediaPipe's
convention. Lead/trail depend on handedness: a right-handed golfer leads with
the left side.
"""

# (name, MediaPipe Pose landmark index). The order defines the column order of
# PoseSequence arrays. The legacy extractor's SELECTED_LANDMARK_INDICES is the
# same set minus the ears; that layout is unchanged and still used by path A.
LANDMARKS = (
    ("nose", 0),
    ("left_ear", 7),
    ("right_ear", 8),
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_pinky", 17),
    ("right_pinky", 18),
    ("left_index", 19),
    ("right_index", 20),
    ("left_thumb", 21),
    ("right_thumb", 22),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
    ("left_foot_index", 31),
    ("right_foot_index", 32),
)

LANDMARK_NAMES = tuple(name for name, _ in LANDMARKS)
LANDMARK_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}
MEDIAPIPE_INDEX = dict(LANDMARKS)

# Segments drawn as the skeleton, by name.
SKELETON_SEGMENTS = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_thumb"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_thumb"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot_index"),
    ("left_ear", "right_ear"),
)

HANDEDNESS = ("right", "left")
VIEWS = ("dtl", "face_on")
VIEW_LABELS = {"dtl": "Down-the-line", "face_on": "Face-on"}


def lead_side(handedness):
    """Anatomical side ('left'/'right') that leads for this handedness."""
    if handedness not in HANDEDNESS:
        raise ValueError(f"handedness must be one of {HANDEDNESS}, got {handedness!r}")
    return "left" if handedness == "right" else "right"


def trail_side(handedness):
    return "right" if lead_side(handedness) == "left" else "left"


def sided(role, joint, handedness):
    """Resolve a role-relative joint to a landmark name.

    sided("lead", "elbow", "right") -> "left_elbow"
    """
    if role == "lead":
        side = lead_side(handedness)
    elif role == "trail":
        side = trail_side(handedness)
    else:
        raise ValueError(f"role must be 'lead' or 'trail', got {role!r}")
    name = f"{side}_{joint}"
    if name not in LANDMARK_INDEX:
        raise KeyError(name)
    return name
