"""What a person can label on a frame, and which pose models produce each point.

Three groups, labelled independently so a frame can get Body only:

- **body**: the 15 joints both pose models produce. These, and only these,
  are scored by the MediaPipe vs RTMPose benchmark (`BENCHMARK_POINTS`).
- **feet**: anatomical foot points (heel, big toe, little toe), shoe contact
  points and floor references, for future heel-clearance, heel-lift timing
  and foot-pivot work. All 2D image positions: no ground pressure, no 3D foot
  rotation.
- **hands_club**: finger points, optional hand/forearm markers, shaft and
  clubhead points.

Each point has a `kind`:

- ``anatomical``: a body landmark a pose model may predict.
- ``contact``: where the shoe meets (or would meet) the floor. A manual label;
  no model predicts it.
- ``reference``: a fixed scene point (the floor line).
- ``marker``: something worn (glove logo, tape). Skip when nothing is worn.
- ``equipment``: a point on the club.

`models` says, per pose model, whether the model has the point at all
(``point``: the model's own keypoint name, or None), how closely that point
matches the label's definition (``match``: "same", "approximate"), and
whether SwingAI's pose layer keeps it today (``extracted``). A point a model
doesn't have is never estimated for it.
"""

from utils.pose_metrics import COMMON_JOINTS

GROUPS = ("body", "feet", "hands_club")
GROUP_TITLES = {"body": "Body", "feet": "Feet", "hands_club": "Hands / Club"}
KINDS = ("anatomical", "contact", "reference", "marker", "equipment")
MODELS = ("mediapipe", "rtmpose")


def _m(point=None, match=None, extracted=False, note=None):
    out = {"point": point, "match": match if point else None, "extracted": bool(point and extracted)}
    if note:
        out["note"] = note
    return out


def _both(name, mp_index, halpe_index):
    return {"mediapipe": _m(f"{name} ({mp_index})", "same", True),
            "rtmpose": _m(f"{name} ({halpe_index})", "same", True)}


_BODY_HELP = {
    "nose": "Tip of the nose.",
    "ear": "The ear canal, not the outer ear's edge.",
    "shoulder": "Centre of the shoulder joint (humeral head), not the top of the shoulder.",
    "elbow": "Centre of the elbow joint.",
    "wrist": "Centre of the wrist joint, between the forearm bones' ends.",
    "hip": "The hip joint centre, not the belt line.",
    "knee": "Centre of the knee joint.",
    "ankle": "Centre of the ankle joint (between the ankle bones).",
}
_MP = {"nose": 0, "left_ear": 7, "right_ear": 8, "left_shoulder": 11, "right_shoulder": 12,
       "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
       "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
       "left_ankle": 27, "right_ankle": 28}
_HALPE = {"nose": 0, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
          "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
          "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
          "left_ankle": 15, "right_ankle": 16}


def _body():
    out = []
    for name in COMMON_JOINTS:
        part = name.split("_", 1)[-1]
        models = _both(name, _MP[name], _HALPE[name])
        if part == "hip":
            note = "COCO-family hips sit on the joint; MediaPipe's somewhat differently"
            models["mediapipe"]["note"] = note
        out.append({"name": name, "group": "body", "kind": "anatomical",
                    "help": _BODY_HELP[part], "models": models})
    return out


def _feet():
    out = []
    for side in ("left", "right"):
        mp_heel = 29 if side == "left" else 30
        h_heel, h_big, h_small = ((24, 20, 22) if side == "left" else (25, 21, 23))
        mp_foot = 31 if side == "left" else 32
        out += [
            {"name": f"{side}_heel", "group": "feet", "kind": "anatomical",
             "help": "Back of the heel bone (calcaneus) through the shoe: the rearmost point "
                     "of the foot, not the shoe's outer heel edge.",
             "models": {"mediapipe": _m(f"{side}_heel ({mp_heel})", "approximate",
                                        note="in the model; SwingAI's pose layer drops it"),
                        "rtmpose": _m(f"{side}_heel ({h_heel})", "approximate",
                                      note="in Halpe26; SwingAI's pose layer drops it")}},
            {"name": f"{side}_big_toe", "group": "feet", "kind": "anatomical",
             "help": "Tip of the big toe through the shoe.",
             "models": {"mediapipe": _m(note=f"its foot_index ({mp_foot}) is the front of "
                                             "the foot, not the big toe, and is never scored "
                                             "as one"),
                        "rtmpose": _m(f"{side}_big_toe ({h_big})", "same", True,
                                      note="kept in the foot_index column for drawing only")}},
            {"name": f"{side}_little_toe", "group": "feet", "kind": "anatomical",
             "help": "Tip of the little toe through the shoe.",
             "models": {"mediapipe": _m(),
                        "rtmpose": _m(f"{side}_small_toe ({h_small})", "same",
                                      note="in Halpe26; SwingAI's pose layer drops it")}},
            {"name": f"{side}_shoe_heel_contact", "group": "feet", "kind": "contact",
             "help": "Bottom of the shoe heel where it touches the floor. When the heel is "
                     "lifted, the bottom of the heel that would touch it.",
             "models": {m: _m() for m in MODELS}},
            {"name": f"{side}_shoe_toe_contact", "group": "feet", "kind": "contact",
             "help": "Bottom of the shoe's toe where it touches (or would touch) the floor.",
             "models": {m: _m() for m in MODELS}},
        ]
    for ab, where in (("a", "nearer the lead foot"), ("b", "nearer the trail foot")):
        out.append({"name": f"floor_{ab}", "group": "feet", "kind": "reference",
                    "help": f"A point on the floor surface {where}, on a visible straight floor "
                            "edge or mat edge. floor_a and floor_b together give the floor line.",
                    "models": {m: _m() for m in MODELS}})
    return out


def _hands_club():
    out = []
    for side in ("left", "right"):
        mp = {"thumb": 21, "index": 19, "pinky": 17} if side == "left" else \
             {"thumb": 22, "index": 20, "pinky": 18}
        pose_note = ("MediaPipe Pose's coarse hand point (not the 21-point hand model); "
                     "the model doesn't define it anatomically, so the match is approximate")
        for finger, help_ in (("thumb", "Tip of the thumb."),
                              ("index", "Base knuckle (MCP joint) of the index finger."),
                              ("pinky", "Base knuckle (MCP joint) of the little finger.")):
            out.append({"name": f"{side}_{finger}", "group": "hands_club", "kind": "anatomical",
                        "help": help_ + " Skip it when the grip hides it.",
                        "models": {"mediapipe": _m(f"{side}_{finger} ({mp[finger]})",
                                                   "approximate", True, note=pose_note),
                                   "rtmpose": _m(note="Body26/Halpe26 has no finger "
                                                      "points")}})
        out += [
            {"name": f"{side}_hand_back", "group": "hands_club", "kind": "marker",
             "help": "Centre of the back of the hand or glove (e.g. the glove logo).",
             "models": {m: _m() for m in MODELS}},
            {"name": f"{side}_forearm_marker", "group": "hands_club", "kind": "marker",
             "help": "A physical marker worn on the forearm (tape, sleeve logo). Skip if none.",
             "models": {m: _m() for m in MODELS}},
        ]
    club_note = ("utils/club_detector.py fits a shaft line by edge detection; that is not a "
                 "keypoint, and it is unvalidated")
    for name, help_ in (
            ("shaft_butt", "Butt end of the grip."),
            ("shaft_hosel", "Where the shaft enters the hosel."),
            ("clubhead_heel", "Heel end of the leading edge."),
            ("clubhead_toe", "Toe end of the leading edge."),
            ("clubface_center", "Centre of the clubface. Only when the face is visible.")):
        out.append({"name": name, "group": "hands_club", "kind": "equipment",
                    "help": help_ + " Skip when motion blur makes it a guess.",
                    "models": {m: _m(note=club_note) for m in MODELS}})
    return out


POINTS = tuple(_body() + _feet() + _hands_club())
POINT_NAMES = tuple(p["name"] for p in POINTS)
POINT = {p["name"]: p for p in POINTS}
GROUP_POINTS = {g: tuple(p["name"] for p in POINTS if p["group"] == g) for g in GROUPS}
# The original benchmark: only these are scored against MediaPipe and RTMPose.
BENCHMARK_POINTS = GROUP_POINTS["body"]


def catalogue():
    """JSON-ready catalogue for the manifest and the labeller."""
    return {"groups": [{"id": g, "title": GROUP_TITLES[g], "points": list(GROUP_POINTS[g])}
                       for g in GROUPS],
            "points": [dict(p) for p in POINTS]}
