"""P1 setup measurements: what each needs, and whether a stream has it yet.

This module computes **no measurement values**. It declares, for each setup
measurement the future comparison will report, which labelled points and
which camera view it needs, and what an uncalibrated 2D image can and can't
support. `setup_readiness` then says, for one shot stream, which measurements
its P1 labels already support.

The raw inputs stay separate: labelled points live in the shot's annotations
(`utils/shot_annotations.py`), the known club specification on the shot
(`club.length_in`, entered, never measured from the image). Derived values,
when they exist, will be computed from those and stored apart from them.

Two limits are recorded on purpose:

- **No real units without calibration.** Ball-to-foot distance or stance
  width in centimetres needs a calibrated camera or a known scale in the
  scene. Until then only ratios to the golfer's own body dimensions and
  image-plane angles are possible.
- **Apparent shaft length is not club length.** Foreshortening depends on
  camera geometry, so club length comes from the specification only.

Setup differences between the user and the reference are to be reported, not
normalized away.
"""

MEASUREMENTS = (
    {"id": "ball_position", "label": "Ball position relative to the feet / stance",
     "views": ("face_on",),
     "points": ("ball_center",),
     "any_of": (("left_ankle", "right_ankle"), ("left_heel", "right_heel"),
                ("left_shoe_toe_contact", "right_shoe_toe_contact")),
     "uncalibrated": "fraction of the way from the trail foot to the lead foot, along the "
                     "image's stance line",
     "calibrated": "distance from the lead heel in real units"},
    {"id": "ball_distance", "label": "Ball-to-golfer distance",
     "views": ("down_the_line",),
     "points": ("ball_center",),
     "any_of": (("left_shoe_toe_contact", "right_shoe_toe_contact"),
                ("left_big_toe", "right_big_toe")),
     "uncalibrated": "ratio to the golfer's own P1 torso length in the same image",
     "calibrated": "toe line to ball centre in real units (pixels are not reliable "
                   "distances: the ball and feet sit at different depths)"},
    {"id": "stance_width", "label": "Stance width",
     "views": ("face_on",),
     "points": (),
     "any_of": (("left_heel", "right_heel"), ("left_ankle", "right_ankle")),
     "uncalibrated": "ratio to shoulder width in the same image",
     "calibrated": "heel-to-heel distance in real units"},
    {"id": "spine_inclination", "label": "Spine inclination (forward bend)",
     "views": ("down_the_line",),
     "points": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
     "any_of": (("floor_a", "floor_b"),),
     "uncalibrated": "image-plane angle of the hip-centre to shoulder-centre line from the "
                     "floor line's perpendicular (camera roll is not assumed level)",
     "calibrated": "3D forward bend from vertical"},
    {"id": "hip_flexion", "label": "Hip flexion",
     "views": ("down_the_line",),
     "points": ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                "left_knee", "right_knee"),
     "any_of": (),
     "uncalibrated": "image-plane angle shoulder-hip-knee on the camera side",
     "calibrated": "3D hip flexion"},
    {"id": "knee_flexion", "label": "Knee flexion",
     "views": ("down_the_line",),
     "points": ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle"),
     "any_of": (),
     "uncalibrated": "image-plane angle hip-knee-ankle on the camera side",
     "calibrated": "3D knee flexion"},
    {"id": "hand_position", "label": "Hand position",
     "views": ("face_on", "down_the_line"),
     "points": ("left_wrist", "right_wrist"),
     "any_of": (("left_hip", "right_hip"),),
     "uncalibrated": "hands relative to the hips (face-on: forward of the lead thigh; DTL: "
                     "away from the body), in torso lengths",
     "calibrated": "hand position in a body-fixed 3D frame"},
    {"id": "shoulder_alignment", "label": "Shoulder alignment",
     "views": ("down_the_line",),
     "points": ("left_shoulder", "right_shoulder"),
     "any_of": (),
     "uncalibrated": "not measurable as open/closed from one uncalibrated view; the "
                     "image-plane shoulder line is recorded only",
     "calibrated": "shoulder line vs. target line, which needs the camera aligned to (or "
                   "calibrated against) the target line"},
    {"id": "pelvis_alignment", "label": "Pelvis alignment",
     "views": ("down_the_line",),
     "points": ("left_hip", "right_hip"),
     "any_of": (),
     "uncalibrated": "as shoulder alignment: the image-plane hip line only",
     "calibrated": "hip line vs. target line"},
    {"id": "shaft_orientation", "label": "Shaft orientation",
     "views": ("face_on", "down_the_line"),
     "points": ("shaft_butt", "shaft_hosel"),
     "any_of": (),
     "uncalibrated": "image-plane shaft angle (face-on: shaft lean; DTL: shaft plane angle)",
     "calibrated": "3D shaft direction"},
)
MEASUREMENT = {m["id"]: m for m in MEASUREMENTS}

def _have(labels, name):
    return labels is not None and name in labels.get("points", {})


def setup_readiness(doc, shot, stream_id):
    """[{id, label, status, missing, uncalibrated, calibrated}] for the stream.

    status: "not_this_view", "no_p1" (P1 not marked), "missing_points", or
    "observable_2d" (every needed point is labelled at P1; only the
    uncalibrated form can be derived). Nothing here is "calibrated" yet:
    no shot has a verified calibration in PR #5's scope.
    """
    stream = next(s for s in shot["streams"] if s["stream_id"] == stream_id)
    view = stream["camera_view"]
    sa = doc["streams"].get(stream_id) or {"positions": {}, "frames": {}}
    p1 = sa["positions"].get("P1")
    labels = sa["frames"].get(str(p1["frame"])) if p1 else None
    out = []
    for m in MEASUREMENTS:
        row = {"id": m["id"], "label": m["label"], "uncalibrated": m["uncalibrated"],
               "calibrated": m["calibrated"], "missing": []}
        if view not in m["views"]:
            row["status"] = "not_this_view"
        elif p1 is None:
            row["status"] = "no_p1"
        else:
            missing = [n for n in m["points"] if not _have(labels, n)]
            for group in m["any_of"]:
                if not all(_have(labels, n) for n in group):
                    missing.append(" + ".join(group))
            # any_of: one complete alternative is enough
            if m["any_of"] and any(all(_have(labels, n) for n in g) for g in m["any_of"]):
                missing = [x for x in missing if " + " not in x]
            row["missing"] = missing
            row["status"] = "missing_points" if missing else "observable_2d"
        out.append(row)
    return out


def club_specification(shot):
    """The entered club facts, kept apart from anything seen in the image."""
    c = shot["club"]
    return {"category": c["category"], "name": c.get("name"), "length_in": c.get("length_in"),
            "length_basis": c.get("length_basis"),
            "length_known": c.get("length_in") is not None}
