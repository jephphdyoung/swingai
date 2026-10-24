"""Reference Editor tab: catalog a video, cut it into shots, crop, and label.

Works on the swing dataset (`utils/catalog.py`, `utils/shot_annotations.py`):
pick a golfer and a registered source video (or register one in place), step
through its exact decoded frames (`utils/frame_access.py`), mark each swing
as a shot with its own start/end window and crop, and label P1-P10 and body,
club and ball points on source frames. Nothing here writes to a video file;
every edit is metadata saved under data/.
"""

import os

import pandas as pd
import streamlit as st

from frame_canvas_component import frame_canvas
from utils import catalog as cm
from utils import setup_measurements as sm
from utils import shot_annotations as sa
from utils import shot_edits as se
from utils.annotations import (P_DEFINITIONS, P_POSITIONS, convention_warnings,
                               load_annotation_record)
from utils.frame_access import FrameAccessError, get_frame_index, read_frame
from utils.paths import get_reference_dir, get_user_dir, list_videos

NEW = "__new__"
VIEW_LABELS = {"face_on": "Face-on", "down_the_line": "Down the line", "overhead": "Overhead",
               "impact": "Impact", "unknown": "Unknown"}
GATE_LABELS = {"stereo_3d": "Same-swing multi-view 3D", "motion_timing": "Velocity / timing",
               "body_landmarks": "Body landmarks", "club_landmarks": "Club landmarks",
               "resolution": "Resolution", "human_review": "Human review"}
READINESS = {"observable_2d": "2D observable", "missing_points": "Points missing",
             "no_p1": "P1 not marked", "not_this_view": "Other view"}


# --- small helpers ------------------------------------------------------------

def _flash(message, **select):
    """Rerun with a success message; `select` sets selections before widgets render."""
    st.session_state["re_flash"] = message
    for k, v in select.items():
        st.session_state[f"re_select_{k}"] = v
    st.rerun()


def _attempt(action, success, **select):
    try:
        action()
    except cm.CatalogError as e:
        st.error("Not saved:\n\n" + "\n".join(f"- {p}" for p in e.problems))
        return
    except cm.ConflictError as e:
        st.error(f"Not saved: {e}")
        return
    except FrameAccessError as e:
        st.error(f"Not saved: {e}")
        return
    _flash(success, **select)


def _apply_selections():
    for k in ("golfer", "source", "shot", "frame"):
        v = st.session_state.pop(f"re_select_{k}", None)
        if v is not None:
            st.session_state[f"re_{k}"] = v


def _operator():
    return (st.session_state.get("re_operator") or "").strip() or None


def _need_operator(what):
    if _operator() is None:
        st.warning(f"Enter your name above to {what}.")
        return True
    return False


def _set_frame(f):
    st.session_state["re_frame"] = int(f)


def _step(delta, lo, hi):
    st.session_state["re_frame"] = max(lo, min(hi, st.session_state.get("re_frame", lo) + delta))


def _date_or_none(text):
    text = (text or "").strip()
    return text or None


# --- sections -----------------------------------------------------------------

def _golfer_section(cat):
    with st.expander("Add a golfer", expanded=not cat.golfers):
        with st.form("re_add_golfer", clear_on_submit=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            name = c1.text_input("Name", key="re_golfer_name")
            role = c2.selectbox("Role", cm.ROLES, format_func=str.title, key="re_golfer_role",
                                help="Reference golfers' shots are shared and can be curated "
                                     "into collections; a user's shots are their own history.")
            hand = c3.selectbox("Handedness", cm.HANDEDNESS, format_func=str.title)
            if st.form_submit_button("Add golfer", key="re_golfer_add"):
                def add():
                    if not name.strip():
                        raise cm.CatalogError("enter a name")
                    rec = cm.add_golfer(cat, name.strip(), role, hand)
                    cm.save_catalog(cat)
                    st.session_state["re_select_golfer"] = rec["golfer_id"]
                _attempt(add, f"Added {name}")
    if not cat.golfers:
        return None
    ids = list(cat.golfers)
    if st.session_state.get("re_golfer") not in ids:
        st.session_state["re_golfer"] = ids[0]
    return st.selectbox("Golfer", ids, key="re_golfer",
                        format_func=lambda g: f"{cat.golfers[g]['display_name']} "
                                              f"({cat.golfers[g]['role']})")


def _registered_paths(cat):
    out = set()
    for s in cat.sources.values():
        for loc in [s["location"]] + s.get("also_found_at", []):
            out.add(os.path.abspath(cm.resolve_location(loc)))
    return out


def _register_form(cat, golfer_id):
    known = _registered_paths(cat)
    candidates = {"": ""}
    for root, base in (("reference", get_reference_dir()), ("user", get_user_dir())):
        for rel in list_videos(base):
            p = os.path.abspath(os.path.join(base, rel))
            if p not in known:
                candidates[p] = f"{root}/{rel}"
    with st.form("re_register"):
        st.caption("Registering reads the file (checksum and a sequential frame index) and "
                   "never moves, copies or changes it.")
        pick = st.selectbox("Unregistered video in the libraries", list(candidates),
                            format_func=candidates.get, key="re_reg_pick")
        other = st.text_input("…or a path to a video elsewhere", key="re_reg_path",
                              help="Absolute paths work on this machine; inside the container "
                                   "only the mounted libraries are visible.")
        c1, c2, c3 = st.columns(3)
        view = c1.selectbox("Camera view of the file", cm.CAMERA_VIEWS,
                            format_func=VIEW_LABELS.get, key="re_reg_view",
                            help="Unknown for compilations; each shot states its own view.")
        stype = c2.selectbox("Source type", cm.SOURCE_TYPES, key="re_reg_type")
        date = c3.text_input("Recording date (YYYY, YYYY-MM or YYYY-MM-DD)", key="re_reg_date")
        c1, c2 = st.columns(2)
        url = c1.text_input("URL (if known)")
        title = c2.text_input("Original title (if known)")
        c1, c2, c3 = st.columns(3)
        motion = c1.selectbox("Is playback time the real swing time?", cm.MOTION_TIME,
                              index=cm.MOTION_TIME.index("unknown"),
                              help="YouTube playback fps says nothing about capture timing. "
                                   "Unknown keeps this video out of velocity/timing data.")
        cap_fps = c2.number_input("Capture fps (if known)", min_value=0.0, value=0.0)
        basis = c3.text_input("Timing basis", help="Why the timing is believed.")
        notes = st.text_input("Provenance notes")
        if st.form_submit_button("Register video", key="re_reg_submit"):
            path = other.strip() or pick

            def register():
                if not path:
                    raise cm.CatalogError("choose a video or enter a path")
                with st.spinner("Checksumming and indexing every frame…"):
                    rec, created = cm.register_source(
                        cat, path, golfer_id=golfer_id, camera_view=view, source_type=stype,
                        url=url.strip() or None, title=title.strip() or None,
                        provenance_notes=notes.strip() or None,
                        recording_date=_date_or_none(date),
                        date_basis="entered" if _date_or_none(date) else None,
                        motion_time=motion, capture_fps=cap_fps or None,
                        timing_basis=basis.strip() or None)
                if not created and rec["golfer_id"] != golfer_id:
                    raise cm.CatalogError(f"these bytes are already registered as "
                                          f"{rec['source_id']} for {rec['golfer_id']}")
                cm.save_catalog(cat)
                st.session_state["re_select_source"] = rec["source_id"]
            _attempt(register, f"Registered {os.path.basename(path)}")


def _source_label(cat, sid):
    s = cat.sources[sid]
    n = len(cm.shots_for_source(cat, sid))
    date = s["recording"]["date"] or "date unknown"
    return (f"{os.path.basename(s['location']['path'])} · "
            f"{VIEW_LABELS[s['recording']['camera_view']]} · {date} · {n} shot(s)")


def _source_section(cat, golfer_id):
    with st.expander("Register a video"):
        _register_form(cat, golfer_id)
    ids = [sid for sid, s in cat.sources.items() if s["golfer_id"] == golfer_id]
    if not ids:
        st.info("No videos registered for this golfer yet.")
        return None
    if st.session_state.get("re_source") not in ids:
        st.session_state["re_source"] = ids[0]
    return st.selectbox("Source video", ids, key="re_source",
                        format_func=lambda s: _source_label(cat, s))


def _source_details(cat, src, status, index):
    v, t, prov, rec = src["video"], src["timing"], src["provenance"], src["recording"]
    fps = (f"{v['nominal_fps']:.3f} nominal / {v['average_fps']:.3f} average fps playback"
           if v["average_fps"] else f"{v['nominal_fps']:.3f} fps")
    st.caption(
        f"`{src['source_id']}` · {v['width']}×{v['height']} · {v['frame_count']} decoded frames"
        f"{' (container says ' + str(v['reported_frame_count']) + ')' if v['reported_frame_count'] != v['frame_count'] else ''}"
        f" · {fps}{' · **variable frame rate**' if v['variable_frame_rate'] else ''} · "
        f"sha256 `{src['sha256'][:12]}…`")
    timing = {"real_time": "playback is real time", "slowed": "slowed playback",
              "retimed": "retimed / edited timing", "unknown": "motion timing unknown"}
    st.caption(f"Timing: {timing[t['motion_time']]}"
               + (f", captured at {t['capture_fps']:g} fps" if t["capture_fps"] else "")
               + (f" ({t['basis']})" if t["basis"] else "")
               + ". Playback fps is not used as real swing time unless stated."
               + (f" · Found at a recorded copy: {status.detail}" if status.state == "relocated"
                  else ""))
    with st.expander("Source metadata"):
        with st.form("re_source_meta"):
            c1, c2, c3 = st.columns(3)
            stype = c1.selectbox("Source type", cm.SOURCE_TYPES,
                                 index=cm.SOURCE_TYPES.index(prov["source_type"]))
            url = c2.text_input("URL", prov["url"] or "")
            title = c3.text_input("Title", prov["title"] or "")
            c1, c2, c3 = st.columns(3)
            view = c1.selectbox("Camera view of the file", cm.CAMERA_VIEWS,
                                index=cm.CAMERA_VIEWS.index(rec["camera_view"]),
                                format_func=VIEW_LABELS.get)
            date = c2.text_input("Recording date", rec["date"] or "")
            basis = c3.text_input("Date basis", rec["date_basis"] or "")
            c1, c2, c3 = st.columns(3)
            motion = c1.selectbox("Motion timing", cm.MOTION_TIME,
                                  index=cm.MOTION_TIME.index(t["motion_time"]))
            cap = c2.number_input("Capture fps", min_value=0.0, value=float(t["capture_fps"] or 0))
            tbasis = c3.text_input("Timing basis", t["basis"] or "")
            pnotes = st.text_input("Provenance notes", prov["notes"] or "")
            rnotes = st.text_input("Recording notes", rec["notes"] or "")
            if st.form_submit_button("Save source metadata"):
                def save():
                    cm.update_source(
                        cat, src["source_id"],
                        provenance={"source_type": stype, "url": url.strip() or None,
                                    "title": title.strip() or None,
                                    "notes": pnotes.strip() or None},
                        recording={"camera_view": view, "date": _date_or_none(date),
                                   "date_basis": basis.strip() or None,
                                   "notes": rnotes.strip() or None},
                        timing={"motion_time": motion, "capture_fps": cap or None,
                                "basis": tbasis.strip() or None})
                    cm.save_catalog(cat)
                _attempt(save, "Saved source metadata")
    if src.get("legacy_annotation_key"):
        legacy = load_annotation_record(status.path)
        if legacy is not None and legacy.frames:
            marks = ", ".join(f"{p} {legacy.frames[p]}" for p in P_POSITIONS
                              if p in legacy.frames)
            st.info(f"Legacy P-positions in data/annotations.json (shown for reference, "
                    f"**not imported**): {marks}. They were marked on seeked images, which "
                    "can be one frame early on high-fps files, and may predate the P7 = "
                    "impact convention. Mark positions for a shot below.")
            for w in convention_warnings(legacy):
                st.caption(f"⚠ {w}")


def _shot_section(cat, source_id):
    shots = cm.shots_for_source(cat, source_id)
    options = [s["shot_id"] for s in shots] + [NEW]
    if st.session_state.get("re_shot") not in options:
        st.session_state["re_shot"] = options[0]

    def label(sid):
        if sid == NEW:
            return "➕ New shot (one swing) from this video"
        s = cat.shots[sid]
        st_ = next(x for x in s["streams"] if x["source_id"] == source_id)
        club = s["club"]["name"] or s["club"]["category"]
        return (f"{sid} · {cm.SWING_TYPE_LABELS[s['swing_type']]} · {club} · frames "
                f"{st_['window']['start_frame']}–{st_['window']['end_frame']} · "
                f"{s['review']['status']}")
    choice = st.selectbox("Shot", options, key="re_shot", format_func=label)
    return None if choice == NEW else cat.shots[choice]


def _navigator(index, lo, hi):
    if st.session_state.get("re_frame") is None or not lo <= st.session_state["re_frame"] <= hi:
        st.session_state["re_frame"] = lo
    cols = st.columns(6)
    for col, (text, delta) in zip(cols, (("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10))):
        col.button(text, key=f"re_step_{delta}", on_click=_step, args=(delta, lo, hi),
                   use_container_width=True)
    cols[4].button("⏮ Start", key="re_to_lo", on_click=_set_frame, args=(lo,),
                   use_container_width=True)
    cols[5].button("End ⏭", key="re_to_hi", on_click=_set_frame, args=(hi,),
                   use_container_width=True)
    if hi > lo:
        st.slider("Source frame", lo, hi, key="re_frame")
    return st.session_state["re_frame"]


def _frame_caption(index, f, stream, doc):
    pts = index.presentation_ms(f)
    marks = [p for p, fr in sa.positions(doc, stream["stream_id"]).items() if fr == f] \
        if stream else []
    where = ""
    if stream:
        k = se.source_to_edit_frame(stream, f)
        where = (f" · swing frame {k} of {se.window_length(stream)}" if k is not None
                 else " · outside the swing window")
    st.caption(f"Source frame **{f}** / {index.frame_count - 1} · t = "
               f"{index.timestamps_ms[f]:.3f} ms"
               + (f" (stream PTS {pts:.3f} ms)" if pts is not None else "")
               + f" · image sha1 `{index.digests[f][:12]}`{where}"
               + (f" · **{', '.join(marks)}**" if marks else ""))


def _canvas_points(doc, stream, f):
    if stream is None:
        return []
    fl = sa.frame_labels(doc, stream["stream_id"], f) or {"points": {}}
    return [{"name": n, "x": p["x"], "y": p["y"], "visible": p["visible"],
             "origin": p["origin"]} for n, p in fl["points"].items()]


def _new_shot_form(cat, golfer_id, src, frame):
    st.markdown("**New shot** — one swing. The window is the swing's span in source "
                "frames; it is separate from P1 and P10, and the video file is untouched.")
    n = src["video"]["frame_count"]
    if st.session_state.get("re_new_for") != src["source_id"]:
        st.session_state.update(re_new_for=src["source_id"], re_new_start=0, re_new_end=n - 1)
    c1, c2 = st.columns(2)
    c1.button("Start = this frame", key="re_new_start_here",
              on_click=lambda: st.session_state.update(re_new_start=frame))
    c2.button("End = this frame", key="re_new_end_here",
              on_click=lambda: st.session_state.update(re_new_end=frame))
    golfer = cat.golfers[golfer_id]
    with st.form("re_new_shot"):
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Start frame", 0, n - 1, key="re_new_start")
        end = c2.number_input("End frame", 0, n - 1, key="re_new_end")
        view = c3.selectbox("Camera view of this swing", cm.CAMERA_VIEWS,
                            index=cm.CAMERA_VIEWS.index(src["recording"]["camera_view"]),
                            format_func=VIEW_LABELS.get, key="re_new_view")
        c1, c2, c3, c4 = st.columns(4)
        swing = c1.selectbox("Swing type", cm.SWING_TYPES, format_func=cm.SWING_TYPE_LABELS.get,
                             key="re_new_swing")
        cat_ = c2.selectbox("Club category", cm.CLUB_CATEGORIES,
                            index=cm.CLUB_CATEGORIES.index("unknown"), key="re_new_club_cat")
        club = c3.text_input("Club (e.g. 6-iron)", key="re_new_club")
        hand = c4.selectbox("Handedness", cm.HANDEDNESS,
                            index=cm.HANDEDNESS.index(golfer["handedness"]))
        c1, c2 = st.columns(2)
        date = c1.text_input("Recording date", src["recording"]["date"] or "")
        notes = c2.text_input("Setup notes (conditions, lie, stance…)")
        crop = st.session_state.get("re_pending_crop")
        if crop:
            st.caption(f"The crop drawn on the frame will be saved with the shot: {crop}")
        if st.form_submit_button("Create shot", type="primary", key="re_new_submit"):
            def create():
                if view == "unknown":
                    raise cm.CatalogError("choose the camera view of this swing")
                shot = cm.add_shot(cat, golfer_id=golfer_id, source_id=src["source_id"],
                                   camera_view=view, start_frame=start, end_frame=end,
                                   swing_type=swing, club={"category": cat_,
                                                           "name": club.strip() or None},
                                   handedness=hand, recording_date=_date_or_none(date),
                                   crop=crop, setup_notes=notes.strip() or None)
                cm.save_catalog(cat)
                st.session_state.pop("re_pending_crop", None)
                st.session_state["re_select_shot"] = shot["shot_id"]
            _attempt(create, "Created shot")


def _window_crop_tab(cat, shot, stream, src, frame):
    sid = shot["shot_id"]
    st.markdown(se.describe(stream, src["video"]["average_fps"]))
    c1, c2 = st.columns(2)
    if c1.button("Swing start = this frame", key="re_win_start"):
        _attempt(lambda: (cm.set_window(cat, sid, stream["stream_id"], frame,
                                        max(frame, stream["window"]["end_frame"])),
                          cm.save_catalog(cat)), f"Swing window now starts at {frame}")
    if c2.button("Swing end = this frame", key="re_win_end"):
        _attempt(lambda: (cm.set_window(cat, sid, stream["stream_id"],
                                        min(frame, stream["window"]["start_frame"]), frame),
                          cm.save_catalog(cat)), f"Swing window now ends at {frame}")
    pending = st.session_state.get("re_pending_crop")
    if pending:
        st.info(f"New crop drawn: {pending['width']}×{pending['height']} at "
                f"({pending['x']}, {pending['y']}).")
        c1, c2 = st.columns(2)
        if c1.button("Save this crop", type="primary", key="re_crop_save"):
            def save():
                cm.set_crop(cat, sid, stream["stream_id"], pending)
                cm.save_catalog(cat)
                st.session_state.pop("re_pending_crop", None)
            _attempt(save, "Saved crop")
        if c2.button("Discard", key="re_crop_discard"):
            st.session_state.pop("re_pending_crop", None)
            st.rerun()
    with st.form("re_crop_numeric"):
        c = stream.get("crop") or {"x": 0, "y": 0, "width": src["video"]["width"],
                                   "height": src["video"]["height"]}
        cols = st.columns(4)
        vals = {k: cols[i].number_input(k, min_value=0, value=int(c[k]), key=f"re_crop_{k}")
                for i, k in enumerate(("x", "y", "width", "height"))}
        b1, b2 = st.columns(2)
        if b1.form_submit_button("Set crop", key="re_crop_set"):
            _attempt(lambda: (cm.set_crop(cat, sid, stream["stream_id"], vals),
                              cm.save_catalog(cat)), "Saved crop")
        if b2.form_submit_button("Clear crop (full frame)", key="re_crop_clear"):
            _attempt(lambda: (cm.set_crop(cat, sid, stream["stream_id"], None),
                              cm.save_catalog(cat)), "Crop cleared")
    st.caption("Crop and window are edit metadata: labels stay in source frames and pixels, "
               "so changing them never moves a label. Switch the canvas to “Draw crop” to drag "
               "a rectangle.")


def _positions_tab(cat, shot, stream, src, index, doc, frame):
    sid = stream["stream_id"]
    marks = sa.positions(doc, sid)
    st.caption("P6 = shaft parallel in the downswing, **P7 = impact**. Each mark records "
               "the exact source frame, its time and its image digest.")
    for p in P_POSITIONS:
        c1, c2, c3, c4 = st.columns([5, 2, 2, 2])
        cur = marks.get(p)
        c1.markdown(f"**{p}** {P_DEFINITIONS[p]}" + (f" — frame {cur}" if cur is not None
                                                   else " — —"))
        if c2.button("Set here", key=f"re_set_{p}", use_container_width=True):
            def set_p(p=p):
                sa.set_position(doc, shot, src, index, sid, p, frame, by=_operator())
                sa.save(doc, cat)
            _attempt(set_p, f"{p} = frame {frame}")
        c3.button("Go", key=f"re_go_{p}", disabled=cur is None, on_click=_set_frame,
                  args=(cur or 0,), use_container_width=True)
        if c4.button("Clear", key=f"re_clear_{p}", disabled=cur is None,
                     use_container_width=True):
            def clear(p=p):
                sa.clear_position(doc, sid, p)
                sa.save(doc, cat)
            _attempt(clear, f"Cleared {p}")
    for w in sa.window_warnings(doc, shot):
        st.warning(w)


def _labels_tab(cat, shot, stream, src, index, doc, frame, active, visible):
    sid = stream["stream_id"]
    fl = sa.frame_labels(doc, sid, frame)
    if fl:
        rows = [{"Point": n, "x": round(p["x"], 1), "y": round(p["y"], 1),
                 "Visible": p["visible"], "Origin": p["origin"], "By": p.get("set_by")}
                for n, p in fl["points"].items()]
        rows += [{"Point": n, "x": None, "y": None, "Visible": None,
                  "Origin": f"unavailable: {sa.UNAVAILABLE_LABELS[r]}", "By": None}
                 for n, r in fl["unavailable"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        rv = fl["review"]
        st.caption(f"Frame review: **{rv['status']}**"
                   + (f" by {rv['by']} at {rv['at']}" if rv.get("by") else ""))
    else:
        st.caption("No labels on this frame yet.")
    if active:
        c1, c2, c3 = st.columns(3)
        with c1.form("re_place_numeric"):
            x = st.number_input("x (source px)", 0.0, float(src["video"]["width"]), step=1.0,
                                key="re_px")
            y = st.number_input("y (source px)", 0.0, float(src["video"]["height"]), step=1.0,
                                key="re_py")
            if st.form_submit_button(f"Place {active}", key="re_place"):
                def place():
                    sa.set_point(doc, shot, src, index, sid, frame, active, x, y, visible,
                                 by=_operator())
                    sa.save(doc, cat)
                _attempt(place, f"Placed {active}")
        with c2.form("re_unavailable"):
            reason = st.selectbox("Can't place it because", sa.UNAVAILABLE_REASONS,
                                  format_func=sa.UNAVAILABLE_LABELS.get, key="re_reason")
            if st.form_submit_button(f"Mark {active} unavailable", key="re_mark_unavailable"):
                def unavailable():
                    sa.mark_unavailable(doc, shot, src, index, sid, frame, active, reason)
                    sa.save(doc, cat)
                _attempt(unavailable, f"{active}: {sa.UNAVAILABLE_LABELS[reason]}")
        if c3.button(f"Remove {active} here", key="re_remove_point"):
            def remove():
                sa.remove_point(doc, sid, frame, active)
                sa.save(doc, cat)
            _attempt(remove, f"Removed {active}")
        st.caption(sa.POINT[active].get("help", ""))
    if fl:
        c1, c2, c3 = st.columns(3)
        for col, status, text in ((c1, "reviewed", "Mark frame reviewed"),
                                  (c2, "rejected", "Reject frame"),
                                  (c3, "unreviewed", "Reset review")):
            if col.button(text, key=f"re_review_{status}"):
                if status != "unreviewed" and _need_operator("review"):
                    continue

                def review(status=status):
                    sa.set_frame_review(doc, sid, frame, status, _operator())
                    sa.save(doc, cat)
                _attempt(review, f"Frame {frame}: {status}")
    labelled = sorted(int(f) for f in (doc["streams"].get(sid) or {"frames": {}})["frames"])
    if labelled:
        st.caption("Labelled frames:")
        cols = st.columns(min(8, len(labelled)))
        for i, f in enumerate(labelled[:32]):
            cols[i % len(cols)].button(str(f), key=f"re_go_label_{f}", on_click=_set_frame,
                                       args=(f,), use_container_width=True)


def _shot_tab(cat, shot, stream, src):
    sid = shot["shot_id"]
    with st.form("re_shot_meta"):
        c1, c2, c3, c4 = st.columns(4)
        swing = c1.selectbox("Swing type", cm.SWING_TYPES,
                             index=cm.SWING_TYPES.index(shot["swing_type"]),
                             format_func=cm.SWING_TYPE_LABELS.get)
        cat_ = c2.selectbox("Club category", cm.CLUB_CATEGORIES,
                            index=cm.CLUB_CATEGORIES.index(shot["club"]["category"]))
        club = c3.text_input("Club", shot["club"]["name"] or "")
        length = c4.number_input("Club length (in)", min_value=0.0,
                                 value=float(shot["club"]["length_in"] or 0),
                                 help="The physical club, from its spec. Never taken from "
                                      "the image: apparent shaft length depends on the camera.")
        c1, c2, c3 = st.columns(3)
        lbasis = c1.text_input("Length basis", shot["club"]["length_basis"] or "")
        hand = c2.selectbox("Handedness", cm.HANDEDNESS,
                            index=cm.HANDEDNESS.index(shot["handedness"]))
        date = c3.text_input("Recording date", shot["recording_date"] or "")
        setup = st.text_input("Setup notes", shot["setup_notes"] or "")
        if st.form_submit_button("Save shot details"):
            _attempt(lambda: (cm.update_shot(
                cat, sid, swing_type=swing, handedness=hand, recording_date=_date_or_none(date),
                setup_notes=setup.strip() or None,
                club={"category": cat_, "name": club.strip() or None,
                      "length_in": length or None, "length_basis": lbasis.strip() or None}),
                cm.save_catalog(cat)), "Saved shot details")
    q = stream["quality"]
    with st.form("re_quality"):
        st.markdown(f"**Quality of this view** (`{stream['stream_id']}`)")
        c1, c2, c3 = st.columns(3)
        body = c1.selectbox("Body visibility", cm.VISIBILITY,
                            index=cm.VISIBILITY.index(q["body_visibility"]))
        clubv = c2.selectbox("Club visibility", cm.VISIBILITY,
                             index=cm.VISIBILITY.index(q["club_visibility"]))
        res = c3.selectbox("Resolution usable", ("unassessed", "yes", "no"),
                           index=("unassessed", "yes", "no").index(q["resolution_usable"]))
        qnotes = st.text_input("Quality notes", q["notes"] or "")
        if st.form_submit_button("Save quality"):
            _attempt(lambda: (cm.set_stream_quality(
                cat, sid, stream["stream_id"], body_visibility=body, club_visibility=clubv,
                resolution_usable=res, notes=qnotes.strip() or None),
                cm.save_catalog(cat)), "Saved quality")
    rv = shot["review"]
    st.markdown(f"**Shot review:** {rv['status']}" + (f" by {rv['by']}" if rv.get("by") else ""))
    c1, c2, c3 = st.columns(3)
    for col, status, text in ((c1, "reviewed", "Mark shot reviewed"),
                              (c2, "rejected", "Reject shot"), (c3, "unreviewed", "Reset")):
        if col.button(text, key=f"re_shot_review_{status}"):
            if status != "unreviewed" and _need_operator("review"):
                continue
            _attempt(lambda status=status: (cm.set_review(cat, sid, status, _operator()),
                                            cm.save_catalog(cat)), f"Shot {status}")
    _multi_view(cat, shot, stream, src)
    with st.expander("Remove this shot"):
        st.caption("Removes the shot record only (never the video). Refused while the shot is "
                   "in a collection or has saved labels.")
        if st.button("Remove shot", key="re_remove_shot"):
            _attempt(lambda: (cm.remove_shot(cat, sid), cm.save_catalog(cat)),
                     f"Removed {sid}", shot=NEW)


def _multi_view(cat, shot, stream, src):
    sid = shot["shot_id"]
    mv = shot["multi_view"]
    st.markdown("**Views of this swing**")
    for s in shot["streams"]:
        other = cat.sources[s["source_id"]]
        st.caption(f"• `{s['stream_id']}` {VIEW_LABELS[s['camera_view']]} — "
                   f"{os.path.basename(other['location']['path'])}, frames "
                   f"{s['window']['start_frame']}–{s['window']['end_frame']}"
                   + ("" if s is stream else " (select that source above to edit it)"))
    if len(shot["streams"]) > 1:
        st.caption(f"Same swing: **{mv['same_swing']['status']}** "
                   f"({mv['same_swing'].get('method') or 'no basis'}) · synchronization: "
                   f"**{mv['synchronization']['status']}** · calibration: "
                   f"**{mv['calibration']['status']}**")
        with st.form("re_mv_status"):
            c1, c2, c3 = st.columns(3)
            which = c1.selectbox("Status", ("same_swing", "synchronization", "calibration"))
            status = c2.selectbox("Set to", ("unknown", "estimated", "verified", "asserted"))
            method = c3.text_input("Method / evidence")
            if st.form_submit_button("Save status"):
                if not _need_operator("record a status"):
                    _attempt(lambda: (cm.set_multi_view(cat, sid, which, status,
                                                        method.strip() or None, _operator()),
                                      cm.save_catalog(cat)), f"{which}: {status}")
    else:
        st.caption("One camera view: an independent 2D observation. Independent recordings "
                   "of other swings are separate shots (group them in a collection).")
    others = [s for s_id, s in cat.sources.items()
              if s["golfer_id"] == shot["golfer_id"]
              and s_id not in {x["source_id"] for x in shot["streams"]}]
    if others:
        with st.expander("Add another camera view of this same swing"):
            st.caption("Only for a recording of **this** swing (e.g. the co-triggered "
                       "partner view). Synchronization and calibration start unknown.")
            with st.form("re_add_stream"):
                pick = st.selectbox("Source", [s["source_id"] for s in others],
                                    format_func=lambda s: _source_label(cat, s))
                c1, c2, c3 = st.columns(3)
                view = c1.selectbox("View", cm.CAMERA_VIEWS, format_func=VIEW_LABELS.get)
                start = c2.number_input("Start frame", min_value=0, value=0)
                end = c3.number_input("End frame", min_value=0, value=0)
                basis = st.text_input("Why is it the same swing?",
                                      placeholder="co-triggered Swing Catalyst pair")
                if st.form_submit_button("Add view"):
                    _attempt(lambda: (cm.add_stream(cat, sid, source_id=pick, camera_view=view,
                                                    start_frame=start, end_frame=end,
                                                    same_swing_method=basis,
                                                    asserted_by=_operator()),
                                      cm.save_catalog(cat)), "Added view")


def _collections_tab(cat, shot):
    golfer = cat.golfers[shot["golfer_id"]]
    if golfer["role"] != "reference":
        st.caption("This is a user's shot: it stays in their own recording history. "
                   "Collections hold reference swings.")
        return
    sid = shot["shot_id"]
    mine = [c for c in cat.collections.values() if c["golfer_id"] == golfer["golfer_id"]]
    for c in mine:
        c1, c2, c3 = st.columns([5, 2, 2])
        member = sid in c["shot_ids"]
        c1.markdown(f"**{c['collection_id']}** [{c['status']}] — {len(c['shot_ids'])} shot(s)"
                    + (" · contains this shot" if member else ""))
        if c["status"] == "draft":
            if c2.button("Remove" if member else "Add", key=f"re_coll_{c['collection_id']}"):
                fn = cm.remove_from_collection if member else cm.add_to_collection
                _attempt(lambda c=c, fn=fn: (fn(cat, c["collection_id"], sid),
                                             cm.save_catalog(cat)),
                         f"{'Removed from' if member else 'Added to'} {c['collection_id']}")
            if c3.button("Freeze", key=f"re_freeze_{c['collection_id']}",
                         help="A frozen collection is a fixed model input and never changes; "
                              "later edits go in a new collection id."):
                if not _need_operator("freeze a collection"):
                    _attempt(lambda c=c: (cm.freeze_collection(cat, c["collection_id"],
                                                               _operator()),
                                          cm.save_catalog(cat)), f"Froze {c['collection_id']}")
        else:
            c2.caption(f"frozen by {c['frozen_by']}")
    with st.form("re_new_collection", clear_on_submit=True):
        c1, c2 = st.columns(2)
        cid = c1.text_input("New collection id", placeholder=f"{golfer['golfer_id']}-full-swing-v2")
        title = c2.text_input("Title")
        types = st.multiselect("Swing types", cm.SWING_TYPES, default=["full"],
                               format_func=cm.SWING_TYPE_LABELS.get)
        if st.form_submit_button("Create collection"):
            _attempt(lambda: (cm.create_collection(cat, cid.strip(), golfer["golfer_id"],
                                                   title.strip() or cid.strip(), None, types),
                              cm.save_catalog(cat)), f"Created {cid}")


def _readiness_tab(cat, shot, stream, doc):
    gates = cm.eligibility(cat, shot["shot_id"])
    st.markdown("**What later stages may use this shot for** — gates from recorded "
                "statuses, not a score. Unknown or unassessed is never eligible.")
    st.dataframe(pd.DataFrame([{"Use": GATE_LABELS[k], "Eligible": "yes" if g["eligible"]
                                else "no", "Why not": "; ".join(g["reasons"])}
                               for k, g in gates.items()]), hide_index=True, width="stretch")
    st.markdown(f"**P1 setup measurements for this view** "
                f"({VIEW_LABELS[stream['camera_view']]}) — what the labels support; no values "
                "are computed yet.")
    rows = sm.setup_readiness(doc, shot, stream["stream_id"])
    st.dataframe(pd.DataFrame([{"Measurement": r["label"], "Status": READINESS[r["status"]],
                                "Missing": ", ".join(r["missing"]),
                                "Without calibration": r["uncalibrated"]}
                               for r in rows if r["status"] != "not_this_view"]),
                 hide_index=True, width="stretch")
    spec = sm.club_specification(shot)
    st.caption(f"Club specification (entered, never measured from the image): "
               f"{spec['name'] or spec['category']}, length "
               + (f"{spec['length_in']} in" if spec["length_known"] else "unknown"))


# --- the tab ------------------------------------------------------------------

def _handle_canvas(event, cat, shot, stream, src, index, doc, frame, visible):
    if not event or event.get("nonce") == st.session_state.get("re_last_canvas"):
        return
    st.session_state["re_last_canvas"] = event["nonce"]
    if event.get("frame") != frame:
        return          # an event from a frame no longer shown
    if event["kind"] == "crop":
        st.session_state["re_pending_crop"] = {k: int(event[k])
                                               for k in ("x", "y", "width", "height")}
        st.rerun()
    if event["kind"] == "point" and stream is not None:
        def place():
            sa.set_point(doc, shot, src, index, stream["stream_id"], frame, event["name"],
                         event["x"], event["y"], visible, by=_operator())
            sa.save(doc, cat)
        _attempt(place, f"Placed {event['name']} at ({event['x']:.0f}, {event['y']:.0f})")


def render_reference_editor_tab():
    flash = st.session_state.pop("re_flash", None)
    if flash:
        st.success(flash)
    _apply_selections()
    st.caption("Curate swings into the dataset: register a video in place, mark each swing "
               "as a shot (window, crop), and label P-positions and body / club / ball points "
               "on exact source frames. Source videos are never modified.")
    try:
        cat = cm.load_catalog()
    except cm.CatalogError as e:
        st.error("The catalog under data/catalog/ is invalid and was not loaded:\n\n"
                 + "\n".join(f"- {p}" for p in e.problems))
        return
    st.text_input("Your name (recorded on labels and reviews)", key="re_operator")

    golfer_id = _golfer_section(cat)
    if golfer_id is None:
        return
    source_id = _source_section(cat, golfer_id)
    if source_id is None:
        return
    src = cat.sources[source_id]
    status = cm.resolve_source(cat, source_id)
    if not status.usable:
        st.error(f"Source video unavailable ({status.state}): {status.detail}. Its catalog "
                 "entry and labels are kept; nothing was changed.")
        return
    try:
        with st.spinner("Indexing frames (once per video)…"):
            index = get_frame_index(status.path)
    except FrameAccessError as e:
        st.error(str(e))
        return
    if index.file_sha256 != src["sha256"]:
        st.error("The file's bytes differ from the registered source; refusing to label it.")
        return
    _source_details(cat, src, status, index)

    shot = _shot_section(cat, source_id)
    stream = None
    if shot is not None:
        stream = next(s for s in shot["streams"] if s["source_id"] == source_id)
    doc = sa.load(shot["shot_id"]) if shot else None
    if doc is not None and stream is not None:
        for problem in sa.identity_problems(doc, stream["stream_id"], index):
            st.error(f"Label identity: {problem}. It is shown as saved, not moved.")

    lo, hi = 0, index.frame_count - 1
    if stream is not None and st.toggle("Only the swing window", value=True,
                                        key="re_only_window"):
        lo, hi = se.window(stream)
    frame = _navigator(index, lo, hi)
    try:
        image = read_frame(status.path, frame).image
    except FrameAccessError as e:
        st.error(str(e))
        return

    c1, c2, c3, c4, c5 = st.columns([2, 2, 3, 2, 2])
    mode = c1.radio("Canvas", ("Place points", "Draw crop"), key="re_mode",
                    label_visibility="collapsed")
    active, visible = None, True
    if stream is not None and mode == "Place points":
        group = c2.selectbox("Group", sa.GROUPS, format_func=sa.GROUP_TITLES.get,
                             key="re_group", label_visibility="collapsed")
        active = c3.selectbox("Point", sa.GROUP_POINTS[group], key=f"re_point_{group}",
                              label_visibility="collapsed")
        visible = c4.checkbox("Visible", value=True, key="re_visible",
                              help="Untick to place a hidden point where you judge it to be; "
                                   "use “unavailable” when you can't judge.")
    crop = st.session_state.get("re_pending_crop") or (stream or {}).get("crop")
    zoom = c5.toggle("Zoom to crop", key="re_zoom", disabled=crop is None)
    event = frame_canvas(image, frame=frame, points=_canvas_points(doc, stream, frame),
                         crop=crop, view=crop if zoom and crop else None,
                         mode="crop" if mode == "Draw crop" else "point", active=active,
                         show_names=True, key="re_canvas")
    _frame_caption(index, frame, stream, doc or sa.empty("-"))
    _handle_canvas(event, cat, shot, stream, src, index, doc, frame, visible)

    if shot is None:
        _new_shot_form(cat, golfer_id, src, frame)
        return
    tabs = st.tabs(["Window & crop", "P-positions", "Labels", "Shot", "Collections",
                    "Readiness"])
    with tabs[0]:
        _window_crop_tab(cat, shot, stream, src, frame)
    with tabs[1]:
        _positions_tab(cat, shot, stream, src, index, doc, frame)
    with tabs[2]:
        _labels_tab(cat, shot, stream, src, index, doc, frame, active, visible)
    with tabs[3]:
        _shot_tab(cat, shot, stream, src)
    with tabs[4]:
        _collections_tab(cat, shot)
    with tabs[5]:
        _readiness_tab(cat, shot, stream, doc)
