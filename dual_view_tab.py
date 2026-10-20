"""Dual-View Comparison tab: your DTL + face-on vs. a reference's, by P-position."""

import dataclasses
import os

import streamlit as st

from annotator import render_annotator
from utils.annotations import (
    P_DEFINITIONS, P_POSITIONS, confirm_annotations, convention_warnings, is_confirmed,
    load_annotation_record,
)
from utils.dual_view import (
    ROLE_LABELS, Slot, available_positions, build_comparison, probe_video, render_panel,
    rows_as_table,
)
from utils.joint_measurements import measurement_catalog
from utils.landmarks import VIEW_LABELS
from utils.paths import list_videos
from utils.pose_data import BACKEND_LABELS, get_pose_sequence, make_backend, video_identity
from utils.pose_rtmpose import RTMPoseUnavailable
from utils.swing_pairing import find_pairs

VIEWS = ("dtl", "face_on")
SLOT_KEYS = [(role, view) for view in VIEWS for role in ("user", "reference")]


def _guess_view(path):
    rec = load_annotation_record(path)
    if rec and rec.view_angle:
        return rec.view_angle
    name = os.path.basename(path).lower().replace(" ", "").replace("-", "").replace("_", "")
    if "dtl" in name or "downtheline" in name:
        return "dtl"
    if "faceon" in name or "face" in name:
        return "face_on"
    return None


def _default_reference(paths, view):
    """Prefer a clip annotated as this view, then one named for it."""
    for p in paths:
        rec = load_annotation_record(p)
        if rec and rec.view_angle == view and rec.frames:
            return p
    for p in paths:
        if _guess_view(p) == view:
            return p
    return paths[0]


def _marked(path):
    rec = load_annotation_record(path)
    return len(rec.frames) if rec else 0


@st.cache_data(show_spinner=False)
def _probe(path, identity):
    # `identity` (name, size, mtime) is part of the cache key: no leading
    # underscore, or Streamlit would skip it when hashing and serve stale info.
    return probe_video(path)


def _pose_store():
    return st.session_state.setdefault("dv_poses", {})


def _pose_key(path, backend_id):
    # The backend identity is part of the key: one backend's landmarks are
    # never served for another.
    ident = video_identity(path)
    return (backend_id, path, ident["size"], ident["mtime_ns"])


def _cached_pose(path, backend, backend_id):
    """From this session, else the disk cache. Never runs inference."""
    store = _pose_store()
    key = _pose_key(path, backend_id)
    if key not in store:
        seq = get_pose_sequence(path, backend, extract_if_missing=False)
        if seq is None:
            return None
        store[key] = seq
    return store[key]


def _extract(paths, backend, backend_id):
    store = _pose_store()
    for i, path in enumerate(paths, 1):
        label = os.path.basename(path)
        bar = st.progress(0.0, text=f"[{i}/{len(paths)}] {label}")

        def progress(done, total, bar=bar, label=label):
            if total and (done % 25 == 0 or done == total):
                bar.progress(min(done / total, 1.0), text=f"[{i}/{len(paths)}] {label}: "
                                                          f"frame {done}/{total}")
        store[_pose_key(path, backend_id)] = get_pose_sequence(path, backend, progress=progress)
        bar.progress(1.0, text=f"[{i}/{len(paths)}] {label}: done")


def _flash(message):
    """Rerun so everything above the annotator -- position counts, panels,
    tables -- is rebuilt from the saved annotations, and show `message` there."""
    st.session_state["dv_flash"] = message
    st.rerun()


def _step_position(delta, options):
    cur = st.session_state.get("dv_position", "P1")
    i = options.index(cur) if cur in options else 0
    st.session_state["dv_position"] = options[max(0, min(len(options) - 1, i + delta))]


def _fmt(v, unit):
    if v is None:
        return "—"
    return f"{v:.1f}°" if unit.startswith("deg") else f"{v:+.3f}"


def render_dual_view_tab(user_dir, reference_dir):
    flash = st.session_state.pop("dv_flash", None)
    if flash:
        st.success(flash)
    st.caption(
        "Compare your down-the-line and face-on views with a reference golfer's at "
        "each P-position. Positions come from saved annotations only; measurements "
        "are 2D image-plane projections from each camera.")

    pairs, unpaired = find_pairs(user_dir)
    ref_files = [os.path.join(reference_dir, f) for f in list_videos(reference_dir)]
    if not pairs:
        st.info(f"No Swing Catalyst DTL/face-on pairs found under `{user_dir}`. "
                "Point SWINGAI_USER_DIR at a capture folder (see docs/dual-view-comparison.md).")
        return
    if not ref_files:
        st.info(f"No reference videos found under `{reference_dir}`.")
        return

    # --- selection --------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("**Your swing**")
        pair_default = next((i for i, p in enumerate(pairs)
                             if _marked(p.dtl.path) or _marked(p.face_on.path)), 0)
        pair_idx = st.selectbox(
            "Recorded DTL / face-on pair", range(len(pairs)), index=pair_default,
            format_func=lambda i: (f"{pairs[i].captured_at} · {pairs[i].golfer} · "
                                   f"DTL {_marked(pairs[i].dtl.path)}/10 · "
                                   f"FO {_marked(pairs[i].face_on.path)}/10 P marked"),
            key="dv_pair")
        if unpaired:
            st.caption(f"{len(unpaired)} clip(s) without a partner view are not listed.")
        pair = pairs[pair_idx]
        rec = load_annotation_record(pair.dtl.path) or load_annotation_record(pair.face_on.path)
        user_hand = st.radio("Your handedness", ["right", "left"], horizontal=True,
                             index=1 if rec and rec.handedness == "left" else 0,
                             format_func=str.title, key="dv_user_hand")
    with right:
        st.markdown("**Reference swing**")
        names = {p: os.path.relpath(p, reference_dir) for p in ref_files}
        ref_dtl = st.selectbox("Reference DTL video", ref_files,
                               index=ref_files.index(_default_reference(ref_files, "dtl")),
                               format_func=lambda p: f"{names[p]} ({_marked(p)}/10 P)",
                               key="dv_ref_dtl")
        ref_fo = st.selectbox("Reference face-on video", ref_files,
                              index=ref_files.index(_default_reference(ref_files, "face_on")),
                              format_func=lambda p: f"{names[p]} ({_marked(p)}/10 P)",
                              key="dv_ref_fo")
        ref_hand = st.radio("Reference handedness", ["right", "left"], horizontal=True,
                            format_func=str.title, key="dv_ref_hand")

    backend_name = st.radio(
        "Pose model", list(BACKEND_LABELS), format_func=BACKEND_LABELS.get, horizontal=True,
        key="dv_backend",
        help="Each model's landmarks are cached separately; switching loads that model's "
             "cache and runs inference only for videos it hasn't processed.")
    backend = make_backend(backend_name)
    try:
        backend_id = backend.identity()
    except RTMPoseUnavailable as e:
        st.error(f"{BACKEND_LABELS[backend_name]} is not available here: {e}")
        st.info("Select MediaPipe to keep working on the CPU.")
        return

    paths = {("user", "dtl"): pair.dtl.path, ("user", "face_on"): pair.face_on.path,
             ("reference", "dtl"): ref_dtl, ("reference", "face_on"): ref_fo}
    hands = {"user": user_hand, "reference": ref_hand}

    # --- poses ------------------------------------------------------------
    unique_paths = list(dict.fromkeys(paths.values()))
    missing = [p for p in unique_paths if _cached_pose(p, backend, backend_id) is None]
    if missing:
        st.warning(
            f"{BACKEND_LABELS[backend_name]} has not processed {len(missing)} of these videos "
            "yet. Extraction runs once per video and model, then is cached (MediaPipe: ~15 s "
            "per 1,244-frame 240fps capture on CPU).")
        for p in missing:
            st.caption(f"• {os.path.basename(p)}")
        if st.button("Extract poses", type="primary", key="dv_extract"):
            try:
                _extract(missing, backend, backend_id)
            except RTMPoseUnavailable as e:
                st.error(f"{BACKEND_LABELS[backend_name]} failed: {e}")
                return
            st.rerun()

    slots = {}
    for key, path in paths.items():
        role, view = key
        pose = _cached_pose(path, backend, backend_id)
        if pose is not None:
            pose = dataclasses.replace(pose, view=view, handedness=hands[role])
        slots[key] = Slot(role=role, view=view, path=path,
                          info=_probe(path, tuple(video_identity(path).values())),
                          record=load_annotation_record(path), handedness=hands[role],
                          pose=pose)

    # --- controls ---------------------------------------------------------
    avail = {k: available_positions(s) for k, s in slots.items()}
    c1, c2, c3 = st.columns([6, 1, 1])
    with c2:
        st.button("◀ Prev", on_click=_step_position, args=(-1, P_POSITIONS),
                  use_container_width=True, key="dv_prev")
    with c3:
        st.button("Next ▶", on_click=_step_position, args=(1, P_POSITIONS),
                  use_container_width=True, key="dv_next")
    with c1:
        position = st.radio(
            "P-position", P_POSITIONS, horizontal=True, key="dv_position",
            format_func=lambda p: f"{p} ({sum(p in a for a in avail.values())}/4)")
    st.caption(f"**{position}** — {P_DEFINITIONS[position]}.  (n/4 = how many of the "
               "four videos have this position annotated)")

    o1, o2, o3 = st.columns([1, 2, 2])
    with o1:
        show_skeleton = st.toggle("Skeleton", value=True, key="dv_skeleton")
    with o2:
        catalog = measurement_catalog("face_on")
        highlight = st.selectbox(
            "Highlight", [None] + [k for k, _, _ in catalog], key="dv_highlight",
            format_func=lambda k: "None" if k is None else
            next(lbl for kk, lbl, _ in catalog if kk == k).split(" (+")[0])
    with o3:
        use_partner = st.checkbox(
            "Fill your missing positions from your other view", value=False,
            key="dv_partner",
            help="Your DTL and face-on clips are co-triggered, so a position marked on one "
                 "view can be shown on the other at the same timestamp. Off by default; "
                 "nothing is saved.")

    with st.expander("Advanced"):
        min_conf = st.slider(
            f"Minimum landmark confidence ({BACKEND_LABELS[backend_name]})", 0.0, 1.0,
            float(backend.min_confidence), 0.05, key=f"dv_min_conf_{backend_name}",
            help="Landmarks below this score are treated as not observed (drawn hollow grey; "
                 "their measurements are unavailable). Scores differ by model: MediaPipe "
                 "reports a visibility probability (default 0.5), RTMPose an uncalibrated "
                 "SimCC peak (default 0.3). From down-the-line the lead arm and knee are "
                 "often occluded.")
        st.caption(f"Pose model identity: `{backend_id}`")

    comparison = build_comparison(slots, position, use_partner_view=use_partner,
                                  min_confidence=min_conf)

    # --- four panels ------------------------------------------------------
    for view in VIEWS:
        st.subheader(VIEW_LABELS[view])
        cols = st.columns(2)
        for col, role in zip(cols, ("user", "reference")):
            res = comparison.slots[(role, view)]
            with col:
                st.markdown(f"**{ROLE_LABELS[role]}** · `{os.path.basename(res.slot.path)}`")
                img = render_panel(res, highlight, show_skeleton, min_confidence=min_conf)
                if img is None:
                    st.info(res.position.note or f"{position} not available.")
                else:
                    pf = res.position
                    fps = res.slot.pose.fps if res.slot.pose else res.slot.info.fps
                    st.image(img, width="stretch",
                             caption=f"frame {pf.frame} · {pf.timestamp_ms:.1f} ms · "
                                     f"{fps:.1f} fps · from {pf.source}")
                    if pf.note:
                        st.caption(pf.note)
                if res.slot.pose is None:
                    st.caption("Skeleton unavailable: pose not extracted.")
                for w in res.warnings:
                    st.warning(w)

    # --- measurement tables -----------------------------------------------
    st.subheader(f"Measurements at {position}")
    st.caption(
        "Difference = you − reference. Angles are projected onto each camera's image "
        "plane (2D), so DTL and face-on values are separate measurements and are never "
        "compared with each other. Displacements are from each video's own P1, in units "
        "of that video's P1 torso length. Elbow/knee flexion is 180° minus the 2D "
        "included angle (0° = straight). A difference is not by itself a swing fault.")
    for view in VIEWS:
        st.markdown(f"**{VIEW_LABELS[view]}**")
        rows = comparison.rows.get(view, [])
        st.dataframe(rows_as_table(rows), hide_index=True, width="stretch",
                     height=35 * (len(rows) + 1) + 3,
                     column_config={"Note": st.column_config.TextColumn(width="large")})

    # --- annotation correction --------------------------------------------
    st.subheader("Annotations")
    unconfirmed = [k for k in SLOT_KEYS
                   if slots[k].record and not is_confirmed(slots[k].record)]
    for k in unconfirmed:
        s = slots[k]
        a, b = st.columns([4, 1])
        a.caption(f"{s.label}: `{os.path.basename(s.path)}` is not confirmed — "
                  + "; ".join(convention_warnings(s.record)))
        if b.button("Confirm reviewed", key=f"dv_confirm_{k[0]}_{k[1]}",
                    help="Every saved position in this video has been checked by a person "
                         "against the definitions (P6 = shaft parallel, P7 = impact)."):
            confirm_annotations(s.path)
            _flash(f"Confirmed annotations for {os.path.basename(s.path)}")

    with st.expander("Add or correct P-positions", expanded=not all(avail.values())):
        target = st.radio("Video", SLOT_KEYS, horizontal=True, key="dv_annot_target",
                          format_func=lambda k: slots[k].label)
        s = slots[target]
        st.caption(f"`{s.path}`")
        render_annotator(s.path, key_prefix=f"dv_annot_{target[0]}_{target[1]}",
                         default_view_angle=target[1], on_save=_flash)
